"""Refit a retained historical family on each strictly earlier temporal prefix."""
import argparse
import fcntl
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from forward_anchor_readout import ROOT, CROPS, fold_root, fold_data
from task_aligned_world import TaskAlignedWorld
from stable_remote_models import build, fit
from audit_stable_remote import read_model
from multimodal_baseline import set_seed, regression_metrics
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json


@torch.no_grad()
def mlp_predict(model, x):
    model.eval(); values = []
    for start in range(0, len(x), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model(torch.from_numpy(x[start:start+256]).cuda()).squeeze(-1)
        values.append(y.float().cpu().numpy())
    return np.concatenate(values).astype(float)


def train_mlp(x, y, labels, norm, root, smoke):
    model = TaskAlignedWorld('history_mlp').yield_head.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
    xx, yy = torch.from_numpy(x['fit']), torch.from_numpy(y['fit'])
    n = len(yy); best = None; best_score = float('inf'); best_epoch = -1; stale = 0; rows = []
    for epoch in range(1, (2 if smoke else 50)+1):
        model.train(); order = torch.randperm(n); loss_total = 0.
        for start in range(0, n, 2048):
            indices = order[start:start+2048]; count = len(indices)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, count, 256):
                ix = indices[offset:offset+256]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(xx[ix].cuda()).squeeze(-1).float()
                    loss = (prediction-yy[ix].cuda()).square().mean()*len(ix)/count
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite forward historical loss')
                loss.backward(); loss_total += float(loss.detach())*count
            nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        prediction = physical(labels['validation'], mlp_predict(model, x['validation']), norm)
        score = regression_metrics(labels['validation']['target'], prediction)['rmse']
        if score < best_score-1e-6:
            best_score = score; best_epoch = epoch; stale = 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        rows.append(dict(epoch=epoch, training_mse=loss_total/n, validation_rmse=score))
        pd.DataFrame(rows).to_csv(root / 'training_history.csv', index=False)
        print(f'[FORWARD HISTORY] {root.parent.name} {root.name} {rows[-1]}', flush=True)
        if stale >= 8:
            break
    if best is None:
        raise RuntimeError('No historical checkpoint')
    model.load_state_dict(best); weight = root / 'model_best.pt'; torch.save(best, weight)
    return model, weight, best_epoch


def physical(labels, residual, norm):
    return labels['baseline'].astype(float)+residual*norm['residual_std']+norm['residual_mean']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--start', type=int, required=True); p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); set_seed(42)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = fold_root(args.crop, args.origin, args.start, args.smoke); root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        x, y, arrays, spec = fold_data(args.crop, args.origin, args.start, args.smoke)
        spec.update(seed=42, versions={k: importlib.metadata.version(k) for k in ('torch', 'lightgbm', 'catboost', 'scikit-learn')},
            mlp_optimization=dict(epochs=50, patience=8, lr=.0003, weight_decay=.0001, batch=2048, micro=256, precision='BF16', clip=1))
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Historical forward recipe changed')
        atomic_json(root / 'config.json', spec)
        if (root / 'audit.json').exists():
            return
        family = spec['family']; norm = spec['normalization']; started = time.monotonic()
        if family == 'history_mlp':
            model, weight, selected = train_mlp(x, y, arrays, norm, root, args.smoke)
            predictions = {s: physical(a, mlp_predict(model, x[s]), norm) for s, a in arrays.items()}
            restored = TaskAlignedWorld('history_mlp').yield_head.cuda()
            restored.load_state_dict(torch.load(weight, weights_only=True, map_location='cuda'))
            replay = {s: physical(a, mlp_predict(restored, x[s]), norm) for s, a in arrays.items()}
        else:
            model = build(family, 42, args.smoke)
            fit(model, family, dict(train=x['fit'], validation=x['validation']), dict(train=y['fit'], validation=y['validation']))
            if family == 'catboost':
                weight = root / 'model.cbm'; model.save_model(str(weight)); selected = int(model.tree_count_)
            elif family == 'xgb_smooth':
                model.set_params(device='cpu', callbacks=None); weight = root / 'model.ubj'; model.save_model(weight)
                selected = int(model.best_iteration)+1
            else:
                weight = root / 'model.joblib'; joblib.dump(model, weight)
                selected = int(model.n_iter_) if family.startswith('hgb') else int(model.best_iteration_)
            predictions = {s: physical(a, model.predict(x[s]).astype(float), norm) for s, a in arrays.items()}
            restored = read_model(family, str(weight))
            replay = {s: physical(a, restored.predict(x[s]).astype(float), norm) for s, a in arrays.items()}
        files = {}; scores = {}
        for split, a in arrays.items():
            np.testing.assert_array_equal(predictions[split], replay[split])
            if not np.isfinite(predictions[split]).all():
                raise FloatingPointError('Nonfinite historical forward prediction')
            file = root / f'{split}_predictions.npz'
            np.savez_compressed(file, prediction=predictions[split], **a)
            files[split] = sha256(file); scores[split] = regression_metrics(a['target'], predictions[split])
        if not (arrays['fit']['year'].max() < arrays['validation']['year'].min()
                and arrays['validation']['year'].max() < arrays['forward']['year'].min()):
            raise ValueError('Noncausal historical fold')
        rebuilt_x, rebuilt_y, rebuilt_arrays, rebuilt_spec = fold_data(args.crop, args.origin, args.start, args.smoke)
        for key in rebuilt_spec:
            if rebuilt_spec[key] != spec[key]:
                raise ValueError('Historical input specification replay failed')
        for split in arrays:
            np.testing.assert_array_equal(rebuilt_x[split], x[split])
            np.testing.assert_array_equal(rebuilt_y[split], y[split])
            for key in arrays[split]:
                np.testing.assert_array_equal(rebuilt_arrays[split][key], arrays[split][key])
        atomic_json(root / 'metrics.json', dict(family=family, selected_epoch_or_trees=selected,
            scores=scores, seconds=time.monotonic()-started, weight=str(weight), weight_sha256=sha256(weight)))
        atomic_json(root / 'audit.json', dict(fits=1, full_array_replay=True, maximum_replay_error=0.,
            temporal_order_verified=True, raw_causal_features_and_training_normalization_replayed=True,
            weight_name=weight.name, weight_sha256=sha256(weight), predictions=files))
        print(f'[FORWARD HISTORY COMPLETE] {args.crop} {args.origin} {args.start} {family}', flush=True)


if __name__ == '__main__':
    main()
