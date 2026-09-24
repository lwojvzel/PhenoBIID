"""Trained historical experts selected before each thirteen-year block."""
import argparse
import copy
import fcntl
import time

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from forecast_bridge_data import history_features
from inseason_13year_data import CACHE, load, partition
from inseason_nested_common import (ROOT, BLOCKS, RECIPES, LABELS, folder,
                                    hashes, verify, finish, register)
from multimodal_baseline import set_seed
from neural_process_readout import NeuralReadout
from review_revision_data import sha256
from run_inseason_direct_baselines import score
from run_review_revision_parallel import atomic_json
from task_aligned_world import TaskAlignedWorld

EXTRA = ('run_inseason_nested_history.py', 'task_aligned_world.py',
         'neural_process_readout.py', 'run_inseason_direct_baselines.py')


def make_model(kind):
    return (TaskAlignedWorld('history_mlp').yield_head if kind == 'mlp'
            else NeuralReadout('tabm', 20))


@torch.no_grad()
def predict(model, features):
    model.eval()
    result = []
    for start in range(0, len(features), 512):
        x = torch.as_tensor(features[start:start+512], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = model(x)
        result.append(value.float().mean(1).cpu().numpy())
    return np.concatenate(result).astype(float)


def fit_model(kind, x, target, other, other_target, years, seed, dest, smoke, epochs=None):
    set_seed(seed)
    model = make_model(kind).cuda()
    lr, batch, maximum = (.0003, 2048, 50) if kind == 'mlp' else (.001, 1024, 60)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    maximum = epochs if epochs is not None else (2 if smoke else maximum)
    batch = 256 if smoke else batch
    rng = np.random.default_rng(seed)
    best, selected, stale, weights, trace = float('inf'), 0, 0, None, []
    steps = 0
    for epoch in range(1, maximum+1):
        model.train()
        order = rng.permutation(len(x))
        summed = 0.
        for start in range(0, len(order), batch):
            chosen = order[start:start+batch]
            opt.zero_grad(set_to_none=True)
            for offset in range(0, len(chosen), 256):
                ix = chosen[offset:offset+256]
                features = torch.as_tensor(x[ix], device='cuda')
                truth = torch.as_tensor(target[ix], dtype=torch.float32, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    value = model(features).float()
                    loss = (value-truth[:, None]).square().mean()*len(ix)/len(chosen)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite nested history loss')
                loss.backward()
                summed += float(loss.detach())*len(chosen)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1. if kind == 'mlp' else 5.)
            opt.step()
            steps += 1
        row = dict(epoch=epoch, loss=summed/len(x), steps=steps)
        if epochs is None:
            error = score(other_target, predict(model, other), years)['mean_annual_rmse']
            row['inner_mean_annual_standardized_rmse'] = error
            if error < best:
                best, selected, stale = error, epoch, 0
                weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        trace.append(row)
        atomic_json(dest / 'training.json', trace)
        if epochs is None and stale >= 8:
            break
    if epochs is None:
        if selected < 1 or weights is None:
            raise ValueError('A trained checkpoint is required')
        model.load_state_dict(weights)
    else:
        selected = epochs
    torch.save(model.state_dict(), dest / 'model.pt')
    combined = np.concatenate((x, other))
    value = predict(model, combined)
    replay_model = make_model(kind).cuda()
    replay_model.load_state_dict(torch.load(dest / 'model.pt', map_location='cpu', weights_only=True))
    np.testing.assert_array_equal(value, predict(replay_model, combined))
    metadata = dict(selected_epochs=selected, optimizer_steps=steps,
        parameter_count=sum(p.numel() for p in model.parameters()),
        weight_replay=True, initialized_candidate_allowed=False)
    atomic_json(dest / 'training_audit.json', metadata)
    del model, replay_model
    torch.cuda.empty_cache()
    return value, metadata


def run(crop, cutoff, seed, smoke=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    dest = folder('history', crop, cutoff, seed, smoke)
    dest.mkdir(parents=True, exist_ok=True)
    code = hashes(EXTRA)
    with (dest / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest / 'complete.json').exists():
            verify(dest, code)
            return
        started = time.monotonic()
        raw, _ = load(crop)
        groups = partition(raw, cutoff)
        if smoke:
            groups = {s: np.concatenate([ix[raw['year'][ix] == y][:12]
                      for y in np.unique(raw['year'][ix])]) for s, ix in groups.items()}
        register(dest, dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke,
            cache_sha256=sha256(CACHE / crop / 'manifest.json'), code_sha256=code,
            partitions={s: np.unique(raw['year'][ix]).tolist() for s, ix in groups.items()},
            selection='Preceding-year annual RMSE; trained epochs only; fresh full-window refit',
            mlp='20-256-128-1 GELU dropout .1; lr .0003 wd .0001; 50 epochs patience 8',
            soybean_tabm='Residual on frozen MLP; k16 width128 blocks3; lr .001; 60 epochs patience8',
            history_training_predictions='In-sample; preserved as a declared original-head design'))
        chosen = {}
        for stage, fit_key, other_key in (('inner', 'inner_fit', 'inner_validation'),
                                         ('full', 'full_fit', 'evaluation')):
            root = dest / stage
            root.mkdir(exist_ok=True)
            fit, other = groups[fit_key], groups[other_key]
            take = np.concatenate((fit, other))
            norm = dict(target_mean=float(raw['target'][fit].mean()),
                        target_std=max(float(raw['target'][fit].std()), 1e-6))
            features, base = history_features(raw, take, norm)
            values = dict(trend=base.astype(float))
            for kind in (('mlp', 'tabm') if crop == 'soybean' else ('mlp',)):
                model_dir = root / kind
                model_dir.mkdir(exist_ok=True)
                anchor = values['trend' if kind == 'mlp' else 'mlp']
                residual = raw['target'][take].astype(float)-anchor
                center, scale = float(residual[:len(fit)].mean()), max(float(residual[:len(fit)].std()), 1e-6)
                x = features.copy()
                if kind == 'tabm':
                    scaler = StandardScaler().fit(x[:len(fit)])
                    x = scaler.transform(x).astype(np.float32)
                    joblib.dump(scaler, model_dir / 'scaler.joblib')
                target = (residual-center)/scale
                pred, audit = fit_model(kind, x[:len(fit)], target[:len(fit)], x[len(fit):],
                    target[len(fit):], raw['year'][other], seed, model_dir, smoke,
                    None if stage == 'inner' else chosen[kind])
                chosen[kind] = audit['selected_epochs']
                values[kind] = anchor+center+scale*pred
                atomic_json(model_dir / 'normalization.json', dict(history=norm, center=center, scale=scale))
            np.savez_compressed(root / 'predictions.npz', raw_indices=take, fit_count=len(fit),
                **values, **{k: raw[k][take] for k in LABELS})
            atomic_json(root / 'metrics.json', {k: score(raw['target'][other], v[len(fit):], raw['year'][other])
                for k, v in values.items()})
            print(f'[NESTED HISTORY] {crop} {cutoff} {stage} epochs={chosen}', flush=True)
        finish(dest, code, crop=crop, cutoff=cutoff, seed=seed, smoke=smoke,
               seconds=time.monotonic()-started, all_checkpoints_trained=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES), required=True)
    p.add_argument('--cutoff', choices=tuple(BLOCKS), type=int, required=True)
    p.add_argument('--seed', choices=(42, 45, 48), type=int, default=42)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.crop, args.cutoff, args.seed, args.smoke)
