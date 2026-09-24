"""Replay registered history MLPs and train their seed-matched history TabM."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json
from multimodal_baseline import set_seed
from task_aligned_world import TaskAlignedWorld
from numeric_embedding_readout import NeuralReadout, predict
from crop_signal_history_reference import IDENTITY
from run_crop_signal_screen import yearly_rmse, sample_rows

RESULT = ROOT / 'benchmark/results/soybean_expert_seed_recheck_v1'
LOGS = ROOT / 'benchmark/logs/soybean_expert_seed_recheck_v1'
ORIGINS = (2004, 2008, 2012)
SEEDS = (45, 48)


def run_root(origin, seed, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / f'origin_{origin}/seed_{seed}'


def mlp_root(origin, seed):
    return ROOT / f'benchmark/results/task_aligned_world_v1/pipelines/soybean/origin_{origin}/history_mlp/seed_{seed}'


@torch.no_grad()
def mlp_predict(model, x, baseline, norm):
    model.eval()
    parts = []
    for start in range(0, len(x), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = model.yield_head(x[start:start+256].cuda()).squeeze(-1)
        parts.append(value.float().cpu().numpy())
    residual = np.concatenate(parts).astype(float)
    return baseline.astype(float)+residual*norm['residual_std']+norm['residual_mean']


def load_inputs(origin, seed):
    registration = json.loads((LOGS / 'registration.json').read_text())
    record = registration['mlps'][f'{origin}_{seed}']
    for name, digest in record['files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Registered MLP/input artifact changed')
    src = mlp_root(origin, seed)
    config = json.loads((src / 'config.json').read_text())
    for name, digest in config['code_hashes'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError('MLP implementation changed')
    norm = config['input_manifest']['normalization']
    old = ROOT / f'benchmark/cache/neural_process_readout_v1/soybean/origin_{origin}'
    bin_root = ROOT / f'benchmark/cache/numeric_embedding_readout_v1/soybean/origin_{origin}'
    meta = json.loads((old / 'manifest.json').read_text())
    bm = json.loads((bin_root / 'manifest.json').read_text())
    bins_file = bin_root / 'history_bins.pt'
    if sha256(bins_file) != bm['files'][bins_file.name]:
        raise ValueError('Old training bins changed')
    bins = torch.load(bins_file, map_location='cpu', weights_only=True)
    model = TaskAlignedWorld('history_mlp').cuda()
    model.load_state_dict(torch.load(src / 'model_best.pt', map_location='cuda', weights_only=True))
    x, arrays, mlp_errors = {}, {}, {}
    source_files = {}
    for split in ('train', 'validation'):
        task_file = ROOT / f'benchmark/cache/task_aligned_world_v1/soybean/origin_{origin}/{split}.npz'
        with np.load(task_file) as f:
            a = {k: f[k] for k in (*IDENTITY, 'baseline')}
            history = torch.from_numpy(np.concatenate((f['history'], f['context']), 1))
        for name in (f'{split}_labels.npz', f'history_{split}.npy'):
            if sha256(old / name) != meta['files'][name]:
                raise ValueError('Old scaled history changed')
            source_files[str(old / name)] = sha256(old / name)
        with np.load(old / f'{split}_labels.npz') as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(a[k], f[k])
        signal = ROOT / f'benchmark/cache/crop_signal_screen_v1/soybean/origin_{origin}'
        sm = json.loads((signal / 'manifest.json').read_text())
        signal_file = signal / f'{split}_labels.npz'
        if sha256(signal_file) != sm['files'][signal_file.name]:
            raise ValueError('Signal cohort changed')
        source_files[str(signal_file)] = sha256(signal_file)
        with np.load(signal_file) as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(a[k], f[k])
        prediction = mlp_predict(model, history, a['baseline'], norm)
        replay = mlp_predict(model, history, a['baseline'], norm)
        np.testing.assert_array_equal(prediction, replay)
        if split == 'validation':
            with np.load(src / 'validation_predictions.npz') as f:
                for k in IDENTITY:
                    np.testing.assert_array_equal(a[k], f[k])
                mlp_errors[split] = float(np.max(np.abs(prediction-f['prediction'])))
                np.testing.assert_allclose(prediction, f['prediction'], rtol=0, atol=1e-7)
        a['mlp_prediction'] = prediction
        arrays[split] = a
        x[split] = torch.from_numpy(np.load(old / f'history_{split}.npy'))
    del model
    torch.cuda.empty_cache()
    return x, arrays, bins, norm['residual_std'], dict(mlp_source=record, files=source_files,
        bin_sha256=sha256(bins_file), mlp_validation_errors=mlp_errors)


def run(origin, seed, smoke=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = run_root(origin, seed, smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg_file = LOGS / 'registration.json'
        reg = json.loads(reg_file.read_text())
        for name, digest in reg['code_sha256'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise ValueError('Expert recipe changed')
        x, a, bins, scale, lineage = load_inputs(origin, seed)
        if smoke:
            for s in a:
                take = sample_rows(a[s], 2048 if s == 'train' else 1024)
                x[s] = x[s][take]
                a[s] = {k: v[take] for k, v in a[s].items()}
        spec = dict(origin=origin, seed=seed, smoke=smoke, lineage=lineage, residual_std=scale,
            registration_sha256=sha256(reg_file), evaluation_split_loaded=False,
            training_predictions_in_sample=True, calibration=False)
        file = root / 'config.json'
        if file.exists() and json.loads(file.read_text()) != spec:
            raise ValueError('Expert configuration changed')
        atomic_json(file, spec)
        if (root / 'audit.json').exists():
            prior = json.loads((root / 'audit.json').read_text())
            for name, digest in prior['files'].items():
                if sha256(root / name) != digest:
                    raise ValueError('Completed expert changed')
            return
        set_seed(seed)
        model = NeuralReadout('tabm', 20, bins).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, fused=True)
        target = torch.from_numpy(((a['train']['target'].astype(float)-a['train']['mlp_prediction'])/scale).astype(np.float32))
        n, trace, stale, best_score, best = len(target), [], 0, float('inf'), None
        started = time.monotonic()
        for epoch in range((2 if smoke else 60)+1):
            loss_sum = 0.
            if epoch:
                model.train()
                order = torch.randperm(n)
                for start in range(0, n, 1024):
                    ix = order[start:start+1024]
                    optimizer.zero_grad(set_to_none=True)
                    for offset in range(0, len(ix), 256):
                        take = ix[offset:offset+256]
                        value = model(x['train'][take].cuda())
                        loss = ((value-target[take, None].cuda())**2).mean()*len(take)/len(ix)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite TabM loss')
                        loss.backward()
                        loss_sum += float(loss.detach())*len(ix)
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
            validation = a['validation']['mlp_prediction']+scale*predict(model, x['validation'])
            score = float(np.sqrt(np.mean((a['validation']['target'].astype(float)-validation)**2)))
            if score < best_score-1e-8:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            trace.append(dict(epoch=epoch, training_mse=loss_sum/n, validation_rmse=score))
            pd.DataFrame(trace).to_csv(root / 'training_history.csv', index=False)
            print(f'[HISTORY TABM] {origin} {seed} epoch={epoch} rmse={score:.8f}', flush=True)
            if epoch and stale >= 8:
                break
        if best is None:
            raise RuntimeError('No finite expert checkpoint')
        weight = root / 'model_best.pt'
        torch.save(best, weight)
        model.load_state_dict(best)
        restored = NeuralReadout('tabm', 20, bins).cuda()
        restored.load_state_dict(torch.load(weight, map_location='cuda', weights_only=True))
        for split in ('train', 'validation'):
            component = predict(model, x[split])
            np.testing.assert_array_equal(component, predict(restored, x[split]))
            prediction = a[split]['mlp_prediction']+scale*component
            np.savez_compressed(root / f'{split}_labels.npz', history_prediction=prediction,
                tabm_component=component, **{k: a[split][k] for k in (*IDENTITY, 'mlp_prediction')})
            with np.load(root / f'{split}_labels.npz') as f:
                np.testing.assert_array_equal(prediction, f['history_prediction'])
                for key in IDENTITY:
                    np.testing.assert_array_equal(a[split][key], f[key])
        atomic_json(root / 'metrics.json', dict(origin=origin, seed=seed, selected_epoch=best_epoch,
            validation_rmse=best_score, n_training=n, weight=str(weight), seconds=time.monotonic()-started,
            annual_rmse=yearly_rmse(a['validation']['target'], prediction, a['validation']['year'])))
        atomic_json(root / 'audit.json', dict(fits=1, smoke=smoke, full_train_validation_replay=True,
            maximum_replay_error=0., reused_mlp_weights=True, freshly_replayed_mlp=True,
            evaluation_split_loaded=False, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            files={name: sha256(root / name) for name in ('config.json', 'metrics.json', 'model_best.pt',
                'training_history.csv', 'train_labels.npz', 'validation_labels.npz')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--origin', type=int, choices=ORIGINS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.origin, args.seed, args.smoke)
