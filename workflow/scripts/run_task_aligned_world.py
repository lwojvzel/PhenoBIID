"""Joint yield/state training with validation-only checkpoint selection."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from multimodal_baseline import regression_metrics, set_seed, write_csv
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from task_aligned_data import load
from task_aligned_world import INPUTS, VARIANTS, TaskAlignedWorld

RESULT = ROOT / 'benchmark/results/task_aligned_world_v1'
CODE = ('run_task_aligned_world.py', 'task_aligned_world.py', 'task_aligned_data.py',
        'biid_world_model.py', 'forward_protocol_revision.py', 'run_history_multimodal_baselines.py')
LABELS = ('target_residual', 'target_lai', 'target_lai_valid', 'lai_scale')


def tensors(a):
    return {k: torch.from_numpy(np.asarray(a[k])) for k in (*INPUTS, *LABELS)}


def batch(a, indices):
    return {k: v[indices].to('cuda', non_blocking=True) for k, v in a.items()}


@torch.no_grad()
def predict(model, data, size=512):
    model.eval(); yields, states = [], []
    n = len(data['history'])
    for start in range(0, n, size):
        b = batch(data, slice(start, start + size))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y, state = model({k: b[k] for k in INPUTS})
        yields.append(y.float().cpu().numpy()); states.append(state.float().cpu().numpy())
    return np.concatenate(yields), np.concatenate(states)


def physical_yield(a, residual, norm):
    return a['baseline'].astype(float) + residual.astype(float) * norm['residual_std'] + norm['residual_mean']


def state_rmse(a, prediction, norm):
    mask = (a['target_lai_valid'] > 0) & (a['relative_valid'] > 0)
    error = (a['target_lai'].astype(float) - prediction.astype(float))[mask] * norm['lai_std']
    return float(np.sqrt(np.mean(error ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), default=2012)
    p.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--micro-batch', type=int, default=256)
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    if min(args.epochs, args.patience, args.batch, args.micro_batch) < 1:
        p.error('Training budgets must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(3000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory)
    set_seed(args.seed)
    root = RESULT / ('smoke' if args.smoke else 'pipelines') / args.crop / f'origin_{args.origin}' / args.variant / f'seed_{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load(args.crop, args.origin)
        spec = dict(**vars(args), input_manifest=meta, code_hashes={f: sha256(ROOT / 'scripts' / f) for f in CODE},
                    torch_version=str(torch.__version__), optimizer='AdamW lr=0.0003 weight_decay=0.0001 clip_norm=1',
                    selection='Lowest validation yield RMSE among epochs with validation LAI RMSE no worse than previous LAI; if none qualify, retain unconstrained best and mark it ineligible. No test feedback.',
                    losses='Yield residual MSE + LAI MSE / training persistence MSE; forcing_joint adds 0.25 local-standardized LAI MSE / matching training persistence MSE',
                    task='Conditional future vegetation and annual yield; target-year weather supplied, no target-year observations fed back',
                    state_supervised=args.variant not in ('history_mlp', 'direct_gru'))
        config = root / 'config.json'
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError('Task-aligned experiment specification changed')
        atomic_json(config, spec)
        if (root / 'metrics.json').exists():
            return
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        data = {s: tensors(a) for s, a in arrays.items()}
        norm = meta['normalization']; scales = meta['loss_scales']
        model = TaskAlignedWorld(args.variant).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
        world = args.variant not in ('history_mlp', 'direct_gru')
        persistence = state_rmse(arrays['validation'], arrays['validation']['previous_lai'], norm)
        best = None; best_score = float('inf'); best_epoch = -1
        fallback = None; fallback_score = float('inf'); fallback_epoch = -1
        stale = 0; history = []; started = time.monotonic()
        n = len(arrays['train']['target'])
        for epoch in range(1, (2 if args.smoke else args.epochs) + 1):
            model.train(); epoch_started = time.monotonic(); total = 0.
            order = torch.randperm(n)
            for start in range(0, n, args.batch):
                indices = order[start:start + args.batch]; count = len(indices)
                full_mass = ((data['train']['target_lai_valid'][indices] > 0) &
                             (data['train']['relative_valid'][indices] > 0)).sum().clamp_min(1).item()
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, count, args.micro_batch):
                    ix = indices[offset:offset + args.micro_batch]
                    b = batch(data['train'], ix)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        y, state = model({k: b[k] for k in INPUTS})
                        loss_y = (y.float() - b['target_residual']).square().mean()
                        loss = loss_y * (len(ix) / count)
                        if world:
                            mask = (b['target_lai_valid'] > 0) & (b['relative_valid'] > 0)
                            error = state.float() - b['target_lai']
                            state_loss = (error.square() * mask).sum() / full_mass
                            loss = loss + state_loss / scales['lai']
                            if args.variant == 'forcing_joint':
                                local = ((error / b['lai_scale']).square() * mask).sum() / full_mass
                                loss = loss + .25 * local / scales['local']
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite joint objective')
                    loss.backward()
                    total += float(loss.detach()) * count
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            y, state = predict(model, data['validation'], args.micro_batch)
            score = regression_metrics(arrays['validation']['target'], physical_yield(arrays['validation'], y, norm))['rmse']
            lai_score = state_rmse(arrays['validation'], state, norm)
            eligible = not world or lai_score <= persistence
            fallback_improved = score < fallback_score - 1e-6
            if fallback_improved:
                fallback_score = score; fallback_epoch = epoch
                fallback = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = eligible and score < best_score - 1e-6
            if improved:
                best_score = score; best_epoch = epoch
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0 if improved or (best is None and fallback_improved) else stale + 1
            row = dict(epoch=epoch, loss=total / n, validation_yield_rmse=score, validation_lai_rmse=lai_score,
                       state_constraint_met=bool(eligible), elapsed_seconds=time.monotonic() - epoch_started)
            history.append(row)
            print(f'[JOINT] {args.crop} {args.variant} epoch={epoch} yield={score:.6f} lai={lai_score:.6f} eligible={eligible} seconds={row["elapsed_seconds"]:.1f}', flush=True)
            write_csv(history, root / 'training_history.csv')
            if stale >= args.patience:
                break
        selected_eligible = best is not None
        chosen = best if selected_eligible else fallback
        selected_epoch = best_epoch if selected_eligible else fallback_epoch
        if chosen is None:
            raise RuntimeError('No valid checkpoint')
        weight = root / 'model_best.pt'; torch.save(chosen, weight); model.load_state_dict(chosen)
        scores = {}; states = {}
        for split in ('validation', 'test'):
            y, prediction_lai = predict(model, data[split], args.micro_batch)
            a = arrays[split]; prediction = physical_yield(a, y, norm)
            scores[split] = regression_metrics(a['target'], prediction)
            states[split] = dict(rmse=state_rmse(a, prediction_lai, norm), persistence_rmse=state_rmse(a, a['previous_lai'], norm))
            np.savez_compressed(root / f'{split}_predictions.npz', prediction=prediction,
                prediction_lai=prediction_lai, **{k: a[k] for k in ('target', 'target_lai', 'target_lai_valid', 'baseline', 'source_indices', 'row', 'col', 'year')})
        atomic_json(root / 'metrics.json', dict(crop=args.crop, origin=args.origin, seed=args.seed, variant=args.variant,
                    scores=scores, state_scores=states, selected_epoch=selected_epoch, state_constraint_met=selected_eligible,
                    parameters=sum(p.numel() for p in model.parameters()), weight=str(weight), weight_sha256=sha256(weight),
                    seconds=time.monotonic() - started, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20))


if __name__ == '__main__':
    main()
