"""Fixed joint-tenth recipe across the registered seeds and temporal origins."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gru_task_priority import RecurrentWorld
from multimodal_baseline import regression_metrics, set_seed, write_csv
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from run_task_aligned_world import RESULT as PRETRAINED, tensors, batch, predict, physical_yield, state_rmse
from task_aligned_data import load
from task_aligned_world import INPUTS, TaskAlignedWorld

RESULT = ROOT / 'benchmark/results/gru_world_replication_v1'
CODE = ('run_gru_world_replication.py', 'gru_task_priority.py', 'task_aligned_world.py',
        'task_aligned_data.py', 'run_task_aligned_world.py')


def world_root(crop, origin, seed):
    if (origin, seed) == (2012, 42):
        return ROOT / 'benchmark/results/gru_task_priority_v1/pipelines' / crop / 'origin_2012/joint_tenth/seed_42'
    return RESULT / 'pipelines' / crop / f'origin_{origin}' / f'seed_{seed}'


def direct_root(crop, origin, seed):
    if (origin, seed) == (2012, 42):
        return ROOT / 'benchmark/results/gru_task_priority_v1/pipelines' / crop / 'origin_2012/direct_continued/seed_42'
    return RESULT / 'direct_controls' / crop / f'origin_{origin}' / f'seed_{seed}'


def initial_model(crop, origin, seed, direct=False):
    root = PRETRAINED / 'pipelines' / crop / f'origin_{origin}/direct_gru/seed_{seed}'
    record = json.loads((root / 'metrics.json').read_text())
    if sha256(Path(record['weight'])) != record['weight_sha256']:
        raise ValueError('Direct pretraining weight changed')
    source = TaskAlignedWorld('direct_gru')
    source.load_state_dict(torch.load(record['weight'], map_location='cpu', weights_only=True))
    if direct:
        model = source
    else:
        model = RecurrentWorld(); model.initialize_from_direct(source)
    return model, dict(weight=record['weight'], sha256=record['weight_sha256'],
                       config_sha256=sha256(root / 'config.json'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--seed', type=int, choices=(42, 45, 48), required=True)
    p.add_argument('--direct', action='store_true')
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    if (args.origin, args.seed) == (2012, 42) and not args.smoke:
        if not ((direct_root if args.direct else world_root)(args.crop, args.origin, args.seed) / 'metrics.json').exists():
            raise RuntimeError('Expected reusable pilot')
        return
    root = (RESULT / ('smoke_direct' if args.direct else 'smoke') / args.crop / f'origin_{args.origin}/seed_{args.seed}') if args.smoke else (direct_root if args.direct else world_root)(args.crop, args.origin, args.seed)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    set_seed(args.seed)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load(args.crop, args.origin)
        model, pretraining = initial_model(args.crop, args.origin, args.seed, args.direct); model = model.cuda()
        mode = 'direct_continued' if args.direct else 'joint_tenth'
        spec = dict(**vars(args), pretraining=pretraining, input_manifest=meta, mode=mode,
            code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
            optimizer=dict(name='AdamW', lr=.0001, weight_decay=.0001, clip=1), epochs=25, patience=6,
            batch=2048, micro_batch=256, state_loss_weight=0 if args.direct else .1, state_constraint_fraction=None if args.direct else .99,
            selection='Validation yield RMSE without a state constraint for direct controls; world candidates require LAI RMSE <= 99 percent of persistence. No test early stopping.',
            recipe='Frozen W5 validation-selected joint-tenth world extension; no architecture or seed selection during replication')
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Replication recipe changed')
        atomic_json(root / 'config.json', spec)
        if (root / 'metrics.json').exists():
            return
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        data = {s: tensors(a) for s, a in arrays.items()}; norm = meta['normalization']
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001, fused=True)
        persistence = state_rmse(arrays['validation'], arrays['validation']['previous_lai'], norm)
        best = fallback = None; best_score = fallback_score = float('inf'); best_epoch = fallback_epoch = -1
        stale = 0; history = []; started = time.monotonic(); n = len(arrays['train']['target'])
        for epoch in range((2 if args.smoke else 25) + 1):
            epoch_started = time.monotonic(); total_y = total_l = 0.
            if epoch:
                model.train(); order = torch.randperm(n)
                for start in range(0, n, 2048):
                    indices = order[start:start+2048]; count = len(indices)
                    mass = ((data['train']['target_lai_valid'][indices] > 0) & (data['train']['relative_valid'][indices] > 0)).sum().clamp_min(1).item()
                    optimizer.zero_grad(set_to_none=True)
                    for offset in range(0, count, 256):
                        ix = indices[offset:offset+256]; b = batch(data['train'], ix)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            y, state = model({k: b[k] for k in INPUTS})
                            loss_y = (y.float() - b['target_residual']).square().sum()/count
                            loss_l = y.new_zeros((), dtype=torch.float32)
                            if not args.direct:
                                mask = (b['target_lai_valid'] > 0) & (b['relative_valid'] > 0)
                                loss_l = ((state.float()-b['target_lai']).square()*mask).sum()/mass/meta['loss_scales']['lai']
                            loss = loss_y + .1*loss_l
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite loss')
                        loss.backward(); total_y += float(loss_y.detach())*count; total_l += float(loss_l.detach())*count
                    nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            y, state = predict(model, data['validation'], size=256)
            score = regression_metrics(arrays['validation']['target'], physical_yield(arrays['validation'], y, norm))['rmse']
            lai_score = state_rmse(arrays['validation'], state, norm); eligible = args.direct or lai_score <= .99*persistence
            fallback_improved = score < fallback_score-1e-6
            if fallback_improved:
                fallback_score = score; fallback_epoch = epoch
                fallback = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = eligible and score < best_score-1e-6
            if improved:
                best_score = score; best_epoch = epoch
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0 if improved or (best is None and fallback_improved) else stale+1
            row = dict(epoch=epoch, validation_yield_rmse=score, validation_lai_rmse=lai_score,
                state_constraint_met=bool(eligible), yield_loss=total_y/n, state_loss=total_l/n, seconds=time.monotonic()-epoch_started)
            history.append(row); write_csv(history, root / 'training_history.csv')
            print(f'[WORLD REPLICATION] {args.crop} {args.origin} {args.seed} {row}', flush=True)
            if epoch and stale >= 6:
                break
        eligible = best is not None; chosen = best if eligible else fallback
        if chosen is None:
            raise RuntimeError('No checkpoint')
        model.load_state_dict(chosen); weight = root / 'model_best.pt'; torch.save(chosen, weight)
        scores, states = {}, {}
        for split in ('validation', 'test'):
            y, state = predict(model, data[split], size=256); a = arrays[split]
            prediction = physical_yield(a, y, norm); scores[split] = regression_metrics(a['target'], prediction)
            states[split] = dict(rmse=state_rmse(a, state, norm), persistence_rmse=state_rmse(a, a['previous_lai'], norm))
            np.savez_compressed(root / f'{split}_predictions.npz', prediction=prediction, prediction_lai=state,
                **{k: a[k] for k in ('target', 'target_lai', 'target_lai_valid', 'source_indices', 'row', 'col', 'year', 'baseline')})
        atomic_json(root / 'metrics.json', dict(crop=args.crop, origin=args.origin, seed=args.seed, mode=mode,
            scores=scores, state_scores=states, state_constraint_met=eligible, selected_epoch=best_epoch if eligible else fallback_epoch,
            weight=str(weight), weight_sha256=sha256(weight), seconds=time.monotonic()-started,
            parameters=sum(p.numel() for p in model.parameters()), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


if __name__ == '__main__':
    main()
