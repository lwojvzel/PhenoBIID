"""LAI-first recurrent-state training followed by frozen or constrained yield fitting."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gru_task_priority import RecurrentWorld
from task_aligned_world import INPUTS
from task_aligned_data import load
from run_gru_world_replication import initial_model
from run_task_aligned_world import tensors, batch, predict, physical_yield, state_rmse
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from multimodal_baseline import regression_metrics, set_seed, write_csv

RESULT = ROOT / 'benchmark/results/staged_recurrent_world_v1'
MODES = ('state', 'frozen', 'joint')
CODE = ('run_staged_recurrent_world.py', 'gru_task_priority.py', 'task_aligned_world.py',
        'task_aligned_data.py', 'run_task_aligned_world.py', 'run_gru_world_replication.py')


def run_root(crop, origin, seed, mode, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}' / f'seed_{seed}' / mode


def configure(model, mode):
    for p in model.parameters():
        p.requires_grad_(True)
    transition = model.transition_parameters()
    ids = {id(p) for p in transition}
    terminal = [p for p in model.parameters() if id(p) not in ids]
    if mode == 'state':
        for p in terminal:
            p.requires_grad_(False)
        return [dict(params=transition, lr=3e-4)]
    if mode == 'frozen':
        for p in transition:
            p.requires_grad_(False)
        return [dict(params=terminal, lr=1e-4)]
    if mode != 'joint':
        raise ValueError('Unknown training mode')
    return [dict(params=transition, lr=3e-5), dict(params=terminal, lr=1e-4)]


def train_mode(model, mode):
    model.train()
    if mode == 'frozen':
        for m in (model.initial, model.direct, model.observation):
            m.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    root = run_root(args.crop, args.origin, args.seed, args.mode, args.smoke); root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    set_seed(args.seed)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load(args.crop, args.origin)
        model, pretraining = initial_model(args.crop, args.origin, args.seed)
        state_source = None
        if args.mode != 'state':
            source = run_root(args.crop, args.origin, args.seed, 'state', args.smoke)
            state_record = json.loads((source / 'metrics.json').read_text())
            state_source = dict(directory=str(source), weight=state_record['weight'], sha256=state_record['weight_sha256'])
            if sha256(Path(state_source['weight'])) != state_source['sha256']:
                raise ValueError('State initialization changed')
            model.load_state_dict(torch.load(state_source['weight'], map_location='cpu', weights_only=True))
        model = model.cuda(); groups = configure(model, args.mode)
        state_only = args.mode == 'state'; frozen = args.mode == 'frozen'
        maximum = 50 if state_only else 25; patience = 8 if state_only else 6
        spec = dict(**vars(args), pretraining=pretraining, state_pretraining=state_source, input_manifest=meta,
            code_hashes={n: sha256(ROOT / 'scripts' / n) for n in CODE},
            optimization=dict(name='AdamW', learning_rates=[g['lr'] for g in groups], weight_decay=.0001,
                maximum_epochs=maximum, patience=patience, batch=2048, micro_batch=256, clip=1),
            state_loss_weight=1 if state_only else 0 if frozen else .1,
            selection='State stage: minimum validation LAI error. Yield stage: validation yield subject to <=99% persistence and <=102% pretrained-state LAI error.',
            protocol='No observed target-season LAI in inference; no test-based selection; old models retained.')
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Staged recipe changed')
        atomic_json(root / 'config.json', spec)
        if (root / 'metrics.json').exists():
            return
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        data = {s: tensors(a) for s, a in arrays.items()}; norm = meta['normalization']
        optimizer = torch.optim.AdamW(groups, weight_decay=.0001, fused=True)
        persistence = state_rmse(arrays['validation'], arrays['validation']['previous_lai'], norm)
        _, initial_lai = predict(model, data['validation'], size=256)
        initial_state = state_rmse(arrays['validation'], initial_lai, norm)
        state_limit = .99*persistence if state_only else min(.99*persistence, 1.02*initial_state)
        best = fallback = None; best_score = fallback_score = float('inf'); best_epoch = fallback_epoch = -1
        stale = 0; history = []; started = time.monotonic(); n = len(arrays['train']['target'])
        for epoch in range((2 if args.smoke else maximum)+1):
            epoch_start = time.monotonic(); total_y = total_l = 0.
            if epoch:
                train_mode(model, args.mode); order = torch.randperm(n)
                for start in range(0, n, 2048):
                    indices = order[start:start+2048]; count = len(indices)
                    mass = ((data['train']['target_lai_valid'][indices] > 0) & (data['train']['relative_valid'][indices] > 0)).sum().clamp_min(1).item()
                    optimizer.zero_grad(set_to_none=True)
                    for offset in range(0, count, 256):
                        ix = indices[offset:offset+256]; b = batch(data['train'], ix)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            y, state = model({k: b[k] for k in INPUTS})
                            ly = (y.float()-b['target_residual']).square().sum()/count
                            mask = (b['target_lai_valid'] > 0) & (b['relative_valid'] > 0)
                            ll = ((state.float()-b['target_lai']).square()*mask).sum()/mass/meta['loss_scales']['lai']
                            loss = ll if state_only else ly if frozen else ly+.1*ll
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite objective')
                        loss.backward(); total_y += float(ly.detach())*count; total_l += float(ll.detach())*count
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1); optimizer.step()
            y, state = predict(model, data['validation'], size=256)
            yr = regression_metrics(arrays['validation']['target'], physical_yield(arrays['validation'], y, norm))['rmse']
            lr = state_rmse(arrays['validation'], state, norm)
            score = lr if state_only else yr
            eligible = state_only or lr <= state_limit
            improved_fallback = score < fallback_score-1e-6
            if improved_fallback:
                fallback_score = score; fallback_epoch = epoch
                fallback = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = eligible and score < best_score-1e-6
            if improved:
                best_score = score; best_epoch = epoch
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0 if improved or (best is None and improved_fallback) else stale+1
            row = dict(epoch=epoch, validation_yield_rmse=yr, validation_lai_rmse=lr,
                state_constraint_met=bool(lr <= state_limit), yield_loss=total_y/n, state_loss=total_l/n,
                seconds=time.monotonic()-epoch_start)
            history.append(row); write_csv(history, root / 'training_history.csv')
            print(f'[STAGED WORLD] {args.crop} {args.origin} {args.seed} {args.mode} {row}', flush=True)
            if epoch and stale >= patience:
                break
        chosen = best if best is not None else fallback
        if chosen is None:
            raise RuntimeError('No saved state')
        model.load_state_dict(chosen); weight = root / 'model_best.pt'; torch.save(chosen, weight)
        scores, states = {}, {}
        for split in ('validation', 'test'):
            y, state = predict(model, data[split], size=256); a = arrays[split]
            states[split] = dict(rmse=state_rmse(a, state, norm), persistence_rmse=state_rmse(a, a['previous_lai'], norm))
            fields = {k: a[k] for k in ('target', 'target_lai', 'target_lai_valid', 'source_indices', 'row', 'col', 'year', 'baseline')}
            fields['prediction_lai'] = state
            if not state_only:
                fields['prediction'] = physical_yield(a, y, norm)
                scores[split] = regression_metrics(a['target'], fields['prediction'])
            np.savez_compressed(root / f'{split}_predictions.npz', **fields)
        atomic_json(root / 'metrics.json', dict(crop=args.crop, origin=args.origin, seed=args.seed, mode=args.mode,
            scores=scores, state_scores=states, state_constraint_met=states['validation']['rmse'] <= state_limit,
            state_limit=state_limit, initial_validation_lai_rmse=initial_state,
            selected_epoch=best_epoch if best is not None else fallback_epoch, weight=str(weight), weight_sha256=sha256(weight),
            seconds=time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            state_only_yield_not_reportable=state_only))


if __name__ == '__main__':
    main()
