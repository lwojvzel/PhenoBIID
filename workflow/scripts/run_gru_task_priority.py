"""Continue direct learning or add supervised, observation-feedback dynamics."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gru_task_priority import MODES, STATE_WEIGHTS, RecurrentWorld, primary_preserving_gradients
from multimodal_baseline import regression_metrics, set_seed, write_csv
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from run_task_aligned_world import RESULT as PRETRAINED, tensors, batch, predict, physical_yield, state_rmse
from task_aligned_data import load
from task_aligned_world import INPUTS, TaskAlignedWorld

RESULT = ROOT / 'benchmark/results/gru_task_priority_v1'
CODE = ('gru_task_priority.py', 'run_gru_task_priority.py', 'task_aligned_world.py',
        'task_aligned_data.py', 'run_task_aligned_world.py')


def make_model(crop, mode):
    root = PRETRAINED / 'pipelines' / crop / 'origin_2012/direct_gru/seed_42'
    metrics = json.loads((root / 'metrics.json').read_text())
    if sha256(Path(metrics['weight'])) != metrics['weight_sha256']:
        raise ValueError('Pretrained direct weight changed')
    source = TaskAlignedWorld('direct_gru')
    source.load_state_dict(torch.load(metrics['weight'], map_location='cpu', weights_only=True))
    if mode == 'direct_continued':
        model = source
    else:
        model = RecurrentWorld(); model.initialize_from_direct(source)
    return model, dict(weight=metrics['weight'], sha256=metrics['weight_sha256'],
                       input_config_sha256=sha256(root / 'config.json'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--mode', choices=MODES, required=True); p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(3000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    set_seed(42)
    root = RESULT / ('smoke' if args.smoke else 'pipelines') / args.crop / 'origin_2012' / args.mode / 'seed_42'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load(args.crop, 2012)
        model, pretraining = make_model(args.crop, args.mode); model = model.cuda()
        spec = dict(**vars(args), pretraining=pretraining, input_manifest=meta, seed=42,
            code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
            optimizer=dict(name='AdamW', lr=.0001, weight_decay=.0001, clip=1),
            epochs=25, patience=6, batch=2048, micro_batch=256, state_loss_weight=STATE_WEIGHTS[args.mode],
            state_constraint='Validation LAI RMSE <= 0.99 times previous-season LAI RMSE for world candidates',
            selection='Validation yield RMSE subject to state constraint; no test checkpoint selection',
            method='Shared past-conditioned recurrent hidden state, current phase weather forcing, predicted LAI feedback; terminal historical plus predicted/latent state readout',
            gradient_method='One-sided conflicting auxiliary gradient projection onto primary-gradient normal plane, shared transition parameters only; not symmetric PCGrad or EMA Bloop')
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('GRU priority configuration changed')
        atomic_json(root / 'config.json', spec)
        if (root / 'metrics.json').exists():
            return
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        data = {s: tensors(a) for s, a in arrays.items()}; norm = meta['normalization']
        world = args.mode != 'direct_continued'; priority = args.mode == 'yield_priority'
        parameters = list(model.parameters()); shared = model.transition_parameters() if world else []
        shared_ids = {id(p) for p in shared}; shared_indices = [i for i, p in enumerate(parameters) if id(p) in shared_ids]
        optimizer = torch.optim.AdamW(parameters, lr=.0001, weight_decay=.0001, fused=True)
        persistence = state_rmse(arrays['validation'], arrays['validation']['previous_lai'], norm)
        best = fallback = None; best_score = fallback_score = float('inf'); best_epoch = fallback_epoch = -1
        stale = 0; history = []; started = time.monotonic(); n = len(arrays['train']['target'])
        for epoch in range(0, (2 if args.smoke else 25) + 1):
            epoch_started = time.monotonic(); total_y = total_l = 0.; diagnostics = []
            if epoch:
                model.train(); order = torch.randperm(n)
                for start in range(0, n, 2048):
                    indices = order[start:start+2048]; count = len(indices)
                    mass = ((data['train']['target_lai_valid'][indices] > 0) & (data['train']['relative_valid'][indices] > 0)).sum().clamp_min(1).item()
                    optimizer.zero_grad(set_to_none=True)
                    if priority:
                        primary = [torch.zeros_like(p) for p in parameters]
                        auxiliary = [torch.zeros_like(p) for p in shared]
                    for offset in range(0, count, 256):
                        ix = indices[offset:offset+256]; b = batch(data['train'], ix)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            y, state = model({k: b[k] for k in INPUTS})
                            loss_y = (y.float() - b['target_residual']).square().sum() / count
                            loss_l = y.new_zeros((), dtype=torch.float32)
                            if world:
                                mask = (b['target_lai_valid'] > 0) & (b['relative_valid'] > 0)
                                loss_l = ((state.float() - b['target_lai']).square() * mask).sum() / mass / meta['loss_scales']['lai']
                        if not torch.isfinite(loss_y + loss_l):
                            raise FloatingPointError('Nonfinite objective')
                        if priority:
                            gy = torch.autograd.grad(loss_y, parameters, retain_graph=True, allow_unused=True)
                            gl = torch.autograd.grad(loss_l, shared, allow_unused=True)
                            for accumulator, g in zip(primary, gy):
                                if g is not None:
                                    accumulator.add_(g)
                            for accumulator, g in zip(auxiliary, gl):
                                if g is not None:
                                    accumulator.add_(g)
                        else:
                            (loss_y + STATE_WEIGHTS[args.mode] * loss_l).backward()
                        total_y += float(loss_y.detach()) * count; total_l += float(loss_l.detach()) * count
                    if priority:
                        merged, diagnostic = primary_preserving_gradients([primary[i] for i in shared_indices], auxiliary, STATE_WEIGHTS[args.mode])
                        diagnostics.append(diagnostic)
                        for i, g in zip(shared_indices, merged):
                            primary[i] = g
                        for p, g in zip(parameters, primary):
                            p.grad = g
                    nn.utils.clip_grad_norm_(parameters, 1); optimizer.step()
            y, state = predict(model, data['validation'], size=256)
            score = regression_metrics(arrays['validation']['target'], physical_yield(arrays['validation'], y, norm))['rmse']
            lai_score = state_rmse(arrays['validation'], state, norm)
            eligible = not world or lai_score <= .99 * persistence
            fallback_improved = score < fallback_score - 1e-6
            if fallback_improved:
                fallback_score = score; fallback_epoch = epoch
                fallback = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = eligible and score < best_score - 1e-6
            if improved:
                best_score = score; best_epoch = epoch
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0 if improved or (best is None and fallback_improved) else stale + 1
            record = dict(epoch=epoch, validation_yield_rmse=score, validation_lai_rmse=lai_score,
                state_constraint_met=bool(eligible), yield_loss=total_y/n, state_loss=total_l/n,
                seconds=time.monotonic()-epoch_started)
            if diagnostics:
                record.update({k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]})
            history.append(record); write_csv(history, root / 'training_history.csv')
            print(f'[GRU PRIORITY] {args.crop} {args.mode} {record}', flush=True)
            if epoch and stale >= 6:
                break
        eligible = best is not None; chosen = best if eligible else fallback
        if chosen is None:
            raise RuntimeError('No checkpoint')
        model.load_state_dict(chosen); weight = root / 'model_best.pt'; torch.save(chosen, weight)
        scores, state_scores = {}, {}
        for split in ('validation', 'test'):
            y, state = predict(model, data[split], size=256); a = arrays[split]
            prediction = physical_yield(a, y, norm); scores[split] = regression_metrics(a['target'], prediction)
            state_scores[split] = dict(rmse=state_rmse(a, state, norm), persistence_rmse=state_rmse(a, a['previous_lai'], norm))
            np.savez_compressed(root / f'{split}_predictions.npz', prediction=prediction, prediction_lai=state,
                **{k: a[k] for k in ('target', 'target_lai', 'target_lai_valid', 'source_indices', 'row', 'col', 'year', 'baseline')})
        atomic_json(root / 'metrics.json', dict(crop=args.crop, mode=args.mode, origin=2012, seed=42, scores=scores,
            state_scores=state_scores, selected_epoch=best_epoch if eligible else fallback_epoch,
            state_constraint_met=eligible, weight=str(weight), weight_sha256=sha256(weight),
            seconds=time.monotonic()-started, parameters=sum(p.numel() for p in parameters),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


if __name__ == '__main__':
    main()
