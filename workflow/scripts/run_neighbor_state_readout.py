"""Fit a registered neighborhood readout and replay the complete held-out arrays."""
import argparse
import fcntl
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from neighbor_state_readout import (ROOT, RESULT, CROPS, CONDITIONS, HEADS, DIMS, run_root,
    load, make, predict, versions, resource_gate, batches, sample_neighbors, library_identity)
from yield_sensitive_readout import save_predictions
from calibrate_world_anchor import coefficient, leave_year_out
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from multimodal_baseline import set_seed, regression_metrics

CODE = ('run_neighbor_state_readout.py', 'neighbor_state_readout.py', 'neighbor_state_head.py',
        'foundation_state_data.py', 'yield_sensitive_readout.py', 'calibrate_world_anchor.py')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--head', choices=HEADS, required=True)
    p.add_argument('--condition', choices=CONDITIONS, required=True)
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    set_seed(42); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    checks = resource_gate()
    root = run_root(args.crop, args.origin, args.head, args.condition, args.smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        x, arrays, meta = load(args.crop, args.origin, args.condition)
        spec = dict(**vars(args), seed=42, input_manifest=meta, versions=versions(),
            code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE}, resource_probes=checks,
            optimization=dict(optimizer='AdamW', learning_rate=.001, weight_decay=.0001,
                epochs=60, patience=8, batch=512, merge_final_singleton=True, precision='FP32', gradient_clip=5),
            retrieval=dict(dim=128, temperature=1., metric='Euclidean', train_candidates=4096,
                inference_candidates='all training rows', encoding_batch=1024, query_batch=256,
                ordering_seed=1729, neighbor_sampling_seed=2718, official_sample_rate=1.),
            selection='Uncalibrated validation RMSE including epoch zero; scalar calibration afterward.',
            inputs='Frozen predicted states and historical anchor; observed target LAI is diagnostic only.')
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Neighborhood training recipe changed')
        atomic_json(root / 'config.json', spec)
        if (root / 'audit.json').exists():
            return
        if args.smoke:
            for split, a in arrays.items():
                years = np.unique(a['year'])
                take = np.concatenate([np.flatnonzero(a['year'] == year)[:max(1, 2048//len(years))] for year in years])
                x[split] = x[split][take]
                arrays[split] = {k: v[take] for k, v in a.items()}
        data = {s: torch.from_numpy(v) for s, v in x.items()}
        anchors = {s: a['history_prediction'] for s, a in arrays.items()}
        scale = meta['sources']['history']['normalization']['residual_std']
        target = ((arrays['train']['target'].astype(float)-anchors['train'])/scale).astype(np.float32)
        identity = library_identity(x['train'], arrays['train'], target)
        atomic_json(root / 'training_library.json', identity)
        train = data['train'].cuda(); target = torch.from_numpy(target).cuda(); n = len(target)
        if n > 400000:
            raise ValueError('Training library exceeds tested resource bound')
        model = make(DIMS[args.condition], args.head).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        ordering = torch.Generator().manual_seed(1729); sampling = torch.Generator().manual_seed(2718)
        sampling_hash = hashlib.sha256()
        best = None; best_score = float('inf'); best_epoch = -1; stale = 0; trace = []
        started = time.monotonic()
        for epoch in range((2 if args.smoke else 60)+1):
            tick = time.monotonic(); total = 0.; count = 0
            if epoch:
                model.train(); order = torch.randperm(n, generator=ordering)
                for ix in batches(order):
                    candidates = sample_neighbors(n, ix, sampling)
                    if epoch == 1:
                        for indices in (ix, candidates):
                            sampling_hash.update(np.asarray([len(indices)], dtype=np.int64).tobytes())
                            sampling_hash.update(indices.numpy().tobytes())
                    take = ix.cuda(); neighbor = candidates.cuda()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(train[take], target[take], train[neighbor], target[neighbor], True)
                    loss = (prediction-target[take]).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite retrieval training loss')
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step(); total += float(loss.detach())*len(ix); count += len(ix)
                if count != n:
                    raise ValueError('Training skipped identities')
                optimizer.zero_grad(set_to_none=True)
            validation = anchors['validation']+scale*predict(model, train, target, data['validation'])
            score = regression_metrics(arrays['validation']['target'], validation)['rmse']
            if score < best_score-1e-8:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_score = score; best_epoch = epoch; stale = 0
            else:
                stale += 1
            row = dict(epoch=epoch, training_mse=total/n, training_rows=count,
                       validation_rmse=score, seconds=time.monotonic()-tick)
            trace.append(row); pd.DataFrame(trace).to_csv(root / 'training_history.csv', index=False)
            print(f'[NEIGHBOR READOUT] {args.crop} {args.origin} {args.head} {args.condition} {row}', flush=True)
            if epoch and stale >= 8:
                break
        if best is None:
            raise RuntimeError('No finite retrieval checkpoint')
        model.load_state_dict(best); weight = root / 'model_best.pt'; torch.save(best, weight)
        raw = {s: anchors[s]+scale*predict(model, train, target, data[s]) for s in ('validation', 'test')}
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()
        restored = make(DIMS[args.condition], args.head).cuda()
        restored.load_state_dict(torch.load(weight, weights_only=True, map_location='cuda'))
        for split in raw:
            np.testing.assert_array_equal(raw[split], anchors[split]+scale*predict(restored, train, target, data[split]))
        np.testing.assert_equal(identity, library_identity(x['train'], arrays['train'], target.cpu().numpy()))
        row, annual = save_predictions(root, arrays, anchors, raw, weight, 0, args.condition,
            args.head, 0., args.crop, args.origin)
        row.update(dimensions=DIMS[args.condition], selected_epoch=best_epoch, seed=42,
            parameters=sum(p.numel() for p in restored.parameters()), weight_sha256=sha256(weight),
            training_library_rows=n, first_epoch_sampling_sha256=sampling_hash.hexdigest(),
            seconds=time.monotonic()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        cal = json.loads((root / 'calibration.json').read_text()); a = arrays['validation']
        if coefficient(a['target'], anchors['validation'], raw['validation'], a['year']) != cal['coefficient']:
            raise ValueError('Retrieval calibration replay failed')
        _, loo = leave_year_out(a['target'], anchors['validation'], raw['validation'], a['year'])
        if loo != cal['leave_year_out_coefficients']:
            raise ValueError('Retrieval leave-year-out calibration replay failed')
        for split in raw:
            with np.load(root / f'{split}_predictions.npz') as saved:
                for key in ('target', 'row', 'col', 'year', 'source_indices'):
                    np.testing.assert_array_equal(saved[key], arrays[split][key])
                np.testing.assert_array_equal(saved['component_prediction'], raw[split])
                np.testing.assert_array_equal(saved['prediction'], anchors[split]+cal['coefficient']*(raw[split]-anchors[split]))
        pd.DataFrame(annual).to_csv(root / 'per_year.csv', index=False)
        atomic_json(root / 'metrics.json', row)
        atomic_json(root / 'audit.json', dict(fits=1, calibrations=1, full_array_replay=True,
            maximum_replay_error=0., weight_sha256=sha256(weight), smoke=args.smoke,
            training_scalers_verified_by_cache=True, source_indices_verified=True,
            inference_memory_rebuilt=True, training_library_sha256=sha256(root / 'training_library.json'),
            first_epoch_sampling_sha256=sampling_hash.hexdigest(), training_rows=n))
        print(f'[NEIGHBOR COMPLETE] {row}', flush=True)


if __name__ == '__main__':
    main()
