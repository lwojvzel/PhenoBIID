"""Train and independently reload each registered nonlinear terminal model."""
import argparse
import fcntl
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from neural_process_readout import (ROOT, RESULT, CROPS, CONDITIONS, HEADS, DIMS,
    run_root, load, NeuralReadout, predict, versions)
from yield_sensitive_readout import save_predictions
from calibrate_world_anchor import coefficient, leave_year_out
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from multimodal_baseline import set_seed, regression_metrics

CODE = ('run_neural_process_readout.py', 'neural_process_readout.py',
        'yield_sensitive_readout.py', 'calibrate_world_anchor.py')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--head', choices=HEADS, required=True); p.add_argument('--condition', choices=CONDITIONS, required=True)
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    set_seed(42); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = run_root(args.crop, args.origin, args.head, args.condition, args.smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        x, arrays, meta = load(args.crop, args.origin, args.condition)
        spec = dict(**vars(args), seed=42, input_manifest=meta, versions=versions(),
            code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            optimization=dict(optimizer='AdamW', learning_rate=.001, weight_decay=.0001,
                epochs=60, patience=8, batch=1024, micro_batch=256, precision='FP32', gradient_clip=5),
            selection='Uncalibrated validation yield RMSE, epoch zero included; common scalar calibration afterward.',
            inputs='All slots retained; frozen recurrent state and history anchor; observed is a diagnosis.')
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Neural training recipe changed')
        atomic_json(root / 'config.json', spec)
        if (root / 'audit.json').exists():
            return
        if args.smoke:
            for s, a in arrays.items():
                years = np.unique(a['year'])
                take = np.concatenate([np.flatnonzero(a['year'] == y)[:max(1, 2048//len(years))] for y in years])
                x[s] = x[s][take]
                arrays[s] = {k: v[take] for k, v in a.items()}
        data = {s: torch.from_numpy(v) for s, v in x.items()}
        anchors = {s: a['history_prediction'] for s, a in arrays.items()}
        scale = meta['normalization']['residual_std']
        target = torch.from_numpy(((arrays['train']['target'].astype(float)-anchors['train'])/scale).astype(np.float32))
        model = NeuralReadout(args.head, DIMS[args.condition]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, fused=True)
        best = None; best_score = float('inf'); best_epoch = -1; stale = 0; trace = []
        n = len(target); started = time.monotonic()
        for epoch in range((2 if args.smoke else 60)+1):
            tick = time.monotonic(); total = 0.
            if epoch:
                model.train(); order = torch.randperm(n)
                for start in range(0, n, 1024):
                    ix = order[start:start+1024]; count = len(ix)
                    optimizer.zero_grad(set_to_none=True)
                    for offset in range(0, count, 256):
                        take = ix[offset:offset+256]
                        members = model(data['train'][take].cuda())
                        loss = ((members-target[take, None].cuda())**2).mean()*len(take)/count
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite neural residual loss')
                        loss.backward(); total += float(loss.detach())*count
                    nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            validation = anchors['validation']+scale*predict(model, data['validation'])
            score = regression_metrics(arrays['validation']['target'], validation)['rmse']
            if score < best_score-1e-8:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_score = score; best_epoch = epoch; stale = 0
            else:
                stale += 1
            row = dict(epoch=epoch, training_mse=total/n, validation_rmse=score, seconds=time.monotonic()-tick)
            trace.append(row); pd.DataFrame(trace).to_csv(root / 'training_history.csv', index=False)
            print(f'[NEURAL READOUT] {args.crop} {args.origin} {args.head} {args.condition} {row}', flush=True)
            if epoch and stale >= 8:
                break
        if best is None:
            raise RuntimeError('No finite validation checkpoint')
        model.load_state_dict(best); weight = root / 'model_best.pt'; torch.save(best, weight)
        raw = {s: anchors[s]+scale*predict(model, data[s]) for s in ('validation', 'test')}
        # Reload before writing the completion marker and replay all rows, not a prefix.
        restored = NeuralReadout(args.head, DIMS[args.condition]).cuda()
        restored.load_state_dict(torch.load(weight, weights_only=True, map_location='cuda'))
        for s in raw:
            replay = anchors[s]+scale*predict(restored, data[s])
            np.testing.assert_array_equal(replay, raw[s])
        row, annual = save_predictions(root, arrays, anchors, raw, weight, 0, args.condition,
            args.head, 0., args.crop, args.origin)
        row.update(dimensions=DIMS[args.condition], selected_epoch=best_epoch, seed=42,
            parameters=sum(p.numel() for p in model.parameters()), weight_sha256=sha256(weight),
            seconds=time.monotonic()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        cal = json.loads((root / 'calibration.json').read_text()); a = arrays['validation']
        if coefficient(a['target'], anchors['validation'], raw['validation'], a['year']) != cal['coefficient']:
            raise ValueError('Calibration replay failed')
        _, loo = leave_year_out(a['target'], anchors['validation'], raw['validation'], a['year'])
        if loo != cal['leave_year_out_coefficients']:
            raise ValueError('Leave-year-out calibration replay failed')
        for s in raw:
            with np.load(root / f'{s}_predictions.npz') as saved:
                for k in ('target', 'row', 'col', 'year', 'source_indices'):
                    np.testing.assert_array_equal(saved[k], arrays[s][k])
                np.testing.assert_array_equal(saved['component_prediction'], raw[s])
                np.testing.assert_array_equal(saved['prediction'], anchors[s]+cal['coefficient']*(raw[s]-anchors[s]))
        pd.DataFrame(annual).to_csv(root / 'per_year.csv', index=False)
        atomic_json(root / 'metrics.json', row)
        atomic_json(root / 'audit.json', dict(fits=1, calibrations=1, full_array_replay=True,
            maximum_replay_error=0., weight_sha256=sha256(weight), smoke=args.smoke,
            training_scalers_verified_by_cache=True, source_indices_verified=True))
        print(f'[NEURAL COMPLETE] {row}', flush=True)


if __name__ == '__main__':
    main()
