"""Export out-of-time state/history predictions for matched yield readouts."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from forecast_bridge_data import (ROOT, CROPS, ORIGINS, PRODUCTS, load, root as raw_root,
                                  forward_blocks, climatology, history_features)
from forecast_bridge_state import ForecastState
from run_forecast_bridge_state import run_root as state_root, predict, state_metrics
from forecast_bridge_history import run_root as history_root, saved_predict
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

CACHE = ROOT / 'benchmark/cache/forecast_state_bridge_v1/predictions'
CODE = ('export_forecast_bridge.py', 'forecast_bridge_data.py', 'forecast_bridge_state.py',
        'run_forecast_bridge_state.py', 'forecast_bridge_history.py')


def run_root(crop, origin, seed, product='ndvi', architecture='biid'):
    return CACHE / crop / product / architecture / f'origin_{origin}/seed_{seed}'


def dependencies(crop, origin, seed, product='ndvi', architecture='biid'):
    cutoffs = sorted({start-1 for start, _ in forward_blocks(origin-3)} | {origin-3})
    out = []
    for cutoff in cutoffs:
        out.append(state_root(crop, product, architecture, cutoff, seed) / 'complete.json')
        out.append(history_root(crop, 'mlp', cutoff, seed) / 'complete.json')
        if crop == 'soybean':
            out.append(history_root(crop, 'tabm', cutoff, seed) / 'complete.json')
    return out


def state_prediction(a, take, crop, product, architecture, cutoff, seed):
    directory = state_root(crop, product, architecture, cutoff, seed)
    record = json.loads((directory / 'complete.json').read_text())
    if record['smoke'] or record['full_fit_cutoff'] != cutoff or np.any(a['year'][take] <= cutoff):
        raise ValueError('State predictions are not strictly forward')
    for name, checksum in record['files'].items():
        if sha256(directory / name) != checksum:
            raise ValueError('Changed state asset')
    config = json.loads((directory / 'config.json').read_text())
    for name, checksum in config['code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != checksum:
            raise ValueError('State implementation changed')
    stats = json.loads((directory / 'normalization.json').read_text())
    model = ForecastState(architecture).cuda()
    model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
    p = predict(model, a, take, stats, product)
    del model
    return p, stats


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for frozen prediction replay')
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    a, _ = load(args.crop)
    parents = dependencies(args.crop, args.origin, args.seed, args.product, args.architecture)
    if any(not p.exists() for p in parents):
        raise ValueError('Upstream dependencies are incomplete; do not occupy GPU while waiting')
    dest = run_root(args.crop, args.origin, args.seed, args.product, args.architecture)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(**vars(args), code_sha256={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            data_sha256=sha256(raw_root(args.crop) / 'manifest.json'),
            parents={str(p): sha256(p) for p in parents}, evaluation_arrays_loaded=False,
            training_predictions='Forward chronological state and history, not in-sample',
            yearly_initialization='Previous calendar year observed RS; given weather and calendar conditions')
        file = dest / 'config.json'
        if file.exists() and json.loads(file.read_text()) != spec:
            raise ValueError('Frozen export changed')
        atomic_json(file, spec)
        if (dest / 'complete.json').exists():
            for n, digest in json.loads((dest / 'complete.json').read_text())['files'].items():
                if sha256(dest / n) != digest:
                    raise ValueError('Changed forecast export')
            return
        take = np.flatnonzero((a['year'] >= 1993) & (a['year'] <= args.origin))
        n = len(take)
        state = np.full((n, 12), np.nan, np.float32)
        climos = {p: np.full_like(state, np.nan) for p in PRODUCTS}
        histories = {k: np.full(n, np.nan) for k in ('trend', 'mlp', *(['tabm'] if args.crop == 'soybean' else []))}
        cutoff_per_row = np.full(n, -1, np.int32)
        blocks = [(s-1, s, e) for s, e in forward_blocks(args.origin-3)]
        blocks.append((args.origin-3, args.origin-2, args.origin))
        block_records = []
        for cutoff, first, last in blocks:
            positions = np.flatnonzero((a['year'][take] >= first) & (a['year'][take] <= last))
            selected = take[positions]
            if not len(selected) or np.any(cutoff_per_row[positions] != -1):
                raise ValueError('Overlapping or empty forecast block')
            p, stats = state_prediction(a, selected, args.crop, args.product, args.architecture, cutoff, args.seed)
            fit = np.flatnonzero(a['year'] <= cutoff)
            state[positions] = p
            for product in PRODUCTS:
                climos[product][positions] = climatology(a, fit, selected, product)
            histories['trend'][positions] = history_features(a, selected, stats)[1]
            for kind in ('mlp', *(['tabm'] if args.crop == 'soybean' else [])):
                histories[kind][positions], _ = saved_predict(a, selected, args.crop, kind, cutoff, args.seed)
            cutoff_per_row[positions] = cutoff
            block_records.append(dict(cutoff=cutoff, first=first, last=last, rows=len(selected),
                state_rmse=state_metrics(a, selected, p, args.product)))
            print(f'[BRIDGE EXPORT] {args.crop} {args.origin}: {cutoff} -> {first}..{last}', flush=True)
        if np.any(cutoff_per_row < 0) or np.any(cutoff_per_row >= a['year'][take]):
            raise ValueError('Forecast coverage or temporal boundary violated')
        if not np.isfinite(state).all() or any(not np.isfinite(v).all() for v in histories.values()):
            raise ValueError('Nonfinite exported prediction')
        arrays = dict(raw_indices=take, source_indices=a['source_indices'][take], year=a['year'][take],
            row=a['row'][take], col=a['col'][take], target=a['target'][take],
            predicted_state=state, climatology=climos[args.product], upstream_cutoff=cutoff_per_row,
            **{f'climatology_{p}': v for p, v in climos.items()},
            **{f'history_{k}': v for k, v in histories.items()})
        file = dest / 'predictions.npz'
        np.savez(file, **arrays)
        with np.load(file) as replay:
            for k, value in arrays.items():
                np.testing.assert_array_equal(replay[k], value)
        atomic_json(dest / 'complete.json', dict(blocks=block_records, training_start=1993,
            evaluation_arrays_loaded=False, row_identity_verified=True, all_upstream_strictly_before_target=True,
            files={n: sha256(dest / n) for n in ('config.json', 'predictions.npz')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', choices=ORIGINS, type=int, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    parser.add_argument('--product', choices=PRODUCTS, default='ndvi')
    parser.add_argument('--architecture', choices=('biid', 'gru'), default='biid')
    run(parser.parse_args())
