"""Matched observed-state readouts for original and crop-weighted GLASS LAI."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from crop_weighted_lai import OUT as PRODUCT
from multimodal_baseline import regression_metrics, set_seed
from observed_remote_anomaly import seasonal_anomalies
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from stable_remote_data import load as load_source
from stable_remote_models import build, fit

CACHE = ROOT / 'benchmark/cache/crop_weighted_diagnostic_v1'
RESULT = ROOT / 'benchmark/results/crop_weighted_diagnostic_v1'
ENGINES = ('hgb_base', 'lgb_base', 'catboost')
CONDITIONS = ('history', 'shared_mask', 'quality_only', 'original_raw', 'weighted_raw',
              'original_anomaly', 'weighted_anomaly', 'original_anomaly_quality', 'weighted_anomaly_quality')
CODE = ('crop_weighted_diagnostic.py', 'crop_weighted_lai.py', 'stable_remote_models.py',
        'stable_remote_data.py', 'observed_remote_anomaly.py')


def extract(crop, a, previous=False):
    kind = 'previous_lai_rel' if previous else 'lai_rel'
    values = np.full_like(a['observed_lai'], np.nan); quality = np.zeros_like(values)
    for year in np.unique(a['year']):
        rows = np.flatnonzero(a['year'] == year)
        folder = PRODUCT / 'crops' / crop / kind
        grid = np.load(folder / f'{kind}_{year}.npy', mmap_mode='r')
        q = np.load(folder / f'valid_area_fraction_{year}.npy', mmap_mode='r')
        if grid.shape != (12, 360, 720) or q.shape != grid.shape:
            raise ValueError('Invalid crop-weighted tensor')
        rr, cc = a['row'][rows, None], a['col'][rows, None]
        phase = np.arange(12)[None]
        active = a['relative_valid'][rows] > 0
        values[rows] = np.where(active, grid[phase, rr, cc], np.nan)
        quality[rows] = np.where(active, q[phase, rr, cc], 0)
    return values, np.nan_to_num(quality, nan=0.)


def prepare(crop, origin):
    root = CACHE / crop / f'origin_{origin}'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'manifest.json').exists():
            return
        if not (PRODUCT / 'complete.json').exists():
            raise RuntimeError('Wait for complete area-weighted aggregation and alignment')
        arrays, source = load_source(crop, origin, 0)
        norm = source['normalization']; summaries = {}
        for split, a in arrays.items():
            values, quality = extract(crop, a)
            _, previous_quality = extract(crop, a, True)
            active = a['relative_valid'] > 0
            shared = active & (a['observed_lai_valid'] > 0) & np.isfinite(values) & (quality > 0)
            original = a['observed_lai'] * norm['lai_std'] + norm['lai_mean']
            difference = (values - original)[shared]
            summaries[split] = dict(samples=len(values), active_slots=int(active.sum()),
                paired_slots=int(shared.sum()), original_available=int((active & (a['observed_lai_valid'] > 0)).sum()),
                weighted_available=int((active & np.isfinite(values)).sum()),
                mean_weighted_area_time_coverage=float(quality[active].mean(dtype=float)),
                physical_lai_mean_difference=float(difference.mean(dtype=float)),
                physical_lai_rmse_difference=float(np.sqrt(np.mean(difference.astype(float) ** 2))))
            a['shared_mask'] = shared.astype(np.float32)
            a['observed_original'] = np.where(shared, a['observed_lai'], 0).astype(np.float32)
            a['observed_weighted'] = np.where(shared, (values - norm['lai_mean']) / norm['lai_std'], 0).astype(np.float32)
            for product in ('original', 'weighted'):
                a[f'observed_{product}_valid'] = a['shared_mask']
            a['quality_current'] = quality.astype(np.float32)
            a['quality_previous'] = previous_quality.astype(np.float32)
            keep = ('history', 'context', 'relative_valid', 'source_month', 'row', 'col', 'year',
                'source_indices', 'target', 'baseline', 'target_residual', 'shared_mask',
                'observed_original', 'observed_original_valid', 'observed_weighted', 'observed_weighted_valid',
                'quality_current', 'quality_previous')
            np.savez(root / f'{split}.npz', **{k: a[k] for k in keep})
        meta = dict(crop=crop, origin=origin, source_manifest=source, normalization=norm, data_summary=summaries,
            product_alignment_sha256=sha256(PRODUCT / 'alignment_manifest.json'),
            code_sha256=sha256(Path(__file__)),
            input_rule='Observed-season diagnostic. Same source rows and labels; both representations zero-fill the shared slot support. No sample removed. Original training normalization applied to both products.',
            files={s: sha256(root / f'{s}.npz') for s in arrays})
        atomic_json(root / 'manifest.json', meta)
        print(f'[WEIGHTED DATA] {crop} {origin} {summaries}', flush=True)


def load(crop, origin):
    root = CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text()); arrays = {}
    for split in ('train', 'validation', 'test'):
        with np.load(root / f'{split}.npz') as f:
            arrays[split] = {k: f[k] for k in f.files}
    return arrays, meta


class Features:
    def __init__(self, arrays):
        self.arrays = arrays; self.anomaly = {}

    def build(self, condition):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        product = condition.split('_')[0]
        if 'anomaly' in condition and product not in self.anomaly:
            self.anomaly[product] = seasonal_anomalies(self.arrays, product)
        result = {}; names = None
        for split, a in self.arrays.items():
            chunks = [a['history'], a['context']]
            labels = [f'history_{i}' for i in range(15)] + [f'context_{i}' for i in range(5)]
            if condition == 'shared_mask':
                chunks.append(a['shared_mask']); labels += [f'support_{k}' for k in range(12)]
            if product in ('original', 'weighted'):
                slots = self.anomaly[product][split] if 'anomaly' in condition else a[f'observed_{product}']
                chunks.append(np.where(a['shared_mask'] > 0, slots, 0))
                labels += [f'{product}_slot_{k}' for k in range(12)]
            if 'quality' in condition:
                chunks.extend((a['quality_current'], a['quality_previous']))
                labels += [f'{time}_quality_{k}' for time in ('current', 'previous') for k in range(12)]
            result[split] = np.concatenate(chunks, 1).astype(np.float32)
            if not np.isfinite(result[split]).all() or result[split].shape[1] != len(labels):
                raise ValueError('Invalid paired feature matrix')
            if names is not None and names != labels:
                raise ValueError('Feature columns changed')
            names = labels
        return result, names


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), default=2012)
    p.add_argument('--engine', choices=ENGINES); p.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare(args.crop, args.origin); return
    if args.engine is None:
        p.error('Specify an engine')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    root = RESULT / ('smoke' if args.smoke else 'pipelines') / args.crop / f'origin_{args.origin}' / args.engine / f'seed_{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load(args.crop, args.origin)
        spec = dict(**vars(args), input_manifest=meta, conditions=list(CONDITIONS),
                    code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
                    selection='Three-year validation only; observed target LAI diagnostic, not world-state forecasting')
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Weighted diagnostic configuration changed')
        atomic_json(root / 'config.json', spec)
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        factory = Features(arrays); norm = meta['normalization']; y = {s: a['target_residual'] for s, a in arrays.items()}
        for condition in CONDITIONS:
            folder = root / condition; folder.mkdir(exist_ok=True)
            if (folder / 'metrics.json').exists():
                continue
            set_seed(args.seed); started = time.monotonic()
            x, names = factory.build(condition)
            model = build(args.engine, args.seed, args.smoke); fit(model, args.engine, x, y)
            if args.engine == 'catboost':
                weight = folder / 'model.cbm'; model.save_model(str(weight))
            else:
                weight = folder / 'model.joblib'; joblib.dump(model, weight)
            scores, yearly = {}, {}
            for split in ('validation', 'test'):
                a = arrays[split]
                prediction = a['baseline'].astype(float) + model.predict(x[split]).astype(float) * norm['residual_std'] + norm['residual_mean']
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite prediction')
                scores[split] = regression_metrics(a['target'], prediction)
                yearly[split] = {str(int(year)): regression_metrics(a['target'][a['year'] == year], prediction[a['year'] == year]) for year in np.unique(a['year'])}
                np.savez_compressed(folder / f'{split}_predictions.npz', prediction=prediction,
                    **{k: a[k] for k in ('target', 'baseline', 'source_indices', 'row', 'col', 'year')})
            atomic_json(folder / 'metrics.json', dict(crop=args.crop, origin=args.origin, engine=args.engine, seed=args.seed,
                condition=condition, scores=scores, per_year=yearly, feature_dim=len(names), features=names,
                weight=str(weight), weight_sha256=sha256(weight), seconds=time.monotonic() - started))
            print(f'[WEIGHTED FIT] {args.crop} {args.engine} {condition} val={scores["validation"]["rmse"]:.6f} test={scores["test"]["rmse"]:.6f}', flush=True)
            del model, x; gc.collect()
        atomic_json(root / 'complete.json', dict(conditions=list(CONDITIONS), fits=len(CONDITIONS)))


if __name__ == '__main__':
    main()
