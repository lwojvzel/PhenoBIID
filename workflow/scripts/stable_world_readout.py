"""Matched tree readouts of forward-only LAI forecasts, without changing states."""
import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from multimodal_baseline import regression_metrics, set_seed
from observed_remote_benchmark import INPUTS, make_features, trajectory_features
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from run_yield_head_redesign import load_forward
from stable_remote_models import ENGINES, build, fit

RESULT = ROOT / 'benchmark/results/stable_world_readout_v1'
CONDITIONS = ('history', 'metadata', 'predicted_raw', 'predicted_anomaly', 'predicted_weighted',
              'previous_raw', 'previous_anomaly', 'observed_anomaly', 'direct_weather')
CODE = ('stable_world_readout.py', 'stable_remote_models.py', 'observed_remote_benchmark.py',
        'run_yield_head_redesign.py', 'forward_protocol_revision.py', 'run_history_multimodal_baselines.py')


def climatology(arrays):
    """Grid/month training climatology; evaluation target observations are never read."""
    def keys(a):
        return (a['row'].astype(np.int64) * 720 + a['col'])[:, None] * 12 + np.minimum(a['source_month'], 11)
    train = arrays['train']
    valid = (train['relative_valid'] > 0) & (train['target_lai_valid'] > 0)
    values = train['target_lai']; index = keys(train); month = np.minimum(train['source_month'], 11)
    sums = np.bincount(index[valid], weights=values[valid], minlength=360 * 720 * 12)
    counts = np.bincount(index[valid], minlength=len(sums))
    ms = np.bincount(month[valid], weights=values[valid], minlength=12)
    mc = np.bincount(month[valid], minlength=12)
    result = {}
    for s, a in arrays.items():
        ix = keys(a); month = np.minimum(a['source_month'], 11)
        numerator, denominator = sums[ix].copy(), counts[ix].copy()
        fm, fc = ms[month].copy(), mc[month].copy()
        if s == 'train':
            own = np.where(valid, values, 0)
            numerator -= own; denominator -= valid; fm -= own; fc -= valid
        fallback = np.divide(fm, fc, out=np.zeros_like(fm), where=fc > 0)
        result[s] = np.divide(numerator, denominator, out=fallback, where=denominator > 0).astype(np.float32)
    return result


class WorldFeatures:
    def __init__(self, arrays):
        self.arrays = arrays
        self.climo = climatology(arrays)

    def build(self, condition):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        source = condition.split('_')[0]
        source = 'previous' if condition == 'direct_weather' else source
        output, names = {}, None
        for split, a in self.arrays.items():
            active = a['relative_valid'] > 0
            # Input-side masks are identical for previous/predicted/observed.
            # The existing cohort has target LAI in every active slot; enforce
            # this instead of silently filtering rows using a future mask.
            if not np.all(a['target_lai_valid'][active] > 0):
                raise ValueError('Observed diagnostic requires complete active target LAI; do not silently change cohort')
            mask = active & (a['previous_lai_valid'] > 0)
            data = {k: a[k] for k in ('history', 'context', 'relative_valid', 'source_month', 'crop_coverage')}
            zero = np.zeros_like(a['previous_lai'])
            remote = a[f'{source}_lai'] if source in ('previous', 'predicted') else a['target_lai'] if source == 'observed' else zero
            data.update(observed_lai=np.where(mask, remote, 0).astype(np.float32), observed_lai_valid=mask.astype(np.float32),
                        observed_ndvi=zero, observed_ndvi_valid=zero)
            variant = condition if condition in ('history', 'metadata') else 'observed_lai'
            x, _, current_names = make_features({k: data[k] for k in INPUTS}, variant)
            chunks = [x]
            if condition not in ('history', 'metadata', 'predicted_raw', 'previous_raw'):
                anomaly = np.where(mask, remote - self.climo[split], 0)
                chunks.append(trajectory_features(anomaly, mask))
                current_names.extend(f'training_climatology_anomaly_{i}' for i in range(18))
                if condition == 'predicted_weighted':
                    for label, q in (('area', np.clip(a['crop_coverage'], 0, 1)),
                                     ('sqrt_area', np.sqrt(np.clip(a['crop_coverage'], 0, 1)))):
                        chunks.append(trajectory_features(anomaly * q[:, None], mask))
                        current_names.extend(f'{label}_anomaly_{i}' for i in range(18))
                if condition == 'direct_weather':
                    chunks.append(np.where(active[..., None], a['weather'], 0).reshape(len(x), -1))
                    current_names.extend(f'weather_slot_{k}_variable_{v}' for k in range(12) for v in range(13))
            output[split] = np.concatenate(chunks, 1).astype(np.float32)
            if not np.isfinite(output[split]).all() or output[split].shape[1] != len(current_names):
                raise ValueError('Invalid world-readout features')
            if names is not None and current_names != names:
                raise ValueError('Feature order changed')
            names = current_names
        return output, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    p.add_argument('--engine', choices=ENGINES, required=True)
    p.add_argument('--conditions', default=','.join(CONDITIONS))
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    conditions = args.conditions.split(',')
    if not set(conditions).issubset(CONDITIONS) or len(set(conditions)) != len(conditions):
        p.error('Invalid conditions')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    root = RESULT / ('smoke' if args.smoke else 'pipelines') / args.crop / f'origin_{args.origin}' / args.engine / f'seed_{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load_forward(args.crop, args.seed, args.origin)
        spec = {k: v for k, v in vars(args).items() if k != 'conditions'}
        spec.update(manifest=meta, code_hashes={f: sha256(ROOT / 'scripts' / f) for f in CODE},
                    versions={k: importlib.metadata.version(k) for k in ('scikit-learn', 'lightgbm', 'xgboost', 'catboost')},
                    state_fits=0, training='Original strict forward state caches; 12 readout training years, one validation year, four evaluation years; distinct from three-year-validation observed screen.',
                    masks='Past availability and calendar only; no future observation mask as a predictor',
                    output='Physical yield = causal trend + independently learned residual; no frozen history expert output as an input')
        config = root / 'config.json'
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError('World-readout configuration changed')
        atomic_json(config, spec)
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        factory = WorldFeatures(arrays); norm = meta['normalization']; y = {s: a['target_residual'] for s, a in arrays.items()}
        for condition in conditions:
            out = root / condition; out.mkdir(exist_ok=True)
            if (out / 'metrics.json').exists():
                continue
            set_seed(args.seed); started = time.monotonic()
            x, names = factory.build(condition)
            model = build(args.engine, args.seed, args.smoke); fit(model, args.engine, x, y)
            if args.engine == 'catboost':
                path = out / 'model.cbm'; model.save_model(str(path))
            elif args.engine == 'xgb_smooth':
                model.set_params(device='cpu', callbacks=None); path = out / 'model.ubj'; model.save_model(path)
            else:
                path = out / 'model.joblib'; joblib.dump(model, path)
            scores = {}; per_year = {}
            for split in ('validation', 'test'):
                a = arrays[split]
                prediction = a['baseline'].astype(float) + model.predict(x[split]).astype(float) * norm['residual_std'] + norm['residual_mean']
                if not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite world readout')
                scores[split] = regression_metrics(a['target'], prediction)
                per_year[split] = {str(int(year)): regression_metrics(a['target'][a['year'] == year], prediction[a['year'] == year]) for year in np.unique(a['year'])}
                np.savez_compressed(out / f'{split}_predictions.npz', prediction=prediction,
                                    **{k: a[k] for k in ('target', 'source_indices', 'year', 'row', 'col', 'baseline')})
            atomic_json(out / 'metrics.json', dict(crop=args.crop, origin=args.origin, engine=args.engine,
                        seed=args.seed, condition=condition, scores=scores, per_year=per_year,
                        features=names, feature_dim=len(names), weight=str(path), weight_sha256=sha256(path),
                        seconds=time.monotonic() - started, state_fits=0))
            print(f'[WORLD READOUT] {args.crop} {args.origin} {args.engine} {condition} val={scores["validation"]["rmse"]:.6f} test={scores["test"]["rmse"]:.6f}', flush=True)
            del model, x; gc.collect()
        if all((root / c / 'metrics.json').exists() for c in CONDITIONS):
            atomic_json(root / 'complete.json', dict(conditions=list(CONDITIONS), source_states_unchanged=True))
        for path, digest in meta['frozen_weight_hashes'].items():
            if sha256(Path(path)) != digest:
                raise RuntimeError('Frozen state checkpoint changed')


if __name__ == '__main__':
    main()
