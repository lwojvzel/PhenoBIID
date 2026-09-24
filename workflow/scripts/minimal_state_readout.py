"""Minimal history + state-anomaly inputs, with matched observation/forecast controls."""
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
from observed_remote_benchmark import trajectory_features
from observed_remote_anomaly import seasonal_anomalies
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from run_yield_head_redesign import load_forward
from stable_remote_data import load as load_observed
from stable_remote_models import build, fit
from stable_world_readout import climatology

RESULT = ROOT / 'benchmark/results/minimal_state_readout_v1'
BASE = ('history', 'metadata_minimal', 'raw_slots', 'anomaly_slots', 'anomaly_summary',
        'change_slots', 'anomaly_area', 'previous_anomaly', 'previous_area')
CONDITIONS = {'world': (*BASE, 'observed_anomaly', 'direct_weather'),
              'observed': (*BASE, 'ndvi_anomaly', 'both_anomaly')}
ENGINES = ('hgb_base', 'lgb_base', 'catboost')
CODE = ('minimal_state_readout.py', 'stable_remote_data.py', 'stable_world_readout.py',
        'stable_remote_models.py', 'observed_remote_anomaly.py', 'observed_remote_benchmark.py',
        'run_yield_head_redesign.py', 'forward_protocol_revision.py')


def load_data(track, crop, origin, seed):
    if track == 'world':
        return load_forward(crop, seed, origin)
    arrays, meta = load_observed(crop, origin, 0)
    for a in arrays.values():
        a['target_lai'] = a['observed_lai']
        a['target_lai_valid'] = a['observed_lai_valid']
    return arrays, meta


class MinimalFeatures:
    def __init__(self, arrays, track):
        if track not in CONDITIONS:
            raise ValueError(track)
        self.arrays, self.track = arrays, track
        self.climo = climatology(arrays)
        self.ndvi = None

    def build(self, condition):
        if condition not in CONDITIONS[self.track]:
            raise ValueError(condition)
        if condition in ('ndvi_anomaly', 'both_anomaly') and self.ndvi is None:
            self.ndvi = seasonal_anomalies(self.arrays, 'ndvi')
        result = {}; names = None
        for s, a in self.arrays.items():
            history = np.concatenate((a['history'], a['context']), 1).astype(np.float32)
            active = a['relative_valid'] > 0
            mask = active & (a['previous_lai_valid'] > 0)
            if self.track == 'observed':
                current = a['observed_lai']
                mask = mask & (a['observed_lai_valid'] > 0)
            else:
                current = a['predicted_lai']
            chunks = [history]; labels = [f'history_{i}' for i in range(15)] + [f'context_{i}' for i in range(5)]
            if condition == 'metadata_minimal':
                chunks.append(np.stack((active.mean(1), mask.sum(1) / np.maximum(active.sum(1), 1)), 1))
                labels.extend(('active_fraction', 'past_lai_fraction'))
            elif condition != 'history':
                source = a['previous_lai'] if condition in ('previous_anomaly', 'previous_area', 'direct_weather') else a['target_lai'] if condition == 'observed_anomaly' else current
                value = source if condition == 'raw_slots' else source - a['previous_lai'] if condition == 'change_slots' else source - self.climo[s]
                if condition in ('anomaly_area', 'previous_area'):
                    value = value * np.sqrt(np.clip(a['crop_coverage'], 0, 1))[:, None]
                if condition != 'ndvi_anomaly':
                    if condition == 'anomaly_summary':
                        chunks.append(trajectory_features(value, mask)); labels.extend(f'state_anomaly_{i}' for i in range(18))
                    else:
                        chunks.append(np.where(mask, value, 0)); labels.extend(f'state_slot_{i}' for i in range(12))
                if condition in ('ndvi_anomaly', 'both_anomaly'):
                    ndvi_mask = active & (a['observed_ndvi_valid'] > 0)
                    chunks.append(np.where(ndvi_mask, self.ndvi[s], 0)); labels.extend(f'ndvi_anomaly_{i}' for i in range(12))
                if condition == 'direct_weather':
                    chunks.append(np.where(active[..., None], a['weather'], 0).reshape(len(history), -1))
                    labels.extend(f'weather_{k}_{v}' for k in range(12) for v in range(13))
            result[s] = np.concatenate(chunks, 1).astype(np.float32)
            if result[s].shape[1] != len(labels) or not np.isfinite(result[s]).all():
                raise ValueError('Invalid minimal feature matrix')
            if names is not None and labels != names:
                raise ValueError('Feature order changed')
            names = labels
        return result, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track', choices=CONDITIONS, required=True)
    p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    p.add_argument('--engine', choices=ENGINES, required=True)
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    root = RESULT / ('smoke' if args.smoke else 'pipelines') / args.track / args.crop / f'origin_{args.origin}' / args.engine / f'seed_{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        arrays, meta = load_data(args.track, args.crop, args.origin, args.seed)
        spec = dict(**vars(args), input_manifest=meta, conditions=CONDITIONS[args.track],
                    code_hashes={f: sha256(ROOT / 'scripts' / f) for f in CODE},
                    versions={k: importlib.metadata.version(k) for k in ('scikit-learn', 'lightgbm', 'catboost')},
                    input_rule='20 history/context features plus complete state slots; no appended calendar/coverage/mask tokens, except named metadata_minimal or anomaly_area controls',
                    normalization='Train-only; local observed LAI climatology excludes own row in training; no evaluation LAI enters forecast features',
                    missing_values='Invalid slots zero-filled identically; missing pattern is still implicitly present')
        spec = json.loads(json.dumps(spec))
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise RuntimeError('Minimal readout specification changed')
        atomic_json(root / 'config.json', spec)
        if args.smoke:
            arrays = {s: {k: v[:256 if s == 'train' else 128] for k, v in a.items()} for s, a in arrays.items()}
        factory = MinimalFeatures(arrays, args.track); norm = meta['normalization']; y = {s: a['target_residual'] for s, a in arrays.items()}
        for condition in CONDITIONS[args.track]:
            out = root / condition; out.mkdir(exist_ok=True)
            if (out / 'metrics.json').exists():
                continue
            started = time.monotonic(); set_seed(args.seed)
            x, names = factory.build(condition)
            model = build(args.engine, args.seed, args.smoke); fit(model, args.engine, x, y)
            if args.engine == 'catboost':
                weight = out / 'model.cbm'; model.save_model(str(weight))
            else:
                weight = out / 'model.joblib'; joblib.dump(model, weight)
            scores, per_year = {}, {}
            for split in ('validation', 'test'):
                a = arrays[split]
                prediction = a['baseline'].astype(float) + model.predict(x[split]).astype(float) * norm['residual_std'] + norm['residual_mean']
                if not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite yield')
                scores[split] = regression_metrics(a['target'], prediction)
                per_year[split] = {str(int(year)): regression_metrics(a['target'][a['year'] == year], prediction[a['year'] == year]) for year in np.unique(a['year'])}
                np.savez_compressed(out / f'{split}_predictions.npz', prediction=prediction,
                                    **{k: a[k] for k in ('target', 'baseline', 'source_indices', 'year', 'row', 'col')})
            atomic_json(out / 'metrics.json', dict(track=args.track, crop=args.crop, origin=args.origin, seed=args.seed,
                        engine=args.engine, condition=condition, scores=scores, per_year=per_year, features=names,
                        feature_dim=len(names), weight=str(weight), weight_sha256=sha256(weight), seconds=time.monotonic() - started))
            print(f'[MINIMAL] {args.track} {args.crop} {args.origin} {args.engine} {condition} val={scores["validation"]["rmse"]:.6f} test={scores["test"]["rmse"]:.6f}', flush=True)
            del model, x; gc.collect()
        atomic_json(root / 'complete.json', dict(conditions=CONDITIONS[args.track], state_fits=0))


if __name__ == '__main__':
    main()
