"""One fixed CPU LightGBM head; no evaluation-period data is opened."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from crop_signal_screen_data import (ROOT, CROPS, ORIGINS, CONDITIONS, CODE as DATA_CODE,
                                     run_root, cache_root, load, select_components)
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

CODE = (*DATA_CODE, 'run_crop_signal_screen.py')


def yearly_rmse(target, prediction, years):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if target.shape != prediction.shape or target.ndim != 1 or len(target) != len(years):
        raise ValueError('Prediction/target/year mismatch')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite yield predictions')
    return {str(int(y)): float(np.sqrt(np.mean((target[years == y]-prediction[years == y])**2)))
            for y in np.unique(years)}


def year_weights(years):
    _, inverse, counts = np.unique(years, return_inverse=True, return_counts=True)
    return len(years)/(len(counts)*counts[inverse])


def sample_rows(a, limit):
    years = np.unique(a['year'])
    return np.concatenate([np.flatnonzero(a['year'] == y)[:max(1, limit//len(years))] for y in years])


def run(crop, origin, condition, smoke=False):
    root = run_root(crop, origin, condition, smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        x, arrays, manifest = load(crop, origin, condition)
        names = [name for c in select_components(condition) for name in manifest['spec']['names'][c]]
        params = dict(n_estimators=20 if smoke else 1200, learning_rate=.03, num_leaves=15,
            min_child_samples=200, reg_lambda=10., subsample=.9, subsample_freq=1,
            colsample_bytree=.9, n_jobs=4, random_state=42, verbosity=-1, deterministic=True,
            force_col_wise=True, objective='regression', metric='None')
        spec = dict(crop=crop, origin=origin, condition=condition, smoke=smoke, seed=42,
            model=params, features=names, code_sha256={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            input_manifest_sha256=sha256(cache_root(crop, origin) / 'manifest.json'),
            versions={k: importlib.metadata.version(k) for k in ('numpy', 'lightgbm', 'joblib')},
            training_weight='Each training year has equal total weight',
            selection='Mean annual validation RMSE; 60-round early stopping; no post-hoc calibration',
            evaluation_split_loaded=False, remote_state='Observed target season, not future-state forecast')
        config = root / 'config.json'
        if config.exists() and json.loads(config.read_text()) != spec:
            raise ValueError('Frozen training specification changed')
        atomic_json(config, spec)
        if (root / 'audit.json').exists():
            a = json.loads((root / 'audit.json').read_text())
            for f, checksum in a['files'].items():
                if sha256(root / f) != checksum:
                    raise ValueError('Previously completed run changed')
            return
        if smoke:
            for s in arrays:
                take = sample_rows(arrays[s], 4096 if s == 'train' else 1024)
                x[s] = x[s][take]
                arrays[s] = {k: v[take] for k, v in arrays[s].items()}
        norm = manifest['spec']['upstream']['upstream']['normalization']
        scale, center = norm['residual_std'], norm['residual_mean']
        y = {s: a['target_residual'] for s, a in arrays.items()}
        for s, a in arrays.items():
            np.testing.assert_allclose(a['baseline'].astype(float)+scale*y[s]+center,
                                       a['target'], rtol=2e-6, atol=2e-6)
        years = arrays['validation']['year']

        def metric(truth, pred):
            scores = yearly_rmse(truth, pred, years)
            return 'year_macro_rmse', float(np.mean(list(scores.values()))), False

        started = time.monotonic()
        trace = {}
        with threadpool_limits(limits=4):
            model = lgb.LGBMRegressor(**params)
            model.fit(x['train'], y['train'], sample_weight=year_weights(arrays['train']['year']),
                eval_set=[(x['validation'], y['validation'])], eval_metric=metric,
                callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False),
                           lgb.record_evaluation(trace), lgb.log_evaluation(100)])
            weight = root / 'model.joblib'
            joblib.dump(model, weight)
            component = model.predict(x['validation'])
            restored = joblib.load(weight)
            replay = restored.predict(x['validation'])
            np.testing.assert_array_equal(component, replay)
        a = arrays['validation']
        prediction = a['baseline'].astype(float)+scale*component+center
        scores = yearly_rmse(a['target'], prediction, a['year'])
        path = root / 'validation_predictions.npz'
        np.savez_compressed(path, prediction=prediction, standardized_residual_prediction=component,
                            **{k: a[k] for k in ('target', 'baseline', 'year', 'row', 'col', 'source_indices')})
        with np.load(path) as saved:
            for k in ('target', 'baseline', 'year', 'row', 'col', 'source_indices'):
                np.testing.assert_array_equal(saved[k], a[k])
            np.testing.assert_array_equal(saved['prediction'], a['baseline'].astype(float)+scale*replay+center)
        atomic_json(root / 'training_history.json', trace)
        metrics = dict(crop=crop, origin=origin, condition=condition, seed=42, smoke=smoke,
            dimensions=x['train'].shape[1], selected_trees=int(model.best_iteration_),
            pooled_rmse=float(np.sqrt(np.mean((a['target'].astype(float)-prediction)**2))),
            mean_annual_rmse=float(np.mean(list(scores.values()))), per_year_rmse=scores,
            n_training=len(y['train']), n_validation=len(y['validation']),
            seconds=time.monotonic()-started, weight=str(weight), evaluation_split_loaded=False)
        atomic_json(root / 'metrics.json', metrics)
        files = {f: sha256(root / f) for f in ('config.json', 'model.joblib', 'validation_predictions.npz',
                                              'training_history.json', 'metrics.json')}
        atomic_json(root / 'audit.json', dict(fits=1, smoke=smoke, full_validation_replay=True,
            maximum_replay_error=0., files=files, evaluation_split_loaded=False,
            training_only_statistics=True, source_indices_verified=True))
        print(f'[SIGNAL FIT COMPLETE] {json.dumps(metrics)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=ORIGINS, required=True)
    p.add_argument('--condition', choices=CONDITIONS, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.crop, args.origin, args.condition, args.smoke)
