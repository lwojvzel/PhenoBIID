"""Fixed crop-specific trees with inner-only early stopping on forward inputs."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from forecast_bridge_data import (ROOT, CROPS, ORIGINS, PRODUCTS, load, fit_stats,
                                  forecast_base, history_features, remote_features)
from export_forecast_bridge import run_root as export_root
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_crop_signal_screen import yearly_rmse, year_weights

RESULT = ROOT / 'benchmark/results/forecast_state_bridge_v1/readouts'
CONDITIONS = ('history', 'weather', 'previous', 'predicted', 'observed')
CODE = ('run_forecast_bridge_readout.py', 'forecast_bridge_data.py', 'export_forecast_bridge.py',
        'run_crop_signal_screen.py', 'observed_remote_benchmark.py')


def run_root(crop, origin, seed, product, architecture, condition):
    return RESULT / 'pipelines' / crop / product / architecture / f'origin_{origin}/seed_{seed}/{condition}'


def features(raw, cache, rows, stats, product, condition):
    take = cache['raw_indices'][rows]
    if condition == 'history':
        return history_features(raw, take, stats)[0]
    base = forecast_base(raw, take, stats)
    if condition == 'weather':
        return base
    if condition == 'predicted':
        physical = cache['predicted_state'][rows]
    elif condition in ('previous', 'observed'):
        physical = raw[f'{condition}_{product}'][take]
    else:
        raise ValueError(condition)
    climo = cache[f'climatology_{product}'] if f'climatology_{product}' in cache else cache['climatology']
    remote = remote_features(physical, raw['relative_valid'][take] > 0, climo[rows], stats[product])
    return np.concatenate((base, remote), 1)


def branches(crop):
    return ('trend', 'mlp') if crop == 'maize' else ('tabm',) if crop == 'soybean' else ('mlp',)


def compose(values, crop):
    if len(values) != len(branches(crop)):
        raise ValueError('Wrong fixed crop branch count')
    return .5*values[0]+.5*values[1] if crop == 'maize' else values[0]


def inner_split(years, origin):
    inner = np.flatnonzero(years <= origin-5)
    validation = np.flatnonzero((years > origin-5) & (years <= origin-3))
    full = np.flatnonzero(years <= origin-3)
    outer = np.flatnonzero((years > origin-3) & (years <= origin))
    if any(not len(x) for x in (inner, validation, full, outer)):
        raise ValueError('Empty readout split')
    return inner, validation, full, outer


def run(args):
    raw, _ = load(args.crop)
    source_product = getattr(args, 'source_product', None) or args.product
    if source_product != args.product and args.condition == 'predicted':
        raise ValueError('Cannot substitute predictions from a different physical product')
    src = export_root(args.crop, args.origin, args.seed, source_product, args.architecture)
    audit = json.loads((src / 'complete.json').read_text())
    if not audit['all_upstream_strictly_before_target'] or audit['evaluation_arrays_loaded']:
        raise ValueError('Unsafe upstream export')
    for name, digest in audit['files'].items():
        if sha256(src / name) != digest:
            raise ValueError('Export changed')
    with np.load(src / 'predictions.npz') as f:
        a = {k: f[k] for k in f.files}
    if source_product != args.product and f'climatology_{args.product}' not in a:
        raise ValueError('Missing product-specific forward climatology')
    for k in ('year', 'row', 'col', 'source_indices', 'target'):
        np.testing.assert_array_equal(a[k], raw[k][a['raw_indices']])
    inner, val, full, outer = inner_split(a['year'], args.origin)
    stats = fit_stats(raw, a['raw_indices'][inner])
    xf = features(raw, a, inner, stats, args.product, args.condition)
    xv = features(raw, a, val, stats, args.product, args.condition)
    full_stats = fit_stats(raw, a['raw_indices'][full])
    xfull = features(raw, a, full, full_stats, args.product, args.condition)
    xo = features(raw, a, outer, full_stats, args.product, args.condition)
    if not all(np.isfinite(x).all() for x in (xf, xv, xfull, xo)):
        raise ValueError('Nonfinite readout features')
    dest = run_root(args.crop, args.origin, args.seed, args.product, args.architecture, args.condition)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        params = dict(n_estimators=1200, learning_rate=.03, num_leaves=15, min_child_samples=200,
            reg_lambda=10., subsample=.9, subsample_freq=1, colsample_bytree=.9, n_jobs=4,
            random_state=args.seed, verbosity=-1, deterministic=True, force_col_wise=True,
            objective='regression', metric='None')
        spec = dict(**vars(args), parameters=params, dimensions=xf.shape[1],
            code_sha256={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            export_sha256=sha256(src / 'complete.json'), inner_years=np.unique(a['year'][val]).tolist(),
            evaluation_years=np.unique(a['year'][outer]).tolist(), full_training_years=np.unique(a['year'][full]).tolist(),
            inner_normalization=stats, full_normalization=full_stats,
            crop_branches=list(branches(args.crop)), forecast_inputs='No target-year RS quality in base',
            selection='Inner validation annual mean RMSE; refit fixed selected tree count on all allowed training rows',
            evaluation_arrays_loaded=False, calibration=False)
        file = dest / 'config.json'
        if file.exists() and json.loads(file.read_text()) != spec:
            raise ValueError('Frozen readout changed')
        atomic_json(file, spec)
        if (dest / 'complete.json').exists():
            for name, digest in json.loads((dest / 'complete.json').read_text())['files'].items():
                if sha256(dest / name) != digest:
                    raise ValueError('Changed readout result')
            return
        started = time.monotonic()
        parts, branch_records, files = [], [], {}
        for kind in branches(args.crop):
            base = a[f'history_{kind}']
            residual = a['target'][inner].astype(float)-base[inner]
            center, scale = float(residual.mean()), max(float(residual.std()), 1e-6)
            y = (residual-center)/scale
            yv = (a['target'][val].astype(float)-base[val]-center)/scale
            trace = {}

            def metric(truth, pred):
                return 'annual_rmse', float(np.mean(list(yearly_rmse(truth, pred, a['year'][val]).values()))), False

            model = lgb.LGBMRegressor(**params)
            model.fit(xf, y, sample_weight=year_weights(a['year'][inner]), eval_set=[(xv, yv)], eval_metric=metric,
                callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False),
                           lgb.record_evaluation(trace), lgb.log_evaluation(200)])
            ntrees = int(model.best_iteration_)
            atomic_json(dest / f'{kind}_inner_history.json', trace)
            residual = a['target'][full].astype(float)-base[full]
            cfull, sfull = float(residual.mean()), max(float(residual.std()), 1e-6)
            final_params = dict(params, n_estimators=ntrees)
            model = lgb.LGBMRegressor(**final_params)
            model.fit(xfull, (residual-cfull)/sfull, sample_weight=year_weights(a['year'][full]))
            weight = dest / f'{kind}.joblib'
            joblib.dump(model, weight)
            prediction = base[outer]+cfull+sfull*model.predict(xo)
            np.testing.assert_array_equal(prediction, base[outer]+cfull+sfull*joblib.load(weight).predict(xo))
            parts.append(prediction)
            branch_records.append(dict(branch=kind, selected_trees=ntrees,
                inner_center=center, inner_scale=scale, center=cfull, scale=sfull))
            for name in (weight.name, f'{kind}_inner_history.json'):
                files[name] = sha256(dest / name)
        p = compose(parts, args.crop)
        np.savez_compressed(dest / 'validation_predictions.npz', prediction=p,
            **{k: a[k][outer] for k in ('target', 'year', 'row', 'col', 'source_indices')},
            **{f'branch_{k}': v for k, v in zip(branches(args.crop), parts)})
        atomic_json(dest / 'metrics.json', dict(crop=args.crop, origin=args.origin, seed=args.seed,
            product=args.product, architecture=args.architecture, condition=args.condition,
            per_year_rmse=yearly_rmse(a['target'][outer], p, a['year'][outer]),
            branches=branch_records, training_rows=len(full), seconds=time.monotonic()-started))
        for name in ('config.json', 'metrics.json', 'validation_predictions.npz'):
            files[name] = sha256(dest / name)
        atomic_json(dest / 'complete.json', dict(full_validation_replay=True, forward_upstreams=True,
            inner_only_early_stopping=True, evaluation_arrays_loaded=False, files=files,
            terminal_refits=len(parts), early_stopping_fits=len(parts)))
        print(f'[READOUT COMPLETE] {args.crop} {args.origin} {args.condition}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', choices=ORIGINS, type=int, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    parser.add_argument('--product', choices=PRODUCTS, default='ndvi')
    parser.add_argument('--source-product', choices=PRODUCTS)
    parser.add_argument('--architecture', choices=('biid', 'gru'), default='biid')
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)
