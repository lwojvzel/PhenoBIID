"""Registered direct seasonal controls; selection never uses test scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from forecast_bridge_data import (ROOT, fit_stats, climatology, history_features,
                                  identity_hash)
from inseason_direct_data import physical_inputs, split_rows, seasonal_features, RECIPES
from review_revision_data import sha256
from run_crop_signal_screen import year_weights, yearly_rmse
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_direct_controls_v1'
CODE = ('inseason_direct_data.py', 'run_inseason_direct_baselines.py',
        'forecast_bridge_data.py', 'ndvi_tail_replacement.py',
        'run_history_multimodal_baselines.py', 'observed_remote_benchmark.py',
        'inseason_signal_matching.py', 'dual_remote_data.py')


def score(y, prediction, year):
    annual = yearly_rmse(y, prediction, year)
    return dict(pooled_rmse=float(np.sqrt(np.mean((np.asarray(y, float)-prediction)**2))),
                mean_annual_rmse=float(np.mean(list(annual.values()))), per_year_rmse=annual)


def run(crop, fit_end=2009, smoke=False):
    root = OUT / ('smoke' if smoke else 'pipelines') / crop / f'fit_{fit_end}' / 'seed_42'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            saved = json.loads((root / 'complete.json').read_text())
            for name, digest in saved['files'].items():
                if sha256(root / name) != digest:
                    raise ValueError('Changed completed output')
            for name, digest in saved['code_sha256'].items():
                if sha256(ROOT / 'scripts' / name) != digest:
                    raise ValueError('Changed completed implementation')
            print(f'[REUSED] {root}', flush=True)
            return
        started = time.monotonic()
        a, provenance = physical_inputs(crop)
        rows = split_rows(a, fit_end)
        if smoke:
            for split, take in rows.items():
                rows[split] = np.concatenate([take[a['year'][take] == year][:128]
                                             for year in np.unique(a['year'][take])])
        stats = fit_stats(a, rows['train'])
        recipe = RECIPES[crop]
        ratios = (.1,) if smoke else (.1, .3, .5)
        code = {name: sha256(ROOT / 'scripts' / name) for name in CODE}
        params = dict(n_estimators=15 if smoke else 1200, learning_rate=.03,
                      num_leaves=15, min_child_samples=200, reg_lambda=10.,
                      subsample=.9, subsample_freq=1, colsample_bytree=.9,
                      n_jobs=2, random_state=42, verbosity=-1, deterministic=True,
                      force_col_wise=True, objective='regression', metric='None')
        spec = dict(crop=crop, recipe=recipe, fit_end=fit_end, seed=42, smoke=smoke,
                    code_sha256=code, normalization=stats, lightgbm=params,
                    ridge_alpha=[.1, 1., 10., 100., 1000.], ratios=ratios,
                    training_weight='Equal total weight for each training year',
                    selection='Mean annual validation RMSE only; no test selection',
                    anchor='Causal linear trend, not a learned or initialized expert',
                    weather='Supplied actual full-year weather condition',
                    input='History20, area1, previous-quality/calendar276, weather156, '
                          'tail12, each selected product: previous36+valid12+prefix36+valid12',
                    cohort='Existing LAI-filtered cohort; source-index exact',
                    comparison='Same issue-time information; not a same-head component ablation',
                    splits={s: dict(years=np.unique(a['year'][ix]).tolist(), rows=len(ix),
                                    identity=identity_hash(a['source_indices'][ix]))
                            for s, ix in rows.items()},
                    versions={n: importlib.metadata.version(n) for n in
                              ('numpy', 'lightgbm', 'scikit-learn', 'joblib')})
        config_file = root / 'config.json'
        # JSON normalization makes tuple/list representations consistent on resume.
        spec = json.loads(json.dumps(spec))
        if config_file.exists() and json.loads(config_file.read_text()) != spec:
            raise ValueError('Registered configuration changed')
        atomic_json(config_file, spec)
        atomic_json(root / 'provenance.json', provenance)
        anchors = {s: history_features(a, ix, stats)[1].astype(float) for s, ix in rows.items()}
        residual = {s: (a['target'][ix]-anchors[s])/stats['target_std'] for s, ix in rows.items()}
        climate = {s: {p: climatology(a, rows['train'], ix, p) for p in recipe.split('_')}
                   for s, ix in rows.items()}
        weights = year_weights(a['year'][rows['train']])
        results = []
        for ratio in ratios:
            features, tails = {}, {}
            for split, ix in rows.items():
                features[split], tails[split] = seasonal_features(a, ix, stats, recipe, ratio, climate[split])
            val_years = a['year'][rows['validation']]

            def metric(truth, pred):
                return 'annual_rmse', score(truth, pred, val_years)['mean_annual_rmse'], False

            for algorithm in ('ridge', 'lightgbm'):
                dest = root / f'tail_{int(round(ratio*100)):02d}' / algorithm
                dest.mkdir(parents=True, exist_ok=True)
                with threadpool_limits(limits=2):
                    if algorithm == 'ridge':
                        candidates = []
                        best = None
                        for alpha in spec['ridge_alpha']:
                            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))
                            model.fit(features['train'], residual['train'], ridge__sample_weight=weights)
                            value = metric(residual['validation'], model.predict(features['validation']))[1]
                            candidates.append(dict(alpha=alpha, validation_rmse=value*stats['target_std']))
                            if best is None or value < best[0]:
                                best = (value, model, alpha)
                        model = best[1]
                        selection = dict(alpha=best[2], candidates=candidates)
                    else:
                        model = lgb.LGBMRegressor(**params)
                        trace = {}
                        model.fit(features['train'], residual['train'], sample_weight=weights,
                                  eval_set=[(features['validation'], residual['validation'])],
                                  eval_metric=metric, callbacks=[
                                      lgb.early_stopping(60, first_metric_only=True, verbose=False),
                                      lgb.record_evaluation(trace)])
                        selection = dict(trees=int(model.best_iteration_), trace=trace)
                    joblib.dump(model, dest / 'model.joblib')
                    restored = joblib.load(dest / 'model.joblib')
                    metrics = {}
                    for split in ('validation', 'test'):
                        ix = rows[split]
                        predictor = model.booster_ if algorithm == 'lightgbm' else model
                        reloaded = restored.booster_ if algorithm == 'lightgbm' else restored
                        pred = anchors[split] + stats['target_std'] * predictor.predict(features[split])
                        replay = anchors[split] + stats['target_std'] * reloaded.predict(features[split])
                        np.testing.assert_array_equal(pred, replay)
                        labels = {k: a[k][ix] for k in ('target', 'year', 'row', 'col', 'source_indices')}
                        np.savez_compressed(dest / f'{split}_predictions.npz', prediction=pred,
                                            baseline=anchors[split], tail=tails[split], **labels)
                        metrics[split] = score(labels['target'], pred, labels['year'])
                record = dict(crop=crop, recipe=recipe, ratio=ratio, algorithm=algorithm,
                              feature_width=features['train'].shape[1], selection=selection,
                              metrics=metrics, model=str(dest / 'model.joblib'),
                              replay_max_error=0., actual_tail_fraction=float(
                                  tails['test'].sum()/(a['relative_valid'][rows['test']]>0).sum()))
                atomic_json(dest / 'metrics.json', record)
                results.append(record)
                print(f'[DIRECT] {crop} {ratio:.1f} {algorithm} '
                      f'val={metrics["validation"]["pooled_rmse"]:.6f} '
                      f'test={metrics["test"]["pooled_rmse"]:.6f}', flush=True)
        atomic_json(root / 'summary.json', results)
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
                 if p.is_file() and p.name not in ('run.lock', 'complete.json')}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=code,
                    smoke=smoke, seconds=time.monotonic()-started,
                    test_used_for_selection=False, trained_models=len(results)))
        print(f'[COMPLETE] {crop} fit={fit_end} {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES) + ('all',), required=True)
    p.add_argument('--fit-end', type=int, choices=(2009, 2011), default=2009)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(run, c, args.fit_end, args.smoke) for c in RECIPES]
            for job in jobs:
                job.result()
    else:
        run(args.crop, args.fit_end, args.smoke)
