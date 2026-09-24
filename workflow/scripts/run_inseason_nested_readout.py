"""Nested original-structure and forward-state-adapted terminal yield heads."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from inseason_13year_data import CACHE, load, partition
from inseason_nested_common import (ROOT, BLOCKS, RECIPES, RATIOS, LABELS,
                                    folder, hashes, verify, finish, register, branches)
from inseason_nested_features import context, features
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_inseason_direct_baselines import score
from run_inseason_nested_states import forward_cache
from run_review_revision_parallel import atomic_json

EXTRA = ('run_inseason_nested_readout.py', 'inseason_nested_features.py',
         'run_inseason_nested_states.py', 'run_inseason_direct_baselines.py')
SUITES = dict(observed=('observed_all', 'observed_common'),
              adapted=('prefix', 'climatology', 'biid'))


def phase_data(raw, crop, cutoff, seed, stage, forward, smoke):
    groups = partition(raw, cutoff)
    fit_key, other_key = ('inner_fit', 'inner_validation') if stage == 'inner' else ('full_fit', 'evaluation')
    statistics_fit = groups[fit_key]
    fit, other = statistics_fit, groups[other_key]
    if smoke:
        fit, other = [np.concatenate([ix[raw['year'][ix] == y][:16]
                       for y in np.unique(raw['year'][ix])]) for ix in (fit, other)]
    take = np.concatenate((fit, other))
    with np.load(folder('history', crop, cutoff, seed) / stage / 'predictions.npz') as saved:
        pos = np.searchsorted(saved['raw_indices'], take)
        np.testing.assert_array_equal(saved['raw_indices'][pos], take)
        for key in LABELS:
            np.testing.assert_array_equal(saved[key][pos], raw[key][take])
        anchors = {k: saved[k][pos] for k in ('trend', 'mlp', *(['tabm'] if crop == 'soybean' else []))}
    ctx = context(raw, statistics_fit, take, RECIPES[crop])
    source, source_ix, _ = forward
    products = RECIPES[crop].split('_')
    warm = np.flatnonzero(raw['year'][fit] >= 1993)
    source_pos = np.searchsorted(source_ix, fit[warm])
    np.testing.assert_array_equal(source_ix[source_pos], fit[warm])
    state = folder('states', crop, cutoff, 42)
    with np.load(state / f'{stage}_identities.npz') as saved:
        state_pos = np.searchsorted(saved['raw_indices'], other)
        np.testing.assert_array_equal(saved['raw_indices'][state_pos], other)
        for key in LABELS:
            np.testing.assert_array_equal(saved[key][state_pos], raw[key][other])
    climate = {p: ctx['climate'][p].copy() for p in products}
    with np.load(source / 'climatology.npz') as saved:
        for p in products:
            climate[p][warm] = saved[p][source_pos]
    predicted = {}
    for ratio in RATIOS:
        predicted[ratio] = {p: np.zeros((len(take), 12), np.float32) for p in products}
        with np.load(source / f'prefix_{round(100*ratio):02d}.npz') as saved:
            for p in products:
                predicted[ratio][p][warm] = saved[p][source_pos]
        with np.load(state / f'{stage}_prefix_{round(100*ratio):02d}.npz') as saved:
            for p in products:
                predicted[ratio][p][len(fit):] = saved[p][state_pos]
    return dict(context=ctx, take=take, train=np.arange(len(fit)), common_train=warm,
        other=np.arange(len(fit), len(take)), anchors=anchors,
        target=raw['target'][take].astype(float), year=raw['year'][take],
        predictions=predicted, forward_climate=climate)


def encoded(data, positions, ratio, mode):
    return features(data['context'], positions, ratio, mode,
        data['predictions'].get(ratio), data['forward_climate'] if mode == 'climatology' else None)


def parameters(seed, smoke):
    return dict(n_estimators=8 if smoke else 1200, learning_rate=.03,
        num_leaves=15, min_child_samples=200, reg_lambda=10., subsample=.9,
        subsample_freq=1, colsample_bytree=.9, n_jobs=2, random_state=seed,
        verbosity=-1, deterministic=True, force_col_wise=True,
        objective='regression', metric='None')


def train_branch(data, indices, xfit, xother, branch, seed, dest, smoke, trees=None):
    other = data['other']
    base = data['anchors'][branch]
    residual = data['target']-base
    center, scale = float(residual[indices].mean()), max(float(residual[indices].std()), 1e-6)
    y = (residual-center)/scale
    params = parameters(seed, smoke)
    if trees is not None:
        params['n_estimators'] = trees
    model = lgb.LGBMRegressor(**params)
    trace = {}
    with threadpool_limits(limits=2):
        kwargs = {}
        if trees is None:
            def metric(truth, prediction):
                return ('annual_rmse', score(truth, prediction, data['year'][other])['mean_annual_rmse'], False)
            kwargs = dict(eval_set=[(xother, y[other])], eval_metric=metric,
                callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.record_evaluation(trace)])
        model.fit(xfit, y[indices], sample_weight=year_weights(data['year'][indices]), **kwargs)
    selected = int(model.best_iteration_) if trees is None else trees
    if selected < 1:
        raise ValueError('Unfitted terminal tree')
    joblib.dump(model, dest / 'model.joblib')
    restored = joblib.load(dest / 'model.joblib')
    output = model.booster_.predict(xother)
    np.testing.assert_array_equal(output, restored.booster_.predict(xother))
    atomic_json(dest / 'normalization.json', dict(center=center, scale=scale))
    atomic_json(dest / 'training.json', dict(selected_trees=selected, parameters=params,
        training_years=np.unique(data['year'][indices]).tolist(), training_rows=len(indices),
        trace=trace, full_weight_replay=True))
    return model, selected, center, scale


def outputs_for(condition, ratio):
    if condition.startswith('observed'):
        return [(0., 'observed')]+[(r, m) for r in RATIOS for m in ('prefix', 'climatology', 'biid')]
    if condition == 'biid':
        return [(ratio, m) for m in ('prefix', 'climatology', 'biid')]
    return [(ratio, condition)]


def run(crop, cutoff, seed, suite, smoke=False):
    root = folder('readout_'+suite, crop, cutoff, seed, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(EXTRA)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        history = folder('history', crop, cutoff, seed)
        states = folder('states', crop, cutoff, 42)
        verify(history)
        verify(states)
        raw, _ = load(crop)
        forward = forward_cache(crop, raw)
        conditions = ('observed_common', 'biid') if smoke else SUITES[suite]
        sources = {str(p / 'complete.json'): sha256(p / 'complete.json') for p in (history, states, forward[0])}
        register(root, dict(crop=crop, cutoff=cutoff, seed=seed, suite=suite, smoke=smoke,
            code_sha256=code, cache_sha256=sha256(CACHE / crop / 'manifest.json'), sources=sources,
            conditions=conditions, recipe=RECIPES[crop], ratios=RATIOS,
            branches=branches(crop), tree_parameters=parameters(seed, smoke),
            observed_selection='Complete observed inner-year trajectories; frozen across issue positions',
            adapted_selection='Causal prefix completion at fixed issue position in preceding inner years',
            objective='Year-balanced residual MSE; mean annual inner validation RMSE',
            state_seed=42, historical_predictions_on_training='In-sample frozen trained expert',
            feature_layout='history20 + metadata289 + weather156 + 36 per product',
            anomaly_training='Leave queried fitting record out of grid/month climatology',
            normalization_fit='All available years to stage fit cutoff; common across conditions',
            common_head_warmup=1993))
        started = time.monotonic()
        phases = {s: phase_data(raw, crop, cutoff, seed, s, forward, smoke) for s in ('inner', 'full')}
        for stage, data in phases.items():
            atomic_json(root / f'{stage}_normalization.json', data['context']['stats'])
        records = []
        for condition in conditions:
            ratios = (0.,) if condition.startswith('observed') else ((.1,) if smoke else RATIOS)
            for ratio in ratios:
                dest = root / condition / f'train_tail_{round(100*ratio):02d}'
                dest.mkdir(parents=True, exist_ok=True)
                if (dest / 'complete.json').exists():
                    verify(dest, code)
                    continue
                mode = 'observed' if condition.startswith('observed') else condition
                results = {key: np.zeros(len(phases['full']['other'])) for key in outputs_for(condition, ratio)}
                selected = {}
                for stage, data in phases.items():
                    index = data['train'] if condition == 'observed_all' else data['common_train']
                    xfit = encoded(data, index, ratio, mode)
                    xother = encoded(data, data['other'], ratio, mode)
                    for branch, weight in branches(crop):
                        destination = dest / stage / branch
                        destination.mkdir(parents=True, exist_ok=True)
                        model, trees, center, scale = train_branch(data, index, xfit, xother,
                            branch, seed, destination, smoke, None if stage == 'inner' else selected[branch])
                        selected[branch] = trees
                        if stage == 'full':
                            other = data['other']
                            for (issue, completion), prediction in results.items():
                                inputs = encoded(data, other, issue, completion)
                                component = model.booster_.predict(inputs)
                                prediction += weight*(data['anchors'][branch][other]+center+scale*component)
                    del xfit, xother
                full = phases['full']
                labels = {k: raw[k][full['take'][full['other']]] for k in LABELS}
                metrics = []
                for (issue, completion), prediction in results.items():
                    name = f'tail_{round(100*issue):03d}_{completion}'
                    np.savez_compressed(dest / f'{name}.npz', prediction=prediction, **labels)
                    row = dict(condition=condition, training_suffix=ratio, suffix=issue,
                        completion=completion, scores=score(labels['target'], prediction, labels['year']))
                    metrics.append(row)
                    records.append(row)
                atomic_json(dest / 'metrics.json', dict(results=metrics, selected_trees=selected))
                finish(dest, code, selection_precedes_evaluation=True, readout_refit=True)
                primary = next(x for x in metrics if x['completion'] == ('biid' if mode == 'observed' else mode)
                               and x['suffix'] == (.1 if mode == 'observed' else ratio))
                print(f'[NESTED READOUT] {crop} {cutoff} {condition} {ratio} annual={primary["scores"]["mean_annual_rmse"]:.6f}', flush=True)
        finish(root, code, smoke=smoke, seconds=time.monotonic()-started,
            historical_weights_trained=True, state_weights_reused=True,
            all_fitting_and_selection_before_evaluation=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES)+('all',), required=True)
    p.add_argument('--cutoff', choices=tuple(BLOCKS), type=int)
    p.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    p.add_argument('--suite', choices=tuple(SUITES), required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(run, c, t, args.seed, args.suite, args.smoke)
                           for c in RECIPES for t in BLOCKS]:
                future.result()
    elif args.cutoff is None:
        p.error('--cutoff is required for a single crop')
    else:
        run(args.crop, args.cutoff, args.seed, args.suite, args.smoke)
