"""Fixed regional regressors and country/year-balanced residual objectives."""
import json

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from cybench_inseason_model_data import (ROOT, DATA_CODE, RECIPES, BLOCKS, SEEDS,
    load_identity, load_targets, load_seasons, history_matrix, mean_std,
    normalized_history, balanced_weights, regional_score)
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

MODELS = ('ridge', 'random_forest', 'lightgbm')
COMMON_CODE = (*DATA_CODE, 'cybench_inseason_yield_models.py', 'inseason_nested_common.py',
    'run_ndvi_signal_permutation.py', 'review_revision_data.py', 'run_review_revision_parallel.py')


def hashes(extra=()):
    return {name: sha256(ROOT / 'scripts' / name) for name in (*COMMON_CODE, *extra)}


def subset(arrays, take):
    return {key: value[take] for key, value in arrays.items()}


def load_data(crop, cutoff, smoke=False, vegetation=False):
    ids, history, parts, sources = load_identity(crop, cutoff)
    target = load_targets(crop, sources)
    if smoke:
        sampled = {}
        for part in ('inner_fit', 'inner_validation', 'evaluation'):
            selected = ids.iloc[parts[part]]
            sampled[part] = selected.groupby(['country', 'year'], sort=True).head(8).sample_index.to_numpy(int)
        sampled['full_fit'] = np.sort(np.concatenate((sampled['inner_fit'], sampled['inner_validation'])))
        parts = sampled
    indices = np.union1d(parts['full_fit'], parts['evaluation'])
    known, targets = load_seasons(crop, ids, indices, sources, targets=vegetation)
    local = {key: np.searchsorted(indices, ix) for key, ix in parts.items()}
    for key, ix in local.items():
        np.testing.assert_array_equal(indices[ix], parts[key])
    selected = ids.iloc[indices].reset_index(drop=True)
    h = subset(history, indices)
    x = history_matrix(h, selected, known)
    y = target[indices]
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Regional cohort contains an invalid paired yield')
    anchor = np.maximum(h['history_summary'][:, 3].astype(np.float64), 0)
    if not np.isfinite(anchor).all():
        raise ValueError('Missing causal historical trend')
    return dict(identities=selected, sample_index=indices, history=x, target=y,
        trend=anchor, known=known, state_targets=targets, parts=local, sources=sources)


def history_stats(history):
    mean, std = mean_std(history, axis=0)
    return dict(history_mean=mean.tolist(), history_std=std.tolist())


def target_stats(target, anchor, weight):
    residual = np.asarray(target, float)-np.asarray(anchor, float)
    weight = np.asarray(weight, float)
    if residual.ndim != 1 or weight.shape != residual.shape or not np.isfinite(residual).all():
        raise ValueError('Invalid residual normalization input')
    if not len(weight) or not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError('Residual weights must be finite and positive')
    center = float(np.average(residual, weights=weight))
    scale = max(float(np.sqrt(np.average((residual-center)**2, weights=weight))), 1e-6)
    return dict(center=center, scale=scale)


def standardized(target, anchor, stats):
    return (np.asarray(target, float)-np.asarray(anchor, float)-stats['center'])/stats['scale']


def physical(raw, anchor, stats):
    result = np.maximum(np.asarray(anchor, float)+stats['center']+stats['scale']*np.asarray(raw, float), 0)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite regional yield prediction')
    return result


def candidates(name, seed, smoke=False):
    if name not in MODELS or seed not in SEEDS:
        raise ValueError('Unregistered regional regressor or seed')
    if name == 'ridge':
        return [dict(alpha=1., random_state=seed)]
    if name == 'random_forest':
        return [dict(n_estimators=8 if smoke else 128, max_depth=20, min_samples_leaf=8,
            max_features=.75, bootstrap=True, random_state=seed, n_jobs=2)]
    base = dict(n_estimators=5 if smoke else 1200, learning_rate=.03,
        subsample=.9, subsample_freq=1, colsample_bytree=.9, random_state=seed,
        n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
        objective='regression', metric='None')
    return [dict(**base, num_leaves=leaves, min_child_samples=child, reg_lambda=penalty)
        for leaves, child, penalty in ((15, 200, 10.), (31, 40, 1.))]


def construct(name, parameters):
    classes = dict(ridge=Ridge, random_forest=RandomForestRegressor, lightgbm=lgb.LGBMRegressor)
    if name not in classes:
        raise ValueError('Unknown regional regressor')
    return classes[name](**parameters)


def raw_prediction(model, features, name):
    if name == 'lightgbm':
        return model.booster_.predict(features, num_threads=2)
    if name == 'random_forest' and model.n_jobs != 1:
        raise ValueError('Forest prediction must use serial tree summation')
    return model.predict(features).astype(np.float64)


def prediction(model, features, anchor, stats, name):
    return physical(raw_prediction(model, features, name), anchor, stats)


def select(name, seed, features, target, anchor, identities, parts, stats, root, cutoff, smoke=False):
    fit, val = parts['inner_fit'], parts['inner_validation']
    if identities.iloc[fit].year.max() > cutoff-2 or set(identities.iloc[val].year) != {cutoff-1, cutoff}:
        raise ValueError('Regional selection must use preceding years only')
    train_weights = balanced_weights(identities.iloc[fit])
    residual = standardized(target, anchor, stats)
    expected = target_stats(target[fit], anchor[fit], train_weights)
    if expected != stats:
        raise ValueError('Selection residual scale not fitted on inner training rows')
    root.mkdir(parents=True, exist_ok=True)
    records, best, selected, best_prediction = [], float('inf'), None, None

    def evaluation_metric(unused_label, raw):
        p = physical(raw, anchor[val], stats)
        value, _ = regional_score(target[val], p, identities.iloc[val])
        return 'country_year_rmse', value, False

    for index, params in enumerate(candidates(name, seed, smoke)):
        model = construct(name, params)
        extra = {}
        if name == 'lightgbm':
            extra = dict(eval_set=[(features['validation'], residual[val])], eval_metric=evaluation_metric,
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        model.fit(features['fit'], residual[fit], sample_weight=train_weights, **extra)
        final_params = dict(params)
        if name == 'random_forest':
            model.set_params(n_jobs=1)
        if name == 'lightgbm':
            final_params['n_estimators'] = max(int(model.best_iteration_ or params['n_estimators']), 1)
        p = prediction(model, features['validation'], anchor[val], stats, name)
        score, annual = regional_score(target[val], p, identities.iloc[val])
        filename = f'candidate_{index}.joblib'
        joblib.dump(model, root / filename)
        np.testing.assert_array_equal(p, prediction(joblib.load(root / filename), features['validation'], anchor[val], stats, name))
        records.append(dict(index=index, score=score, annual=annual, parameters=params,
            full_refit_parameters=final_params, weight=filename, weight_sha256=sha256(root / filename)))
        if score < best:
            best, selected, best_prediction = score, index, p.copy()
    if selected is None:
        raise ValueError('No finite trained regional regressor selected')
    result = dict(model=name, seed=seed, selected_candidate=selected, score=best,
        candidates=records, full_refit_parameters=records[selected]['full_refit_parameters'],
        criterion='Country-equal mean of annual physical yield RMSE after nonnegative clipping',
        selected_weight=str(root / records[selected]['weight']), untrained_candidate_allowed=False)
    atomic_json(root / 'selection.json', result)
    return result, best_prediction


def refit(name, parameters, features, target, anchor, identities, stats):
    weights = balanced_weights(identities)
    if target_stats(target, anchor, weights) != stats:
        raise ValueError('Full residual scale not fitted on full training rows')
    model = construct(name, parameters)
    model.fit(features, standardized(target, anchor, stats), sample_weight=weights)
    if name == 'random_forest':
        model.set_params(n_jobs=1)
    return model


def history_features(data, take, stats):
    return normalized_history(data['history'][take], stats)
