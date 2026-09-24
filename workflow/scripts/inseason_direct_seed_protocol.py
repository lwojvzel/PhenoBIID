"""Seed-aware adapter for the two existing strongest direct seasonal families."""
import json

import joblib
import lightgbm as lgb
import numpy as np

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES
from inseason_complete_inputs import flat_features
from inseason_nested_common import LABELS
from review_revision_data import sha256
from run_inseason_13year_direct import CODE as LGB_CODE, root_for as lgb_root
from run_inseason_forest_cpu import CODE as RF_CODE, root_for as rf_root
from run_inseason_direct_baselines import score
from run_ndvi_signal_permutation import check_files
from run_yield_only_classical_baselines import build_model

SEEDS = (42, 45, 48)
PERCENTS = (10, 30, 50)
MODELS = ('lightgbm', 'random_forest')
OUT = ROOT / 'benchmark/results/inseason_direct_seeds_v1'
SOURCE_CODE = tuple(dict.fromkeys((*LGB_CODE, *RF_CODE)))


def identity(crop, cutoff, seed, model, percent):
    if crop not in RECIPES or cutoff not in BLOCKS or seed not in SEEDS or model not in MODELS or percent not in PERCENTS:
        raise ValueError('Unregistered direct-model repeat identity')
    return dict(crop=crop, cutoff=cutoff, seed=seed, model=model, percent=percent, recipe=RECIPES[crop])


def root_for(crop, cutoff, seed, model, percent):
    identity(crop, cutoff, seed, model, percent)
    return OUT / 'pipelines' / crop / f'cutoff_{cutoff}/seed_{seed}/{model}/tail_{percent:02d}'


def candidate_parameters(seed, smoke=False):
    if seed not in SEEDS:
        raise ValueError('Unregistered seed')
    return [dict(n_estimators=15 if smoke else 1200, learning_rate=.03,
        num_leaves=leaves, min_child_samples=child, reg_lambda=penalty,
        subsample=.9, subsample_freq=1, colsample_bytree=.9, n_jobs=2,
        random_state=seed, verbosity=-1, deterministic=True, force_col_wise=True,
        objective='regression', metric='None')
        for leaves, child, penalty in ((15, 200, 10.), (31, 40, 1.))]


def selection_contract(selection, seed, smoke=False):
    candidates = selection['candidates']
    if len(candidates) != 2:
        raise ValueError('Changed LightGBM candidate set')
    for row, parameters in zip(candidates, candidate_parameters(seed, smoke)):
        if row['parameters'] != parameters:
            raise ValueError('Changed candidate parameters or seed')
        if set(row['trace']) != {'valid_0'} or set(row['trace']['valid_0']) != {'annual_rmse'}:
            raise ValueError('Unexpected selection metric or validation dataset')
        trace = np.asarray(row['trace']['valid_0']['annual_rmse'], dtype=float)
        if (trace.ndim != 1 or not 1 <= len(trace) <= parameters['n_estimators'] or
                not np.isfinite(trace).all() or row['selected_trees'] != int(trace.argmin())+1):
            raise ValueError('Invalid pre-cutoff tree selection trace')
        # LightGBM's callback sees float32 labels; the final candidate score uses float64.
        np.testing.assert_allclose(row['score'], trace.min(), rtol=0, atol=1e-6)
    winner = int(np.argmin([row['score'] for row in candidates]))
    if selection['selected_candidate'] != winner:
        raise ValueError('Candidate did not minimize the registered inner metric')
    return dict(candidates[winner]['parameters'], n_estimators=candidates[winner]['selected_trees'])


def select_lightgbm(x, residual, weights, years, dest, seed, smoke=False):
    """Preserve the original arithmetic, callbacks, tie rule, and two candidates."""
    features = {s: flat_features(v) for s, v in x.items()}
    best, candidates = None, []
    for parameters in candidate_parameters(seed, smoke):
        def metric(truth, prediction):
            return 'annual_rmse', score(truth, prediction, years)['mean_annual_rmse'], False

        model = lgb.LGBMRegressor(**parameters)
        trace = {}
        model.fit(features['train'], residual['train'], sample_weight=weights,
            eval_set=[(features['other'], residual['other'])], eval_metric=metric,
            callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.record_evaluation(trace)])
        value = metric(residual['other'], model.booster_.predict(features['other']))[1]
        candidates.append(dict(parameters=parameters, selected_trees=int(model.best_iteration_), score=value, trace=trace))
        if best is None or value < best[0]:
            best = value, model, len(candidates)-1
    model = best[1]
    joblib.dump(model, dest / 'model.joblib')
    prediction = model.booster_.predict(features['other'])
    np.testing.assert_array_equal(prediction, joblib.load(dest / 'model.joblib').booster_.predict(features['other']))
    selection = dict(selected_candidate=best[2], candidates=candidates)
    selection_contract(selection, seed, smoke)
    return prediction, selection


def original(crop, cutoff, model, percent):
    identity(crop, cutoff, 42, model, percent)
    root = (lgb_root if model == 'lightgbm' else rf_root)(crop, cutoff, model)
    marker = json.loads((root / 'complete.json').read_text())
    code = LGB_CODE if model == 'lightgbm' else RF_CODE
    if marker['smoke'] or marker['code_sha256'] != {n: sha256(ROOT / 'scripts' / n) for n in code}:
        raise ValueError('Original baseline is changed or not a formal result')
    check_files(root, {'config.json': marker['files']['config.json']})
    config = json.loads((root / 'config.json').read_text())
    for key, value in dict(crop=crop, cutoff=cutoff, seed=42, model=model, recipe=RECIPES[crop], smoke=False).items():
        if config[key] != value:
            raise ValueError(f'Original baseline identity mismatch: {key}')
    if (config['ratios'] != [.1, .3, .5] or config['periods']['evaluation'] != list(BLOCKS[cutoff]) or
            config['periods']['inner_validation'] != [cutoff-1, cutoff] or
            config['cache_sha256'] != sha256(CACHE / crop / 'manifest.json')):
        raise ValueError('Original baseline cohort or time protocol mismatch')
    dest = root / f'tail_{percent:02d}'
    check_files(root, {f'tail_{percent:02d}/audit.json': marker['files'][f'tail_{percent:02d}/audit.json']})
    audit = json.loads((dest / 'audit.json').read_text())
    for name, digest in audit['files'].items():
        if marker['files'][f'tail_{percent:02d}/{name}'] != digest:
            raise ValueError('Original completion and ratio audit disagree')
    check_files(dest, audit['files'])
    metrics = json.loads((dest / 'metrics.json').read_text())
    if model == 'lightgbm':
        expected = selection_contract(metrics['selection'], 42)
    else:
        expected = build_model('random_forest', 42, 8).get_params()
        if config['forest'] != expected or metrics['selection'] != dict(fixed_parameters=True):
            raise ValueError('Changed original forest capacity')
        expected = dict(expected, n_jobs=1)
    fitted = joblib.load(dest / 'model.joblib')
    for key, value in expected.items():
        if fitted.get_params()[key] != value:
            raise ValueError(f'Original model parameters differ: {key}')
    return root, dest, config, metrics, fitted


def compare_labels(saved, labels):
    for key in LABELS:
        np.testing.assert_array_equal(saved[key], labels[key])
    ids = np.rec.fromarrays([labels[k] for k in ('year', 'row', 'col')])
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate evaluation identities')


def normalization_contract(actual, original_value):
    if json.loads(json.dumps(actual)) != original_value:
        raise ValueError('Seed repeat changed train-only feature or target normalization')


def physical_prediction(model, features, anchor, norm, name):
    if name == 'lightgbm':
        output = model.booster_.predict(features)
    else:
        if model.n_jobs != 1:
            raise ValueError('Forest inference must preserve serial summation')
        output = model.predict(features).astype(float)
    result = anchor+norm['center']+norm['scale']*output
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite direct yield prediction')
    return result
