"""Fixed direct-prefix regressors and observed-trained residual terminal heads."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from cybench_inseason_model_data import (ROOT, RESULT, RECIPES, SEEDS, BLOCKS, PERCENTS,
    fit_feature_stats, feature_matrix, state_arrays, normalized_history,
    balanced_weights, regional_score)
from cybench_seasonal_inputs import issue_view
from cybench_inseason_yield_models import (hashes, subset, load_data, target_stats,
    select, refit, prediction)
from inseason_nested_common import register, finish, verify
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_cybench_inseason_history import verify_completed as verify_history

CODE = ('run_cybench_inseason_readout.py', 'run_cybench_inseason_history.py')


def validate_job(route, crop, cutoff, seed, model, percent, smoke=False):
    if crop not in RECIPES or cutoff not in BLOCKS or seed not in SEEDS:
        raise ValueError('Unregistered regional crop/cutoff/seed')
    if route == 'direct':
        valid = model in ('random_forest', 'lightgbm') and percent in PERCENTS
    elif route == 'terminal':
        valid = model == 'lightgbm' and percent == 0
    else:
        valid = False
    if not valid:
        raise ValueError('Unregistered regional route/model/cutoff percentage')
    if smoke and (cutoff != 2001 or seed != 42):
        raise ValueError('Smoke checks use the registered first cutoff and seed')


def root_for(route, crop, cutoff, seed, model, percent, smoke=False):
    return RESULT / ('heads_smoke' if smoke else 'heads') / route / crop / f'cutoff_{cutoff}/seed_{seed}' / model / f'tail_{percent:03d}'


def history_expert(crop, cutoff, seed, sources, smoke=False):
    gate = RESULT / ('history_smoke_verification.json' if smoke else 'history_verification.json')
    record = json.loads(gate.read_text())
    if not record['passed'] or record['models'] != (6 if smoke else 54) or record['selection_uses_outer_scores']:
        raise ValueError('Independently verified inner-selected historical experts required')
    if record['verifier_sha256'] != sha256(ROOT / 'scripts/verify_cybench_inseason_history.py'):
        raise ValueError('Historical verification implementation changed')
    path = RESULT / ('history_smoke_summary' if smoke else 'history_summary') / 'experts.json'
    if sha256(path) != record['sources'][str(path)]:
        raise ValueError('Frozen historical expert definition changed')
    matches = [row for row in json.loads(path.read_text())
               if (row['crop'], row['cutoff'], row['seed']) == (crop, cutoff, seed)]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate same-seed historical expert')
    expert = matches[0]
    verify_history(crop, cutoff, seed, expert['model'], smoke)
    for field in ('full_weight', 'full_normalization', 'inner_weight', 'inner_normalization', 'completed_model'):
        file = Path(expert[field])
        digest = sha256(file)
        if digest != record['sources'][str(file)]:
            raise ValueError('Historical expert asset differs from independent audit')
        sources[str(file)] = digest
    for file in (gate, path):
        sources[str(file)] = sha256(file)
    return expert


def historical_prediction(data, expert, stage):
    norm = json.loads(Path(expert[f'{stage}_normalization']).read_text())
    fitted = joblib.load(expert[f'{stage}_weight'])
    features = normalized_history(data['history'], norm['features'])
    return prediction(fitted, features, data['trend'], norm['residual'], expert['model'])


def design(data, fit, other, products, percent, cutoff):
    kfit, tfit = subset(data['known'], fit), subset(data['state_targets'], fit)
    norm = fit_feature_stats(data['history'][fit], kfit, tfit, products, cutoff)
    training = state_arrays(kfit, tfit)
    training_values = {p: training[f'observed_{p}'] for p in products}
    train_view = issue_view(kfit, tfit, percent)
    other_view = issue_view(subset(data['known'], other), subset(data['state_targets'], other), percent)
    xfit = feature_matrix(data['history'][fit], train_view, norm, products, training_values=training_values)
    xother = feature_matrix(data['history'][other], other_view, norm, products)
    return dict(fit=xfit, validation=xother), norm


def verify_completed(route, crop, cutoff, seed, model, percent, smoke=False):
    job = dict(route=route, crop=crop, cutoff=cutoff, seed=seed, model=model, percent=percent, smoke=smoke)
    validate_job(**job)
    root = root_for(**job)
    record = verify(root, hashes(CODE))
    expected_width = 578+96*len(RECIPES[crop])
    if record['job'] != job or record['input_width'] != expected_width:
        raise ValueError('Regional readout identity or dimensions changed')
    config = json.loads((root / 'config.json').read_text())
    if config['job'] != job:
        raise ValueError('Regional readout configuration identity mismatch')
    for path, digest in config['sources'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Regional readout source changed')
    return record


def run(route, crop, cutoff, seed, model, percent, smoke=False):
    job = dict(route=route, crop=crop, cutoff=cutoff, seed=seed, model=model, percent=percent, smoke=smoke)
    validate_job(**job)
    root = root_for(**job)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            return verify_completed(**job)
        data = load_data(crop, cutoff, smoke, vegetation=True)
        ids, target, parts = [data[k] for k in ('identities', 'target', 'parts')]
        fit, val, full, evaluation = [parts[key] for key in ('inner_fit', 'inner_validation', 'full_fit', 'evaluation')]
        expert = history_expert(crop, cutoff, seed, data['sources'], smoke) if route == 'terminal' else None
        inner_anchor = historical_prediction(data, expert, 'inner') if expert else data['trend']
        full_anchor = historical_prediction(data, expert, 'full') if expert else data['trend']
        products = RECIPES[crop]
        code = hashes(CODE)
        spec = dict(schema=1, job=job, sources=data['sources'], code_sha256=code,
            products=list(products), input_width=578+96*len(products), historical_expert=expert,
            anchor='Frozen inner-selected historical expert' if expert else 'Nonnegative causal historical trend',
            periods={key: sorted(ids.iloc[ix].year.unique().tolist()) for key, ix in parts.items()},
            counts={key: len(ix) for key, ix in parts.items()},
            feature_builder='Same regional calendar/weather/support/history/trajectory interface for direct and terminal',
            state_training='Separate fixed state models; no state optimization in this driver',
            head_training='Complete observed trajectories' if route == 'terminal' else 'Only issued observed prefixes',
            full_weather_supplied=True, evaluation_targets_loaded=True, evaluation_targets_used_for_selection=False,
            hidden_remote_values_or_support_in_direct_input=False, history_expert_refitted=False,
            model_optimization=False, fit_threads=2, forest_prediction_threads=1)
        register(root, spec)
        started = time.monotonic()
        np.savez_compressed(root / 'partition_indices.npz', **{key: data['sample_index'][ix] for key, ix in parts.items()})
        np.savez_compressed(root / 'history_anchors.npz', sample_index=data['sample_index'], inner=inner_anchor, full=full_anchor)
        with threadpool_limits(limits=2):
            features, stats = design(data, fit, val, products, percent, cutoff-2)
            residual = target_stats(target[fit], inner_anchor[fit], balanced_weights(ids.iloc[fit]))
            atomic_json(root / 'inner_normalization.json', dict(features=stats, residual=residual))
            selection, pval = select(model, seed, features, target, inner_anchor, ids, parts, residual, root / 'inner', cutoff, smoke)
            np.savez_compressed(root / 'inner_validation_predictions.npz', sample_index=data['sample_index'][val],
                prediction=pval, target=target[val], year=ids.iloc[val].year.to_numpy(), anchor=inner_anchor[val])
            ids.iloc[val].to_csv(root / 'inner_validation_identities.csv', index=False)
            features, stats = design(data, full, evaluation, products, percent, cutoff)
            residual = target_stats(target[full], full_anchor[full], balanced_weights(ids.iloc[full]))
            atomic_json(root / 'normalization.json', dict(features=stats, residual=residual))
            fitted = refit(model, selection['full_refit_parameters'], features['fit'], target[full], full_anchor[full], ids.iloc[full], residual)
            joblib.dump(fitted, root / 'model.joblib')
            p = prediction(fitted, features['validation'], full_anchor[evaluation], residual, model)
            np.testing.assert_array_equal(p, prediction(joblib.load(root / 'model.joblib'), features['validation'], full_anchor[evaluation], residual, model))
        score, annual = regional_score(target[evaluation], p, ids.iloc[evaluation])
        np.savez_compressed(root / 'evaluation_predictions.npz', sample_index=data['sample_index'][evaluation],
            prediction=p, target=target[evaluation], year=ids.iloc[evaluation].year.to_numpy(), anchor=full_anchor[evaluation])
        ids.iloc[evaluation].to_csv(root / 'evaluation_identities.csv', index=False)
        np.savez_compressed(root / 'replay_features.npz', sample_index=data['sample_index'][evaluation[:32]],
            features=features['validation'][:32], prediction=p[:32], anchor=full_anchor[evaluation[:32]])
        atomic_json(root / 'metrics.json', dict(score=score, annual=annual, selection=selection,
            input_width=features['fit'].shape[1], weight_replay_error=0., final_parameters=fitted.get_params(),
            evaluation_condition='Full observed diagnostic' if route == 'terminal' else 'Issued prefix only',
            predicted_state_yield_evaluated=False))
        if hashes(CODE) != code:
            raise ValueError('Readout implementation changed during fit')
        for path, digest in data['sources'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('Readout source changed during fit')
        finish(root, code, job=job, input_width=features['fit'].shape[1], new_final_models=0 if smoke else 1,
            seconds=time.monotonic()-started, full_training_rows=len(full), evaluation_rows=len(evaluation),
            predicted_state_yield_evaluated=False)
        print(f'[REGIONAL READOUT] {job} inner={selection["score"]:.6f} outer={score:.6f}', flush=True)
        return verify_completed(**job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('direct', 'terminal'), required=True)
    parser.add_argument('--crop', choices=RECIPES, required=True)
    parser.add_argument('--cutoff', type=int, choices=BLOCKS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--model', choices=('random_forest', 'lightgbm'), required=True)
    parser.add_argument('--percent', type=int, choices=(0, *PERCENTS), required=True)
    parser.add_argument('--smoke', action='store_true')
    run(**vars(parser.parse_args()))
