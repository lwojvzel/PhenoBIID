"""Train every registered regional historical reference, without vegetation input."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from cybench_inseason_model_data import ROOT, RESULT, RECIPES, BLOCKS, SEEDS, balanced_weights, regional_score
from cybench_inseason_yield_models import (MODELS, hashes, load_data, history_stats, target_stats,
    history_features, select, refit, prediction)
from inseason_nested_common import register, finish, verify
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

CODE = ('run_cybench_inseason_history.py',)


def root_for(crop, cutoff, seed, model, smoke=False):
    return RESULT / ('history_smoke' if smoke else 'history') / crop / f'cutoff_{cutoff}/seed_{seed}' / model


def verify_completed(crop, cutoff, seed, model, smoke=False):
    root = root_for(crop, cutoff, seed, model, smoke)
    record = verify(root, hashes(CODE))
    job = dict(crop=crop, cutoff=cutoff, seed=seed, model=model, smoke=smoke)
    if record['job'] != job or record['input_width'] != 23 or record['untrained_candidate_allowed']:
        raise ValueError('Incorrect regional historical model record')
    config = json.loads((root / 'config.json').read_text())
    if config['job'] != job:
        raise ValueError('Incorrect regional historical model configuration')
    for path, digest in config['sources'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Regional historical source changed')
    return record


def run(crop, cutoff, seed, model, smoke=False):
    if crop not in RECIPES or cutoff not in BLOCKS or seed not in SEEDS or model not in MODELS:
        raise ValueError('Unregistered regional historical model')
    job = dict(crop=crop, cutoff=cutoff, seed=seed, model=model, smoke=smoke)
    root = root_for(**job)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            return verify_completed(**job)
        data = load_data(crop, cutoff, smoke, vegetation=False)
        ids, target, anchor, parts = [data[k] for k in ('identities', 'target', 'trend', 'parts')]
        fit, val, full, evaluation = [parts[k] for k in ('inner_fit', 'inner_validation', 'full_fit', 'evaluation')]
        code = hashes(CODE)
        config = dict(schema=1, job=job, sources=data['sources'], code_sha256=code,
            periods={key: sorted(ids.iloc[ix].year.unique().tolist()) for key, ix in parts.items()},
            counts={key: len(ix) for key, ix in parts.items()}, input_width=23,
            inputs='Five calendar yield lags, five valid flags, seven past summaries, lat/lon/year/three country indicators',
            anchor='Nonnegative causal past linear trend', objective='Country-year-balanced standardized residual squared loss',
            selection='Preceding two years; equal-country mean annual physical RMSE; no untrained candidates',
            evaluation_yield_file_loaded=True, evaluation_yield_used_for_selection=False,
            state_target_arrays_loaded=False, numerical_weather_or_vegetation_features=False,
            final_fits=0 if smoke else 1, fit_threads=2, forest_prediction_threads=1,
            model_optimization=False)
        register(root, config)
        started = time.monotonic()
        np.savez_compressed(root / 'partition_indices.npz', **{k: data['sample_index'][ix] for k, ix in parts.items()})
        with threadpool_limits(limits=2):
            inner_stats = history_stats(data['history'][fit])
            inner_residual = target_stats(target[fit], anchor[fit], balanced_weights(ids.iloc[fit]))
            atomic_json(root / 'inner_normalization.json', dict(features=inner_stats, residual=inner_residual))
            design = dict(fit=history_features(data, fit, inner_stats), validation=history_features(data, val, inner_stats))
            selection, pval = select(model, seed, design, target, anchor, ids, parts, inner_residual, root / 'inner', cutoff, smoke)
            np.savez_compressed(root / 'inner_validation_predictions.npz', sample_index=data['sample_index'][val],
                prediction=pval, target=target[val], year=ids.iloc[val].year.to_numpy(), anchor=anchor[val])
            ids.iloc[val].to_csv(root / 'inner_validation_identities.csv', index=False)
            full_stats = history_stats(data['history'][full])
            full_residual = target_stats(target[full], anchor[full], balanced_weights(ids.iloc[full]))
            atomic_json(root / 'normalization.json', dict(features=full_stats, residual=full_residual))
            features = history_features(data, full, full_stats)
            fitted = refit(model, selection['full_refit_parameters'], features, target[full], anchor[full], ids.iloc[full], full_residual)
            joblib.dump(fitted, root / 'model.joblib')
            other = history_features(data, evaluation, full_stats)
            p = prediction(fitted, other, anchor[evaluation], full_residual, model)
            np.testing.assert_array_equal(p, prediction(joblib.load(root / 'model.joblib'), other, anchor[evaluation], full_residual, model))
        score, annual = regional_score(target[evaluation], p, ids.iloc[evaluation])
        np.savez_compressed(root / 'evaluation_predictions.npz', sample_index=data['sample_index'][evaluation],
            prediction=p, target=target[evaluation], year=ids.iloc[evaluation].year.to_numpy(), anchor=anchor[evaluation])
        ids.iloc[evaluation].to_csv(root / 'evaluation_identities.csv', index=False)
        atomic_json(root / 'metrics.json', dict(selection=selection, score=score, annual=annual,
            inner_candidate_count=len(selection['candidates']), input_width=23, weight_replay_error=0.,
            final_parameters=fitted.get_params(), score_scope='Regional adaptation; countries retain unequal year coverage'))
        if hashes(CODE) != code:
            raise ValueError('Regional historical source changed during fit')
        for path, digest in data['sources'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('Regional historical data changed during fit')
        finish(root, code, job=job, new_final_models=0 if smoke else 1, seconds=time.monotonic()-started,
            input_width=23, untrained_candidate_allowed=False, full_training_rows=len(full),
            evaluation_rows=len(evaluation), external_world_performance_evaluated=False)
        print(f'[REGIONAL HISTORY] {job} inner={selection["score"]:.6f} outer={score:.6f}', flush=True)
        return verify_completed(**job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=RECIPES, required=True)
    parser.add_argument('--cutoff', type=int, choices=BLOCKS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--smoke', action='store_true')
    run(**vars(parser.parse_args()))
