"""Repeat direct baselines without changing the original completed drivers."""
import argparse
import fcntl
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES, load, partition
from inseason_complete_inputs import flat_features
from inseason_direct_seed_protocol import (OUT, SOURCE_CODE, SEEDS, PERCENTS, MODELS,
    root_for, identity, original, select_lightgbm, selection_contract,
    normalization_contract, compare_labels, physical_prediction)
from inseason_nested_common import LABELS, hashes, verify, register, finish
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_inseason_13year_direct import make_design
from run_inseason_direct_baselines import score
from run_review_revision_parallel import atomic_json
from run_yield_only_classical_baselines import build_model

CODE = (*SOURCE_CODE, 'inseason_direct_seed_protocol.py', 'run_inseason_direct_seed.py')


def require_replay():
    path = OUT / 'original_replay_verification.json'
    value = json.loads(path.read_text())
    if (not value['passed'] or value['conditions'] != 72 or value['maximum_original_error'] != 0 or
            value['training_code_sha256'] != hashes(CODE) or not value['original_main_scores_reproduced'] or
            value['verifier_sha256'] != sha256(ROOT / 'scripts/verify_inseason_direct_seed_outputs.py')):
        raise ValueError('All original direct baseline replays must be verified first')
    for filename, expected in value['completed_sources'].items():
        if sha256(ROOT / filename) != expected:
            raise ValueError('Original direct replay gate changed')
    return path


def run(crop, cutoff, seed, model, percent):
    key = identity(crop, cutoff, seed, model, percent)
    root = root_for(**{k: key[k] for k in ('crop', 'cutoff', 'seed', 'model', 'percent')})
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = hashes(CODE)
        if (root / 'complete.json').exists():
            return verify(root, code)
        gate = require_replay() if seed != 42 else None
        old_root, old_dest, old_config, old_metrics, old_model = original(crop, cutoff, model, percent)
        raw, _ = load(crop)
        indices = partition(raw, cutoff)
        periods = {name: np.unique(raw['year'][ix]).tolist() for name, ix in indices.items()}
        if periods != old_config['periods']:
            raise ValueError('Repeat changed original temporal cohort')
        sources = {str(p.relative_to(ROOT)): sha256(p) for p in [old_root / 'complete.json', old_root / 'config.json',
            old_dest / 'audit.json', old_dest / 'metrics.json', old_dest / 'model.joblib',
            old_dest / 'evaluation_predictions.npz', CACHE / crop / 'manifest.json']}
        if gate:
            sources[str(gate.relative_to(ROOT))] = sha256(gate)
        config = dict(**key, code_sha256=code, smoke=False, periods=periods,
            counts={name: len(ix) for name, ix in indices.items()}, source_sha256=sources,
            original_weight_reused=seed == 42, fit_threads=2 if model == 'lightgbm' else 8,
            prediction_threads=2 if model == 'lightgbm' else 1,
            selection='Original two-candidate pre-cutoff annual RMSE' if model == 'lightgbm' else 'Original fixed 128-tree forest',
            weather='Supplied complete actual ERA5-Land; conditional evaluation',
            inputs='Unchanged complete-quality direct builder; visible remote prefix, past state, history, calendar and support',
            objective='Year-balanced standardized causal-trend residual MSE',
            paired_world_prediction=str((ROOT / 'benchmark/results/inseason_pipeline_seeds_v1/inference' / crop /
                f'cutoff_{cutoff}/seed_{seed}/tail_{percent:03d}_biid.npz').relative_to(ROOT)))
        register(root, config)
        started = time.monotonic()
        with threadpool_limits(limits=2):
            if model == 'lightgbm':
                x, residual, norm = make_design(raw, indices['inner_fit'], indices['inner_validation'], RECIPES[crop], percent/100)
                normalization_contract(norm, json.loads((old_dest / 'inner_normalization.json').read_text()))
                atomic_json(root / 'inner_normalization.json', norm)
                if seed == 42:
                    selection = old_metrics['selection']
                    fitted_inner = joblib.load(old_dest / 'inner/model.joblib')
                    inner_prediction = fitted_inner.booster_.predict(flat_features(x['other']))
                    value = score(residual['other'], inner_prediction, raw['year'][indices['inner_validation']])['mean_annual_rmse']
                    winner = selection['candidates'][selection['selected_candidate']]
                    np.testing.assert_allclose(value, winner['score'], rtol=0, atol=1e-12)
                else:
                    dest = root / 'inner'
                    dest.mkdir(exist_ok=True)
                    inner_prediction, selection = select_lightgbm(x, residual,
                        year_weights(raw['year'][indices['inner_fit']]), raw['year'][indices['inner_validation']], dest, seed)
                parameters = selection_contract(selection, seed)
                inner_labels = {k: raw[k][indices['inner_validation']] for k in LABELS}
                np.savez_compressed(root / 'inner_predictions.npz', prediction=inner_prediction,
                    standardized_target=residual['other'], **inner_labels)
                atomic_json(root / 'selection.json', selection)
                del x, residual, inner_prediction
            else:
                selection = dict(fixed_parameters=True)
                parameters = build_model('random_forest', seed, 8).get_params()
            x, residual, norm = make_design(raw, indices['full_fit'], indices['evaluation'], RECIPES[crop], percent/100)
            normalization_contract(norm, old_metrics['full_normalization'])
            if list(x['train']['sequence'].shape[1:]) != old_metrics['sequence_shape'] or x['train']['static'].shape[1] != old_metrics['static_width']:
                raise ValueError('Seed repeat feature width changed')
            train_features, test_features = flat_features(x['train']), flat_features(x['other'])
            if seed == 42:
                fitted = old_model
                weight = old_dest / 'model.joblib'
            else:
                fitted = lgb.LGBMRegressor(**parameters) if model == 'lightgbm' else build_model('random_forest', seed, 8)
                fitted.fit(train_features, residual['train'], sample_weight=year_weights(raw['year'][indices['full_fit']]))
                if model == 'random_forest':
                    fitted.set_params(n_jobs=1)
                weight = root / 'model.joblib'
                joblib.dump(fitted, weight)
            prediction = physical_prediction(fitted, test_features, x['other']['anchor'], norm, model)
            np.testing.assert_array_equal(prediction, physical_prediction(joblib.load(weight), test_features, x['other']['anchor'], norm, model))
        labels = {k: raw[k][indices['evaluation']] for k in LABELS}
        with np.load(ROOT / config['paired_world_prediction']) as paired:
            compare_labels(paired, labels)
        with np.load(old_dest / 'evaluation_predictions.npz') as saved:
            compare_labels(saved, labels)
            if seed == 42:
                np.testing.assert_array_equal(prediction, saved['prediction'])
        np.savez_compressed(root / 'evaluation_predictions.npz', prediction=prediction, **labels)
        measured = score(labels['target'], prediction, labels['year'])
        atomic_json(root / 'metrics.json', dict(scores=measured, selection=selection, full_normalization=norm,
            sequence_shape=list(x['train']['sequence'].shape[1:]), static_width=x['train']['static'].shape[1],
            flat_width=train_features.shape[1], weight=str(weight.relative_to(ROOT)), weight_sha256=sha256(weight),
            weight_replay_error=0., original_replay_error=0. if seed == 42 else None,
            final_parameters=fitted.get_params(), all_fitting_before_evaluation=True))
        for filename, expected in sources.items():
            if sha256(ROOT / filename) != expected:
                raise ValueError('Source changed while direct repeat ran')
        if hashes(CODE) != code:
            raise ValueError('Direct repeat code changed while running')
        finish(root, code, **key, smoke=False, original_weight_reused=seed == 42,
            new_final_fits=0 if seed == 42 else 1, seconds=time.monotonic()-started)
        print(f'[DIRECT SEED] {crop} cutoff={cutoff} seed={seed} {model} suffix={percent} annual={measured["mean_annual_rmse"]:.6f}', flush=True)
        return verify(root, code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=tuple(RECIPES), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--percent', type=int, choices=PERCENTS, required=True)
    run(**vars(parser.parse_args()))
