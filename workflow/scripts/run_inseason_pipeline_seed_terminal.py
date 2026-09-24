"""Rebuild original terminal heads, then fit fixed-capacity seed counterparts.

This stage saves readout weights only. Seed45/48 world-model evaluation must use
their own state rollouts in the subsequent inference stage.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from crop_signal_screen_data import cache_root
from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_nested_common import LABELS, hashes, register, finish, verify
from inseason_pipeline_seed_terminal import (paired_prediction, fixed_parameters,
    residual_labels, combine, validate_feature_schema)
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_inseason_pipeline_seed_history import root_for as history_root, CODE as HISTORY_CODE
from run_inseason_terminal_paths import prepare, fit_branch, selected_rows, CODE as SOURCE_CODE
from run_ndvi_signal_permutation import compose
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_pipeline_seeds_v1'
CODE = (*SOURCE_CODE, *HISTORY_CODE, 'inseason_pipeline_seed_terminal.py',
        'run_inseason_pipeline_seed_terminal.py')


def root_for(crop, cutoff, seed, smoke=False):
    return OUT / ('terminal_smoke' if smoke else 'terminal') / crop / f'cutoff_{cutoff}/seed_{seed}'


def history(crop, cutoff, seed, train_labels, evaluation_labels):
    component = 'tabm' if crop == 'soybean' else 'mlp'
    root = history_root(crop, cutoff, seed, component)
    record = verify(root, hashes(HISTORY_CODE))
    config = json.loads((root / 'config.json').read_text())
    identity = dict(crop=crop, cutoff=cutoff, seed=seed, component=component, smoke=False)
    if any(record.get(k) != v or config.get(k) != v for k, v in identity.items()):
        raise ValueError('Wrong historical component')
    if component == 'mlp' and record['fixed_epochs'] < 1:
        raise ValueError('Untrained historical MLP')
    arrays, sources = {}, {}
    for split in ('train', 'validation', 'test'):
        file = root / f'{split}_predictions.npz'
        with np.load(file) as f:
            arrays[split] = {key: f[key] for key in (*LABELS, 'prediction')}
        sources[str(file)] = sha256(file)
    parts = ('validation', 'test') if cutoff == 2009 else ('validation',)
    evaluation = {k: np.concatenate([arrays[s][k] for s in parts]) for k in (*LABELS, 'prediction')}
    bases = dict(train=paired_prediction(arrays['train'], train_labels),
                 evaluation=paired_prediction(evaluation, evaluation_labels))
    for file in (root / 'complete.json', root / 'config.json', root / 'model.pt'):
        sources[str(file)] = sha256(file)
    return bases, sources, dict(component=component, selected_epochs=record['fixed_epochs'],
        initialization_component=record['initialization_component'], root=str(root))


def require_original_replays():
    for crop in RECIPES:
        for cutoff in BLOCKS:
            record = verify(root_for(crop, cutoff, 42), hashes(CODE))
            if record['smoke'] or not record['original_predictions_replayed']:
                raise ValueError('All twelve original terminal replays must pass first')


def run(crop, cutoff, seed, smoke=False):
    root = root_for(crop, cutoff, seed, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(CODE)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        if seed != 42 and not smoke:
            require_original_replays()
        started = time.monotonic()
        with threadpool_limits(limits=4):
            data = prepare(crop, cutoff, smoke)
            cache = cache_root(crop, cutoff + 3)
            manifest = json.loads((cache / 'manifest.json').read_text())
            validate_feature_schema(data['train'], manifest['spec']['names'])
            with np.load(cache / 'train_labels.npz') as f:
                take = selected_rows(f['year'], smoke)
                train_labels = {k: f[k][take] for k in LABELS}
            np.testing.assert_array_equal(train_labels['year'], data['years'])
            bases, history_sources, historical = history(crop, cutoff, seed, train_labels, data['labels'])
            sources = dict(data['sources'], **history_sources)
            source_file = cache / 'train_labels.npz'
            sources[str(source_file)] = sha256(source_file)
            branches = []
            for b in data['branches']:
                parameters = fixed_parameters(b['original'].get_params(), b['original_trees'], seed, smoke)
                residual = (b['residual'] if b['name'] == 'trend' else
                    residual_labels(train_labels['target'], bases['train'], b['config']['normalization']))
                base = b['base'] if b['name'] == 'trend' else bases['evaluation']
                if seed == 42:
                    np.testing.assert_array_equal(residual, b['residual'])
                    np.testing.assert_array_equal(base, b['base'])
                branches.append(dict(b, parameters=parameters, residual=residual, base=base))
            register(root, dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke, recipe=RECIPES[crop],
                code_sha256=code, sources=sources, historical=historical,
                features='Unchanged raw history20 + metadata289 + weather156 + 36 per product',
                training_input='Original complete observed trajectories',
                selection='Fixed original selected tree count and normalization; no new early stopping',
                original_development_selection_retained=True, same_seed_historical_anchor=True,
                training_rows=len(data['train']), training_years=np.unique(data['years']).tolist(),
                evaluation_years=np.unique(data['labels']['year']).tolist(),
                feature_width=data['train'].shape[1],
                branches=[{k: b[k] for k in ('name', 'parameters', 'original_trees', 'config')} for b in branches],
                scope='Terminal weights only; full corresponding-seed state inference is a later stage'))
            weights = year_weights(data['years'])
            reference_parts = {key: [] for key in data['evaluation']}
            fit_records = []
            for b in branches:
                directory = root / b['name']
                directory.mkdir(exist_ok=True)
                model, removed = fit_branch(data['train'], b['residual'], weights,
                                            b['parameters'], 'full', directory)
                if removed:
                    raise ValueError('A terminal input path was accidentally removed')
                max_error = None
                if seed == 42 and not smoke:
                    max_error = 0.
                    for key, x in data['evaluation'].items():
                        component = model.booster_.predict(x)
                        original = b['original'].booster_.predict(x)
                        np.testing.assert_allclose(component, original, rtol=0, atol=1e-12)
                        max_error = max(max_error, float(np.max(np.abs(component-original))))
                        reference_parts[key].append(compose(b['config'], b['base'], component))
                fit_records.append(dict(branch=b['name'], selected_trees=b['parameters']['n_estimators'],
                    fitted_trees=int(model.booster_.current_iteration()), original_max_error=max_error))
                print(f'[FIXED TERMINAL] {crop} {cutoff} seed={seed} branch={b["name"]} trees={model.booster_.current_iteration()}', flush=True)
            if seed == 42 and not smoke:
                for (ratio, mode), parts in reference_parts.items():
                    prediction = combine(crop, parts)
                    expected = data['original'][ratio, mode]
                    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
                    np.savez_compressed(root / f'replay_{round(ratio*100):03d}_{mode}.npz',
                        prediction=prediction, original=expected, **data['labels'])
            np.savez_compressed(root / 'train_anchor.npz', prediction=bases['train'], **train_labels)
            np.savez_compressed(root / 'evaluation_anchor.npz', prediction=bases['evaluation'], **data['labels'])
            atomic_json(root / 'fits.json', dict(branches=fit_records,
                original_predictions_replayed=seed == 42 and not smoke,
                corresponding_seed_world_predictions_generated=False))
            for name, digest in sources.items():
                if sha256(Path(name)) != digest:
                    raise ValueError(f'Upstream asset changed during terminal fitting: {name}')
            finish(root, code, crop=crop, cutoff=cutoff, seed=seed, smoke=smoke,
                terminal_branch_fits=len(branches), original_predictions_replayed=seed == 42 and not smoke,
                complete_pipeline=False, seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=(*RECIPES, 'all'), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS))
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, c, k, args.seed, args.smoke) for c in RECIPES for k in BLOCKS]
            for future in futures:
                future.result()
    elif args.cutoff is None:
        parser.error('A single crop requires --cutoff')
    else:
        run(args.crop, args.cutoff, args.seed, args.smoke)
