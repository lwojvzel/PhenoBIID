"""Replace only observed scalar products under the currently selected crop heads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from crop_signal_screen_data import ROOT, CROPS, ORIGINS, load as signal_load
from crop_signal_history_reference import IDENTITY
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_crop_signal_screen import yearly_rmse, year_weights
from run_ndvi_signal_permutation import compose, check_files

RESULT = ROOT / 'benchmark/results/crop_head_signal_match_v1'
INVENTORY = ROOT / 'visualize/paper_experiments/observed_yield_majority_v1/inventory.csv'
PRODUCTS = ('lai', 'gpp')


def source_root(crop, origin):
    inventory = pd.read_csv(INVENTORY)
    row = inventory[(inventory.crop == crop) & (inventory.origin == origin) &
                    (inventory.seed == 42) & (inventory.condition == 'ndvi')]
    if len(row) != 1:
        raise ValueError('Expected one current seed42 NDVI reference')
    return Path(row.iloc[0]['config']).parent


def expert_root(crop, origin):
    if crop in ('maize', 'wheat'):
        return ROOT / f'benchmark/cache/cereal_mlp_correction_v1/{crop}/origin_{origin}/seed_42'
    name = 'rice_frozen_mlp_v1' if crop == 'rice' else 'soybean_frozen_history_v1'
    return ROOT / f'benchmark/cache/{name}/origin_{origin}'


def references(crop, origin):
    src = source_root(crop, origin)
    cfg = json.loads((src / 'config.json').read_text())
    if crop == 'maize':
        if cfg['blend_weights'] != [.5, .5]:
            raise ValueError('Unexpected maize composition')
        return [(name, Path(cfg['parents'][name]['root'])) for name in ('trend', 'mlp')]
    return [('expert', src)]


def register():
    audit = json.loads((ROOT / 'visualize/paper_experiments/forecast_state_bridge_v1/input_audit.json').read_text())
    if not audit['passed'] or audit['global_harvest_season_repair']:
        raise ValueError('Old-interface screen requires audited unchanged calendar')
    files = {str(INVENTORY): sha256(INVENTORY), str(Path(__file__)): sha256(Path(__file__))}
    for crop in CROPS:
        for origin in ORIGINS:
            for _, src in references(crop, origin):
                record = json.loads((src / 'audit.json').read_text())
                check_files(src, record['files'])
                for name in ('audit.json', 'config.json', 'model.joblib', 'validation_predictions.npz'):
                    files[str(src / name)] = sha256(src / name)
            ex = expert_root(crop, origin)
            audit = json.loads((ex / 'audit.json').read_text())
            if not audit['full_train_validation_replay'] or audit['evaluation_split_loaded']:
                raise ValueError('Unaudited historical expert')
            check_files(ex, audit['files'])
            for name in ('audit.json', 'config.json', 'train_labels.npz', 'validation_labels.npz'):
                files[str(ex / name)] = sha256(ex / name)
    spec = dict(schema=1, files=files, seed=42, products=list(PRODUCTS),
        logical_runs=24, terminal_tree_fits=30, worker_count=2, threads_per_worker=4,
        old_interface=True, evaluation_arrays_loaded=False,
        selection='Existing observed-protocol outer validation early stopping, not formal forward confirmation',
        scope='Candidates only; admission to world model requires new-interface observed recheck')
    RESULT.mkdir(parents=True, exist_ok=True)
    file = RESULT / 'registration.json'
    if file.exists() and json.loads(file.read_text()) != spec:
        raise ValueError('Observed signal registration changed')
    atomic_json(file, spec)
    return spec


def run_group(crop, origin):
    registration = json.loads((RESULT / 'registration.json').read_text())
    for path, digest in registration['files'].items():
        if (f'/{crop}/' in path or f'origin_{origin}' in path or path == str(Path(__file__))) and sha256(Path(path)) != digest:
            raise ValueError(f'Changed registered source {path}')
    reference_x, a, meta = signal_load(crop, origin, 'ndvi')
    bases = {}
    ex = expert_root(crop, origin)
    for split in ('train', 'validation'):
        with np.load(ex / f'{split}_labels.npz') as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(a[split][k], f[k])
            bases[split] = f['history_prediction']
    branch_info = []
    for name, src in references(crop, origin):
        cfg = json.loads((src / 'config.json').read_text())
        base = {s: a[s]['baseline'].astype(float) if name == 'trend' else bases[s] for s in a}
        model = joblib.load(src / 'model.joblib')
        p = compose(cfg, base['validation'], model.predict(reference_x['validation']))
        with np.load(src / 'validation_predictions.npz') as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(a['validation'][k], f[k])
            np.testing.assert_allclose(p, f['prediction'], rtol=0, atol=1e-12)
        branch_info.append((name, src, cfg, base, model.get_params()))
    del reference_x
    for product in PRODUCTS:
        dest = RESULT / 'pipelines' / crop / f'origin_{origin}/seed_42/{product}'
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (dest / 'audit.json').exists():
                check_files(dest, json.loads((dest / 'audit.json').read_text())['files'])
                continue
            x, current, source = signal_load(crop, origin, product)
            for s in a:
                for k in IDENTITY:
                    np.testing.assert_array_equal(a[s][k], current[s][k])
            predictions, branch_records, files = [], [], {}
            started = time.monotonic()
            for name, src, cfg, base, params in branch_info:
                norm = cfg['normalization']
                center = norm.get('center', norm.get('residual_mean'))
                scale = norm.get('scale', norm.get('residual_std'))
                y = {s: (a[s]['target'].astype(float)-base[s]-center)/scale for s in a}
                if name == 'trend':
                    y = {s: a[s]['target_residual'] for s in a}
                trace = {}
                years = a['validation']['year']

                def metric(truth, pred):
                    return 'annual_rmse', float(np.mean(list(yearly_rmse(truth, pred, years).values()))), False

                model = lgb.LGBMRegressor(**params)
                model.fit(x['train'], y['train'], sample_weight=year_weights(a['train']['year']),
                    eval_set=[(x['validation'], y['validation'])], eval_metric=metric,
                    callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False),
                               lgb.record_evaluation(trace), lgb.log_evaluation(200)])
                weight = dest / f'{name}.joblib'
                joblib.dump(model, weight)
                part = model.predict(x['validation'])
                np.testing.assert_array_equal(part, joblib.load(weight).predict(x['validation']))
                prediction = compose(cfg, base['validation'], part)
                predictions.append(prediction)
                np.savez_compressed(dest / f'{name}_predictions.npz', prediction=prediction,
                                    component=part, history_prediction=base['validation'])
                atomic_json(dest / f'{name}_training_history.json', trace)
                branch_records.append(dict(name=name, source=str(src), source_config_sha256=sha256(src / 'config.json'),
                    parameters=params, normalization=norm, selected_trees=int(model.best_iteration_)))
                for filename in (weight.name, f'{name}_predictions.npz', f'{name}_training_history.json'):
                    files[filename] = sha256(dest / filename)
            prediction = .5*predictions[0]+.5*predictions[1] if crop == 'maize' else predictions[0]
            np.savez_compressed(dest / 'validation_predictions.npz', prediction=prediction,
                                **{k: a['validation'][k] for k in IDENTITY})
            atomic_json(dest / 'config.json', dict(crop=crop, origin=origin, seed=42, product=product,
                branches=branch_records, blend_weights=[.5, .5] if crop == 'maize' else [1.],
                registration_sha256=sha256(RESULT / 'registration.json'),
                source_manifest=source, evaluation_arrays_loaded=False, interface='old_observed_501',
                new_fits=len(branch_info), fixed_expert=True))
            atomic_json(dest / 'metrics.json', dict(crop=crop, origin=origin, seed=42, product=product,
                per_year_rmse=yearly_rmse(a['validation']['target'], prediction, years),
                seconds=time.monotonic()-started, terminal_tree_fits=len(branch_info)))
            for filename in ('config.json', 'metrics.json', 'validation_predictions.npz'):
                files[filename] = sha256(dest / filename)
            atomic_json(dest / 'audit.json', dict(full_reference_replay=True, full_validation_replay=True,
                maximum_reference_error_tolerance=1e-12, evaluation_arrays_loaded=False, files=files))
            print(f'[SIGNAL MATCH COMPLETE] {crop} {origin} {product}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS)
    parser.add_argument('--origin', choices=ORIGINS, type=int)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        register()
        if args.crop is not None:
            if args.origin is None:
                parser.error('--origin is required with --crop')
            run_group(args.crop, args.origin)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_group, crop, origin) for crop in CROPS for origin in ORIGINS]
                for future in futures:
                    future.result()
