"""Five signal-value controls with the original thirteen-year samples."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from inseason_signal_values import CONDITIONS, design
from run_inseason_13year_direct import (ROOT, CACHE, RECIPES, BLOCKS, CODE as BASE_CODE,
    load, partition, flat_features, year_weights, train_tree, score, sha256, atomic_json, check_files)

OUT = ROOT / 'benchmark/results/inseason_signal_values_v1'
CODE = (*BASE_CODE, 'inseason_signal_values.py', 'run_inseason_signal_values.py',
        'run_crop_signal_screen.py', 'run_inseason_direct_baselines.py')


def root_for(crop, cutoff, condition, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_42' / condition


def verify(root):
    marker = json.loads((root / 'complete.json').read_text())
    check_files(root, marker['files'])
    if marker['code_sha256'] != {name: sha256(ROOT / 'scripts' / name) for name in CODE}:
        raise ValueError('Changed signal-value source')
    return marker


def run(crop, cutoff, condition, smoke=False):
    root = root_for(crop, cutoff, condition, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = {name: sha256(ROOT / 'scripts' / name) for name in CODE}
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root)
            return
        raw, _ = load(crop)
        groups = partition(raw, cutoff)
        if smoke:
            groups = {name: np.concatenate([ix[raw['year'][ix] == y][:16]
                for y in np.unique(raw['year'][ix])]) for name, ix in groups.items()}
        config = dict(crop=crop, cutoff=cutoff, condition=condition, smoke=smoke, seed=42,
            ratio=.1, code_sha256=code, cache_sha256=sha256(CACHE / crop / 'manifest.json'),
            periods={name: np.unique(raw['year'][ix]).tolist() for name, ix in groups.items()},
            model='Direct LightGBM with the original two capacities and preceding validation',
            shared='History, weather, calendar, crop coverage, six issue-masked product-support fields',
            previous_product_quality='Constant zero for every product; not native quality for LAI',
            metadata_control='Remove selected-product values, anomalies, summaries and validity; keep shared support',
            primary_recipe_selection=False, new_sample_filter=False)
        path = root / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Changed signal-value registration')
        atomic_json(path, config)
        started = time.monotonic()
        x, residual, norm = design(raw, groups['inner_fit'], groups['inner_validation'], condition)
        atomic_json(root / 'inner_normalization.json', norm)
        inner = root / 'inner'
        inner.mkdir(exist_ok=True)
        with threadpool_limits(limits=2):
            _, selection = train_tree(dict(train=x['train'], validation=x['other'], test=x['other']),
                dict(train=residual['train'], validation=residual['other'], test=residual['other']),
                year_weights(raw['year'][groups['inner_fit']]), raw['year'][groups['inner_validation']], inner, smoke)
        del x, residual
        x, residual, norm = design(raw, groups['full_fit'], groups['evaluation'], condition)
        winner = selection['candidates'][selection['selected_candidate']]
        if winner['selected_trees'] < 1:
            raise ValueError('Untrained signal-value model')
        model = lgb.LGBMRegressor(**dict(winner['parameters'], n_estimators=winner['selected_trees']))
        with threadpool_limits(limits=2):
            model.fit(flat_features(x['train']), residual['train'],
                sample_weight=year_weights(raw['year'][groups['full_fit']]))
            output = model.booster_.predict(flat_features(x['other']))
            joblib.dump(model, root / 'model.joblib')
            restored = joblib.load(root / 'model.joblib')
            np.testing.assert_array_equal(output, restored.booster_.predict(flat_features(x['other'])))
        prediction = x['other']['anchor'] + norm['center'] + norm['scale']*output
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite signal-value prediction')
        labels = {key: raw[key][groups['evaluation']] for key in ('target', 'year', 'row', 'col', 'source_indices')}
        metrics = score(labels['target'], prediction, labels['year'])
        np.savez_compressed(root / 'evaluation_predictions.npz', prediction=prediction, **labels)
        atomic_json(root / 'metrics.json', dict(scores=metrics, selection=selection, full_normalization=norm,
            weight_replay=True, sequence_shape=list(x['train']['sequence'].shape[1:]), static_width=x['train']['static'].shape[1]))
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
            if p.is_file() and p.name not in ('run.lock', 'complete.json')}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=code, smoke=smoke,
            seconds=time.monotonic()-started, final_models=1))
        print(f'[SIGNAL VALUE] {crop} {cutoff} {condition} annual={metrics["mean_annual_rmse"]:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=(*RECIPES, 'all'), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS))
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(run, c, end, s, args.smoke)
                for c in RECIPES for end in BLOCKS for s in CONDITIONS]:
                future.result()
    elif args.cutoff is None or args.condition is None:
        parser.error('A single crop requires --cutoff and --condition')
    else:
        run(args.crop, args.cutoff, args.condition, args.smoke)
