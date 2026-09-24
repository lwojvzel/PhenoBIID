"""Refit only terminal input paths, replaying frozen states and historical anchors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from crop_signal_screen_data import load as signal_load, cache_root
from inseason_13year_data import ROOT, RECIPES, BLOCKS, load, partition
from inseason_nested_common import LABELS, hashes, register, finish, verify
from inseason_signal_matching import load_group, feature_matrix
from inseason_terminal_paths import CONDITIONS, RATIOS, MODES, mask_features, check_splits
from ndvi_tail_replacement import tail_mask, mix_trajectory
from run_crop_head_signal_match import references, expert_root
from run_crop_signal_screen import year_weights
from run_inseason_direct_baselines import score
from run_ndvi_signal_permutation import check_files, compose
from run_ndvi_tail_replacement import yield_prediction
from run_review_revision_parallel import atomic_json
from review_revision_data import sha256

OUT = ROOT / 'benchmark/results/inseason_terminal_paths_v1'
PLAN = ROOT / 'Paper/task/十三年主实验_世界模型终端路径消融_20260908.md'
CODE = ('inseason_terminal_paths.py', 'run_inseason_terminal_paths.py',
    'inseason_signal_matching.py', 'crop_signal_screen_data.py',
    'run_crop_head_signal_match.py', 'run_crop_signal_screen.py',
    'run_ndvi_signal_permutation.py', 'run_ndvi_tail_replacement.py',
    'run_inseason_direct_baselines.py', 'observed_remote_anomaly.py',
    'observed_remote_benchmark.py')


def root_for(crop, cutoff, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_42'


def selected_rows(years, smoke):
    if not smoke:
        return np.arange(len(years))
    return np.concatenate([np.flatnonzero(years == year)[:24] for year in np.unique(years)])


def prepare(crop, cutoff, smoke):
    origin, recipe = cutoff+3, RECIPES[crop]
    raw, groups, scales, epochs, sources = load_group(crop, origin)
    reference = ROOT / f'benchmark/results/inseason_signal_match_v1/pipelines/{crop}/origin_{origin}/seed_42'
    check_files(reference, json.loads((reference / 'complete.json').read_text())['files'])
    sources[str(reference / 'complete.json')] = sha256(reference / 'complete.json')
    x, labels, _ = signal_load(crop, origin, recipe.split('_')[0])
    if recipe == 'ndvi_gpp':
        extra, paired, _ = signal_load(crop, origin, 'gpp')
        for split in x:
            for key in LABELS:
                np.testing.assert_array_equal(labels[split][key], paired[split][key])
            np.testing.assert_array_equal(x[split][:, :465], extra[split][:, :465])
            x[split] = np.concatenate((x[split], extra[split][:, -36:]), 1)
    cache = cache_root(crop, origin)
    for name in ('manifest.json', 'audit.json'):
        sources[str(cache / name)] = sha256(cache / name)
    current, _ = load(crop)
    fit_ix = partition(current, cutoff)['full_fit']
    evaluation_ix = partition(current, cutoff)['evaluation']
    all_labels = {k: np.concatenate([g['labels'][k] for g in groups.values()]) for k in LABELS}
    for key in LABELS:
        np.testing.assert_array_equal(labels['train'][key], current[key][fit_ix])
        np.testing.assert_array_equal(all_labels[key], current[key][evaluation_ix])
    fit_take = selected_rows(labels['train']['year'], smoke)
    evaluation_take = selected_rows(all_labels['year'], smoke)
    expert = expert_root(crop, origin)
    check_files(expert, json.loads((expert / 'audit.json').read_text())['files'])
    with np.load(expert / 'train_labels.npz') as saved:
        for key in LABELS:
            np.testing.assert_array_equal(labels['train'][key], saved[key])
        history = saved['history_prediction'].astype(float)
    sources[str(expert / 'train_labels.npz')] = sha256(expert / 'train_labels.npz')
    names = [name for name, _ in references(crop, origin)]
    first = next(iter(groups.values()))
    branches = []
    for i, (name, (model, config, _)) in enumerate(zip(names, first['heads'][recipe])):
        base = labels['train']['baseline'].astype(float) if name == 'trend' else history
        norm = config['normalization']
        center, scale = norm.get('center', norm.get('residual_mean')), norm.get('scale', norm.get('residual_std'))
        residual = (labels['train']['target'].astype(float)-base-center)/scale
        if name == 'trend':
            residual = labels['train']['target_residual']
        trees = int(model.best_iteration_ or model.booster_.current_iteration())
        if trees < 1:
            raise ValueError('Unfitted original terminal model')
        params = dict(model.get_params(), n_estimators=min(trees, 8) if smoke else trees)
        if params['random_state'] != 42 or params['n_jobs'] != 4:
            raise ValueError('Unexpected original seed or CPU setting')
        bases = np.concatenate([g['heads'][recipe][i][2] for g in groups.values()])
        branches.append(dict(name=name, original=model, config=config, parameters=params,
            original_trees=trees, residual=residual[fit_take], base=bases[evaluation_take]))
    prepared = {(r, m): [] for r in RATIOS for m in MODES}
    original_predictions = {(r, m): [] for r in RATIOS for m in MODES}
    for split, group in groups.items():
        take = group['take']
        active = raw['relative_valid'][take] > 0
        observed = {p: raw[f'observed_{p}'][take] for p in recipe.split('_')}
        zeros = np.zeros_like(active)
        observed_x = feature_matrix(group['common'], group['observed_encodings'], zeros, group['support'], recipe)
        if split == 'validation':
            np.testing.assert_array_equal(observed_x, x['validation'])
        climates = {}
        for p in observed:
            encoded = group['encoders'][p].encode(np.full_like(observed[p], scales[p]['mean']), active, True)
            climates[p] = -encoded[:, -18:-6]*scales[p]['std']+scales[p]['mean']
        for ratio in RATIOS:
            percent = round(100*ratio)
            tail = tail_mask(active, ratio)
            path = reference / f'{split}_trajectories_{percent:03d}.npz'
            sources[str(path)] = sha256(path)
            with np.load(path) as saved:
                mixed = {p: saved[p] for p in observed}
            for mode in MODES:
                values = mixed if mode == 'biid' else {p: mix_trajectory(observed[p], climates[p], tail) for p in observed}
                encodings = {p: group['encoders'][p].encode(values[p], tail, True) for p in observed}
                matrix = feature_matrix(group['common'], encodings, tail, group['support'], recipe)
                prediction = yield_prediction(group['heads'][recipe], matrix, crop)
                path = reference / f'{split}_{recipe}_{mode}_{percent:03d}.npz'
                sources[str(path)] = sha256(path)
                with np.load(path) as saved:
                    np.testing.assert_array_equal(prediction, saved['prediction'])
                prepared[ratio, mode].append(matrix)
                original_predictions[ratio, mode].append(prediction)
    return dict(train=x['train'][fit_take], years=labels['train']['year'][fit_take],
        branches=branches, evaluation={k: np.concatenate(v)[evaluation_take] for k, v in prepared.items()},
        original={k: np.concatenate(v)[evaluation_take] for k, v in original_predictions.items()},
        labels={k: v[evaluation_take] for k, v in all_labels.items()}, sources=sources, epochs=epochs)


def fit_branch(x, y, weights, params, condition, dest):
    model = lgb.LGBMRegressor(**params)
    model.fit(mask_features(x, condition), y, sample_weight=weights)
    removed = check_splits(model, condition)
    joblib.dump(model, dest / 'model.joblib')
    restored = joblib.load(dest / 'model.joblib')
    probe = mask_features(x[:min(1024, len(x))], condition)
    np.testing.assert_array_equal(model.booster_.predict(probe), restored.booster_.predict(probe))
    return model, removed


def run(crop, cutoff, smoke=False):
    root = root_for(crop, cutoff, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(CODE)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        started = time.monotonic()
        with threadpool_limits(limits=4):
            data = prepare(crop, cutoff, smoke)
            register(root, dict(crop=crop, cutoff=cutoff, smoke=smoke, seed=42,
                recipe=RECIPES[crop], conditions=CONDITIONS, ratios=RATIOS, modes=MODES,
                code_sha256=code, sources=data['sources'], original_expert_epochs=data['epochs'],
                training_years=np.unique(data['years']).tolist(), training_rows=len(data['train']),
                evaluation_years=np.unique(data['labels']['year']).tolist(),
                selection='Fixed original selected tree count per branch; no new stopping or search',
                original_development_selection_retained=True, state_and_history_weights_unchanged=True,
                trajectory_training='Original complete observed values',
                mask_scope='Terminal paths only; native missing-value effects and state quality remain',
                branches=[{k: b[k] for k in ('name', 'parameters', 'original_trees', 'config')} for b in data['branches']]))
            weights = year_weights(data['years'])
            for condition in CONDITIONS:
                dest = root / condition
                dest.mkdir(exist_ok=True)
                if (dest / 'complete.json').exists():
                    verify(dest, code)
                    continue
                predictions = {key: [] for key in data['evaluation']}
                fit_records = []
                for branch in data['branches']:
                    directory = dest / branch['name']
                    directory.mkdir(exist_ok=True)
                    model, removed = fit_branch(data['train'], branch['residual'], weights,
                        branch['parameters'], condition, directory)
                    max_replay = 0.
                    for key, x in data['evaluation'].items():
                        component = model.booster_.predict(mask_features(x, condition))
                        if condition == 'full' and not smoke:
                            original = branch['original'].booster_.predict(x)
                            max_replay = max(max_replay, float(np.max(np.abs(component-original))))
                            np.testing.assert_allclose(component, original, rtol=0, atol=1e-12)
                        predictions[key].append(compose(branch['config'], branch['base'], component))
                    fit_records.append(dict(branch=branch['name'], removed_columns=removed,
                        fitted_trees=int(model.booster_.current_iteration()), parameters=branch['parameters'],
                        original_prediction_max_error=max_replay if condition == 'full' and not smoke else None))
                rows = []
                for (ratio, mode), parts in predictions.items():
                    prediction = .5*parts[0]+.5*parts[1] if crop == 'maize' else parts[0]
                    if condition == 'full' and not smoke:
                        np.testing.assert_allclose(prediction, data['original'][ratio, mode], rtol=0, atol=1e-12)
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite terminal prediction')
                    name = f'tail_{round(100*ratio):02d}_{mode}.npz'
                    np.savez_compressed(dest / name, prediction=prediction, **data['labels'])
                    rows.append(dict(percent=round(100*ratio), mode=mode,
                        **score(data['labels']['target'], prediction, data['labels']['year'])))
                atomic_json(dest / 'metrics.json', dict(scores=rows, fits=fit_records))
                finish(dest, code, smoke=smoke, terminal_branch_fits=len(data['branches']),
                    full_rebuild_verified=condition == 'full' and not smoke,
                    mask_applied_in_fit_and_inference=True)
                print(f'[TERMINAL PATHS] {crop} {cutoff} {condition} annual={rows[0]["mean_annual_rmse"]:.6f}', flush=True)
            for path, digest in data['sources'].items():
                if sha256(Path(path)) != digest:
                    raise ValueError('Upstream asset changed during terminal ablation')
            finish(root, code, smoke=smoke, seconds=time.monotonic()-started,
                final_configurations=len(CONDITIONS), branch_fits=len(CONDITIONS)*len(data['branches']),
                original_predictions_replayed=True, full_rebuild_verified=not smoke)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=(*RECIPES, 'all'), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(run, crop, cutoff, args.smoke) for crop in RECIPES for cutoff in BLOCKS]:
                future.result()
    elif args.cutoff is None:
        parser.error('A single crop requires --cutoff')
    else:
        run(args.crop, args.cutoff, args.smoke)
