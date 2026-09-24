"""Frozen-head validation-only block permutation; no fitting or model selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from crop_signal_screen_data import ROOT, CODE, cache_root
from crop_signal_history_reference import IDENTITY
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_crop_signal_screen import yearly_rmse

SOURCE = ROOT / 'visualize/paper_experiments/crop_specific_ndvi_repeats_v3/audit.json'
OUT = ROOT / 'benchmark/results/ndvi_signal_permutation_v1'
PLAN = ROOT / 'Paper/task/固定NDVI信号置换_阶段登记_20260907.md'
REPEATS = (1101, 1102, 1103)
MODES = ('within_year', 'calendar_mask_matched')
CROPS = ('maize', 'rice', 'soybean', 'wheat')
ORIGINS = (2004, 2008, 2012)
SEEDS = (42, 45, 48)


def check_files(root, files):
    for name, digest in files.items():
        if sha256(root / name) != digest:
            raise ValueError(f'Artifact changed: {root / name}')


def strata(year, metadata, mode):
    if metadata.shape != (len(year), 289) or mode not in MODES:
        raise ValueError('Unexpected support representation or mode')
    fields = metadata[:, 1:].reshape(-1, 12, 24)[:, :, [0, 1, 2, 6]]
    keys = year[:, None] if mode == 'within_year' else np.column_stack((year, fields.reshape(len(year), -1)))
    return np.unique(keys, axis=0, return_inverse=True)[1]


def donors(groups, repeat):
    rng = np.random.default_rng(repeat)
    order = np.argsort(groups, kind='stable')
    breaks = np.flatnonzero(np.diff(groups[order]))+1
    out = np.arange(len(groups))
    for take in np.split(order, breaks):
        out[take] = rng.permutation(take)
    np.testing.assert_array_equal(np.sort(out), np.arange(len(groups)))
    np.testing.assert_array_equal(groups[out], groups)
    return out


def perturb(x, donor):
    if x.ndim != 2 or x.shape[1] != 501 or donor.shape != (len(x),):
        raise ValueError('Expected registered 501-dimensional NDVI input')
    changed = x.copy()
    changed[:, -36:] = x[donor, -36:]
    return changed


def compose(config, base, component):
    norm = config['normalization']
    center = norm.get('center', norm.get('residual_mean'))
    scale = norm.get('scale', norm.get('residual_std'))
    if config['head'] == 'lightgbm':
        return base+scale*component+center
    return base+center+scale*component


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    if source['terminal_models'] != 144 or source['evaluation_split_loaded']:
        raise ValueError('Wrong source scope')
    paths = [SOURCE, PLAN, Path(__file__)]
    models = []
    for name, digest in source['sources'].items():
        path = Path(name)
        if sha256(path) != digest:
            raise ValueError('Changed source audit')
        audit = json.loads(path.read_text())
        check_files(path.parent, audit['files'])
        paths.extend([path, *[path.parent / n for n in audit['files']]])
        config = json.loads((path.parent / 'config.json').read_text())
        if config['condition'] == 'ndvi':
            models.append(str(path.parent))
    if len(models) != 36:
        raise ValueError('Missing NDVI models')
    for crop in CROPS:
        for origin in ORIGINS:
            root = cache_root(crop, origin)
            audit = json.loads((root / 'audit.json').read_text())
            manifest = json.loads((root / 'manifest.json').read_text())
            if not audit['full_array_rebuild'] or audit['manifest_sha256'] != sha256(root / 'manifest.json'):
                raise ValueError('Cache not audited')
            if manifest['spec']['code_sha256'] != {n: sha256(ROOT / 'scripts' / n) for n in CODE}:
                raise ValueError('Cache encoding changed')
            paths.extend([root / 'manifest.json', root / 'audit.json'])
            for n in ('history_validation.npy', 'metadata_validation.npy', 'weather_validation.npy',
                      'ndvi_validation.npy', 'validation_labels.npz'):
                check_files(root, {n: manifest['files'][n]})
                paths.append(root / n)
    paths.extend(ROOT / 'scripts' / n for n in (*CODE, 'review_revision_data.py',
        'run_review_revision_parallel.py', 'run_crop_signal_screen.py', 'crop_signal_history_reference.py'))
    registration = dict(models=sorted(models), modes=MODES, repeats=REPEATS, evaluation_split_loaded=False,
        new_fits=0, files={str(p): sha256(p) for p in paths})
    file = OUT / 'registration.json'
    if file.exists():
        if json.loads(file.read_text()) != json.loads(json.dumps(registration)):
            raise ValueError('Registration already frozen with different content')
    else:
        atomic_json(file, registration)


def run_group(crop, origin, registration):
    started = time.monotonic()
    root = OUT / crop / f'origin_{origin}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'audit.json').exists():
            check_files(root, json.loads((root / 'audit.json').read_text())['files'])
            return
        cache = cache_root(crop, origin)
        x = np.concatenate([np.load(cache / f'{k}_validation.npy')
                            for k in ('history', 'metadata', 'weather', 'ndvi')], axis=1)
        with np.load(cache / 'validation_labels.npz') as f:
            a = {k: f[k] for k in IDENTITY}
        years = a['year']
        if list(np.unique(years)) != list(range(origin-2, origin+1)):
            raise ValueError('Wrong validation years')
        mapping, support = {}, []
        for mode in MODES:
            groups = strata(years, x[:, 20:309], mode)
            for repeat in REPEATS:
                donor = donors(groups, repeat)
                mapping[f'{mode}_{repeat}'] = donor
                for year in np.unique(years):
                    take = years == year
                    support.append(dict(crop=crop, origin=origin, mode=mode, repeat=repeat, year=int(year),
                        moved_fraction=float(np.mean(donor[take] != np.flatnonzero(take))),
                        changed_fraction=float(np.mean(np.any(x[donor[take], -36:] != x[take, -36:], axis=1))),
                        n=int(take.sum())))
        np.savez_compressed(root / 'donors.npz', **mapping, **a)
        annual, found = [], []
        for name in registration['models']:
            src = Path(name)
            config = json.loads((src / 'config.json').read_text())
            if (config['crop'], config['origin']) != (crop, origin):
                continue
            seed = config['seed']
            found.append(seed)
            with np.load(src / 'validation_predictions.npz') as f:
                for k in IDENTITY:
                    np.testing.assert_array_equal(a[k], f[k])
                correct, base = f['prediction'], f['history_prediction']
            model = joblib.load(src / 'model.joblib')
            if isinstance(model, dict):
                if model['scaler'] is not None:
                    raise ValueError('Unexpected scaling')
                model = model['model']
            replay = compose(config, base, model.predict(x, num_threads=4))
            np.testing.assert_allclose(replay, correct, rtol=0, atol=1e-12)
            with np.load(src.parent / 'weather/validation_predictions.npz') as f:
                for k in IDENTITY:
                    np.testing.assert_array_equal(a[k], f[k])
                weather = yearly_rmse(a['target'], f['prediction'], years)
            original = yearly_rmse(a['target'], correct, years)
            saved = dict(original=correct, **a)
            for mode in MODES:
                for repeat in REPEATS:
                    key = f'{mode}_{repeat}'
                    changed = perturb(x, mapping[key])
                    np.testing.assert_array_equal(changed[:, :465], x[:, :465])
                    prediction = compose(config, base, model.predict(changed, num_threads=4))
                    saved[key] = prediction
                    scores = yearly_rmse(a['target'], prediction, years)
                    for year, rmse in scores.items():
                        annual.append(dict(crop=crop, origin=origin, seed=seed, mode=mode, repeat=repeat,
                            year=int(year), rmse=rmse, original_rmse=original[year], weather_rmse=weather[year]))
            file = root / f'predictions_seed_{seed}.npz'
            np.savez_compressed(file, **saved)
            with np.load(file) as f:
                for k, value in saved.items():
                    np.testing.assert_array_equal(value, f[k])
        if sorted(found) != list(SEEDS) or len(annual) != 54:
            raise ValueError('Incomplete seed/repeat/year scope')
        atomic_json(root / 'metrics.json', dict(annual=annual, support=support))
        files = ['metrics.json', 'donors.npz', *[f'predictions_seed_{s}.npz' for s in SEEDS]]
        atomic_json(root / 'audit.json', dict(full_validation_replay=True, evaluation_split_loaded=False,
            registration_sha256=sha256(OUT / 'registration.json'), files={n: sha256(root / n) for n in files},
            seconds=time.monotonic()-started))
        print(f'COMPLETE {crop} {origin} | 3 replays + 18 permutations | {time.monotonic()-started:.1f}s', flush=True)


def run():
    reg = json.loads((OUT / 'registration.json').read_text())
    check_files(Path('/'), reg['files'])
    with threadpool_limits(limits=4), ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_group, crop, origin, reg) for crop in CROPS for origin in ORIGINS]
        for future in futures:
            future.result()
    audits = {str(OUT / c / f'origin_{o}/audit.json'): sha256(OUT / c / f'origin_{o}/audit.json')
              for c in CROPS for o in ORIGINS}
    atomic_json(OUT / 'complete.json', dict(models=36, perturbed_inferences=216, annual_rows=648,
        sources=audits, evaluation_split_loaded=False, new_fits=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    register() if args.register else run()
