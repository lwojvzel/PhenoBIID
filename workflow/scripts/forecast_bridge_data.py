"""Physical train/development data and explicit conditional-forecast inputs."""
import argparse
import calendar
import fcntl
import json
from pathlib import Path

import numpy as np

from crop_signal_screen_data import source_inputs, active_slots, ORIGINS
from dual_remote_data import extract
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from run_history_multimodal_baselines import build_causal_history_features
from observed_remote_benchmark import trajectory_features

CACHE = ROOT / 'benchmark/cache/forecast_state_bridge_v1/raw'
PRODUCTS = ('lai', 'ndvi', 'gpp')
MAX_YEAR = 2012
KEYS = ('source_indices', 'year', 'row', 'col', 'target', 'history', 'baseline',
        'context', 'crop_coverage', 'relative_valid', 'source_month', 'weather',
        'previous_ndvi_quality', 'previous_gpp_quality',
        *[f'{prefix}_{p}' for p in PRODUCTS for prefix in ('observed', 'previous')])


def root(crop):
    if crop not in CROPS:
        raise ValueError(crop)
    return CACHE / crop


def identity_hash(a):
    import hashlib
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()


def forward_blocks(end):
    return [(start, min(start+2, end)) for start in range(1993, end+1, 3)]


def physical_history(crop, sources):
    folder = ROOT / 'benchmark/cache/multimodal_main' / crop
    years = np.load(folder / 'year.npy', mmap_mode='r')
    take = np.flatnonzero(years <= MAX_YEAR)
    raw = {'year': years[take]}
    for k in ('target', 'row', 'col'):
        raw[k] = np.load(folder / f'{k}.npy', mmap_mode='r')[take]
    h, b = build_causal_history_features(raw, 0., 1.)
    # Count scaling is a fixed protocol constant, not the observed future range.
    h[:, 14] *= (int(raw['year'].max())-int(raw['year'].min())) / 35.
    ix = np.searchsorted(take, sources)
    np.testing.assert_array_equal(take[ix], sources)
    return h[ix], b[ix], {str(folder / f'{k}.npy'): sha256(folder / f'{k}.npy')
                          for k in ('year', 'target', 'row', 'col')}


def previous_gpp(a):
    folder = ROOT / 'Data/processed/reclue_monthly_gpp_v1/monthly_0p5'
    value = np.full(a['source_month'].shape, np.nan, np.float32)
    quality = np.zeros_like(value)
    files = {}
    for year in np.unique(a['year']):
        if year-1 < 1982:
            continue
        take = np.flatnonzero(a['year'] == year)
        mm = np.minimum(a['source_month'][take], 11)
        rr, cc = a['row'][take, None], a['col'][take, None]
        valid = a['relative_valid'][take] > 0
        for label, name in (('value', 'gpp_daily_rate'), ('quality', 'valid_area_fraction')):
            file = folder / f'{name}_{year-1}.npy'
            grid = np.load(file, mmap_mode='r')
            selected = grid[mm, rr, cc]
            if label == 'value':
                value[take] = np.where(valid, selected, np.nan)
            else:
                quality[take] = np.where(valid, selected, 0)
            files[str(file)] = sha256(file)
    return value, quality, files


def prepare(crop):
    destination = root(crop)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / 'manifest.json').exists():
            load(crop, verify=True)
            return
        inputs, labels, lineage = source_inputs(crop, MAX_YEAR)
        arrays = {k: np.concatenate([labels[s][k] for s in ('train', 'validation')])
                  for k in ('source_indices', 'year', 'row', 'col', 'target')}
        a = {k: np.concatenate([inputs[s][k] for s in ('train', 'validation')])
             for k in inputs['train']}
        if np.any(arrays['year'] > MAX_YEAR):
            raise ValueError('Evaluation-year values are forbidden')
        if len(np.unique(arrays['source_indices'])) != len(arrays['year']):
            raise ValueError('Duplicate row identities')
        for k in ('context', 'crop_coverage', 'relative_valid', 'source_month'):
            arrays[k] = a[k]
        active, months = active_slots(arrays)
        world = ROOT / 'benchmark/cache/biid_world_model' / crop
        source = np.load(world / 'source_indices.npy', mmap_mode='r')
        ix = np.searchsorted(source, arrays['source_indices'])
        np.testing.assert_array_equal(source[ix], arrays['source_indices'])
        arrays['weather'] = np.load(world / 'weather_rel.npy', mmap_mode='r')[ix]
        arrays['observed_lai'] = np.load(world / 'target_lai_rel.npy', mmap_mode='r')[ix]
        old = np.load(world / 'previous_lai.npy', mmap_mode='r')[ix]
        valid = np.load(world / 'previous_lai_valid.npy', mmap_mode='r')[ix] > 0
        arrays['previous_lai'] = np.where(valid & active, old, np.nan)
        arrays['observed_ndvi'], _ = extract(arrays)
        arrays['previous_ndvi'], arrays['previous_ndvi_quality'] = extract(arrays, True)
        gnorm = lineage['gpp_training_normalization']
        arrays['observed_gpp'] = np.where(a['observed_gpp_valid'] > 0,
            a['observed_gpp']*gnorm['std']+gnorm['mean'], np.nan).astype(np.float32)
        arrays['previous_gpp'], arrays['previous_gpp_quality'], gfiles = previous_gpp(arrays)
        arrays['history'], arrays['baseline'], hfiles = physical_history(crop, arrays['source_indices'])
        wfiles = {str(world / f'{k}.npy'): sha256(world / f'{k}.npy') for k in
                  ('source_indices', 'weather_rel', 'target_lai_rel', 'previous_lai', 'previous_lai_valid')}
        for p in ('ndvi', 'gpp'):
            k = f'previous_{p}_quality'
            arrays[k] = np.where(np.isfinite(arrays[f'previous_{p}']) & active,
                                  np.nan_to_num(arrays[k], nan=0.), 0).astype(np.float32)
        for p in PRODUCTS:
            for prefix in ('observed', 'previous'):
                k = f'{prefix}_{p}'
                arrays[k] = np.where(active, arrays[k], np.nan).astype(np.float32)
        arrays['weather'] = np.where(active[..., None], arrays['weather'], np.nan).astype(np.float32)
        if set(arrays) != set(KEYS):
            raise ValueError('Unexpected physical cache keys')
        files = {}
        for k, v in arrays.items():
            file = destination / f'{k}.npy'
            np.save(file, v)
            np.testing.assert_array_equal(np.load(file, mmap_mode='r'), v)
            files[file.name] = sha256(file)
        origin_checks = {}
        for origin in ORIGINS:
            cache = ROOT / 'benchmark/cache/crop_signal_screen_v1' / crop / f'origin_{origin}'
            checks = {}
            for split in ('train', 'validation'):
                take = ((arrays['year'] <= origin-3) if split == 'train' else
                        ((arrays['year'] > origin-3) & (arrays['year'] <= origin)))
                with np.load(cache / f'{split}_labels.npz') as f:
                    for k in ('source_indices', 'year', 'row', 'col', 'target'):
                        np.testing.assert_array_equal(arrays[k][take], f[k])
                checks[split] = dict(rows=int(take.sum()), identity=identity_hash(arrays['source_indices'][take]))
            origin_checks[str(origin)] = checks
        coverage = {p: dict(observed=float(np.isfinite(arrays[f'observed_{p}'])[active].mean()),
                            previous=float(np.isfinite(arrays[f'previous_{p}'])[active].mean())) for p in PRODUCTS}
        multi = np.any(active[:, 1:] & (np.diff(months.astype(int), axis=1) > 1), axis=1)
        spec = dict(crop=crop, schema=1, maximum_loaded_target_year=MAX_YEAR,
            evaluation_arrays_loaded=False, years=np.unique(arrays['year']).tolist(),
            rows=len(arrays['year']), code_sha256=sha256(Path(__file__)),
            protocol='Conditional annual active-slot prediction, not repaired harvest-season forecasting',
            initialization='Previous calendar-year observations aligned by supplied target-year calendar',
            cutoff='January 1 of target calendar year; actual target-year weather is an explicit condition',
            supplied_calendar='Retrospective nearest MIRCA snapshot/area treated as given for all conditions; not operationally known historical data',
            target_quality_in_forecast_features=False, cohort='Unchanged original LAI-filtered cohort',
            disjoint_month_row_fraction=float(multi.mean()), coverage=coverage,
            split_identity_checks=origin_checks, lineage=lineage, previous_gpp_files=gfiles,
            historical_files=hfiles, world_files=wfiles,
            dependency_sha256={k: sha256(ROOT / 'scripts' / k) for k in
                               ('crop_signal_screen_data.py', 'dual_remote_data.py',
                                'run_history_multimodal_baselines.py', 'observed_remote_benchmark.py')},
            files=files)
        atomic_json(destination / 'manifest.json', spec)
        print(f'[BRIDGE RAW COMPLETE] {crop}: {spec["rows"]} rows; {coverage}', flush=True)


def load(crop, verify=False):
    directory = root(crop)
    meta = json.loads((directory / 'manifest.json').read_text())
    if meta['code_sha256'] != sha256(Path(__file__)) or meta['evaluation_arrays_loaded']:
        raise ValueError('Raw cache contract changed')
    for k, checksum in meta['dependency_sha256'].items():
        if sha256(ROOT / 'scripts' / k) != checksum:
            raise ValueError(f'Changed data dependency: {k}')
    arrays = {}
    for k in KEYS:
        file = directory / f'{k}.npy'
        if verify and sha256(file) != meta['files'][file.name]:
            raise ValueError(f'Changed raw cache: {file}')
        arrays[k] = np.load(file, mmap_mode='r')
    return arrays, meta


def fit_stats(a, fit):
    if not len(fit):
        raise ValueError('Empty training window')
    weather = np.asarray(a['weather'][fit], float)
    mean, std = np.nanmean(weather, (0, 1)), np.nanstd(weather, (0, 1))
    stats = dict(weather_mean=mean.tolist(), weather_std=np.maximum(std, 1e-6).tolist(),
                 target_mean=float(a['target'][fit].mean()), target_std=max(float(a['target'][fit].std()), 1e-6),
                 fit_years=np.unique(a['year'][fit]).tolist(), identity=identity_hash(a['source_indices'][fit]))
    for p in PRODUCTS:
        v = a[f'observed_{p}'][fit]
        stats[p] = dict(mean=float(np.nanmean(v)), std=max(float(np.nanstd(v)), 1e-6))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Missing training weather')
    return stats


def history_features(a, take, stats):
    h = np.array(a['history'][take], copy=True)
    mean, std = stats['target_mean'], stats['target_std']
    valid = h[:, 5:10] > 0
    h[:, :5] = np.where(valid, (h[:, :5]-mean)/std, 0)
    available = h[:, 14] > 0
    h[:, 10:12] = np.where(available[:, None], (h[:, 10:12]-mean)/std, 0)
    h[:, 12:14] /= std
    baseline = np.where(available, a['baseline'][take], mean).astype(np.float32)
    return np.concatenate((h, a['context'][take]), 1).astype(np.float32), baseline


def forecast_base(a, take, stats):
    history, _ = history_features(a, take, stats)
    active = a['relative_valid'][take] > 0
    month = np.minimum(a['source_month'][take], 11)
    days = np.zeros(active.shape, np.float32)
    for year in np.unique(a['year'][take]):
        rows = a['year'][take] == year
        lengths = np.array([calendar.monthrange(int(year), m)[1] for m in range(1, 13)])
        days[rows] = lengths[month[rows]] / 31.
    gap = np.zeros_like(days)
    gap[:, 1:] = np.diff(month.astype(float), axis=1) / 12.
    columns = [active, np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), days, gap]
    columns += [np.isfinite(a[f'previous_{p}'][take]) for p in PRODUCTS]
    columns += [a[f'previous_{p}_quality'][take] for p in ('ndvi', 'gpp')]
    weather = a['weather'][take]
    columns += [np.isfinite(weather[..., j]) for j in range(13)]
    metadata = np.where(active[..., None], np.stack(columns, -1), 0).reshape(len(take), -1)
    weather = np.nan_to_num((weather-np.array(stats['weather_mean']))/np.array(stats['weather_std']),
                            nan=0., posinf=0., neginf=0.).reshape(len(take), -1)
    return np.concatenate((history, a['crop_coverage'][take, None], metadata, weather), 1).astype(np.float32)


def climatology(a, fit, take, product):
    grid = a['row'].astype(np.int64)*720+a['col']
    month = np.minimum(a['source_month'], 11)
    keys = grid[fit, None]*12+month[fit]
    v = a[f'observed_{product}'][fit]
    valid = np.isfinite(v) & (a['relative_valid'][fit] > 0)
    sums = np.bincount(keys[valid], weights=v[valid], minlength=360*720*12)
    counts = np.bincount(keys[valid], minlength=len(sums))
    ms = np.bincount(month[fit][valid], weights=v[valid], minlength=12)
    mc = np.bincount(month[fit][valid], minlength=12)
    global_mean = float(v[valid].mean())
    fallback = np.divide(ms, mc, out=np.full(12, global_mean), where=mc > 0)
    query = grid[take, None]*12+month[take]
    return np.divide(sums[query], counts[query], out=fallback[month[take]].copy(), where=counts[query] > 0).astype(np.float32)


def remote_features(physical, active, climo, scale):
    valid = np.isfinite(physical) & active
    raw = np.where(valid, (physical-scale['mean'])/scale['std'], 0).astype(np.float32)
    anomaly = np.where(valid, (physical-climo)/scale['std'], 0).astype(np.float32)
    return np.concatenate((trajectory_features(raw, valid), trajectory_features(anomaly, valid)), 1).astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    args = parser.parse_args()
    prepare(args.crop)
