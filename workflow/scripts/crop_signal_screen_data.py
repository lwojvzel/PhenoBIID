"""Paired train/validation-only signal screen on the unchanged original cohort."""
import argparse
import calendar
import fcntl
import json
from pathlib import Path

import numpy as np

from observed_remote_anomaly import seasonal_anomalies
from observed_remote_benchmark import trajectory_features
from prepare_pku_ndvi import OUT as NDVI
from prepare_reclue_monthly_gpp import OUT as GPP, AUDIT as GPP_AUDIT
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from stable_remote_data import cache_root as old_cache_root

CACHE = ROOT / 'benchmark/cache/crop_signal_screen_v1'
RESULT = ROOT / 'benchmark/results/crop_signal_screen_v1'
ORIGINS = (2004, 2008, 2012)
SPLITS = ('train', 'validation')
PRODUCTS = ('lai', 'ndvi', 'gpp')
CONDITIONS = ('history', 'metadata', 'weather', *PRODUCTS)
COMPONENTS = ('history', 'metadata', 'weather', *PRODUCTS)
LABELS = ('target', 'target_residual', 'baseline', 'year', 'row', 'col', 'source_indices')
INPUT_KEYS = ('history', 'context', 'crop_coverage', 'relative_valid', 'source_month', 'row', 'col', 'year',
              'observed_lai', 'observed_lai_valid', 'observed_ndvi', 'observed_ndvi_valid')
CODE = ('crop_signal_screen_data.py', 'observed_remote_anomaly.py', 'observed_remote_benchmark.py')


def cache_root(crop, origin):
    return CACHE / crop / f'origin_{origin}'


def run_root(crop, origin, condition, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42/lightgbm' / condition


def select_components(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == 'history':
        return ('history',)
    if condition == 'metadata':
        return ('history', 'metadata')
    return ('history', 'metadata', 'weather', *((condition,) if condition in PRODUCTS else ()))


def active_slots(a):
    active = a['relative_valid'] > 0
    month = a['source_month'].astype(np.int64)
    if active.shape != month.shape or active.shape[1] != 12:
        raise ValueError('Expected twelve aligned crop slots')
    if np.any(active & ((month < 0) | (month > 11))) or np.any(active.sum(1) == 0):
        raise ValueError('Invalid active source months')
    for slot in range(1, 12):
        paired = active[:, slot] & active[:, slot-1]
        if np.any(month[paired, slot] <= month[paired, slot-1]):
            raise ValueError('This screen expects the registered ascending natural-month packing')
    if np.any(np.diff(active.astype(np.int8), axis=1) > 0):
        raise ValueError('Active slots must be packed before padding')
    return active, month


def calendar_fields(a):
    active, month = active_slots(a)
    days = np.zeros(month.shape, np.float32)
    for year in np.unique(a['year']):
        take = a['year'] == year
        lengths = np.array([calendar.monthrange(int(year), m)[1] for m in range(1, 13)], np.float32)
        days[take] = lengths[np.minimum(month[take], 11)]
    gap = np.zeros_like(days)
    gap[:, 1:] = np.diff(month, axis=1)
    return [active.astype(np.float32), np.where(active, np.sin(2*np.pi*month/12), 0),
            np.where(active, np.cos(2*np.pi*month/12), 0), np.where(active, days/31, 0),
            np.where(active, gap/12, 0)]


def normalize_values(raw, active):
    train_valid = np.isfinite(raw['train']) & active['train']
    values = raw['train'][train_valid]
    if not len(values):
        raise ValueError('No training product observations')
    mean = float(values.mean(dtype=np.float64))
    std = max(float(values.std(dtype=np.float64)), 1e-6)
    normalized, masks = {}, {}
    for s, v in raw.items():
        mask = np.isfinite(v) & active[s]
        normalized[s] = np.where(mask, (v-mean)/std, 0).astype(np.float32)
        masks[s] = mask.astype(np.float32)
    return normalized, masks, dict(mean=mean, std=std, count=int(len(values)))


def construct_components(inputs):
    allowed = set(INPUT_KEYS) | {'observed_gpp', 'observed_gpp_valid', 'ndvi_quality', 'ndvi_available',
                                'gpp_area', 'weather', 'weather_valid'}
    if any(set(a) != allowed for a in inputs.values()):
        raise ValueError('Feature encoder accepts only registered inputs, not yield labels')
    anomalies = {p: seasonal_anomalies(inputs, p) for p in PRODUCTS}
    out = {s: {} for s in inputs}
    names = dict(history=[f'history_{i}' for i in range(15)]+[f'context_{i}' for i in range(5)])
    slot_names = ['active', 'source_month_sin', 'source_month_cos', 'calendar_days_div31', 'month_gap_div12',
                  'lai_valid', 'ndvi_valid', 'ndvi_good_area_time', 'ndvi_available_area_time',
                  'gpp_valid', 'gpp_valid_area']+[f'weather_{j}_valid' for j in range(13)]
    names['metadata'] = ['crop_area_fraction']+[f'slot_{k}_{v}' for k in range(12) for v in slot_names]
    names['weather'] = [f'slot_{k}_era5_variable_{j}' for k in range(12) for j in range(13)]
    summary = [f'slot_{k}' for k in range(12)]+['mean', 'std', 'max', 'min', 'slot_sum', 'peak_slot']
    for p in PRODUCTS:
        names[p] = [f'{p}_{representation}_{v}' for representation in ('raw_standardized', 'local_anomaly') for v in summary]
    for s, a in inputs.items():
        n = len(a['year'])
        active, _ = active_slots(a)
        out[s]['history'] = np.concatenate((a['history'], a['context']), axis=1).astype(np.float32)
        columns = calendar_fields(a)
        for field in ('observed_lai_valid', 'observed_ndvi_valid', 'ndvi_quality', 'ndvi_available',
                      'observed_gpp_valid', 'gpp_area'):
            columns.append(np.where(active, a[field], 0))
        columns.extend(a['weather_valid'][..., j] for j in range(13))
        metadata = np.stack(columns, axis=-1).reshape(n, -1)
        out[s]['metadata'] = np.concatenate((a['crop_coverage'][:, None], metadata), axis=1).astype(np.float32)
        out[s]['weather'] = a['weather'].reshape(n, -1).astype(np.float32)
        for p in PRODUCTS:
            valid = active & (a[f'observed_{p}_valid'] > 0)
            out[s][p] = np.concatenate((trajectory_features(a[f'observed_{p}'], valid),
                                        trajectory_features(anomalies[p][s], valid)), axis=1)
        for component, value in out[s].items():
            if value.shape != (n, len(names[component])) or not np.isfinite(value).all():
                raise ValueError(f'Invalid component {s} {component}')
    return out, names


def source_inputs(crop, origin):
    ready_path = GPP_AUDIT / 'full_period_ready.json'
    ready = json.loads(ready_path.read_text())
    if not ready['complete'] or not ready['all_relative_arrays_rebuilt'] or ready['yield_targets_read']:
        raise ValueError('Full-period GPP correctness gate incomplete')
    inputs_hashes = {}

    def record(path, expected=None):
        path = Path(path)
        if str(path) not in inputs_hashes:
            inputs_hashes[str(path)] = sha256(path)
        value = inputs_hashes[str(path)]
        if expected is not None and value != expected:
            raise ValueError(f'Upstream source changed: {path}')
        return value

    record(ready_path)
    record(GPP / 'manifest.json', ready['manifest_sha256'])
    old = old_cache_root(crop, origin, 0)
    meta = json.loads((old / 'manifest.json').read_text())
    record(old / 'manifest.json')
    record(NDVI / 'manifest.json', meta['ndvi_manifest_sha256'])
    a, labels = {}, {}
    for s in SPLITS:
        path = old / f'{s}.npz'
        record(path, meta['splits'][s]['file_sha256'])
        with np.load(path, allow_pickle=False) as f:
            a[s] = {k: f[k] for k in INPUT_KEYS}
            labels[s] = {k: f[k] for k in LABELS}
        if np.any(a[s]['year'] > origin) or (s == 'train' and np.any(a[s]['year'] > origin-3)):
            raise ValueError('Chronological split violated')
        if s == 'validation' and list(np.unique(a[s]['year'])) != list(range(origin-2, origin+1)):
            raise ValueError('Three validation years required')
    if np.intersect1d(labels['train']['source_indices'], labels['validation']['source_indices']).size:
        raise ValueError('Overlapping train/validation identities')
    world = ROOT / 'benchmark/cache/biid_world_model' / crop
    world_arrays = {}
    for k in ('source_indices', 'source_month', 'relative_valid', 'weather_rel'):
        path = world / f'{k}.npy'
        record(path)
        world_arrays[k] = np.load(path, mmap_mode='r')
    identity = world_arrays['source_indices']
    if np.any(np.diff(identity) <= 0):
        raise ValueError('World identity ordering invalid')
    raw_gpp, active = {}, {}
    norm = meta['normalization']
    for s, b in a.items():
        active[s], month = active_slots(b)
        ix = np.searchsorted(identity, labels[s]['source_indices'])
        np.testing.assert_array_equal(identity[ix], labels[s]['source_indices'])
        for k in ('source_month', 'relative_valid'):
            np.testing.assert_array_equal(world_arrays[k][ix], b[k])
        physical_weather = np.asarray(world_arrays['weather_rel'][ix])
        valid = np.isfinite(physical_weather) & active[s][..., None]
        b['weather_valid'] = valid.astype(np.float32)
        b['weather'] = np.where(valid, (physical_weather-np.array(norm['weather_mean'], np.float32)) /
                                np.array(norm['weather_std'], np.float32), 0).astype(np.float32)
        raw_gpp[s] = np.full(month.shape, np.nan, np.float32)
        for field in ('ndvi_quality', 'ndvi_available', 'gpp_area'):
            b[field] = np.zeros(month.shape, np.float32)
        for year in np.unique(b['year']):
            take = np.flatnonzero(b['year'] == year)
            rr, cc = b['row'][take, None], b['col'][take, None]
            mm = np.minimum(month[take], 11)
            for field, path in (
                ('gpp', GPP / 'monthly_0p5' / f'gpp_daily_rate_{year}.npy'),
                ('gpp_area', GPP / 'monthly_0p5' / f'valid_area_fraction_{year}.npy'),
                ('ndvi_quality', NDVI / 'monthly_0p5' / f'quality_fraction_monthly_{year}.npy'),
                ('ndvi_available', NDVI / 'monthly_0p5' / f'available_fraction_monthly_{year}.npy'),
                ('ndvi', NDVI / 'monthly_0p5' / f'ndvi_monthly_{year}.npy')):
                record(path, ready['files'].get(str(path)))
                grid = np.load(path, mmap_mode='r')
                value = grid[mm, rr, cc]
                if field == 'gpp':
                    raw_gpp[s][take] = np.where(active[s][take], value, np.nan)
                    rel_path = GPP / 'crops' / crop / 'gpp_daily_rate' / f'gpp_daily_rate_rel_{year}.npy'
                    record(rel_path, ready['files'][str(rel_path)])
                    rel = np.load(rel_path, mmap_mode='r')[:, b['row'][take], b['col'][take]].T
                    np.testing.assert_array_equal(value[active[s][take]], rel[active[s][take]])
                elif field == 'ndvi':
                    ndvi = meta['ndvi_normalization']
                    finite = np.isfinite(value) & active[s][take]
                    np.testing.assert_array_equal(finite, b['observed_ndvi_valid'][take] > 0)
                    expected = np.where(finite, (value-ndvi['mean'])/ndvi['std'], 0).astype(np.float32)
                    np.testing.assert_array_equal(expected, b['observed_ndvi'][take])
                else:
                    if not np.isfinite(value[active[s][take]]).all() or np.any(value[active[s][take]] < 0) or np.any(value[active[s][take]] > 1.000001):
                        raise ValueError(f'Invalid product support {field}')
                    b[field][take] = np.where(active[s][take], value, 0)
        print(f'[SIGNAL INPUT] {crop} {origin} {s}: {len(b["year"])} unchanged rows', flush=True)
    values, masks, stats = normalize_values(raw_gpp, active)
    for s in SPLITS:
        a[s]['observed_gpp'], a[s]['observed_gpp_valid'] = values[s], masks[s]
    return a, labels, dict(upstream=meta, input_sha256=inputs_hashes, gpp_training_normalization=stats,
        evaluation_split_loaded=False, gpp_source_units='gC m-2 day-1, model-derived',
        calendar='Unchanged same-year active natural-month packing')


def prepare(crop, origin, audit=False):
    root = cache_root(crop, origin)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs, labels, upstream = source_inputs(crop, origin)
        components, names = construct_components(inputs)
        spec = dict(crop=crop, origin=origin, conditions=list(CONDITIONS), upstream=upstream,
                    names=names, code_sha256={k: sha256(ROOT / 'scripts' / k) for k in CODE})
        marker = root / 'manifest.json'
        old = json.loads(marker.read_text()) if marker.exists() else None
        if old and old['spec'] != spec:
            raise ValueError('Frozen signal cache recipe changed')
        if audit and not old:
            raise ValueError('No cache to independently rebuild')
        replay = bool(old)
        files = {}
        for s in SPLITS:
            path = root / f'{s}_labels.npz'
            if replay:
                with np.load(path) as saved:
                    if set(saved.files) != set(labels[s]):
                        raise ValueError('Label schema changed')
                    for k, v in labels[s].items():
                        np.testing.assert_array_equal(saved[k], v)
            else:
                np.savez(path, **labels[s])
            files[path.name] = sha256(path)
            for component, value in components[s].items():
                path = root / f'{component}_{s}.npy'
                if replay:
                    np.testing.assert_array_equal(np.load(path, mmap_mode='r'), value)
                else:
                    np.save(path, value)
                files[path.name] = sha256(path)
        if old and old['files'] != files:
            raise ValueError('Cache output identity changed')
        if not old:
            atomic_json(marker, dict(spec=spec, files=files))
        if audit:
            atomic_json(root / 'audit.json', dict(full_array_rebuild=True, maximum_replay_error=0.,
                input_lineage_verified=True, training_only_statistics=True, evaluation_split_loaded=False,
                manifest_sha256=sha256(marker), files=files))
        print(f'[SIGNAL CACHE] {crop} {origin} audit={audit}; dimensions '+str({k: len(v) for k, v in names.items()}), flush=True)


def load(crop, origin, condition):
    root = cache_root(crop, origin)
    audit = json.loads((root / 'audit.json').read_text())
    manifest_path = root / 'manifest.json'
    if not audit['full_array_rebuild'] or audit['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Signal cache audit missing or stale')
    manifest = json.loads(manifest_path.read_text())
    if manifest['spec']['code_sha256'] != {k: sha256(ROOT / 'scripts' / k) for k in CODE}:
        raise ValueError('Feature code changed after audit')
    components = select_components(condition)
    x, labels = {}, {}
    for s in SPLITS:
        path = root / f'{s}_labels.npz'
        if sha256(path) != manifest['files'][path.name]:
            raise ValueError('Labels changed')
        with np.load(path) as saved:
            labels[s] = {k: saved[k] for k in saved.files}
        chunks = []
        for component in components:
            path = root / f'{component}_{s}.npy'
            if sha256(path) != manifest['files'][path.name]:
                raise ValueError('Features changed')
            chunks.append(np.load(path, mmap_mode='r'))
        x[s] = np.concatenate(chunks, axis=1)
    return x, labels, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', type=int, choices=ORIGINS, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    prepare(args.crop, args.origin, args.audit)
