"""Verified regional inputs with separate state targets and train-only features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cybench_label_protocol import BLOCKS
from cybench_seasonal_inputs import KNOWN_FIELDS, PRODUCTS, issue_masks
from forecast_bridge_data import remote_features
from review_revision_data import ROOT, sha256

SEASONS = ROOT / 'Data/processed/cybench_seasonal_v1'
LABELS = ROOT / 'benchmark/cache/cybench_inseason_labels_v1'
RESULT = ROOT / 'benchmark/results/cybench_inseason_models_v1'
RECIPES = {'maize': ('gpp',), 'wheat': ('ndvi', 'gpp')}
SEEDS = (42, 45, 48)
PERCENTS = (10, 30, 50)
COUNTRIES = ('DE', 'FR', 'PL')
VIEW_FIELDS = KNOWN_FIELDS | frozenset(('observed_vegetation', 'observed_vegetation_support',
    'observed_valid', 'visible_mask', 'hidden_mask', 'issue_day', 'days_to_supplied_eos'))
DATA_CODE = ('cybench_inseason_model_data.py', 'cybench_label_protocol.py',
    'cybench_seasonal_inputs.py', 'forecast_bridge_data.py', 'observed_remote_benchmark.py')


def provenance():
    sources = {}
    definitions = ((SEASONS, ROOT / 'benchmark/results/cybench_inseason_inputs_v1/seasonal_verification.json'),
        (LABELS, ROOT / 'benchmark/results/cybench_inseason_labels_v1/label_verification.json'))
    manifests = []
    for folder, audit_path in definitions:
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        audit = json.loads(audit_path.read_text())
        if not audit['passed'] or audit['manifest_sha256'] != sha256(manifest_path):
            raise ValueError('Regional source no longer matches independent verification')
        registration = folder / 'registration.json'
        if sha256(registration) != manifest['registration_sha256']:
            raise ValueError('Changed input registration')
        for name, digest in json.loads(registration.read_text())['code_sha256'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise ValueError('Changed regional preprocessing implementation')
        for path in (manifest_path, registration, audit_path):
            sources[str(path)] = sha256(path)
        manifests.append(manifest)
    return manifests, sources


def checked(path, digest, sources):
    if sha256(path) != digest:
        raise ValueError(f'Changed regional input: {path}')
    sources[str(path)] = digest
    return path


def load_identity(crop, cutoff, *, with_history=True):
    if crop not in RECIPES or cutoff not in BLOCKS:
        raise ValueError('Unregistered regional crop or cutoff')
    (_, manifest), sources = provenance()
    folder = LABELS / crop
    paths = {}
    for name in ('identities.csv', f'block_{cutoff}.npz') + (('history.npz',) if with_history else ()):
        paths[name] = checked(folder / name, manifest['files'][f'{crop}/{name}'], sources)
    ids = pd.read_csv(paths['identities.csv'], dtype={'country': str, 'adm_id': str})
    np.testing.assert_array_equal(ids.sample_index, np.arange(len(ids)))
    if ids.duplicated(['country', 'adm_id', 'year']).any() or not ids.country.isin(COUNTRIES).all():
        raise ValueError('Unexpected regional identity')
    history = None
    if with_history:
        with np.load(paths['history.npz'], allow_pickle=False) as f:
            history = {k: f[k] for k in f.files}
    with np.load(paths[f'block_{cutoff}.npz'], allow_pickle=False) as f:
        partitions = {k: f[k] for k in f.files}
    if set(partitions) != {'inner_fit', 'inner_validation', 'full_fit', 'evaluation'}:
        raise ValueError('Regional partition schema changed')
    for part, ix in partitions.items():
        if ix.dtype.kind not in 'iu' or ix.ndim != 1 or not len(ix) or len(np.unique(ix)) != len(ix):
            raise ValueError('Empty or duplicate regional partition')
        if np.any((ix < 0) | (ix >= len(ids))):
            raise ValueError('Partition index outside population')
        years = ids.year.to_numpy()[ix]
        valid = (years <= cutoff-2 if part == 'inner_fit' else
            (years >= cutoff-1) & (years <= cutoff) if part == 'inner_validation' else
            years <= cutoff if part == 'full_fit' else np.isin(years, BLOCKS[cutoff]))
        if not valid.all():
            raise ValueError('Temporal partition crossed its boundary')
    return ids, history, partitions, sources


def load_targets(crop, sources):
    manifest = json.loads((LABELS / 'manifest.json').read_text())
    path = checked(LABELS / crop / 'targets.npz', manifest['files'][f'{crop}/targets.npz'], sources)
    with np.load(path, allow_pickle=False) as f:
        if f.files != ['yield']:
            raise ValueError('Unexpected yield target fields')
        return f['yield'].astype(np.float64)


def load_seasons(crop, identities, indices, sources, *, targets=True, maximum_year=None):
    ix = np.asarray(indices)
    if ix.ndim != 1 or not len(ix) or len(np.unique(ix)) != len(ix):
        raise ValueError('Expected nonempty unique seasonal indices')
    selected = identities.iloc[ix]
    if maximum_year is not None and selected.year.max() > maximum_year:
        raise ValueError('Attempt to load a future seasonal file for training')
    manifest = json.loads((SEASONS / 'manifest.json').read_text())
    known, supervision = {}, {}
    for year in sorted(selected.year.unique()):
        positions = np.flatnonzero(selected.year.to_numpy() == year)
        rows = selected.iloc[positions].season_row.to_numpy(int)
        for suffix, destination in [('known', known)] + ([('state_targets', supervision)] if targets else []):
            name = f'{crop}/{int(year)}_{suffix}.npz'
            path = checked(SEASONS / name, manifest['outputs'][name], sources)
            with np.load(path, allow_pickle=False) as f:
                for key in f.files:
                    data = f[key][rows]
                    if key not in destination:
                        destination[key] = np.empty((len(ix), *data.shape[1:]), data.dtype)
                    destination[key][positions] = data
    if set(known) != KNOWN_FIELDS:
        raise ValueError('Known-condition schema changed')
    np.testing.assert_array_equal(known['region_index'], selected.region_index)
    np.testing.assert_array_equal(known['year'], selected.year)
    issue_masks(known['active_mask'], 0)
    if targets and set(supervision) != {'vegetation', 'vegetation_support'}:
        raise ValueError('Unexpected state targets')
    return known, supervision


def state_population(identities, partitions, cutoff):
    seen = np.unique(identities.iloc[partitions['inner_fit']].region_index)
    years = identities.year.to_numpy()
    eligible = identities.region_index.isin(seen).to_numpy() & (years <= cutoff)
    return dict(inner_fit=np.flatnonzero(eligible & (years <= cutoff-2)),
        inner_validation=np.flatnonzero(eligible & (years >= cutoff-1)),
        full_fit=np.flatnonzero(eligible))


def context5(known):
    context = np.asarray(known['context'], np.float32)
    if context.shape != (len(known['year']), 4) or not np.isfinite(context).all():
        raise ValueError('Invalid regional background')
    return np.column_stack((context[:, 0]/90, context[:, 1]/180,
        (known['year']-2000)/40, context[:, 2]/366, context[:, 3]/366)).astype(np.float32)


def state_arrays(known, targets=None):
    if set(known) != KNOWN_FIELDS:
        raise ValueError('State initialization accepts known fields only')
    active = known['active_mask']
    issue_masks(active, 0)
    n = len(active)
    weather = np.asarray(known['weather'])
    if weather.shape != (n, 12, 13):
        raise ValueError('Expected thirteen regional meteorological channels')
    a = dict(weather=np.where(active[..., None] & (known['weather_finite_grid_area_fraction'] > 0),
        weather, np.nan).astype(np.float32), relative_valid=active.astype(np.float32),
        context=context5(known), year=known['year'].copy())
    for j, p in enumerate(PRODUCTS):
        value = known['previous_vegetation'][..., j]
        quality = known['previous_vegetation_support'][..., j]
        valid = active & known['previous_valid'][..., j] & np.isfinite(value) & (quality > 0)
        a[f'previous_{p}'] = np.where(valid, value, np.nan).astype(np.float32)
        a[f'previous_{p}_quality'] = np.where(valid, quality, 0).astype(np.float32)
        if targets is not None:
            value = targets['vegetation'][..., j]
            valid = active & np.isfinite(value) & (targets['vegetation_support'][..., j] > 0)
            a[f'observed_{p}'] = np.where(valid, value, np.nan).astype(np.float32)
    return a


def mean_std(value, axis=None):
    value = np.where(np.isfinite(value), value, np.nan).astype(np.float64)
    count = np.isfinite(value).sum(axis=axis)
    mean = np.divide(np.nansum(value, axis=axis), count,
        out=np.zeros_like(np.nansum(value, axis=axis)), where=count > 0)
    variance = np.divide(np.nansum((value-mean)**2, axis=axis), count,
        out=np.zeros_like(mean), where=count > 0)
    return np.asarray(mean), np.maximum(np.sqrt(variance), 1e-6)


def fit_state_stats(known, targets, products, cutoff):
    if np.max(known['year']) > cutoff:
        raise ValueError('Normalization attempted to use future seasons')
    a = state_arrays(known, targets)
    active = known['active_mask']
    mean, std = mean_std(a['weather'][active], axis=0)
    stats = dict(weather_mean=mean.tolist(), weather_std=std.tolist())
    for p in products:
        values = a[f'observed_{p}'][active]
        if not np.isfinite(values).any():
            raise ValueError('No valid training vegetation; cannot fit state scale')
        mean, std = mean_std(values)
        stats[p] = dict(mean=float(mean), std=float(std))
    return stats


def history_matrix(history, identities, known):
    n = len(identities)
    if not identities.country.isin(COUNTRIES).all():
        raise ValueError('Country outside registered regional study')
    np.testing.assert_array_equal(history['lag_valid'], np.isfinite(history['lag_yield']))
    x = np.concatenate((history['lag_yield'], history['lag_valid'], history['history_summary'],
        known['context'][:, :2], identities.year.to_numpy()[:, None],
        np.column_stack([identities.country.to_numpy() == c for c in COUNTRIES])), axis=1)
    if x.shape != (n, 23):
        raise ValueError('Regional historical feature schema changed')
    return x.astype(np.float32)


def balanced_weights(identities):
    counts = identities.groupby(['country', 'year']).size()
    years = counts.groupby(level=0).size()
    weight = np.array([1/(len(years)*years.loc[r.country]*counts.loc[(r.country, r.year)])
                      for r in identities.itertuples()], dtype=np.float64)
    return weight / weight.mean()


def regional_score(target, prediction, identities):
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    if target.shape != prediction.shape or target.shape != (len(identities),):
        raise ValueError('Wrong regional prediction shape')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite paired regional labels or prediction')
    data = identities[['country', 'year']].copy()
    data['squared_error'] = (target-prediction)**2
    data['absolute_error'] = abs(target-prediction)
    annual = data.groupby(['country', 'year']).agg(
        mse=('squared_error', 'mean'), mae=('absolute_error', 'mean'), samples=('squared_error', 'size'))
    annual['rmse'] = np.sqrt(annual.pop('mse'))
    score = float(annual.rmse.groupby(level=0).mean().mean())
    return score, annual.reset_index().to_dict(orient='records')


def fit_feature_stats(history, known, targets, products, cutoff):
    stats = fit_state_stats(known, targets, products, cutoff)
    mean, std = mean_std(history, axis=0)
    stats.update(history_mean=mean.tolist(), history_std=std.tolist(), cutoff=int(cutoff))
    active = known['active_mask']
    month = np.maximum(known['source_month_id'] % 100 - 1, 0)
    keys = known['region_index'][:, None]*12 + month
    size = int(keys.max())+1
    a = state_arrays(known, targets)
    for p in products:
        values = a[f'observed_{p}']
        valid = active & np.isfinite(values)
        stats[p]['climatology'] = dict(
            sums=np.bincount(keys[valid], weights=values[valid], minlength=size).tolist(),
            counts=np.bincount(keys[valid], minlength=size).tolist(),
            month_sums=np.bincount(month[valid], weights=values[valid], minlength=12).tolist(),
            month_counts=np.bincount(month[valid], minlength=12).tolist())
    stats['fit_region_years'] = np.column_stack((known['region_index'], known['year'])).tolist()
    return stats


def regional_climatology(known, stats, product, training_values=None):
    fitted = stats[product]['climatology']
    month = np.maximum(known['source_month_id'] % 100-1, 0)
    keys = known['region_index'][:, None]*12+month
    sums, counts = np.asarray(fitted['sums'], float), np.asarray(fitted['counts'], float)
    in_bounds = keys < len(sums)
    safe = np.minimum(keys, len(sums)-1)
    numerator, denominator = np.where(in_bounds, sums[safe], 0), np.where(in_bounds, counts[safe], 0)
    month_sum = np.asarray(fitted['month_sums'], float)[month].copy()
    month_count = np.asarray(fitted['month_counts'], float)[month].copy()
    if training_values is not None:
        registered = set(map(tuple, stats['fit_region_years']))
        if any((r, y) not in registered for r, y in zip(known['region_index'], known['year'])):
            raise ValueError('Leave-self-out subtraction on a non-training season')
        valid = known['active_mask'] & np.isfinite(training_values)
        numerator -= np.where(valid, training_values, 0)
        denominator -= valid
        month_sum -= np.where(valid, training_values, 0)
        month_count -= valid
    fallback = np.divide(month_sum, month_count, out=np.full_like(month_sum, stats[product]['mean']), where=month_count > 0)
    return np.divide(numerator, denominator, out=fallback, where=denominator > 0).astype(np.float32)


def normalized_history(history, stats):
    mean, std = np.asarray(stats['history_mean']), np.asarray(stats['history_std'])
    return np.nan_to_num((history-mean)/std, nan=0., posinf=0., neginf=0.).astype(np.float32)


def feature_matrix(history, view, stats, products, *, completed=None, training_values=None):
    if set(view) != VIEW_FIELDS:
        raise ValueError('Regional head receives only issued known/prefix fields')
    if not products or not set(products).issubset(PRODUCTS):
        raise ValueError('Unexpected regional products')
    active, visible, hidden = view['active_mask'], view['visible_mask'], view['hidden_mask']
    if np.any(visible & hidden) or not np.array_equal(visible | hidden, active):
        raise ValueError('Invalid issued mask partition')
    if np.any(view['observed_valid'][hidden]) or np.any(view['observed_vegetation_support'][hidden]):
        raise ValueError('Hidden remote support entered prediction input')
    if np.isfinite(view['observed_vegetation'][hidden]).any():
        raise ValueError('Hidden remote values entered issued prefix')
    known = {k: view[k] for k in KNOWN_FIELDS}
    n = len(active)
    a = state_arrays(known)
    month = np.maximum(view['source_month_id'] % 100-1, 0)
    calendar = np.stack((active, np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
        view['source_month_id']//100-view['year'][:, None], view['calendar_overlap_days']/31), axis=-1)
    calendar = np.where(active[..., None], calendar, 0)
    weather = a['weather']
    weather_values = np.nan_to_num((weather-np.asarray(stats['weather_mean']))/np.asarray(stats['weather_std']),
        nan=0., posinf=0., neginf=0.)
    weather_features = np.concatenate((weather_values, np.isfinite(weather),
        np.where(active[..., None], view['weather_finite_grid_area_fraction'], 0)), axis=-1)
    chunks = [normalized_history(history, stats),
        np.column_stack((view['context'][:, 2:]/366, view['cross_year'])),
        calendar.reshape(n, 60), weather_features.reshape(n, 468), visible, hidden]
    encoded = []
    if completed is not None and set(completed) != set(products):
        raise ValueError('Completion products differ from registered recipe')
    for p in products:
        j = PRODUCTS.index(p)
        prior = a[f'previous_{p}']
        previous_valid = np.isfinite(prior)
        previous = np.where(previous_valid, (prior-stats[p]['mean'])/stats[p]['std'], 0)
        prefix = view['observed_vegetation'][..., j]
        valid = view['observed_valid'][..., j]
        chunks.extend((previous, previous_valid, a[f'previous_{p}_quality'], valid,
                       view['observed_vegetation_support'][..., j]))
        value = prefix if completed is None else np.asarray(completed[p])
        if value.shape != (n, 12):
            raise ValueError('Wrong completed trajectory shape')
        if completed is not None:
            np.testing.assert_array_equal(value[~hidden], prefix[~hidden])
            if not np.isfinite(value[hidden]).all():
                raise ValueError('Nonfinite hidden completion')
        climo = regional_climatology(known, stats, p,
            None if training_values is None else training_values[p])
        encoded.append(remote_features(value, active, climo, stats[p]))
    result = np.concatenate((*chunks, *encoded), axis=1).astype(np.float32)
    if result.shape != (n, 578+96*len(products)) or not np.isfinite(result).all():
        raise ValueError('Invalid regional terminal feature matrix')
    return result
