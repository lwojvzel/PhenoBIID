"""Rebuild the frozen 501-column interface beyond the original screen years."""
import json
import numpy as np

from forecast_bridge_data import ROOT, load
from crop_signal_screen_data import cache_root, calendar_fields, NDVI, GPP
from stable_remote_data import cache_root as stable_root
from dual_remote_data import extract
from inseason_ndvi_reuse import TrajectoryEncoder, historical_support
from review_revision_data import sha256


def prepare(crop):
    original, _ = load(crop)
    old = stable_root(crop, 2012, 0)
    meta = json.loads((old/'manifest.json').read_text())
    sources = {str(old/'manifest.json'): sha256(old/'manifest.json')}
    arrays = {}
    for split in ('validation', 'test'):
        file = old/f'{split}.npz'
        sources[str(file)] = sha256(file)
        if sources[str(file)] != meta['splits'][split]['file_sha256']:
            raise ValueError('Stable cache changed')
        with np.load(file) as f:
            arrays[split] = {k: f[k] for k in f.files}
    world = ROOT/'benchmark/cache/biid_world_model'/crop
    identity = np.load(world/'source_indices.npy', mmap_mode='r')
    scale = meta['ndvi_normalization']
    weather = np.load(world/'weather_rel.npy', mmap_mode='r')
    for name in ('source_indices', 'weather_rel', 'source_month', 'relative_valid'):
        sources[str(world/f'{name}.npy')] = sha256(world/f'{name}.npy')
    for a in arrays.values():
        index = np.searchsorted(identity, a['source_indices'])
        np.testing.assert_array_equal(identity[index], a['source_indices'])
        for name in ('source_month', 'relative_valid'):
            np.testing.assert_array_equal(np.load(world/f'{name}.npy', mmap_mode='r')[index], a[name])
        active = a['relative_valid'] > 0
        a['weather'] = np.where(active[..., None], weather[index], np.nan).astype(np.float32)
        a['physical_ndvi'], _ = extract(a)
        a['physical_previous'], quality = extract(a, True)
        a['previous_ndvi_quality'] = np.where(np.isfinite(a['physical_previous']) & active, quality, 0).astype(np.float32)
        finite = np.isfinite(a['physical_ndvi']) & active
        np.testing.assert_array_equal(finite, a['observed_ndvi_valid'] > 0)
        np.testing.assert_array_equal(np.where(finite, (a['physical_ndvi']-scale['mean'])/scale['std'], 0).astype(np.float32), a['observed_ndvi'])
        a['normalized_weather'] = np.where(np.isfinite(a['weather']),
            (a['weather']-np.array(meta['normalization']['weather_mean'], np.float32)) /
            np.array(meta['normalization']['weather_std'], np.float32), 0).astype(np.float32)
        fields = {k: np.zeros(active.shape, np.float32) for k in ('ndvi_quality', 'ndvi_available', 'gpp_valid', 'gpp_area')}
        for year in np.unique(a['year']):
            ix = np.flatnonzero(a['year'] == year)
            mm = np.minimum(a['source_month'][ix], 11)
            rr, cc = a['row'][ix, None], a['col'][ix, None]
            for field, file in (
                ('ndvi_quality', NDVI/'monthly_0p5'/f'quality_fraction_monthly_{year}.npy'),
                ('ndvi_available', NDVI/'monthly_0p5'/f'available_fraction_monthly_{year}.npy'),
                ('gpp_valid', GPP/'monthly_0p5'/f'gpp_daily_rate_{year}.npy'),
                ('gpp_area', GPP/'monthly_0p5'/f'valid_area_fraction_{year}.npy')):
                sources[str(file)] = sha256(file)
                value = np.load(file, mmap_mode='r')[mm, rr, cc]
                fields[field][ix] = np.where(active[ix], np.isfinite(value) if field == 'gpp_valid' else value, 0)
            for source_year in (year-1, year):
                for prefix in ('ndvi_monthly', 'quality_fraction_monthly'):
                    file = NDVI/'monthly_0p5'/f'{prefix}_{source_year}.npy'
                    sources[str(file)] = sha256(file)
        columns = calendar_fields(a)
        columns.extend(np.where(active, value, 0) for value in (
            a['observed_lai_valid'], a['observed_ndvi_valid'], fields['ndvi_quality'],
            fields['ndvi_available'], fields['gpp_valid'], fields['gpp_area']))
        columns.extend(np.isfinite(a['weather'][..., j]).astype(np.float32) for j in range(13))
        a['metadata'] = np.concatenate((a['crop_coverage'][:, None], np.stack(columns, -1).reshape(len(active), -1)), 1).astype(np.float32)

    test = arrays['test']
    needed = ('source_indices', 'year', 'row', 'col', 'target', 'context', 'relative_valid', 'source_month', 'weather', 'previous_ndvi_quality')
    raw = {k: np.concatenate((original[k], test[k])) for k in needed}
    raw['observed_ndvi'] = np.concatenate((original['observed_ndvi'], test['physical_ndvi']))
    raw['previous_ndvi'] = np.concatenate((original['previous_ndvi'], test['physical_previous']))
    fit = np.flatnonzero(raw['year'] <= 2009)
    cache = cache_root(crop, 2012)
    cache_meta = json.loads((cache/'manifest.json').read_text())
    sources[str(cache/'manifest.json')] = sha256(cache/'manifest.json')
    encoders, features, indices = {}, {}, {}
    for split, a in arrays.items():
        take = np.flatnonzero((raw['year'] >= 2010) & (raw['year'] <= 2012)) if split == 'validation' else np.arange(len(original['year']), len(raw['year']))
        for k in ('source_indices', 'year', 'row', 'col', 'target'):
            np.testing.assert_array_equal(raw[k][take], a[k])
        encoder = TrajectoryEncoder(raw, fit, take, scale)
        parts = dict(history=np.concatenate((a['history'], a['context']), 1).astype(np.float32),
            metadata=a['metadata'], weather=a['normalized_weather'].reshape(len(take), -1),
            ndvi=encoder.encode(raw['observed_ndvi'][take]))
        for name, value in parts.items():
            if split == 'validation':
                file = cache/f'{name}_{split}.npy'
                sources[str(file)] = sha256(file)
                if sources[str(file)] != cache_meta['files'][file.name]:
                    raise ValueError('Screen cache changed')
                np.testing.assert_array_equal(value, np.load(file, mmap_mode='r'))
        features[split] = np.concatenate(list(parts.values()), 1)
        if features[split].shape[1] != 501 or not np.isfinite(features[split]).all():
            raise ValueError('Invalid extension features')
        indices[split], encoders[split] = take, encoder
    file = cache/'metadata_train.npy'
    sources[str(file)] = sha256(file)
    if sources[str(file)] != cache_meta['files'][file.name]:
        raise ValueError('Support fit data changed')
    support = historical_support(raw, fit, indices['test'], np.load(file, mmap_mode='r'))
    return raw, arrays, indices, encoders, features, support, scale, sources
