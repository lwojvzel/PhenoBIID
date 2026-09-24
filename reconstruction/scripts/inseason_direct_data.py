from __future__ import annotations
import json
import numpy as np
from dual_remote_data import extract, OUT as NDVI_ROOT
from forecast_bridge_data import ROOT, PRODUCTS, load
from inseason_signal_matching import physical_gpp
from review_revision_data import sha256
from run_history_multimodal_baselines import build_causal_history_features
from stable_remote_data import cache_root
RECIPES = dict(maize='gpp', rice='ndvi', soybean='ndvi', wheat='ndvi_gpp')

def physical_inputs(crop):
    original, manifest = load(crop, verify=True)
    folder = cache_root(crop, 2012, 0)
    meta = json.loads((folder / 'manifest.json').read_text())
    file = folder / 'test.npz'
    sources = {str(file): sha256(file)}
    if sources[str(file)] != meta['splits']['test']['file_sha256']:
        raise ValueError('Changed test cohort')
    keys = ('source_indices', 'year', 'row', 'col', 'target', 'context',
            'crop_coverage', 'relative_valid', 'source_month')
    with np.load(file) as saved:
        test = {k: saved[k] for k in keys}
    if not np.array_equal(np.unique(test['year']), [2013, 2014, 2015, 2016]):
        raise ValueError('Unexpected test years')
    world = ROOT / 'benchmark/cache/biid_world_model' / crop
    ids = np.load(world / 'source_indices.npy', mmap_mode='r')
    sources[str(world / 'source_indices.npy')] = sha256(world / 'source_indices.npy')
    ix = np.searchsorted(ids, test['source_indices'])
    np.testing.assert_array_equal(ids[ix], test['source_indices'])
    for name in ('relative_valid', 'source_month'):
        path = world / f'{name}.npy'
        sources[str(path)] = sha256(path)
        np.testing.assert_array_equal(np.load(path, mmap_mode='r')[ix], test[name])
    active = test['relative_valid'] > 0
    for key, name in (('weather', 'weather_rel'), ('observed_lai', 'target_lai_rel'),
                      ('previous_lai', 'previous_lai')):
        path = world / f'{name}.npy'
        sources[str(path)] = sha256(path)
        values = np.load(path, mmap_mode='r')[ix]
        mask = active[..., None] if key == 'weather' else active
        if key == 'previous_lai':
            valid_file = world / 'previous_lai_valid.npy'
            sources[str(valid_file)] = sha256(valid_file)
            mask = mask & (np.load(valid_file, mmap_mode='r')[ix] > 0)
        test[key] = np.where(mask, values, np.nan).astype(np.float32)
    test['observed_ndvi'], _ = extract(test)
    test['previous_ndvi'], quality = extract(test, True)
    test['previous_ndvi_quality'] = np.where(
        active & np.isfinite(test['previous_ndvi']), quality, 0).astype(np.float32)
    for year in range(2012, 2017):
        for prefix in ('ndvi_monthly', 'quality_fraction_monthly'):
            path = NDVI_ROOT / 'monthly_0p5' / f'{prefix}_{year}.npy'
            sources[str(path)] = sha256(path)
    test['observed_gpp'], _, files = physical_gpp(test)
    sources.update(files)
    test['previous_gpp'], test['previous_gpp_quality'], files = physical_gpp(test, True)
    sources.update(files)

    # Include all original yield rows when constructing causal grid histories.
    # Filtering to the remote-sensing cohort first would discard valid past yields.
    history_root = ROOT / 'benchmark/cache/multimodal_main' / crop
    history_raw = {}
    for name in ('year', 'row', 'col', 'target'):
        path = history_root / f'{name}.npy'
        history_raw[name] = np.load(path, mmap_mode='r')
        sources[str(path)] = sha256(path)
    history, baseline = build_causal_history_features(history_raw, 0., 1.)
    span = max(int(history_raw['year'].max())-int(history_raw['year'].min()), 1)
    history[:, 14] *= span / 35.
    np.testing.assert_allclose(history[original['source_indices']], original['history'],
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(baseline[original['source_indices']], original['baseline'])
    for key in ('year', 'row', 'col', 'target'):
        np.testing.assert_array_equal(history_raw[key][test['source_indices']], test[key])
    test['history'] = history[test['source_indices']]
    test['baseline'] = baseline[test['source_indices']]
    arrays = {k: np.concatenate((original[k], test[k])) for k in original}
    if len(np.unique(arrays['source_indices'])) != len(arrays['year']):
        raise ValueError('Duplicate identities')
    return arrays, dict(raw_manifest=manifest, extension_files=sources,
                       physical_history_replay=True)

def split_rows(a, fit_end):
    if fit_end not in (2009, 2011):
        raise ValueError('Only registered training windows are allowed')
    y = a['year']
    return dict(train=np.flatnonzero(y <= fit_end),
                validation=np.flatnonzero((y > fit_end) & (y <= 2012)),
                test=np.flatnonzero((y >= 2013) & (y <= 2016)))
