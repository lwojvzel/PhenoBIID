"""Aggregate existing monthly products to audited CY-Bench polygons, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'benchmark/results/cybench_inseason_inputs_v1'
NDVI = ROOT / 'Data/processed/pku_gimms_ndvi_v1p2'
GPP = ROOT / 'Data/processed/reclue_monthly_gpp_v1'
WEATHER = ROOT / 'Data/era5land/monthly_npy_lon180_0p5deg'
OUT = ROOT / 'Data/processed/cybench_regional_monthly_v1'
VARIABLES = ('d2m', 't2m', 'stl1', 'stl2', 'swvl1', 'swvl2', 'swvl3',
             'ssrd', 'pev', 'u10', 'v10', 'sp', 'tp')
FIELDS = ('ndvi', 'ndvi_observed_area_fraction', 'ndvi_available_area_fraction',
          'gpp', 'gpp_observed_area_fraction', 'weather',
          'weather_finite_grid_area_fraction')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fractions(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite & ((values < -1e-6) | (values > 1+1e-6))):
        raise ValueError('Support fractions outside [0, 1]')
    return np.where(finite, np.clip(values, 0, 1), 0)


def supported_mean(values, support, weights):
    """Return [region, time] means and fractions of full regional area."""
    values = np.asarray(values, dtype=np.float64)
    support = fractions(support)
    if values.ndim != 2 or values.shape != support.shape or values.shape[1] != weights.shape[1]:
        raise ValueError('Expected matching [time, source_cell] values/support')
    valid = np.isfinite(values) & (support > 0)
    mass = np.where(valid, support, 0)
    numerator = weights @ (np.where(valid, values, 0)*mass).T
    denominator = weights @ mass.T
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return result.astype(np.float32), denominator.astype(np.float32)


def load_weights():
    audit = json.loads((AUDIT/'input_audit.json').read_text())
    path = AUDIT/'polygon_grid_weights.csv'
    if sha256(path) != audit['outputs'][path.name]:
        raise ValueError('Polygon overlap audit changed')
    frame = pd.read_csv(path)
    regions = frame[['country', 'adm_id']].drop_duplicates().sort_values(['country', 'adm_id']).reset_index(drop=True)
    regions.insert(0, 'region_index', np.arange(len(regions), dtype=np.int32))
    frame = frame.merge(regions, validate='many_to_one')
    cells = np.sort(frame.grid_index.unique())
    weights = csr_matrix((frame.region_area_fraction.to_numpy(),
                          (frame.region_index.to_numpy(), np.searchsorted(cells, frame.grid_index))),
                         shape=(len(regions), len(cells)))
    if np.any(weights.data <= 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError('Nonconservative polygon weights')
    return regions, cells, weights, {str(path): sha256(path), str(AUDIT/'input_audit.json'): sha256(AUDIT/'input_audit.json')}


def coordinates():
    sources = {}
    for coord in ('lat', 'lon'):
        expected = np.load(ROOT/f'Data/processed/crop_yield_growing_season/{coord}.npy')
        for directory in (NDVI, GPP, WEATHER):
            file = directory/(coord+'.npy')
            np.testing.assert_array_equal(expected, np.load(file))
            sources[str(file)] = sha256(file)
    path = WEATHER/'variable_order.txt'
    if tuple(path.read_text().split()) != VARIABLES:
        raise ValueError('Weather order changed')
    sources[str(path)] = sha256(path)
    return sources


def source_hashes(year):
    ndvi_meta = NDVI/f'metadata/{year}.json'
    gpp_meta = ROOT/f'visualize/paper_experiments/reclue_monthly_gpp_alignment_v1/{year}.json'
    nm, gm = json.loads(ndvi_meta.read_text()), json.loads(gpp_meta.read_text())
    expected = {str(NDVI/k): v for k,v in nm['output_hashes'].items()}
    expected.update({p['path']: p['sha256'] for p in gm['outputs']})
    checks = gm['independent_rebuild']
    if (not gm['source_format_passed'] or len(checks) != 12
            or [c['month'] for c in checks] != list(range(1, 13))
            or any(c['independent_cells'] <= 0 for c in checks)):
        raise ValueError('GPP source audit did not pass')
    np.testing.assert_allclose([c['coarse_integral'] for c in checks],
                               [c['fine_integral'] for c in checks], rtol=2e-7, atol=1e-7)
    sources = {str(ndvi_meta): sha256(ndvi_meta), str(gpp_meta): sha256(gpp_meta)}
    return expected, sources


def aggregate_year(year, cells, weights):
    expected, sources = source_hashes(year)

    def read(path, weather=False):
        before = path.stat()
        digest = sha256(path)
        if not weather and expected.get(str(path)) != digest:
            raise ValueError(f'Product hash mismatch: {path}')
        a = np.load(path, mmap_mode='r')
        shape = (12, 13, 360, 720) if weather else (12, 360, 720)
        if a.shape != shape or a.dtype != np.float32:
            raise ValueError(f'Unexpected monthly array: {path}')
        a = np.asarray(a.reshape(-1, 360*720)[:, cells])
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError('Source changed during aggregation')
        sources[str(path)] = digest
        return a

    result = {}
    n = read(NDVI/f'monthly_0p5/ndvi_monthly_{year}.npy')
    q = read(NDVI/f'monthly_0p5/quality_fraction_monthly_{year}.npy')
    available = read(NDVI/f'monthly_0p5/available_fraction_monthly_{year}.npy')
    result['ndvi'], result['ndvi_observed_area_fraction'] = supported_mean(n, q, weights)
    result['ndvi_available_area_fraction'] = (weights @ fractions(available).T).astype(np.float32)
    g = read(GPP/f'monthly_0p5/gpp_daily_rate_{year}.npy')
    q = read(GPP/f'monthly_0p5/valid_area_fraction_{year}.npy')
    result['gpp'], result['gpp_observed_area_fraction'] = supported_mean(g, q, weights)
    weather = read(WEATHER/f'era5land_monthly_0p5deg_{year}.npy', weather=True)
    mean, support = supported_mean(weather, np.isfinite(weather), weights)
    result['weather'] = mean.reshape(len(mean), 12, 13)
    result['weather_finite_grid_area_fraction'] = support.reshape(len(mean), 12, 13)
    if set(result) != set(FIELDS):
        raise ValueError('Output whitelist changed')
    return result, sources


def main(smoke=False):
    out = OUT.with_name(OUT.name+'_smoke') if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    regions, cells, weights, sources = load_weights()
    sources.update(coordinates())
    years = [1982] if smoke else list(range(1982, 2017))
    config = dict(years=years, fields=FIELDS, weather_order=VARIABLES,
        n_regions=len(regions), n_source_cells=len(cells),
        source_sha256=sources, code_sha256=sha256(Path(__file__)),
        aggregation='EPSG:6933 polygon overlap times product support; finite-only normalization',
        ndvi_support='Monthly good-quality area-time fraction of preaggregated fields',
        gpp_support='Provider-valid area fraction of preaggregated fields',
        weather_support='Finite coarse-grid area, not native ERA5 retrieval coverage',
        approximation='Within-cell support is assumed spatially uniform; monthly aggregation cannot recover native pixel-time pooling',
        labels_loaded=False, normalization_fitted=False, temporal_interpolation=False,
        crop_area_weighted=False, seasons_packed=False, smoke=smoke)
    registration = out/'registration.json'
    encoded = json.dumps(config, indent=2)+'\n'
    if registration.exists() and registration.read_text() != encoded:
        raise ValueError('Registered regional input construction changed')
    registration.write_text(encoded)
    regions.to_csv(out/'regions.csv', index=False)
    monthly = out/'calendar_months'
    monthly.mkdir(exist_ok=True)
    reports = []
    for year in years:
        marker = monthly/f'{year}.json'
        output = monthly/f'{year}.npz'
        if marker.exists():
            record = json.loads(marker.read_text())
            if (record['registration_sha256'] != sha256(registration)
                    or record['output_sha256'] != sha256(output)
                    or any(sha256(Path(p)) != h for p,h in record['sources'].items())):
                raise ValueError('Existing regional year provenance mismatch')
        else:
            start = time.monotonic()
            arrays, source = aggregate_year(year, cells, weights)
            temporary = output.with_suffix('.part')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(output)
            record = dict(year=year, output_sha256=sha256(output),
                registration_sha256=sha256(registration), sources=source,
                shapes={k:list(a.shape) for k,a in arrays.items()},
                finite_fraction={k:float(np.isfinite(arrays[k]).mean()) for k in ('ndvi', 'gpp', 'weather')},
                mean_supported_fraction={k:float(arrays[k].mean()) for k in FIELDS if k.endswith('fraction')},
                seconds=time.monotonic()-start)
            marker.write_text(json.dumps(record, indent=2)+'\n')
        reports.append(dict(year=year, **record['finite_fraction']))
        print(f'[REGION] {year}: {record["finite_fraction"]}', flush=True)
    pd.DataFrame(reports).to_csv(out/'annual_support.csv', index=False)
    manifest = dict(years=years, complete=True, smoke=smoke, models_fitted=0,
        labels_loaded=False, seasons_packed=False, registration_sha256=sha256(registration),
        regions_sha256=sha256(out/'regions.csv'), support_sha256=sha256(out/'annual_support.csv'),
        year_markers={str(y):sha256(monthly/f'{y}.json') for y in years})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(directory=str(out), years=len(years), regions=len(regions))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
