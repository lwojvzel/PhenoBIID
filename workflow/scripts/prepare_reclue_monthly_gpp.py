"""Verified monthly GPP aggregation and reuse of the fixed crop-month mappings."""
import argparse
import calendar
import json
from pathlib import Path
import re
import stat
import time
import zipfile

import numpy as np
import requests
from rasterio.io import MemoryFile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fetch_reclue_gpp_sample import RAW as SAMPLE, md5
from process_glass_lai_avhrr_to_growing_season import nearest_mirca_year, reorder_relative_months
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

RAW = ROOT / 'Data/external/reclue_monthly_gpp_v1'
OUT = ROOT / 'Data/processed/reclue_monthly_gpp_v1'
AUDIT = ROOT / 'visualize/paper_experiments/reclue_monthly_gpp_alignment_v1'
GROWING = ROOT / 'Data/processed/crop_yield_growing_season'
YEARS = tuple(range(1982, 2017))


def frozen_array(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, mmap_mode='r')
        if old.dtype != values.dtype or old.shape != values.shape:
            raise ValueError(f'Existing array identity mismatch: {path}')
        np.testing.assert_array_equal(old, values)
    else:
        part = path.with_suffix('.part.npy')
        if part.exists():
            raise ValueError(f'Unresolved partial array: {part}')
        np.save(part, values)
        part.replace(path)
    return dict(path=str(path), sha256=sha256(path), shape=list(values.shape), dtype=str(values.dtype))


def source_month(name, year):
    match = re.fullmatch(fr'{year}/GLASSGPP_{year}(0[1-9]|1[0-2])_005D\.tif', name)
    if match is None:
        raise ValueError(f'Unexpected monthly member {name}')
    return int(match.group(1))


def row_areas(height, degrees):
    edges = np.deg2rad(90 - np.arange(height+1)*degrees)
    weights = np.sin(edges[:-1]) - np.sin(edges[1:])
    if not np.all(weights > 0):
        raise ValueError('Invalid latitude row areas')
    return weights


def aggregate(codes, factor=10, degrees=.05):
    if codes.dtype != np.uint16 or codes.ndim != 2:
        raise ValueError('Expected two-dimensional uint16 source')
    h, w = codes.shape
    if factor < 1 or h % factor or w % factor:
        raise ValueError('Aggregation factor does not tile the source')
    weights = row_areas(h, degrees)[:, None]
    valid = codes <= 60000
    shape = (h//factor, factor, w//factor, factor)
    denominator = (valid*weights).reshape(shape).sum((1, 3))
    numerator = (np.where(valid, codes, 0).astype(np.float64)*.01*weights).reshape(shape).sum((1, 3))
    means = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    full_area = factor*weights.reshape(h//factor, factor).sum(1)[:, None]
    support = denominator/full_area
    return np.flipud(means).astype(np.float32), np.flipud(support).astype(np.float32)


def verify_aggregation(codes, mean, support, month):
    weights = row_areas(3600, .05)
    rng = np.random.default_rng(4200+month)
    locations = [(0, 0), (359, 719), (0, 719), (359, 0), (179, 359)]
    locations += [(int(r), int(c)) for r, c in zip(rng.integers(0, 360, 251), rng.integers(0, 720, 251))]
    max_error = 0.
    for r, c in locations:
        start = (359-r)*10
        values, areas = [], []
        for sr in range(start, start+10):
            for sc in range(c*10, (c+1)*10):
                if int(codes[sr, sc]) <= 60000:
                    values.append(float(codes[sr, sc])*.01)
                    areas.append(float(weights[sr]))
        denominator = sum(areas)
        total_area = sum(float(weights[j])*10 for j in range(start, start+10))
        expected = sum(v*a for v, a in zip(values, areas))/denominator if denominator else np.nan
        np.testing.assert_allclose(mean[r, c], expected, rtol=2e-7, atol=3e-5, equal_nan=True)
        np.testing.assert_allclose(support[r, c], denominator/total_area, rtol=2e-7, atol=1e-7)
        if np.isfinite(expected):
            max_error = max(max_error, abs(float(mean[r, c])-expected))
    fine_integral = 0.
    for r in range(3600):
        row = codes[r]
        fine_integral += float(row[row <= 60000].astype(np.float64).sum())*.01*weights[r]
    coarse_area = np.flipud(10*weights.reshape(360, 10).sum(1))[:, None]
    coarse_integral = np.nansum(mean.astype(np.float64)*support.astype(np.float64)*coarse_area)
    np.testing.assert_allclose(coarse_integral, fine_integral, rtol=2e-7, atol=1e-7)
    return dict(independent_cells=len(locations), max_cell_abs_error=max_error,
                fine_integral=float(fine_integral), coarse_integral=float(coarse_integral))


def mapping_checked(monthly, src):
    if not np.all((src < 12) | (src == 255)):
        raise ValueError('Invalid mapping index')
    rel = reorder_relative_months(monthly, src)
    for k in range(12):
        rows, cols = np.nonzero(src[k] != 255)
        expected = monthly[src[k, rows, cols], rows, cols]
        np.testing.assert_array_equal(rel[k, rows, cols], expected)
        if not np.isnan(rel[k][src[k] == 255]).all():
            raise ValueError('Padding contains a finite state')
    return rel


def source_record():
    original = json.loads((SAMPLE / 'record.json').read_text())
    if original['id'] != 14350035 or original['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Unexpected pinned source or license')
    selected = [v for v in original['files'] if v['key'] in {f'{y}.zip' for y in YEARS}]
    if len(selected) != 35:
        raise ValueError('Incomplete pinned source years')
    path = RAW / 'record.json'
    RAW.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != original:
        raise ValueError('Previously pinned source differs')
    if not path.exists():
        atomic_json(path, original)
    return {int(v['key'][:4]): v for v in selected}


def fetch(year, entry):
    if year not in YEARS or entry['key'] != f'{year}.zip':
        raise ValueError('Unregistered source year')
    size = int(entry['size'])
    algorithm, checksum = entry['checksum'].split(':')
    if algorithm != 'md5':
        raise ValueError('Unexpected provider checksum')
    path = SAMPLE / '1982.zip' if year == 1982 else RAW / entry['key']
    if not path.exists():
        part = path.with_suffix('.zip.part')
        retry = Retry(total=3, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
        with requests.Session() as session:
            session.headers['User-Agent'] = 'AgroClimate-data-audit/1.0'
            session.mount('https://', HTTPAdapter(max_retries=retry))
            for attempt in range(4):
                start = part.stat().st_size if part.exists() else 0
                if start > size:
                    raise ValueError('Oversized partial source; preserve for inspection')
                if start == size:
                    break
                try:
                    headers = {'Range': f'bytes={start}-'} if start else {}
                    with session.get(entry['links']['self'], headers=headers, stream=True, timeout=(30, 180)) as response:
                        response.raise_for_status()
                        if start and (response.status_code != 206 or not response.headers.get(
                                'Content-Range', '').startswith(f'bytes {start}-')):
                            raise ValueError('Source resume not honored')
                        with part.open('ab' if start else 'xb') as stream:
                            for chunk in response.iter_content(2**20):
                                if start+len(chunk) > size:
                                    raise ValueError('Oversized download response')
                                stream.write(chunk)
                                start += len(chunk)
                    if start == size:
                        break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(5*(attempt+1))
            if part.stat().st_size != size or md5(part) != checksum:
                raise ValueError('Downloaded source checksum mismatch')
            part.replace(path)
    if path.stat().st_size != size or md5(path) != checksum:
        raise ValueError(f'Existing source checksum mismatch: {year}')
    print(f'[GPP SOURCE VERIFIED] {year}', flush=True)
    return path


def crop_support(crop, year, rel):
    base = ROOT / 'benchmark/cache/multimodal_main' / crop
    world = ROOT / 'benchmark/cache/biid_world_model' / crop
    identity = np.load(world / 'source_indices.npy', mmap_mode='r')
    years = np.load(base / 'year.npy', mmap_mode='r')[identity]
    selected = np.flatnonzero(years == year)
    indices = identity[selected]
    rows = np.load(base / 'row.npy', mmap_mode='r')[indices]
    cols = np.load(base / 'col.npy', mmap_mode='r')[indices]
    valid = np.load(world / 'relative_valid.npy', mmap_mode='r')[selected] > 0
    values = rel[:, rows, cols].T
    count = int(valid.sum())
    finite = int((np.isfinite(values) & valid).sum())
    return dict(samples=len(selected), original_valid_slots=count, finite_slots=finite,
                finite_fraction=finite/count if count else None,
                zero_valid_slots=int(((values == 0) & valid).sum()),
                target_values_accessed=False)


def process(year, path, audit_cohort=True):
    AUDIT.mkdir(parents=True, exist_ok=True)
    marker = AUDIT / f'{year}.json'
    if marker.exists():
        audit = json.loads(marker.read_text())
        if audit['archive_sha256'] != sha256(path):
            raise ValueError('Processed source changed')
        for item in audit['outputs']:
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('Processed output changed')
        print(f'[GPP OUTPUTS REVERIFIED] {year}', flush=True)
        return audit
    monthly = np.full((12, 360, 720), np.nan, np.float32)
    support = np.empty_like(monthly)
    seen, checks = set(), []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir() and member.filename == f'{year}/' and member.file_size == 0:
                continue
            month = source_month(member.filename, year)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if month in seen or mode not in (0, stat.S_IFREG) or member.flag_bits & 1 or not 0 < member.file_size <= 64*2**20:
                raise ValueError('Unsafe or duplicate source member')
            seen.add(month)
            with MemoryFile(archive.read(member)) as memory, memory.open() as source:
                if source.shape != (3600, 7200) or source.count != 1 or source.dtypes != ('uint16',) or source.crs.to_epsg() != 4326:
                    raise ValueError('Unexpected source raster')
                np.testing.assert_allclose(tuple(source.transform)[:6], (.05, 0, -180, 0, -.05, 90), rtol=0, atol=1e-10)
                codes = source.read(1)
            monthly[month-1], support[month-1] = aggregate(codes)
            check = verify_aggregation(codes, monthly[month-1], support[month-1], month)
            checks.append(dict(month=month, **check))
            print(f'[GPP AGGREGATED AND REBUILT] {year}-{month:02d}', flush=True)
    if seen != set(range(1, 13)):
        raise ValueError('Missing calendar month')
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)], dtype=np.float32)
    daily = monthly/days[:, None, None]
    np.testing.assert_allclose(daily*days[:, None, None], monthly, rtol=2e-7, atol=3e-5)
    outputs = []
    for name, expected in [('lat', np.arange(360)*.5-89.75), ('lon', np.arange(720)*.5-179.75)]:
        coord = np.load(GROWING / f'{name}.npy')
        np.testing.assert_allclose(coord, expected, rtol=0, atol=1e-8)
        outputs.append(frozen_array(OUT / f'{name}.npy', coord))
    quantities = {'gpp_monthly_total': monthly, 'gpp_daily_rate': daily, 'valid_area_fraction': support}
    for name, values in quantities.items():
        outputs.append(frozen_array(OUT / 'monthly_0p5' / f'{name}_{year}.npy', values))
    crops = {}
    for crop in CROPS:
        src_path = GROWING / crop / 'mirca' / str(nearest_mirca_year(year)) / 'src_rel.npy'
        src = np.load(src_path)
        crop_record = dict(source_mapping=str(src_path), source_mapping_sha256=sha256(src_path),
                           mapped_slots=int((src != 255).sum()))
        for name, values in quantities.items():
            rel = mapping_checked(values, src)
            outputs.append(frozen_array(OUT / 'crops' / crop / name / f'{name}_rel_{year}.npy', rel))
            if name == 'gpp_daily_rate' and audit_cohort:
                crop_record['original_benchmark_support'] = crop_support(crop, year, rel)
        crops[crop] = crop_record
    audit = dict(year=year, source_format_passed=True, phenology_aligned=True,
        archive_path=str(path), archive_sha256=sha256(path), outputs=outputs,
        independent_rebuild=checks, month_days=days.astype(int).tolist(), crops=crops,
        code_sha256=sha256(Path(__file__)), mapping_code_sha256=sha256(ROOT / 'scripts/process_glass_lai_avhrr_to_growing_season.py'),
        units=dict(gpp_monthly_total='gC m-2 month-1', gpp_daily_rate='gC m-2 day-1',
                   valid_area_fraction='provider-valid area / full grid area, not observation quality'),
        orientation='latitude south-to-north; longitude west-to-east; same project pixel centers',
        ecosystem_model_derived=True, independently_observed_remote_sensing=False,
        models_fitted=0, yield_targets_read=False, finished_unix=time.time())
    audit['cohort_support_audited'] = audit_cohort
    if year == 1982 and audit_cohort:
        audit['sample_support_passed'] = all(
            (item['original_benchmark_support']['finite_fraction'] or 0) >= .90 for item in crops.values())
    atomic_json(marker, audit)
    print(f'[GPP YEAR COMPLETE] {year} {json.dumps(crops)}', flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    record = source_record()
    sample = process(1982, fetch(1982, record[1982]))
    if not sample['sample_support_passed']:
        raise ValueError('Registered 1982 crop support threshold failed; no full download')
    if not args.full:
        print('[GPP SAMPLE ALIGNMENT PASSED] Full-period ingestion not requested.', flush=True)
        return
    for year in YEARS[1:]:
        process(year, fetch(year, record[year]))
    atomic_json(OUT / 'manifest.json', dict(years=list(YEARS), complete=True, models_fitted=0,
        audits=[str(AUDIT / f'{y}.json') for y in YEARS], dataset_record='https://zenodo.org/records/14350035',
        data_dependency='rEC-LUE model with GLASS LAI v6, meteorology and CO2; not independent observations.',
        license='CC-BY-4.0', source_metadata_sha256=sha256(RAW / 'record.json'),
        units='Monthly totals and daily mean rates stored separately.',
        current_model_inputs_changed=False))
    print('[MONTHLY GPP FULL PERIOD COMPLETE] 35 years; no model fitted.', flush=True)


if __name__ == '__main__':
    main()
