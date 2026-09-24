"""Conservative crop-area weighting before spatial and temporal LAI aggregation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

from audit_calendar_admin_join import AREA_NAMES
from process_glass_lai_avhrr_to_growing_season import (
    read_lai_hdf, parse_year_doy, period_month_overlaps, reorder_relative_months,
    FILL_VALUE, VALID_MIN, VALID_MAX, SCALE_FACTOR,
)
from multimodal_baseline import nearest_mirca_year
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'Data/processed/glass_lai_cropweighted_v1'
RAW = ROOT / 'Data/GLASS_LAI_AVHRR_005D'
CALENDAR = ROOT / 'Data/processed/crop_yield_growing_season'


def native_area_path(snapshot, crop, system):
    folder = ROOT / f'Data/MIRCA-OS/Annual Harvested Area Grids/Annual Harvested Area Grids/{snapshot}/5-arcminute'
    path = folder / f'MIRCA-OS_{AREA_NAMES[crop]}_{snapshot}_{system}.tif'
    if not path.exists() and (snapshot, crop, system) == (2000, 'soybean', 'ir'):
        path = folder / 'MIRCA-OS_Soybeans2000_ir.tif'
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_native_area(snapshot, crop, system):
    path = native_area_path(snapshot, crop, system)
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 4326:
            raise ValueError(f'Unexpected area CRS: {path}')
        t = src.transform
        np.testing.assert_allclose((t.a, t.b, t.d, t.e), (1/12, 0, 0, -1/12), atol=1e-9)
        offsets = np.array([(90-t.f)*12, (t.c+180)*12])
        top, left = np.rint(offsets).astype(int)
        np.testing.assert_allclose(offsets, (top, left), atol=1e-6)
        if not (0 <= top < top+src.height <= 2160 and 0 <= left < left+src.width <= 4320):
            raise ValueError(f'Area raster outside the global grid: {path}')
        original = src.read(1, masked=True).filled(0).astype(np.float64)
        original[~np.isfinite(original) | (original < 0)] = 0
        area = np.zeros((2160, 4320))
        area[top:top+src.height, left:left+src.width] = original
    return area[::-1], dict(path=str(path.relative_to(ROOT)), sha256=sha256(path),
                           native_origin_row_col=[int(top), int(left)], native_shape=list(original.shape))


def remap_mass(area, source_lat_edges, source_lon_edges, dest_lat_edges, dest_lon_edges):
    """Bilinear interpolation of cumulative mass gives conservative cell integrals."""
    area = np.asarray(area, dtype=np.float64)
    if np.any(area < 0) or not np.isfinite(area).all():
        raise ValueError('Invalid source harvested area')
    cumulative = np.zeros((area.shape[0] + 1, area.shape[1] + 1), dtype=np.float64)
    cumulative[1:, 1:] = area.cumsum(0).cumsum(1)
    source_y = np.sin(np.deg2rad(source_lat_edges))
    dest_y = np.sin(np.deg2rad(dest_lat_edges))
    interpolator = RegularGridInterpolator((source_y, source_lon_edges), cumulative,
                                            method='linear', bounds_error=True)
    output = np.zeros((len(dest_y) - 1, len(dest_lon_edges) - 1), dtype=np.float64)
    for start in range(0, len(dest_y) - 1, 96):
        stop = min(start + 96, len(dest_y) - 1)
        yy, xx = np.meshgrid(dest_y[start:stop + 1], dest_lon_edges, indexing='ij')
        values = interpolator(np.stack((yy, xx), -1))
        output[start:stop] = np.diff(np.diff(values, axis=0), axis=1)
    tolerance = max(float(area.sum()) * 1e-13, 1e-8)
    if output.min() < -tolerance:
        raise ValueError('Conservative interpolation created substantive negative area')
    output = np.maximum(output, 0)
    np.testing.assert_allclose(output.sum(), area.sum(), rtol=1e-7, atol=1e-6)
    return output


def weight_path(crop, snapshot):
    return OUT / 'weights_005' / str(snapshot) / f'{crop}.npy'


def prepare_weights(crop, snapshot):
    destination = weight_path(crop, snapshot); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.with_suffix('.json').exists():
            return
        ir, ir_meta = read_native_area(snapshot, crop, 'ir')
        rf, rf_meta = read_native_area(snapshot, crop, 'rf')
        area = ir + rf
        fine = remap_mass(area, np.linspace(-90, 90, 2161), np.linspace(-180, 180, 4321),
                         np.linspace(-90, 90, 3601), np.linspace(-180, 180, 7201))
        coarse = area.reshape(360, 6, 720, 6).sum((1, 3))
        # Eliminate roundoff-only support outside cropped coarse cells.
        fine *= np.repeat(np.repeat(coarse > 0, 10, axis=0), 10, axis=1)
        fine = fine.astype(np.float32)
        reproduced = fine.reshape(360, 10, 720, 10).sum((1, 3), dtype=np.float64)
        np.testing.assert_allclose(reproduced, coarse, rtol=1e-6, atol=1e-3)
        np.save(destination, fine)
        atomic_json(destination.with_suffix('.json'), dict(crop=crop, snapshot=snapshot,
            source=[ir_meta, rf_meta], source_area_ha=float(area.sum()), output_area_ha=float(fine.sum(dtype=float)),
            maximum_coarse_area_error_ha=float(np.max(np.abs(reproduced - coarse))),
            shape=list(fine.shape), latitude='south to north', longitude='-180 to 180',
            method='Conservative cell-integrated harvested-area remap via cumulative mass interpolation in sin(latitude), longitude; uniform harvested-area density inside each native 5-arcminute cell.',
            code_sha256=sha256(Path(__file__)), sha256=sha256(destination)))
        print(f'[CROP WEIGHT] {crop} {snapshot} max coarse error {np.max(np.abs(reproduced-coarse)):.6g} ha', flush=True)


def weighted_sum(raw_south_to_north, weight):
    valid = (raw_south_to_north != FILL_VALUE) & (raw_south_to_north >= VALID_MIN) & (raw_south_to_north <= VALID_MAX)
    mass = np.where(valid, weight, 0).astype(np.float32)
    values = np.where(valid, raw_south_to_north, 0).astype(np.float32) * SCALE_FACTOR
    rows, cols = raw_south_to_north.shape
    shape = (rows // 10, 10, cols // 10, 10)
    numerator = (values * mass).reshape(shape).sum((1, 3), dtype=np.float64)
    denominator = mass.reshape(shape).sum((1, 3), dtype=np.float64)
    return numerator, denominator


def month_path(crop, snapshot, year, quality=False):
    name = 'valid_area_fraction' if quality else 'lai'
    return OUT / 'snapshots' / str(snapshot) / crop / f'{name}_monthly_{year}.npy'


def process_year(year):
    started = time.monotonic()
    marker = OUT / 'year_manifests' / f'{year}.json'; marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return year
        snapshots = sorted({nearest_mirca_year(year), nearest_mirca_year(min(year + 1, 2016))})
        keys = [(crop, snapshot) for crop in CROPS for snapshot in snapshots]
        weights = {key: np.load(weight_path(*key), mmap_mode='r') for key in keys}
        totals = {key: weights[key].reshape(360, 10, 720, 10).sum((1, 3), dtype=float) for key in keys}
        sums = {key: np.zeros((12, 360, 720), dtype=np.float64) for key in keys}
        masses = {key: np.zeros((12, 360, 720), dtype=np.float64) for key in keys}
        files = sorted((RAW / str(year)).glob('*.hdf'))
        doys = [parse_year_doy(p)[1] for p in files]
        if doys != list(range(1, 366, 8)):
            raise ValueError(f'{year}: expected all 46 distinct 8-day composites')
        for path in files:
            file_year, doy = parse_year_doy(path)
            if file_year != year:
                raise ValueError('Wrong HDF year')
            raw = read_lai_hdf(path)[::-1]
            for key in keys:
                numerator, denominator = weighted_sum(raw, weights[key])
                for month, days in period_month_overlaps(year, doy):
                    sums[key][month] += numerator * days
                    masses[key][month] += denominator * days
        import calendar
        days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
        outputs = {}
        for key in keys:
            path = month_path(*key, year); path.parent.mkdir(parents=True, exist_ok=True)
            monthly = np.divide(sums[key], masses[key], out=np.full_like(sums[key], np.nan), where=masses[key] > 1e-8).astype(np.float32)
            possible = totals[key][None] * days[:, None, None]
            fraction = np.divide(masses[key], possible, out=np.zeros_like(masses[key]), where=possible > 1e-8)
            if fraction.min() < -1e-6 or fraction.max() > 1 + 1e-5:
                raise ValueError('Invalid crop-area-time coverage')
            np.save(path, monthly)
            quality_path = month_path(*key, year, quality=True)
            np.save(quality_path, np.clip(fraction, 0, 1).astype(np.float32))
            outputs[str(path)] = sha256(path); outputs[str(quality_path)] = sha256(quality_path)
        atomic_json(marker, dict(year=year, snapshots=snapshots, sources={str(p): sha256(p) for p in files},
                    outputs=outputs, seconds=time.monotonic() - started, code_sha256=sha256(Path(__file__)),
                    weights={str(weight_path(*k)): sha256(weight_path(*k)) for k in keys}))
        print(f'[CROP LAI] {year} snapshots={snapshots} seconds={time.monotonic()-started:.1f}', flush=True)
        return year


def align(years):
    output = {}
    for year in years:
        snapshot = nearest_mirca_year(year)
        for crop in CROPS:
            src_path = CALENDAR / crop / 'mirca' / str(snapshot) / 'src_rel.npy'
            src = np.load(src_path)
            for previous in (False, True):
                source_year = year - int(previous)
                source = month_path(crop, snapshot, source_year)
                if not source.exists():
                    if previous and year == min(years):
                        continue
                    raise FileNotFoundError(source)
                kind = 'previous_lai_rel' if previous else 'lai_rel'
                folder = OUT / 'crops' / crop / kind; folder.mkdir(parents=True, exist_ok=True)
                destination = folder / f'{kind}_{year}.npy'
                np.save(destination, reorder_relative_months(np.load(source), src))
                q_source = month_path(crop, snapshot, source_year, quality=True)
                q_destination = folder / f'valid_area_fraction_{year}.npy'
                np.save(q_destination, reorder_relative_months(np.load(q_source), src))
                output[str(destination)] = dict(sha256=sha256(destination), monthly_source=str(source),
                    monthly_sha256=sha256(source), calendar=str(src_path), calendar_sha256=sha256(src_path),
                    quality=str(q_destination), quality_sha256=sha256(q_destination))
    for name in ('lat', 'lon'):
        np.save(OUT / f'{name}.npy', np.load(CALENDAR / f'{name}.npy'))
    atomic_json(OUT / 'alignment_manifest.json', dict(years=years, outputs=output,
        alignment='Unchanged MIRCA packed natural-month indices. Previous-year LAI uses target-year area snapshot and target-year source-month mapping. Cross-year season reconstruction is not claimed.',
        interpretation='Harvested-area-weighted GLASS LAI, not a retrieval of pure crop LAI inside mixed 0.05-degree pixels.',
        code_sha256=sha256(Path(__file__))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--start', type=int, default=1981)
    p.add_argument('--end', type=int, default=2016); p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.start > args.end or args.start < 1981 or args.end > 2016 or not 1 <= args.workers <= 2:
        p.error('Use 1981--2016 and at most two CPU workers')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshots = OUT / 'code_snapshots'; snapshots.mkdir(exist_ok=True)
        shutil.copy2(__file__, snapshots / f'{sha256(Path(__file__))}.py')
        for snapshot in (2000, 2005, 2010, 2015):
            for crop in CROPS:
                prepare_weights(crop, snapshot)
        years = list(range(args.start, args.end + 1))
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[variable] = '1'
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(process_year, years))
        align(years)
        atomic_json(OUT / 'complete.json', dict(years=years, crops=list(CROPS), aligned=True,
                    original_products_modified=False, weighting='Native harvested area before spatial aggregation'))


if __name__ == '__main__':
    main()
