"""Conservative administrative-calendar support audit, without model refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import shapefile
from multimodal_baseline import nearest_mirca_year, save_json
from review_revision_data import ROOT, CROPS, load_shared, sha256
from summarize_review_revision import ensemble, rmse
from audit_spatial_block_bootstrap import block_bootstrap

EXTERNAL = ROOT/'Data/external/MIRCA_OS_admin_v2_review20260906'
OUTPUT = ROOT/'visualize/paper_experiments/review_revision_v4'
DERIVED = ROOT/'Data/processed/mirca_admin_calendar_audit_v4'
CROP_NAMES = {'Maize': 'maize', 'Soybeans': 'soybean', 'Soyabeans': 'soybean',
              'Rice1': 'rice', 'Rice2': 'rice', 'Rice3': 'rice',
              'Wheat1': 'wheat', 'Wheat2': 'wheat'}
AREA_NAMES = dict(maize='Maize', soybean='Soybeans', rice='Rice', wheat='Wheat')


def read_calendar(year, system):
    path = ROOT/f'Data/MIRCA-OS/Crop Calendar/MIRCA-OS_{year}_{system}.csv'
    frame = pd.read_csv(path, encoding='cp1252')
    frame['crop'] = frame.Crop.str.strip().map(CROP_NAMES)
    frame['unit_code'] = pd.to_numeric(frame.unit_code, errors='raise').astype(np.int64)
    return frame, path


def unique_window(group):
    """Multiple positive records remain unknown, even if their dates coincide."""
    if len(group) != 1:
        return 0
    p, m = int(group.Planting_Month.iloc[0]), int(group.Maturity_Month.iloc[0])
    if not 1 <= p < m <= 12:
        return 0
    return p*16+m


def apply_unit_burn(view, occupied, code):
    conflict = occupied & (view != 0) & (view != code)
    view[occupied & (view == 0)] = code
    view[conflict] = -1


def sum_six(values):
    return values.reshape(360, 6, 720, 6).sum((1, 3))


def read_native_area(year, crop, system):
    path = ROOT/f'Data/MIRCA-OS/Annual Harvested Area Grids/Annual Harvested Area Grids/{year}/5-arcminute/MIRCA-OS_{AREA_NAMES[crop]}_{year}_{system}.tif'
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326
        t = src.transform
        np.testing.assert_allclose((t.a,t.b,t.d,t.e), (1/12,0,0,-1/12), atol=1e-9)
        offsets = np.array([(90-t.f)*12, (t.c+180)*12])
        top, left = np.rint(offsets).astype(int)
        np.testing.assert_allclose(offsets, (top,left), atol=1e-6)
        assert 0 <= top < top+src.height <= 2160 and 0 <= left < left+src.width <= 4320
        original = src.read(1, masked=True).filled(0).astype(np.float64)
        original[~np.isfinite(original) | (original < 0)] = 0
        area = np.zeros((2160,4320))
        area[top:top+src.height, left:left+src.width] = original
    return area[::-1], {'path':str(path.relative_to(ROOT)), 'sha256':sha256(path),
                        'native_origin_row_col':[int(top),int(left)], 'native_shape':list(original.shape)}


def rasterize_units(year):
    source = next(EXTERNAL.rglob(f'MIRCAOS_{year}_*.shp'))
    dest = DERIVED/f'unit_codes_{year}_5arcmin_south_to_north.npy'
    meta = dest.with_suffix('.json')
    signature = {p.suffix: sha256(p) for p in (source, source.with_suffix('.dbf'), source.with_suffix('.prj'))}
    if dest.exists() and meta.exists():
        assert json.loads(meta.read_text())['source_sha256'] == signature
        return np.load(dest, mmap_mode='r')
    reference = ROOT/f'Data/MIRCA-OS/Annual Harvested Area Grids/Annual Harvested Area Grids/{year}/5-arcminute/MIRCA-OS_Maize_{year}_ir.tif'
    with rasterio.open(reference) as src:
        transform = src.transform
        assert src.shape == (2160, 4320) and src.crs.to_epsg() == 4326
        np.testing.assert_allclose(tuple(transform)[:6], (1/12, 0, -180, 0, -1/12, 90), atol=1e-9)
    crs = rasterio.crs.CRS.from_wkt(source.with_suffix('.prj').read_text())
    assert crs.to_epsg() == 4326
    units = np.zeros((2160, 4320), dtype=np.int32)
    reader = shapefile.Reader(str(source))
    # Rasterize one bounded polygon at a time instead of materializing all vertices.
    for i, record in enumerate(reader.iterShapeRecords(fields=['unit_code'])):
        code = int(record.record[0])
        xmin, ymin, xmax, ymax = record.shape.bbox
        left = max(0, int(np.floor((xmin+180)*12))-1)
        right = min(4320, int(np.ceil((xmax+180)*12))+1)
        top = max(0, int(np.floor((90-ymax)*12))-1)
        bottom = min(2160, int(np.ceil((90-ymin)*12))+1)
        if right > left and bottom > top:
            mask = rasterize([(record.shape.__geo_interface__, 1)],
                             out_shape=(bottom-top, right-left),
                             transform=from_origin(-180+left/12, 90-top/12, 1/12, 1/12),
                             dtype='uint8', all_touched=False).astype(bool)
            apply_unit_burn(units[top:bottom, left:right], mask, code)
        if i % 500 == 0:
            print(f'[RASTER] {year}: {i}/{len(reader)}', flush=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    np.save(dest, units[::-1])
    save_json({'source_sha256': signature, 'reference_tiff': str(reference.relative_to(ROOT)),
               'unit_nodata': 0, 'overlapping_distinct_units': -1,
               'rasterization': '5-arcminute pixel centers; south-to-north; not subpixel boundary fractions'}, meta)
    return np.load(dest, mmap_mode='r')


def join_metadata():
    records, missing, sources = [], [], []
    for year in (2000, 2005, 2010, 2015):
        shape = next(EXTERNAL.rglob(f'MIRCAOS_{year}_*.shp'))
        reader = shapefile.Reader(str(shape))
        ids = {int(r[0]) for r in reader.iterRecords(fields=['unit_code'])}
        for system in ('ir', 'rf'):
            frame, path = read_calendar(year, system)
            missing.extend(frame.loc[~frame.unit_code.isin(ids)].assign(year=year, system=system).to_dict('records'))
            positive = frame[frame.crop.notna() & (frame.Growing_area > 0)]
            for crop, group in positive.groupby('crop'):
                absent = ~group.unit_code.isin(ids)
                records.append(dict(year=year, system=system, crop=crop, records=len(group),
                    matched_records=int((~absent).sum()), missing_units=group.loc[absent, 'unit_code'].nunique(),
                    unmatched_calendar_area_percent=float(100*group.loc[absent, 'Growing_area'].sum()/group.Growing_area.sum())))
            sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    pd.DataFrame(records).to_csv(OUTPUT/'calendar_admin_id_join.csv', index=False)
    pd.DataFrame(missing).to_csv(OUTPUT/'calendar_admin_unmatched_records.csv', index=False)
    return sources


def coarse_calendar(year, crop, units):
    total, known = np.zeros((360,720)), np.zeros((360,720))
    lo, hi = np.full((360,720), 1000), np.zeros((360,720), dtype=int)
    area_paths = []
    for system in ('ir','rf'):
        frame, _ = read_calendar(year, system)
        d = frame[(frame.crop == crop) & (frame.Growing_area > 0)]
        mapping = {int(uid): unique_window(g) for uid, g in d.groupby('unit_code')}
        unique, inverse = np.unique(units, return_inverse=True)
        dates = np.array([mapping.get(int(u), 0) for u in unique], dtype=np.int16)[inverse].reshape(units.shape)
        area, source = read_native_area(year, crop, system)
        area_paths.append(source)
        assert area.shape == units.shape
        area[~np.isfinite(area) | (area <= 0)] = 0
        total += sum_six(area)
        known += sum_six(np.where(dates > 0, area, 0))
        supported = (dates > 0) & (area > 0)
        dates_lo = np.where(supported, dates, 1000).reshape(360,6,720,6).min((1,3))
        dates_hi = np.where(supported, dates, 0).reshape(360,6,720,6).max((1,3))
        lo, hi = np.minimum(lo, dates_lo), np.maximum(hi, dates_hi)
    fraction = np.divide(known, total, out=np.zeros_like(known), where=total > 0)
    selected = (fraction >= .99) & (lo == hi) & (hi > 0)
    np.savez_compressed(DERIVED/f'{crop}_{year}_calendar_support.npz', selected=selected,
                        known_area_fraction=fraction, calendar_code=hi, total_harvested_area=total)
    return selected, hi, area_paths


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    sources = join_metadata()
    all_records, metrics = [], []
    # All current main evaluation years map to the 2015 snapshot.
    assert {nearest_mirca_year(y) for y in (2013,2014,2015,2016)} == {2015}
    units = rasterize_units(2015)
    for crop in CROPS:
        selected, dates, area_sources = coarse_calendar(2015, crop, units)
        sources.extend(area_sources)
        arrays, _ = load_shared(crop, 42)
        a = arrays['test']; row, col = a['row'], a['col']
        code = dates[row, col]; p, m = code//16, code % 16
        mapping = a['source_month']
        expected = p[:,None]-1+np.arange(12)
        expected = np.where(expected < m[:,None], expected, 255)
        exact = np.all(mapping == expected, axis=1)
        calendar = selected[row, col]
        keep = calendar & exact
        np.savez_compressed(DERIVED/f'{crop}_current_evaluation_subset.npz',
                            source_indices=a['source_indices'], keep=keep,
                            calendar_agreement=calendar, packed_months_match=exact)
        all_records.append(dict(crop=crop, samples=len(keep), calendar_agreement_samples=int(calendar.sum()),
            exact_window_samples=int(keep.sum()), retained_percent=float(100*keep.mean())))
        for pathway in ('state_only','climate'):
            forecast = ensemble(crop, f'fusion__predicted__{pathway}__coverage__bce', 'test')
            previous = ensemble(crop, f'fusion__previous__{pathway}__coverage__bce', 'test')
            np.testing.assert_array_equal(forecast['source_indices'], a['source_indices'])
            np.testing.assert_array_equal(forecast['source_indices'], previous['source_indices'])
            np.testing.assert_array_equal(forecast['target'], previous['target'])
            for scope, k in (('all',np.ones(len(keep),bool)), ('calendar_consistent',keep)):
                target, f, b = forecast['target'][k].astype(float), forecast['prediction'][k], previous['prediction'][k]
                for degrees in (10,20):
                    blocks = np.unique((row[k]//(degrees*2))*(360//degrees)+col[k]//(degrees*2)).size
                    if len(target) and blocks >= 8:
                        gain, low, high, blocks = block_bootstrap(target,b,f,row[k],col[k],degrees,20260906)
                    else:
                        gain = low = high = float('nan')
                    metrics.append(dict(crop=crop, pathway=pathway, subset=scope, samples=len(target),
                        predicted_rmse=rmse(target,f) if len(target) else None,
                        previous_rmse=rmse(target,b) if len(target) else None,
                        block_degrees=degrees, spatial_blocks=blocks, gain_percent=gain, ci_low=low, ci_high=high))
    pd.DataFrame(all_records).to_csv(OUTPUT/'calendar_consistent_subset_counts.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT/'calendar_consistent_subset_metrics.csv', index=False)
    save_json({'sources':sources, 'boundary_release': 'MIRCA-OS v2 2026; matching historical-year IDs, not 2020 calendars',
        'criteria_frozen_before_scoring': 'one positive calendar record per unit/crop/system; 1<=plant<maturity<=12; >=99% native annual crop area with known non-cross-year dates; all supported dates equal within coarse cell and both systems; exact match to current packed active months',
        'unknowns': 'missing IDs, overlapping units, multiple records, plant==maturity, conflicting periods, and unmatched packed months excluded',
        'scope': 'retained three-seed current-main predictions, no subset-specific retraining or model selection',
        'limitations': 'native-pixel-center admin assignment, administrative calendar not field-level ground truth, no cross-year reconstruction or season-specific yield labels'}, OUTPUT/'calendar_consistent_subset_manifest.json')
    print(pd.DataFrame(all_records).to_string(index=False), flush=True)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
