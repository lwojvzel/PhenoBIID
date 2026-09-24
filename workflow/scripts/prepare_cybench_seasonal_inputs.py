"""Build label-free crop-season records from verified regional monthly products."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cybench_seasonal_inputs import PRODUCTS, SLOTS, issue_view, season_calendar
from prepare_cybench_regional_monthly import sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'Data/processed/cybench_regional_monthly_v1'
LABEL_SOURCE = ROOT/'Data/external/CYBench/full_v1_10'
AUDIT = ROOT/'benchmark/results/cybench_inseason_inputs_v1'
OUT = ROOT/'Data/processed/cybench_seasonal_v1'
YEARS = list(range(1983, 2017))


def atomic_json(path, value):
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def atomic_arrays(path, value):
    tmp = path.with_suffix('.part')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **value)
    tmp.replace(path)


def crop_regions(crop, regions, sources):
    allowed = json.loads((AUDIT/'input_audit.json').read_text())['sources']
    rows = []
    for country in ('DE', 'FR', 'PL'):
        tables = []
        for name, columns in (('crop_calendar', ['adm_id', 'sos', 'eos']),
                              ('location', ['adm_id', 'latitude', 'longitude'])):
            path = LABEL_SOURCE/crop/country/f'{name}_{crop}_{country}.csv'
            digest = sha256(path)
            if digest != allowed[str(path)]:
                raise ValueError('Provider metadata changed after audit')
            sources[str(path)] = digest
            table = pd.read_csv(path, usecols=columns)
            if table.adm_id.duplicated().any():
                raise ValueError('Nonunique provider region')
            tables.append(table)
        record = regions[regions.country.eq(country)].merge(tables[0], on='adm_id', how='left', validate='one_to_one')
        record = record.merge(tables[1], on='adm_id', how='left', validate='one_to_one')
        rows.append(record)
    frame = pd.concat(rows, ignore_index=True)
    ok = (np.isfinite(frame[['sos', 'eos', 'latitude', 'longitude']]).all(axis=1)
          & frame.sos.between(0, 366) & frame.eos.between(0, 366)
          & frame.latitude.between(-90, 90) & frame.longitude.between(-180, 180))
    return frame[ok].reset_index(drop=True), frame[~ok].copy()


def take_months(monthly, ids, region_index, field):
    suffix = (13,) if field.startswith('weather') else ()
    result = np.full(ids.shape+suffix, np.nan, dtype=np.float32)
    for y in np.unique(ids//100):
        if y not in monthly:
            continue
        r, k = np.where(ids//100 == y)
        months = ids[r, k] % 100
        if not np.all((months >= 1) & (months <= 12)):
            raise ValueError('Invalid source month identifier')
        result[r, k] = monthly[y][field][region_index[r], months-1]
    return result


def pack_year(year, regions, monthly):
    calendars = [season_calendar(year, row.sos, row.eos) for row in regions.itertuples()]
    known = {key:np.stack([row[key] for row in calendars]) for key in calendars[0]}
    known['region_index'] = regions.region_index.to_numpy(np.int32)
    known['year'] = np.full(len(regions), year, dtype=np.int16)
    known['context'] = regions[['latitude', 'longitude', 'sos', 'eos']].to_numpy(np.float32)
    ids, previous = known['source_month_id'], known['previous_source_month_id']
    index = known['region_index']
    known['weather'] = take_months(monthly, ids, index, 'weather')
    known['weather_finite_grid_area_fraction'] = np.nan_to_num(
        take_months(monthly, ids, index, 'weather_finite_grid_area_fraction'), nan=0)
    known['previous_vegetation'] = np.stack([take_months(monthly, previous, index, p) for p in PRODUCTS], axis=-1)
    known['previous_vegetation_support'] = np.nan_to_num(np.stack([
        take_months(monthly, previous, index, p+'_observed_area_fraction') for p in PRODUCTS], axis=-1), nan=0)
    targets = dict(vegetation=np.stack([take_months(monthly, ids, index, p) for p in PRODUCTS], axis=-1),
        vegetation_support=np.nan_to_num(np.stack([
            take_months(monthly, ids, index, p+'_observed_area_fraction') for p in PRODUCTS], axis=-1), nan=0))
    known['previous_valid'] = np.isfinite(known['previous_vegetation']) & (known['previous_vegetation_support'] > 0)
    return known, targets


def main():
    monthly_manifest = json.loads((SOURCE/'manifest.json').read_text())
    verification_path = AUDIT/'regional_verification.json'
    verification = json.loads(verification_path.read_text())
    if not verification['passed'] or verification['manifest_sha256'] != sha256(SOURCE/'manifest.json'):
        raise ValueError('Independent monthly verification missing or stale')
    registration = json.loads((SOURCE/'registration.json').read_text())
    if sha256(SOURCE/'registration.json') != monthly_manifest['registration_sha256']:
        raise ValueError('Monthly registration changed')
    if sha256(SOURCE/'regions.csv') != monthly_manifest['regions_sha256']:
        raise ValueError('Monthly region order changed')
    monthly = {}
    sources = {str(SOURCE/'manifest.json'):sha256(SOURCE/'manifest.json'),
               str(verification_path):sha256(verification_path),
               str(AUDIT/'input_audit.json'):sha256(AUDIT/'input_audit.json')}
    for rel in ('Data/external/CYBench/code/cybench/datasets/alignment.py',
                'Data/external/CYBench/code/README.md'):
        sources[str(ROOT/rel)] = sha256(ROOT/rel)
    for y in monthly_manifest['years']:
        marker_path = SOURCE/f'calendar_months/{y}.json'
        if sha256(marker_path) != monthly_manifest['year_markers'][str(y)]:
            raise ValueError('Monthly year marker changed')
        marker = json.loads(marker_path.read_text())
        path = SOURCE/f'calendar_months/{y}.npz'
        if sha256(path) != marker['output_sha256']:
            raise ValueError('Monthly arrays changed')
        with np.load(path, allow_pickle=False) as data:
            monthly[y] = {k:data[k] for k in data.files}
    base_regions = pd.read_csv(SOURCE/'regions.csv')
    region_sets = {crop:crop_regions(crop, base_regions, sources) for crop in ('maize', 'wheat')}
    config = dict(years=YEARS, products=PRODUCTS, slots=SLOTS,
        weather_order=registration['weather_order'], context_order=['latitude', 'longitude', 'sos', 'eos'],
        code_sha256={n:sha256(ROOT/'scripts'/n) for n in (
            'prepare_cybench_seasonal_inputs.py', 'cybench_seasonal_inputs.py', 'prepare_cybench_regional_monthly.py')},
        source_sha256=sources, calendar='Supplied CY-Bench WorldCereal active-growth SOS/EOS; truncation and cross-year shift follow pinned implementation',
        monthly_support='Full calendar months intersecting supplied SOS/EOS; partial boundary months not reconstructed',
        issue='End of last visible full month; hide ceil(percent*active_months/100) terminal months',
        previous='Same source calendar months one year earlier, not a crop-year label lookup',
        labels_loaded=False, models_fitted=0, normalization_fitted=False, interpolation=False,
        selection='Metadata-ready audited regions, not a label-availability or skill-selected cohort',
        weather='Given full seasonal weather; not official truncated-weather CY-Bench protocol',
        support='Future product-specific observation support hidden with remote values',
        percent_zero='Full observed monthly-support reference, may be later than supplied EOS; not a forecast')
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT/'registration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Registered season construction changed')
    atomic_json(config_path, config)
    records, summaries, output_hashes = [], [], {}
    for crop, (regions, excluded) in region_sets.items():
        folder = OUT/crop
        folder.mkdir(exist_ok=True)
        for name, data in (('regions', regions), ('excluded_regions', excluded)):
            path = folder/f'{name}.csv'
            data.to_csv(path, index=False)
            output_hashes[str(path.relative_to(OUT))] = sha256(path)
        for year in YEARS:
            known, targets = pack_year(year, regions, monthly)
            for kind, data in (('known', known), ('state_targets', targets)):
                path = folder/f'{year}_{kind}.npz'
                atomic_arrays(path, data)
                output_hashes[str(path.relative_to(OUT))] = sha256(path)
            active = known['active_mask']
            missing_prior = active[..., None] & ~known['previous_valid']
            records.append(dict(crop=crop, year=year, regions=len(regions),
                active_slots=int(active.sum()), cross_year_regions=int(known['cross_year'].sum()),
                missing_prior_product_slots=int(missing_prior.sum())))
            for percent in range(0, 101, 10):
                view = issue_view(known, targets, percent)
                summaries.append(dict(crop=crop, year=year, percent=percent,
                    regions=len(regions), mean_visible_slots=float(view['visible_mask'].sum(1).mean()),
                    mean_hidden_fraction=float((view['hidden_mask'].sum(1)/active.sum(1)).mean()),
                    minimum_days_to_supplied_eos=int(view['days_to_supplied_eos'].min()),
                    median_days_to_supplied_eos=float(np.median(view['days_to_supplied_eos'])),
                    maximum_days_to_supplied_eos=int(view['days_to_supplied_eos'].max())))
            print(f'[SEASON INPUT] {crop} {year}: {len(regions)} regions', flush=True)
    for name, rows in (('season_summary', records), ('issue_summary', summaries)):
        path = OUT/f'{name}.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        output_hashes[path.name] = sha256(path)
    atomic_json(OUT/'manifest.json', dict(complete=True, models_fitted=0, labels_loaded=False,
        crop_year_files=len(records), source_monthly_verification=True,
        registration_sha256=sha256(config_path), outputs=output_hashes,
        sample_scope='Input-ready region-years; no claim that every row has a yield label',
        exclusions='Only missing or invalid provider calendar/location; excluded IDs retained'))
    print(json.dumps(dict(crop_year_files=len(records), rows=sum(r['regions'] for r in records))))


if __name__ == '__main__':
    main()
