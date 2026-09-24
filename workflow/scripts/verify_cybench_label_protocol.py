"""Independently reconstruct saved regional labels, histories and scores."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'benchmark/cache/cybench_inseason_labels_v1'
SEASONS = ROOT / 'Data/processed/cybench_seasonal_v1'
PROVIDER = ROOT / 'Data/external/CYBench/full_v1_10'
OUT = ROOT / 'benchmark/results/cybench_inseason_labels_v1'
BLOCKS = {2001: (2002, 2003, 2004), 2005: (2006, 2007, 2008),
          2009: tuple(range(2010, 2017))}
METHODS = ('latest_available', 'mean_last_three', 'mean_last_five',
           'past_mean', 'past_linear_trend')
SUMMARY = ('latest', 'mean', 'std', 'linear_trend', 'slope_per_year',
           'count', 'latest_age')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_array(actual, expected, name, atol=0):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f'{name}: shape or dtype differs')
    if np.issubdtype(expected.dtype, np.floating):
        if not np.array_equal(np.isnan(actual), np.isnan(expected)):
            raise ValueError(f'{name}: missingness differs')
        if np.isinf(actual).any() or np.isinf(expected).any():
            raise ValueError(f'{name}: unexpected infinity')
        selected = np.isfinite(expected)
        delta = np.abs(actual[selected].astype(np.float64) - expected[selected])
        maximum = float(delta.max()) if delta.size else 0.
        if maximum > atol:
            raise ValueError(f'{name}: absolute error {maximum} exceeds {atol}')
        return maximum
    if not np.array_equal(actual, expected):
        raise ValueError(f'{name}: entries differ')
    return 0.


def reference_history(source, identities):
    """Use scalar records and least-squares fitting, not the production helpers."""
    grouped = defaultdict(list)
    for (country, region, year), record in source.items():
        value = record['yield']
        if math.isfinite(value) and value >= 0:
            grouped[country, region].append((year, value))
    n = len(identities)
    lags = np.full((n, 5), np.nan, dtype=np.float32)
    years = np.empty((n, 5), dtype=np.int32)
    summary = np.full((n, 7), np.nan, dtype=np.float32)
    summary[:, 5] = 0
    latest = np.zeros(n, dtype=np.int32)
    for i, row in enumerate(identities):
        year = int(row['year'])
        years[i] = [year - k for k in range(1, 6)]
        past = sorted((y, v) for y, v in grouped[row['country'], row['adm_id']]
                      if y < year)
        if not past:
            continue
        lookup = dict(past)
        lags[i] = [lookup.get(int(y), math.nan) for y in years[i]]
        values = [v for _, v in past]
        mean = math.fsum(values) / len(values)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
        # Center at the prediction year, so the fitted intercept is the forecast.
        x = np.array([y - year for y, _ in past], dtype=np.float64)
        if len(past) > 1:
            level, slope = np.linalg.lstsq(
                np.column_stack((np.ones(len(x)), x)), values, rcond=None)[0]
        else:
            level, slope = mean, 0.
        latest[i] = past[-1][0]
        summary[i] = [past[-1][1], mean, std, level, slope, len(past),
                      year - int(latest[i])]
    return dict(lag_yield=lags, lag_valid=np.isfinite(lags),
                lag_source_year=years, history_summary=summary,
                latest_source_year=latest)


def reference_predictions(history):
    output = {name: np.empty(len(history['history_summary']), np.float32)
              for name in METHODS}
    for i, summary in enumerate(history['history_summary']):
        values = dict(latest_available=float(summary[0]), past_mean=float(summary[1]),
                      past_linear_trend=float(summary[3]))
        for count, name in ((3, 'mean_last_three'), (5, 'mean_last_five')):
            available = [float(v) for v in history['lag_yield'][i, :count]
                         if math.isfinite(float(v))]
            values[name] = math.fsum(available) / len(available) if available else float(summary[1])
        for name, value in values.items():
            output[name][i] = max(value, 0.)
    return output


def compare_records(actual, expected, keys, numeric=()):
    def keyed(records):
        result = {tuple(str(r[k]) for k in keys): r for r in records}
        if len(result) != len(records):
            raise ValueError('Duplicate summary identity')
        return result
    a, e = keyed(actual), keyed(expected)
    if a.keys() != e.keys():
        raise ValueError('Summary identities differ')
    for key, expected_row in e.items():
        actual_row = a[key]
        if actual_row.keys() != expected_row.keys():
            raise ValueError('Summary fields differ')
        for name, value in expected_row.items():
            if name in numeric:
                if not math.isclose(float(actual_row[name]), float(value), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f'Summary value differs: {key}, {name}')
            elif str(actual_row[name]) != str(value):
                raise ValueError(f'Summary value differs: {key}, {name}')


def source_records(crop):
    source = {}
    for country in ('DE', 'FR', 'PL'):
        for row in rows(PROVIDER / crop / country / f'yield_{crop}_{country}.csv'):
            if row['country_code'] != country:
                raise ValueError('Source country mismatch')
            year = float(row['harvest_year'])
            if not math.isfinite(year) or year != int(year):
                raise ValueError('Invalid source year')
            if year > 2016:
                continue
            key = country, row['adm_id'], int(year)
            if key in source:
                raise ValueError('Duplicate source row')
            value = float(row['yield']) if row['yield'].strip() else math.nan
            source[key] = {'yield': value, 'crop_name': row['crop_name']}
    return source


def verify():
    registration = json.loads((BASE / 'registration.json').read_text())
    manifest = json.loads((BASE / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['registration_sha256'] != digest(BASE / 'registration.json'):
        raise ValueError('Incomplete or changed registration')
    expected_config = dict(blocks={str(k): list(v) for k, v in BLOCKS.items()},
        years=list(range(1983, 2017)), history_lag_years=5,
        history_summary_names=list(SUMMARY), baseline_methods=list(METHODS),
        target_unit='t/ha', maximum_source_label_year=2016,
        source_columns=['country_code', 'adm_id', 'harvest_year', 'yield', 'crop_name'],
        fitted_models=0, normalization_fitted=False, hyperparameter_search=False)
    if any(registration.get(k) != v for k, v in expected_config.items()):
        raise ValueError('Unexpected registered study')
    hashes = dict(registration['source_sha256'])
    hashes.update({str(ROOT / 'scripts' / n): d for n, d in registration['code_sha256'].items()})
    hashes.update({str(BASE / n): d for n, d in manifest['files'].items()})
    hashes.update(manifest['simple_score_files'])
    seasonal = json.loads((SEASONS / 'manifest.json').read_text())
    hashes.update({str(SEASONS / n): d for n, d in seasonal['outputs'].items()})
    for name, value in hashes.items():
        if digest(name) != value:
            raise ValueError(f'Changed dependency: {name}')
    counts, cohort_rows, annual, crop_checks = [], [], [], []
    total_entries = 0
    for crop in ('maize', 'wheat'):
        folder = BASE / crop
        regions = rows(SEASONS / crop / 'regions.csv')
        identities = []
        for year in range(1983, 2017):
            for season_row, region in enumerate(regions):
                identities.append(dict(sample_index=str(len(identities)), country=region['country'],
                    adm_id=region['adm_id'], region_index=region['region_index'],
                    year=str(year), season_row=str(season_row)))
        if identities != rows(folder / 'identities.csv'):
            raise ValueError('Saved identity order differs from seasonal inputs')
        source = source_records(crop)
        target = np.full(len(identities), np.nan, np.float32)
        expected_join = []
        history = reference_history(source, identities)
        for i, row in enumerate(identities):
            record = source.get((row['country'], row['adm_id'], int(row['year'])))
            if record is not None:
                target[i] = record['yield']
            valid = bool(np.isfinite(target[i]) and target[i] >= 0)
            has_history = bool(history['history_summary'][i, 5] > 0)
            reason = ('no_label' if record is None else 'invalid_target' if not valid
                      else 'no_past_label' if not has_history else 'eligible')
            expected_join.append(dict(row, crop_name='' if record is None else record['crop_name'],
                label_present=str(record is not None), target_valid=str(valid),
                has_history=str(has_history), eligible_before_block=str(valid and has_history),
                exclusion_reason=reason))
        if expected_join != rows(folder / 'label_join.csv'):
            raise ValueError('Label join or eligibility differs from raw sources')
        with np.load(folder / 'targets.npz', allow_pickle=False) as saved:
            if saved.files != ['yield']:
                raise ValueError('Unexpected target fields')
            check_array(saved['yield'], target, 'yield')
        predictions = reference_predictions(history)
        errors = {}
        for filename, expected in (('history', history), ('simple_history_predictions', predictions)):
            with np.load(folder / f'{filename}.npz', allow_pickle=False) as saved:
                if set(saved.files) != set(expected):
                    raise ValueError('Unexpected saved array field')
                for name, value in expected.items():
                    tolerance = 2e-6 if name in ('history_summary', 'past_linear_trend') else 0
                    errors[name] = check_array(saved[name], value, f'{crop}/{name}', tolerance)
                    total_entries += value.size
        total_entries += target.size
        for country in ('DE', 'FR', 'PL'):
            group = [r for r in expected_join if r['country'] == country]
            counts.append(dict(crop=crop, country=country, input_rows=len(group),
                label_rows=sum(r['label_present'] == 'True' for r in group),
                valid_targets=sum(r['target_valid'] == 'True' for r in group),
                history_supported=sum(r['eligible_before_block'] == 'True' for r in group)))
        eligible = [i for i, r in enumerate(expected_join) if r['eligible_before_block'] == 'True']
        eval_ids = set()
        for cutoff, evaluation_years in BLOCKS.items():
            inner_ids = [i for i in eligible if int(identities[i]['year']) <= cutoff - 2]
            seen = {(identities[i]['country'], identities[i]['adm_id']) for i in inner_ids}
            chosen = [i for i in eligible if (identities[i]['country'], identities[i]['adm_id']) in seen]
            indices = dict(inner_fit=inner_ids,
                inner_validation=[i for i in chosen if cutoff - 1 <= int(identities[i]['year']) <= cutoff],
                full_fit=[i for i in chosen if int(identities[i]['year']) <= cutoff],
                evaluation=[i for i in chosen if int(identities[i]['year']) in evaluation_years])
            if eval_ids.intersection(indices['evaluation']):
                raise ValueError('Evaluation blocks overlap')
            eval_ids.update(indices['evaluation'])
            with np.load(folder / f'block_{cutoff}.npz', allow_pickle=False) as saved:
                if set(saved.files) != set(indices):
                    raise ValueError('Unexpected partition fields')
                for split, ix in indices.items():
                    check_array(saved[split], np.array(ix, dtype=np.int64), split)
                    for country in ('DE', 'FR', 'PL'):
                        group = [identities[i] for i in ix if identities[i]['country'] == country]
                        cohort_rows.append(dict(crop=crop, cutoff=cutoff, split=split, country=country,
                            samples=len(group), regions=len({r['adm_id'] for r in group}),
                            years=';'.join(map(str, sorted({int(r['year']) for r in group})))))
            grouped = defaultdict(list)
            for i in indices['evaluation']:
                grouped[identities[i]['country'], int(identities[i]['year'])].append(i)
            # Score archived prediction vectors, separately from their reconstruction.
            with np.load(folder / 'simple_history_predictions.npz', allow_pickle=False) as saved:
                for (country, year), ix in grouped.items():
                    for method in METHODS:
                        selected = saved[method][ix]
                        error = [float(v) - float(target[i]) for v, i in zip(selected, ix)]
                        if not all(math.isfinite(e) for e in error):
                            raise ValueError('Missing prediction on paired evaluation cohort')
                        annual.append(dict(crop=crop, cutoff=cutoff, country=country, year=year,
                            method=method, samples=len(ix),
                            rmse=math.sqrt(math.fsum(e * e for e in error) / len(ix)),
                            mae=math.fsum(abs(e) for e in error) / len(ix)))
        crop_checks.append(dict(crop=crop, rows=len(identities), evaluation_rows=len(eval_ids),
                                maximum_absolute_errors=errors))
        print(f'[LABEL VERIFY] {crop}: {len(identities)} rows, {len(eval_ids)} evaluation rows', flush=True)
    compare_records(rows(BASE / 'label_readiness.csv'), counts, ('crop', 'country'))
    compare_records(rows(BASE / 'partition_counts.csv'), cohort_rows, ('crop', 'cutoff', 'split', 'country'))
    compare_records(rows(OUT / 'simple_history_annual.csv'), annual,
                    ('crop', 'cutoff', 'country', 'year', 'method'), ('rmse', 'mae'))
    grouped = defaultdict(list)
    for r in annual:
        grouped[r['crop'], r['country'], r['method']].append(r)
    summary = []
    for (crop, country, method), group in grouped.items():
        if len(group) != len({r['year'] for r in group}):
            raise ValueError('Repeated evaluation year in summary')
        summary.append(dict(crop=crop, country=country, method=method,
            mean_annual_rmse=math.fsum(r['rmse'] for r in group) / len(group),
            mean_annual_mae=math.fsum(r['mae'] for r in group) / len(group),
            years=len(group), samples=sum(r['samples'] for r in group)))
    compare_records(rows(OUT / 'simple_history_scores.csv'), summary,
                    ('crop', 'country', 'method'), ('mean_annual_rmse', 'mean_annual_mae'))
    manifest_expected = dict(rows=sum(r['input_rows'] for r in counts),
        valid_label_rows=sum(r['valid_targets'] for r in counts),
        feature_targets_separate=True, fitted_models=0, world_model_evaluated=False)
    if any(manifest.get(k) != v for k, v in manifest_expected.items()):
        raise ValueError('Manifest totals or scope do not match verified outputs')
    return dict(passed=True, timestamp=datetime.now().astimezone().isoformat(),
        scope='All saved label/history rows and simple reference scores; no world-model performance claim',
        manifest_sha256=digest(BASE / 'manifest.json'), code_sha256=digest(__file__),
        source_and_output_files_verified=len(hashes), array_entries_checked=total_entries,
        rows=sum(r['input_rows'] for r in counts), valid_label_rows=sum(r['valid_targets'] for r in counts),
        history_supported_rows=sum(r['history_supported'] for r in counts),
        evaluation_rows=sum(c['evaluation_rows'] for c in crop_checks),
        annual_score_rows=len(annual), summary_score_rows=len(summary),
        identities_match_seasonal_order=True, labels_match_provider=True,
        histories_reconstructed_from_strictly_earlier_records=True,
        partitions_reconstructed_with_seen_region_rule=True,
        annual_scores_recomputed_on_same_paired_indices=True,
        labels_and_features_physically_separate=True,
        summary_absolute_tolerance=2e-6, score_absolute_tolerance=1e-12,
        whole_model_evaluated=False, models_fitted=0, checks=crop_checks,
        limitations=['Statistical release dates are not verified.',
                     'German winter wheat and French/Polish soft wheat retain distinct source definitions.',
                     'Only maize/wheat have labels here; Poland has seven evaluation years.',
                     'This does not establish current world-model performance on independent labels.'])


if __name__ == '__main__':
    result = verify()
    path = OUT / 'label_verification.json'
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(result, indent=2))
