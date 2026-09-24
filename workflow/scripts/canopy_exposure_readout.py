"""A process-inspired accumulated-light state, not a calibrated biomass simulator."""
import argparse
import calendar
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from yield_sensitive_readout import inputs, save_predictions, RESULT as PRECEDING
from linear_state_yield import PENALTIES, yield_features
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/canopy_exposure_readout_v1'
CONDITIONS = ('predicted', 'exposure_only', 'previous', 'climatology', 'constant', 'observed')
CODE = ('canopy_exposure_readout.py', 'yield_sensitive_readout.py', 'linear_state_yield.py',
        'calibrate_world_anchor.py', 'export_yield_sensitive_state.py', 'yield_sensitive_state.py')


def days_in_slots(year, month):
    year = np.asarray(year, dtype=int); month = np.asarray(month, dtype=int)
    if month.shape != (len(year), 12) or np.any((month < 0) | (month > 11)):
        raise ValueError('Expected twelve zero-indexed civil-calendar slots')
    days = np.empty(month.shape, dtype=float)
    for y in np.unique(year):
        lookup = np.array([calendar.monthrange(int(y), m)[1] for m in range(1, 13)])
        keep = year == y; days[keep] = lookup[month[keep]]
    return days


def temperature_basis(temperature):
    result = [np.ones_like(temperature)]
    for optimum in (15., 25., 35.):
        delta = np.where(temperature <= optimum, (optimum-temperature)/optimum,
                         (temperature-optimum)/(45.-optimum))
        result.append(np.clip(1-delta**2, 0, 1))
    return np.stack(result, -1)


def exposure_state(a, normalized_lai, normalized_weather, norm):
    active = a['relative_valid'] > 0
    if normalized_lai.shape != active.shape or normalized_weather.shape != (*active.shape, 13):
        raise ValueError('Canopy state dimensions differ')
    physical_lai = normalized_lai.astype(float)*norm['lai_std']+norm['lai_mean']
    physical_weather = normalized_weather.astype(float)*np.asarray(norm['weather_std'])+np.asarray(norm['weather_mean'])
    lai = np.maximum(np.where(active, physical_lai, 0), 0)
    temperature = np.where(active, physical_weather[..., 1]-273.15, 0)
    radiation = np.maximum(np.where(active, physical_weather[..., 7], 0), 0)/1e6
    days = days_in_slots(a['year'], np.minimum(a['source_month'], 11))
    absorbed = .48*radiation*(-np.expm1(-.5*lai))
    increments = absorbed[..., None]*days[..., None]*temperature_basis(temperature)*active[..., None]
    if not np.isfinite(increments).all():
        raise ValueError('Nonfinite accumulated light')
    cumulative = np.cumsum(increments, axis=1)
    rank = np.cumsum(active, axis=1)-.5
    count = np.maximum(active.sum(1), 1)
    group = np.clip(np.floor(3*rank/count[:, None]), 0, 2).astype(int)
    summaries = np.concatenate([(increments*((group == k) & active)[..., None]).sum(1) for k in range(3)], 1)
    return cumulative, summaries


def canopy_features(a, predicted, norm, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown canopy condition')
    if condition in ('predicted', 'exposure_only'):
        state = predicted
    elif condition == 'previous':
        state = a['previous_lai']
    elif condition == 'climatology':
        state = a['lai_climo']
    elif condition == 'constant':
        state = np.full_like(a['previous_lai'], (1.-norm['lai_mean'])/norm['lai_std'])
    else:
        active = a['relative_valid'] > 0
        if not np.all(a['target_lai_valid'][active] > 0):
            raise ValueError('Observed diagnostic may not change the sample cohort')
        state = a['target_lai']
    _, current = exposure_state(a, state, a['weather'], norm)
    _, climo = exposure_state(a, a['lai_climo'], a['weather_climo'], norm)
    base = yield_features(a, None, {}, 'known_state') if condition == 'exposure_only' else yield_features(
        a, None, {'state_innovation': state}, 'state_innovation')
    return np.concatenate((base, current, current-climo), 1)


def verify_physical_inputs(arrays, norm):
    records = []
    years = np.unique(np.concatenate([a['year'] for a in arrays.values()]))
    for year in years:
        source = ROOT / f'Data/era5land/monthly/era5land_monthly_{year}.nc'
        with Dataset(source) as ds:
            history = ds.getncattr('history')
            match = re.search(r'(\{.*\})$', history)
            request = json.loads(match.group(1)) if match else {}
            if request.get('filter_by_keys', {}).get('stream') != ['moda']:
                raise ValueError(f'Unknown radiation accumulation stream: {year}')
            units = {name: ds[name].getncattr('units') for name in ('ssrd', 't2m')}
            if units != {'ssrd': 'J m**-2', 't2m': 'K'}:
                raise ValueError('Unexpected physical weather units')
        errors, hashes = {}, {}
        for name, column, tolerance in (('ssrd', 7, 5.), ('t2m', 1, .0001)):
            path = ROOT / f'Data/era5land/monthly_npy_lon180_0p5deg_by_var/{name}/{name}_{year}.npy'
            natural = np.load(path, mmap_mode='r'); maximum = 0.
            for a in arrays.values():
                i, k = np.where((a['year'] == year)[:, None] & (a['relative_valid'] > 0))
                if not len(i):
                    continue
                expected = natural[a['source_month'][i, k], a['row'][i], a['col'][i]].astype(float)
                actual = a['weather'][i, k, column].astype(float)*norm['weather_std'][column]+norm['weather_mean'][column]
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=tolerance)
                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            errors[name] = maximum; hashes[name] = sha256(path)
        records.append(dict(year=int(year), netcdf=str(source), history=history, units=units,
            source_array_sha256=hashes, maximum_inverse_normalization_errors=errors))
    return records


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    args = p.parse_args()
    preceding = json.loads((PRECEDING / 'validation_selection.json').read_text())
    if preceding['passes_pilot']:
        raise ValueError('W14 was conditional on failure of the registered W13 validation pilot')
    arrays, meta, anchors, states, sources = inputs(args.crop, args.origin)
    norm = meta['normalization']; physical = verify_physical_inputs(arrays, norm)
    root = RESULT / args.crop / f'origin_{args.origin}'; root.mkdir(parents=True, exist_ok=True)
    spec = dict(crop=args.crop, origin=args.origin, seed=42, conditions=list(CONDITIONS), penalties=PENALTIES.tolist(),
        state_mode='state_only', sources=sources, input_manifest=meta, physical_audit=physical,
        code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
        process=dict(par_fraction=.48, extinction=.5, minimum_temperature=0, maximum_temperature=45,
                     optima=[15, 25, 35], slot_groups=3, radiation_units='MJ/m2/day', cumulative_units='MJ/m2'),
        scope='Process-inspired accumulated-light states, not calibrated biomass or a replication of daily SAFY.',
        protocol='Fixed LAI-only recurrent state, unchanged cohort/history, validation-only ridge and scalar calibration.')
    if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
        raise ValueError('Canopy exposure recipe changed')
    atomic_json(root / 'config.json', spec)
    if (root / 'complete.json').exists():
        return
    y = (arrays['train']['target'].astype(float)-anchors['train'])/norm['residual_std']
    rows, annual, weights = [], [], {}
    with threadpool_limits(limits=2):
        for condition in CONDITIONS:
            x = {split: canopy_features(a, states[split]['state_only'], norm, condition) for split, a in arrays.items()}
            scaler = StandardScaler(); train = scaler.fit_transform(x['train'])
            model = Ridge(alpha=len(train)*PENALTIES, solver='cholesky').fit(train, np.repeat(y[:, None], 5, axis=1))
            weight = root / f'{condition}.joblib'; joblib.dump(dict(model=model, scaler=scaler), weight)
            weights[str(weight)] = sha256(weight)
            restored = joblib.load(weight); raw = {}
            for split in ('validation', 'test'):
                delta = model.predict(scaler.transform(x[split]))
                np.testing.assert_array_equal(delta, restored['model'].predict(restored['scaler'].transform(x[split])))
                raw[split] = anchors[split][:, None]+norm['residual_std']*delta
            for j, penalty in enumerate(PENALTIES):
                row, years = save_predictions(root / condition / f'penalty_{penalty:g}', arrays, anchors,
                    {split: value[:, j] for split, value in raw.items()}, weight, j, condition, 'canopy_ridge',
                    penalty, args.crop, args.origin)
                row['dimensions'] = x['train'].shape[1]; rows.append(row); annual.extend(years)
            print(f'[CANOPY EXPOSURE] {args.crop} {args.origin} {condition}', flush=True)
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(annual).to_csv(root / 'per_year.csv', index=False)
    atomic_json(root / 'complete.json', dict(fits=30, calibrations=30, weights_sha256=weights,
        physical_inputs_verified=True, initial_full_replay_error=0.))


if __name__ == '__main__':
    main()
