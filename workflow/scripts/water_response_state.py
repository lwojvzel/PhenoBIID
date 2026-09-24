"""Spline-weighted light accumulation with explicit weather and state controls."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from canopy_exposure_readout import days_in_slots, verify_physical_inputs
from yield_sensitive_readout import inputs, save_predictions
from linear_state_yield import PENALTIES, yield_features
from calibrate_world_anchor import coefficient, leave_year_out
from multimodal_baseline import regression_metrics
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/water_response_state_v1'
CONDITIONS = ('predicted', 'previous', 'climatology', 'constant', 'observed')
CODE = ('water_response_state.py', 'canopy_exposure_readout.py', 'yield_sensitive_readout.py',
        'linear_state_yield.py', 'calibrate_world_anchor.py', 'export_yield_sensitive_state.py')
VARIABLES = (('d2m', 0, 'K'), ('t2m', 1, 'K'), ('swvl1', 4, 'm**3 m**-3'),
             ('swvl2', 5, 'm**3 m**-3'), ('swvl3', 6, 'm**3 m**-3'))


def water_drivers(weather, active, norm):
    physical = np.where(active[..., None], weather, 0).astype(float)
    physical = physical*np.asarray(norm['weather_std'])+np.asarray(norm['weather_mean'])
    temperature = physical[..., 1]-273.15
    dewpoint = physical[..., 0]-273.15
    if not np.isfinite(physical[active]).all() or np.any(temperature[active] < -100):
        raise ValueError('Invalid active meteorology')
    # Applying this to monthly means is a proxy, not the mean of daily VPD.
    saturation = lambda t: .6108*np.exp(17.27*t/(t+237.3))
    deficit = np.maximum(saturation(temperature)-saturation(dewpoint), 0)
    moisture = physical[..., 4:7]@np.array([.07, .21, .72])
    return np.where(active[..., None], np.stack((temperature, deficit, moisture), -1), 0)


def fit_response(a, norm):
    active = a['relative_valid'] > 0
    x = water_drivers(a['weather'], active, norm)[active]
    return SplineTransformer(n_knots=4, degree=2, knots='quantile',
        include_bias=False, extrapolation='constant').fit(x)


def response_state(a, state, weather, norm, response):
    active = a['relative_valid'] > 0
    if state.shape != active.shape or weather.shape != (*active.shape, 13):
        raise ValueError('Unexpected state/weather dimensions')
    drivers = water_drivers(weather, active, norm)
    basis = response.transform(drivers.reshape(-1, 3)).reshape(*active.shape, 12)
    basis = np.concatenate((np.ones((*active.shape, 1)), basis), -1)
    lai = np.maximum(np.where(active, state, 0)*norm['lai_std']+norm['lai_mean'], 0)
    radiation = np.maximum(np.where(active, weather[..., 7], 0)*norm['weather_std'][7]
        +norm['weather_mean'][7], 0)/1e6
    days = days_in_slots(a['year'], np.minimum(a['source_month'], 11))
    absorbed = .48*radiation*(-np.expm1(-.5*lai))*days
    increment = np.where(active[..., None], absorbed[..., None]*basis, 0)
    if not np.isfinite(increment).all():
        raise ValueError('Nonfinite water-response state')
    cumulative = np.cumsum(increment, axis=1)
    rank = np.cumsum(active, axis=1)-.5
    groups = np.clip(np.floor(3*rank/np.maximum(active.sum(1, keepdims=True), 1)), 0, 2).astype(int)
    summary = np.concatenate([(increment*((groups == g) & active)[..., None]).sum(1) for g in range(3)], 1)
    return cumulative, summary


def state_for(a, predicted, norm, condition):
    if condition == 'predicted':
        return predicted
    if condition == 'previous':
        return a['previous_lai']
    if condition == 'climatology':
        return a['lai_climo']
    if condition == 'constant':
        return np.full_like(predicted, (1.-norm['lai_mean'])/norm['lai_std'])
    if condition == 'observed':
        if not np.all(a['target_lai_valid'][a['relative_valid'] > 0] > 0):
            raise ValueError('Observed diagnosis must retain the same cohort')
        return a['target_lai']
    raise ValueError('Unknown water-state condition')


def water_features(a, predicted, norm, response, condition):
    state = state_for(a, predicted, norm, condition)
    base = yield_features(a, None, {'state_innovation': state}, 'state_innovation')
    blocks = []
    for start in range(0, len(state), 4096):
        stop = start+4096
        b = {k: a[k][start:stop] for k in ('year', 'source_month', 'relative_valid')}
        _, current = response_state(b, state[start:stop], a['weather'][start:stop], norm, response)
        _, climo = response_state(b, a['lai_climo'][start:stop], a['weather_climo'][start:stop], norm, response)
        blocks.append(np.concatenate((current, current-climo), 1))
    x = np.concatenate((base, np.concatenate(blocks)), 1)
    if x.shape[1] != 182 or not np.isfinite(x).all():
        raise ValueError('Invalid response feature layout')
    return x


def verify_water_inputs(arrays, norm):
    records = []
    for year in np.unique(np.concatenate([a['year'] for a in arrays.values()])):
        source = ROOT / f'Data/era5land/monthly/era5land_monthly_{year}.nc'
        units, errors, hashes = {}, {}, {}
        with Dataset(source) as ds:
            for variable, _, expected in VARIABLES:
                units[variable] = ds[variable].getncattr('units')
                if units[variable] != expected:
                    raise ValueError(f'Unexpected {variable} unit: {units[variable]}')
        for variable, channel, _ in VARIABLES:
            path = ROOT / f'Data/era5land/monthly_npy_lon180_0p5deg_by_var/{variable}/{variable}_{year}.npy'
            natural = np.load(path, mmap_mode='r'); maximum = 0.
            for a in arrays.values():
                i, k = np.where((a['year'] == year)[:, None] & (a['relative_valid'] > 0))
                if not len(i):
                    continue
                expected = natural[a['source_month'][i, k], a['row'][i], a['col'][i]].astype(float)
                actual = a['weather'][i, k, channel].astype(float)*norm['weather_std'][channel]+norm['weather_mean'][channel]
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-4 if channel < 2 else 1e-6)
                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            errors[variable] = maximum; hashes[variable] = sha256(path)
        records.append(dict(year=int(year), units=units, source_sha256=hashes, inverse_errors=errors))
    return records


def context(crop, origin):
    arrays, meta, anchors, states, sources = inputs(crop, origin)
    norm = meta['normalization']
    spec = dict(crop=crop, origin=origin, seed=42, conditions=list(CONDITIONS),
        penalties=PENALTIES.tolist(), sources=sources, input_manifest=meta,
        code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
        light_source_audit=verify_physical_inputs(arrays, norm), water_source_audit=verify_water_inputs(arrays, norm),
        state_mode='state_only', spline=dict(n_knots=4, degree=2, knots='quantile',
            include_bias=False, extrapolation='constant', fitting='Training active slots only'),
        dimensions=182, scope='Monthly water/thermal exposure proxy, not calibrated GPP or daily extremes.')
    return arrays, meta, anchors, states, spec


def fit(crop, origin):
    selection = json.loads((ROOT / 'benchmark/results/outcome_feedback_state_v1/validation_selection.json').read_text())
    if selection['passes_pilot']:
        raise ValueError('W17 is conditional on the W16 validation screen failing')
    arrays, meta, anchors, states, spec = context(crop, origin)
    root = RESULT / crop / f'origin_{origin}'; root.mkdir(parents=True, exist_ok=True)
    if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
        raise ValueError('Water-response specification changed')
    atomic_json(root / 'config.json', spec)
    if (root / 'complete.json').exists():
        return
    norm = meta['normalization']; response = fit_response(arrays['train'], norm)
    y = (arrays['train']['target'].astype(float)-anchors['train'])/norm['residual_std']
    rows, annual, weights = [], [], {}
    for condition in CONDITIONS:
        x = {s: water_features(a, states[s]['state_only'], norm, response, condition) for s, a in arrays.items()}
        scaler = StandardScaler(); train = scaler.fit_transform(x['train'])
        model = Ridge(alpha=len(train)*PENALTIES, solver='cholesky').fit(train, np.repeat(y[:, None], 5, axis=1))
        weight = root / f'{condition}.joblib'
        joblib.dump(dict(response=response, scaler=scaler, model=model), weight)
        weights[str(weight)] = sha256(weight)
        raw = {s: anchors[s][:, None]+norm['residual_std']*model.predict(scaler.transform(x[s])) for s in ('validation', 'test')}
        for j, penalty in enumerate(PENALTIES):
            row, years = save_predictions(root / condition / f'penalty_{penalty:g}', arrays, anchors,
                {s: v[:, j] for s, v in raw.items()}, weight, j, condition, 'water_response_ridge', penalty, crop, origin)
            row['dimensions'] = 182; rows.append(row); annual.extend(years)
        print(f'[WATER STATE] {crop} {origin} {condition}', flush=True)
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(annual).to_csv(root / 'per_year.csv', index=False)
    atomic_json(root / 'complete.json', dict(fits=25, calibrations=25, weights_sha256=weights))


def audit(crop, origin):
    arrays, meta, anchors, states, expected = context(crop, origin)
    root = RESULT / crop / f'origin_{origin}'
    if json.loads((root / 'config.json').read_text()) != expected:
        raise ValueError('Water-state source specification changed')
    complete = json.loads((root / 'complete.json').read_text())
    if (complete['fits'], complete['calibrations']) != (25, 25):
        raise ValueError('Incomplete response study')
    norm = meta['normalization']; fitted_response = fit_response(arrays['train'], norm)
    frame = pd.read_csv(root / 'metrics.csv'); records = []
    for condition, part in frame.groupby('mode'):
        weight = Path(part.iloc[0].weight)
        if sha256(weight) != complete['weights_sha256'][str(weight)]:
            raise ValueError('Water-state weights changed')
        bundle = joblib.load(weight)
        for actual, ref in zip(bundle['response'].bsplines_, fitted_response.bsplines_):
            np.testing.assert_array_equal(actual.t, ref.t)
            np.testing.assert_array_equal(actual.c, ref.c)
        xtrain = water_features(arrays['train'], states['train']['state_only'], norm, fitted_response, condition)
        fitted_scaler = StandardScaler().fit(xtrain)
        for key in ('mean_', 'scale_', 'var_'):
            np.testing.assert_array_equal(getattr(bundle['scaler'], key), getattr(fitted_scaler, key))
        raw = {s: anchors[s][:, None]+norm['residual_std']*bundle['model'].predict(bundle['scaler'].transform(
            water_features(arrays[s], states[s]['state_only'], norm, bundle['response'], condition))) for s in ('validation', 'test')}
        for row in part.itertuples():
            path = Path(row.directory); j = row.output_column
            cal = json.loads((path / 'calibration.json').read_text()); a = arrays['validation']
            value = coefficient(a['target'], anchors['validation'], raw['validation'][:, j], a['year'])
            _, loo = leave_year_out(a['target'], anchors['validation'], raw['validation'][:, j], a['year'])
            if value != cal['coefficient'] or loo != cal['leave_year_out_coefficients']:
                raise ValueError('Water-state calibration changed')
            for s in ('validation', 'test'):
                pred = anchors[s]+value*(raw[s][:, j]-anchors[s])
                with np.load(path / f'{s}_predictions.npz') as saved:
                    for key in ('target', 'row', 'col', 'year', 'source_indices'):
                        np.testing.assert_array_equal(saved[key], arrays[s][key])
                    np.testing.assert_array_equal(saved['prediction'], pred)
                    np.testing.assert_array_equal(saved['component_prediction'], raw[s][:, j])
                    np.testing.assert_array_equal(saved['history_prediction'], anchors[s])
                    np.testing.assert_allclose(regression_metrics(arrays[s]['target'], pred)['rmse'],
                        getattr(row, s+'_rmse'), rtol=0, atol=1e-8)
            records.append(dict(mode=condition, penalty=row.penalty, replay_error=0.))
    output = ROOT / 'visualize/paper_experiments' / RESULT.name / 'audit'; output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f'{crop}_{origin}.json', dict(fits=25, calibrations=25, maximum_replay_error=0.,
        full_array_replay=True, training_only_spline_scaler_verified=True, sources_verified=True, records=records))
    print(f'[WATER AUDIT] {crop} {origin}: 25 full-array replays', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True); p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=2):
        (audit if args.audit else fit)(args.crop, args.origin)
