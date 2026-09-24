"""Low-capacity yield residuals from independently supervised linear LAI forecasts."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from diagnose_weather_innovation import RESULT as STATES, features as state_features
from paired_weather_world import CACHE as WEATHER
from anchored_residual_world import CACHE as ANCHORS
from task_aligned_data import load
from calibrate_world_anchor import coefficient, leave_year_out
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from multimodal_baseline import regression_metrics

RESULT = ROOT / 'benchmark/results/linear_state_yield_v1'
PENALTIES = np.array([.001, .01, .1, 1., 10.])
CONDITIONS = ('history', 'known_state', 'climatology', 'state_current', 'state_innovation',
              'observed_state', 'direct_current', 'direct_paired')
CODE = ('linear_state_yield.py', 'diagnose_weather_innovation.py', 'calibrate_world_anchor.py',
        'task_aligned_data.py')


def forecast_state(a, difference, bundle, column):
    # Supply synthetic labels/masks only to reuse the fitted feature ordering.
    # Actual target-season labels and observation availability are never read.
    allowed = ('weather', 'weather_anomaly', 'previous_lai', 'previous_lai_valid',
               'lai_climo', 'source_month', 'relative_valid', 'context', 'year')
    b = {k: a[k] for k in allowed}
    b['target_lai'] = np.zeros_like(a['previous_lai'])
    b['target_lai_valid'] = np.ones_like(a['relative_valid'])
    x, _, _ = state_features(b, np.zeros_like(a['weather']))
    i, k = np.where(a['relative_valid'] > 0)
    x[:, 59:72] = difference[i, k]
    predicted_change = bundle['model'].predict(bundle['scaler'].transform(x[:, bundle['columns']]))[:, column]
    prediction = a['previous_lai'].astype(float).copy()
    prediction[i, k] += predicted_change
    return prediction


def yield_features(a, difference, forecasts, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown input condition')
    h = np.concatenate((a['history'], a['context']), 1).astype(float)
    if condition == 'history':
        return h
    active = a['relative_valid'] > 0
    phase = np.minimum(a['source_month'], 11).astype(float)*(2*np.pi/12)
    previous = np.where(active, a['previous_lai'], 0)
    climo = np.where(active, a['lai_climo'], 0)
    parts = [h, previous, a['previous_lai_valid']*active, climo,
             np.sin(phase)*active, np.cos(phase)*active, active.astype(float)]
    if condition in ('state_current', 'state_innovation', 'observed_state', 'climatology'):
        if condition == 'observed_state':
            if not np.all(a['target_lai_valid'][active] > 0):
                raise ValueError('Observed diagnostic lacks active labels; do not change the cohort')
            state = a['target_lai']
        elif condition == 'climatology':
            state = a['lai_climo']
        else:
            state = forecasts[condition]
        parts.append(np.where(active, state-a['previous_lai'], 0))
    elif condition.startswith('direct'):
        for field in ('weather_climo', 'weather_anomaly'):
            parts.append(np.where(active[..., None], a[field], 0).reshape(len(h), -1))
        if condition == 'direct_paired':
            parts.append(np.where(active[..., None], difference, 0).reshape(len(h), -1))
    output = np.concatenate(parts, 1).astype(float)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite yield features')
    return output


def inputs(crop, origin):
    arrays, meta = load(crop, origin)
    source = STATES / crop / f'origin_{origin}'
    diagnostic = json.loads((source / 'complete.json').read_text())
    if diagnostic['fits'] != 40 or diagnostic['maximum_validation_replay_error'] != 0:
        raise ValueError('State diagnostic incomplete')
    frame = pd.read_csv(source / 'validation_metrics.csv')
    choices = {}
    for group in ('current', 'innovation'):
        rows = frame[(frame.window == 0) & (frame.group == group)]
        best = rows.sort_values(['mean_year_mse_ratio', 'penalty']).iloc[0]
        weight = Path(best.weight)
        if sha256(weight) != diagnostic['weights_sha256'][str(weight)]:
            raise ValueError('Linear state weight changed')
        choices['state_'+group] = dict(weight=str(weight), sha256=sha256(weight),
            output_column=int(best.output_column), penalty=float(best.penalty),
            validation_rmse=float(best.validation_rmse))
    weather_root = WEATHER / crop / f'origin_{origin}'
    anchor_root = ANCHORS / crop / f'origin_{origin}/seed_42'
    weather_spec = json.loads((weather_root / 'manifest.json').read_text())
    anchor_spec = json.loads((anchor_root / 'manifest.json').read_text())
    differences, anchors, forecasts = {}, {}, {}
    for split, a in arrays.items():
        if sha256(weather_root / f'{split}.npy') != weather_spec['files'][split]:
            raise ValueError('Weather difference changed')
        if sha256(anchor_root / f'{split}.npy') != anchor_spec['files'][split]:
            raise ValueError('Historical anchor changed')
        differences[split] = np.load(weather_root / f'{split}.npy')
        anchors[split] = np.load(anchor_root / f'{split}.npy')
        forecasts[split] = {}
        for condition, choice in choices.items():
            forecasts[split][condition] = forecast_state(a, differences[split],
                joblib.load(choice['weight']), choice['output_column'])
            if split == 'validation':
                valid = (a['relative_valid'] > 0) & (a['target_lai_valid'] > 0)
                error = np.sqrt(np.mean((forecasts[split][condition][valid]-a['target_lai'][valid])**2))*meta['normalization']['lai_std']
                np.testing.assert_allclose(error, choice['validation_rmse'], rtol=0, atol=1e-7)
        if split != 'train':
            with np.load(Path(anchor_spec['spec']['source']) / f'{split}_predictions.npz') as old:
                np.testing.assert_array_equal(old['prediction'], anchors[split])
                for key in ('target', 'source_indices', 'row', 'col', 'year'):
                    np.testing.assert_array_equal(old[key], a[key])
    if sha256(Path(anchor_spec['spec']['weight'])) != anchor_spec['spec']['weight_sha256']:
        raise ValueError('Historical expert changed')
    return arrays, meta, differences, anchors, forecasts, dict(states=choices, weather=weather_spec, anchor=anchor_spec)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    args = parser.parse_args()
    root = RESULT / args.crop / f'origin_{args.origin}'; root.mkdir(parents=True, exist_ok=True)
    arrays, meta, differences, anchors, forecasts, sources = inputs(args.crop, args.origin)
    spec = dict(crop=args.crop, origin=args.origin, penalties=PENALTIES.tolist(), conditions=list(CONDITIONS),
        sources=sources, input_manifest=meta, code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
        protocol='Frozen linear LAI maps, fixed historical anchor, ridge residuals, validation-only calibration and penalty selection.',
        observed_scope='Observed-state branch is a diagnostic, never an eligible forecasting candidate.',
        state_scope='Linear slot-state maps, not an autoregressive transition; no claim of final world architecture.')
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != spec:
        raise ValueError('Linear state-yield specification changed')
    atomic_json(config_path, spec)
    if (root / 'complete.json').exists():
        return
    scale = meta['normalization']['residual_std']
    y = (arrays['train']['target'].astype(float)-anchors['train'])/scale
    rows, annual, models, replays = [], [], {}, []
    with threadpool_limits(limits=2):
        for condition in CONDITIONS:
            x = {split: yield_features(a, differences[split], forecasts[split], condition) for split, a in arrays.items()}
            scaler = StandardScaler(); train = scaler.fit_transform(x['train'])
            model = Ridge(alpha=len(train)*PENALTIES, solver='cholesky').fit(train, np.repeat(y[:, None], 5, axis=1))
            weight = root / f'{condition}.joblib'; joblib.dump(dict(scaler=scaler, model=model), weight)
            models[str(weight)] = sha256(weight); restored = joblib.load(weight)
            raw = {}
            for split in ('validation', 'test'):
                prediction = model.predict(scaler.transform(x[split]))
                replay = restored['model'].predict(restored['scaler'].transform(x[split]))
                np.testing.assert_array_equal(prediction, replay)
                replays.append(float(np.max(np.abs(prediction-replay))))
                raw[split] = anchors[split][:, None]+scale*prediction
            for j, penalty in enumerate(PENALTIES):
                destination = root / condition / f'penalty_{penalty:g}'
                destination.mkdir(parents=True, exist_ok=True)
                a = arrays['validation']; h = anchors['validation']; p = raw['validation'][:, j]
                value = coefficient(a['target'], h, p, a['year'])
                held, held_values = leave_year_out(a['target'], h, p, a['year'])
                ratios = [np.sqrt(np.mean((a['target'][a['year'] == year]-held[a['year'] == year])**2)/
                                  np.mean((a['target'][a['year'] == year]-h[a['year'] == year])**2))
                          for year in np.unique(a['year'])]
                calibration = dict(coefficient=value, leave_year_out_coefficients=held_values,
                    leave_year_out_mean_ratio=float(np.mean(ratios)), penalty=float(penalty),
                    weight=str(weight), output_column=j)
                atomic_json(destination / 'calibration.json', calibration)
                record = dict(crop=args.crop, origin=args.origin, condition=condition, penalty=float(penalty),
                    coefficient=value, loo_ratio=float(np.mean(ratios)), dimensions=x['train'].shape[1],
                    directory=str(destination), weight=str(weight), output_column=j)
                for split in ('validation', 'test'):
                    a = arrays[split]; h = anchors[split]; component = raw[split][:, j]
                    prediction = h+value*(component-h)
                    metric = regression_metrics(a['target'], prediction)
                    raw_metric = regression_metrics(a['target'], component)
                    base = regression_metrics(a['target'], h)['rmse']
                    record[split+'_rmse'] = metric['rmse']
                    record[split+'_gain'] = 100*(1-metric['rmse']/base)
                    record[split+'_raw_gain'] = 100*(1-raw_metric['rmse']/base)
                    np.savez_compressed(destination / f'{split}_predictions.npz', prediction=prediction,
                        component_prediction=component, history_prediction=h,
                        **{key: a[key] for key in ('target', 'row', 'col', 'year', 'source_indices')})
                    for year in np.unique(a['year']):
                        keep = a['year'] == year
                        rmse = float(np.sqrt(np.mean((a['target'][keep]-prediction[keep])**2)))
                        reference = float(np.sqrt(np.mean((a['target'][keep]-h[keep])**2)))
                        annual.append(dict(crop=args.crop, origin=args.origin, condition=condition,
                            penalty=float(penalty), split=split, year=int(year), rmse=rmse, gain=100*(1-rmse/reference)))
                rows.append(record)
            print(f'[LINEAR STATE YIELD] {args.crop} {args.origin} {condition}', flush=True)
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(annual).to_csv(root / 'per_year.csv', index=False)
    atomic_json(root / 'complete.json', dict(fits=len(rows), weights_sha256=models,
        maximum_replay_error=max(replays), history_alignment_verified=True, calibration_count=len(rows)))


if __name__ == '__main__':
    main()
