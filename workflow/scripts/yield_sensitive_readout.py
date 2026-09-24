"""Matched frozen-teacher and refitted yield heads for recurrent state corrections."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from yield_sensitive_state import CACHE as STATE_CACHE, MODES
from export_yield_sensitive_state import export_root
from linear_state_yield import yield_features, PENALTIES, RESULT as LINEAR
from task_aligned_data import load, CACHE as TASK_CACHE
from anchored_residual_world import CACHE as ANCHORS
from calibrate_world_anchor import coefficient, leave_year_out
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from multimodal_baseline import regression_metrics

RESULT = ROOT / 'benchmark/results/yield_sensitive_readout_v1'
CODE = ('yield_sensitive_readout.py', 'linear_state_yield.py', 'calibrate_world_anchor.py',
        'export_yield_sensitive_state.py', 'yield_sensitive_state.py', 'task_aligned_data.py')


def features(a, state):
    if state.shape != a['previous_lai'].shape:
        raise ValueError('State and sample/slot dimensions differ')
    return yield_features(a, None, {'state_innovation': state}, 'state_innovation')


def inputs(crop, origin):
    arrays, meta = load(crop, origin)
    state_cache = STATE_CACHE / crop / f'origin_{origin}'
    state_spec = json.loads((state_cache / 'manifest.json').read_text())
    if state_spec['spec']['input_manifest'] != meta:
        raise ValueError('State and yield cohorts differ')
    anchor_root = ANCHORS / crop / f'origin_{origin}/seed_42'
    anchor_spec = json.loads((anchor_root / 'manifest.json').read_text())
    teacher = Path(state_spec['spec']['teacher_weight'])
    if sha256(teacher) != state_spec['spec']['teacher_sha256']:
        raise ValueError('Frozen teacher changed')
    exports = {mode: json.loads((export_root(crop, origin, mode) / 'manifest.json').read_text()) for mode in MODES}
    for mode, export in exports.items():
        if export['spec']['input_manifest'] != state_spec or export['audit']['maximum_replay_error'] != 0:
            raise ValueError('Unverified state export')
        if sha256(Path(export['spec']['weight'])) != export['spec']['weight_sha256']:
            raise ValueError('State weight changed')
    anchors, states = {}, {}
    for split, a in arrays.items():
        if sha256(TASK_CACHE / crop / f'origin_{origin}/{split}.npz') != meta['files'][split]:
            raise ValueError('Original inputs changed')
        if sha256(state_cache / f'{split}.npz') != state_spec['files'][split]:
            raise ValueError('State inputs changed')
        if sha256(anchor_root / f'{split}.npy') != anchor_spec['files'][split]:
            raise ValueError('Frozen history predictions changed')
        anchors[split] = np.load(anchor_root / f'{split}.npy')
        with np.load(state_cache / f'{split}.npz') as cached:
            for key in ('source_indices', 'row', 'col', 'year'):
                np.testing.assert_array_equal(cached[key], a[key])
            states[split] = {'prior': cached['prior_lai']}
        for mode in MODES:
            path = export_root(crop, origin, mode) / f'{split}.npy'
            if sha256(path) != exports[mode]['files'][split]:
                raise ValueError('Exported free rollout changed')
            states[split][mode] = np.load(path)
        if split != 'train':
            with np.load(Path(anchor_spec['spec']['source']) / f'{split}_predictions.npz') as saved:
                for key in ('target', 'row', 'col', 'year', 'source_indices'):
                    np.testing.assert_array_equal(saved[key], a[key])
                np.testing.assert_array_equal(saved['prediction'], anchors[split])
    return arrays, meta, anchors, states, dict(state_cache=state_spec, exports=exports, anchor=anchor_spec,
        teacher_weight=str(teacher), teacher_sha256=sha256(teacher))


def save_predictions(destination, arrays, anchors, raw, weight, column, mode, readout, penalty, crop, origin):
    destination.mkdir(parents=True, exist_ok=True)
    a = arrays['validation']; h = anchors['validation']; component = raw['validation']
    value = coefficient(a['target'], h, component, a['year'])
    held, held_values = leave_year_out(a['target'], h, component, a['year'])
    ratios = [np.sqrt(np.mean((a['target'][a['year'] == year]-held[a['year'] == year])**2)/
                      np.mean((a['target'][a['year'] == year]-h[a['year'] == year])**2))
              for year in np.unique(a['year'])]
    calibration = dict(coefficient=value, leave_year_out_coefficients=held_values,
        leave_year_out_mean_ratio=float(np.mean(ratios)), weight=str(weight), output_column=column)
    atomic_json(destination / 'calibration.json', calibration)
    record = dict(crop=crop, origin=origin, mode=mode, readout=readout, penalty=float(penalty),
        coefficient=value, loo_ratio=float(np.mean(ratios)), dimensions=104,
        directory=str(destination), weight=str(weight), output_column=column)
    annual = []
    for split in ('validation', 'test'):
        a = arrays[split]; h = anchors[split]; component = raw[split]
        prediction = h+value*(component-h)
        metric = regression_metrics(a['target'], prediction)
        raw_metric = regression_metrics(a['target'], component)
        base = regression_metrics(a['target'], h)['rmse']
        record[split+'_rmse'] = metric['rmse']; record[split+'_gain'] = 100*(1-metric['rmse']/base)
        record[split+'_raw_gain'] = 100*(1-raw_metric['rmse']/base)
        np.savez_compressed(destination / f'{split}_predictions.npz', prediction=prediction,
            component_prediction=component, history_prediction=h,
            **{key: a[key] for key in ('target', 'row', 'col', 'year', 'source_indices')})
        for year in np.unique(a['year']):
            keep = a['year'] == year
            rmse = float(np.sqrt(np.mean((a['target'][keep]-prediction[keep])**2)))
            reference = float(np.sqrt(np.mean((a['target'][keep]-h[keep])**2)))
            annual.append(dict(crop=crop, origin=origin, mode=mode, readout=readout, penalty=float(penalty),
                split=split, year=int(year), rmse=rmse, gain=100*(1-rmse/reference)))
    return record, annual


def main():
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    args = p.parse_args(); root = RESULT / args.crop / f'origin_{args.origin}'
    root.mkdir(parents=True, exist_ok=True)
    arrays, meta, anchors, states, sources = inputs(args.crop, args.origin)
    spec = dict(crop=args.crop, origin=args.origin, seed=42, penalties=PENALTIES.tolist(),
        sources=sources, input_manifest=meta, code_hashes={name: sha256(ROOT / 'scripts' / name) for name in CODE},
        fixed_teacher_penalty=.01, teacher_scope='Observed-state training teacher evaluated on predicted states only.',
        protocol='Same 104-dimensional residual features, frozen history, free state rollouts, validation-only calibration.',
        prior_precision='Frozen-teacher prior uses W12 float32 state; reused W11 refit control used float64 linear output.',
        refitted_prior_reused=str(LINEAR / args.crop / f'origin_{args.origin}/state_innovation'))
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != spec:
        raise ValueError('State-sensitive readout recipe changed')
    atomic_json(config_path, spec)
    if (root / 'complete.json').exists():
        return
    scale = meta['normalization']['residual_std']
    y = (arrays['train']['target'].astype(float)-anchors['train'])/scale
    rows, annual, weights, replay_errors = [], [], {}, []
    teacher = joblib.load(sources['teacher_weight']); column = int(np.flatnonzero(PENALTIES == .01)[0])
    weights[sources['teacher_weight']] = sources['teacher_sha256']
    with threadpool_limits(limits=2):
        for mode in ('prior', *MODES):
            x = {split: features(a, states[split][mode]) for split, a in arrays.items()}
            raw = {split: anchors[split]+scale*teacher['model'].predict(teacher['scaler'].transform(x[split]))[:, column]
                   for split in ('validation', 'test')}
            row, years = save_predictions(root / mode / 'fixed_teacher', arrays, anchors, raw,
                sources['teacher_weight'], column, mode, 'fixed_teacher', .01, args.crop, args.origin)
            rows.append(row); annual.extend(years)
            if mode == 'prior':
                continue
            scaler = StandardScaler(); train = scaler.fit_transform(x['train'])
            model = Ridge(alpha=len(train)*PENALTIES, solver='cholesky').fit(train, np.repeat(y[:, None], 5, axis=1))
            weight = root / f'{mode}.joblib'; joblib.dump(dict(model=model, scaler=scaler), weight)
            weights[str(weight)] = sha256(weight); restored = joblib.load(weight)
            raw = {}
            for split in ('validation', 'test'):
                prediction = model.predict(scaler.transform(x[split]))
                replay = restored['model'].predict(restored['scaler'].transform(x[split]))
                np.testing.assert_array_equal(prediction, replay)
                replay_errors.append(float(np.max(np.abs(prediction-replay))))
                raw[split] = anchors[split][:, None]+scale*prediction
            for j, penalty in enumerate(PENALTIES):
                row, years = save_predictions(root / mode / f'refit/penalty_{penalty:g}', arrays, anchors,
                    {split: v[:, j] for split, v in raw.items()}, weight, j, mode, 'refit', penalty, args.crop, args.origin)
                rows.append(row); annual.extend(years)
            print(f'[SENSITIVE READOUT] {args.crop} {args.origin} {mode}', flush=True)
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(annual).to_csv(root / 'per_year.csv', index=False)
    atomic_json(root / 'complete.json', dict(fits=15, fixed_teacher_replacements=4,
        calibrations=len(rows), maximum_replay_error=max(replay_errors), weights_sha256=weights))


if __name__ == '__main__':
    main()
