"""Diagnostic-only trajectory substitution into unchanged observed-support heads."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from forecast_bridge_data import ROOT, CROPS, ORIGINS, load
from export_forecast_bridge import run_root as export_root
from crop_signal_screen_data import cache_root
from observed_remote_anomaly import seasonal_anomalies
from observed_remote_benchmark import trajectory_features
from run_crop_head_signal_match import references, source_root, expert_root
from run_ndvi_signal_permutation import compose, check_files
from run_crop_signal_screen import yearly_rmse
from summarize_forecast_bridge import verify
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/forecast_state_bridge_v1/frozen_replacement'


def encode(raw, fit, take, physical, scale):
    arrays = {}
    for split, rows, value in (('train', fit, raw['observed_ndvi'][fit]), ('validation', take, physical)):
        active = raw['relative_valid'][rows] > 0
        # This intentionally retains target-observation support for a paired
        # frozen-head diagnostic, not for operational forecast inputs.
        valid = np.isfinite(raw['observed_ndvi'][rows]) & active
        observed = np.where(valid, (value-scale['mean'])/scale['std'], 0).astype(np.float32)
        arrays[split] = dict(observed_ndvi=observed, observed_ndvi_valid=valid.astype(np.float32),
            **{k: raw[k][rows] for k in ('row', 'col', 'source_month', 'relative_valid')})
    a = arrays['validation']
    mask = a['observed_ndvi_valid'] > 0
    anomalies = seasonal_anomalies(arrays, 'ndvi')['validation']
    return np.concatenate((trajectory_features(a['observed_ndvi'], mask), trajectory_features(anomalies, mask)), 1)


def run(crop, origin):
    dest = OUT / 'pipelines' / crop / f'origin_{origin}/seed_42'
    if (dest / 'complete.json').exists():
        verify(dest)
        return
    raw, _ = load(crop)
    fit = np.flatnonzero(raw['year'] <= origin-3)
    take = np.flatnonzero((raw['year'] > origin-3) & (raw['year'] <= origin))
    src = export_root(crop, origin, 42)
    verify(src)
    with np.load(src / 'predictions.npz') as f:
        rows = np.flatnonzero(f['year'] > origin-3)
        np.testing.assert_array_equal(f['source_indices'][rows], raw['source_indices'][take])
        predicted = f['predicted_state'][rows]
    cache = cache_root(crop, origin)
    manifest = json.loads((cache / 'manifest.json').read_text())
    names = [f'{k}_validation.npy' for k in ('history', 'metadata', 'weather', 'ndvi')]
    check_files(cache, {n: manifest['files'][n] for n in names})
    x = np.concatenate([np.load(cache / n) for n in names], 1)
    scale = manifest['spec']['upstream']['upstream']['ndvi_normalization']
    observed = encode(raw, fit, take, raw['observed_ndvi'][take], scale)
    np.testing.assert_array_equal(observed, x[:, -36:])
    zero_normalized = np.full_like(predicted, scale['mean'])
    climo = -encode(raw, fit, take, zero_normalized, scale)[:, -18:-6]*scale['std']+scale['mean']
    previous = np.where(np.isfinite(raw['previous_ndvi'][take]), raw['previous_ndvi'][take], climo)
    files, source_files, records = {}, {str(src / 'complete.json'): sha256(src / 'complete.json')}, []
    dest.mkdir(parents=True, exist_ok=True)
    labels = {k: raw[k][take] for k in ('target', 'source_indices', 'year', 'row', 'col')}
    branches = []
    for name, parent in references(crop, origin):
        check_files(parent, json.loads((parent / 'audit.json').read_text())['files'])
        cfg = json.loads((parent / 'config.json').read_text())
        file = cache / 'validation_labels.npz' if name == 'trend' else expert_root(crop, origin) / 'validation_labels.npz'
        key = 'baseline' if name == 'trend' else 'history_prediction'
        with np.load(file) as f:
            for k in labels:
                np.testing.assert_array_equal(labels[k], f[k])
            base = f[key].astype(float)
        branches.append((joblib.load(parent / 'model.joblib'), cfg, base))
        for p in (parent / 'model.joblib', parent / 'config.json', file):
            source_files[str(p)] = sha256(p)
    for name, physical in (('observed', raw['observed_ndvi'][take]), ('predicted', predicted),
            ('previous', previous), ('climatology', climo)):
        features = np.concatenate((x[:, :-36], encode(raw, fit, take, physical, scale)), 1)
        pieces = [compose(cfg, base, model.predict(features)) for model, cfg, base in branches]
        p = .5*pieces[0]+.5*pieces[1] if crop == 'maize' else pieces[0]
        if name == 'observed':
            with np.load(source_root(crop, origin) / 'validation_predictions.npz') as f:
                np.testing.assert_array_equal(f['prediction'], p)
        np.savez_compressed(dest / f'{name}_predictions.npz', prediction=p, **labels)
        files[f'{name}_predictions.npz'] = sha256(dest / f'{name}_predictions.npz')
        for year, error in yearly_rmse(labels['target'], p, labels['year']).items():
            records.append(dict(crop=crop, origin=origin, year=int(year), condition=name, rmse=error))
    pd.DataFrame(records).to_csv(dest / 'annual.csv', index=False)
    atomic_json(dest / 'config.json', dict(crop=crop, origin=origin, seed=42, source_files=source_files,
        target_quality_retained=True, operational_forecast=False, refits=0,
        purpose='Frozen observed-head substitution diagnostic, not new-interface main results',
        normalization=scale, code_sha256=sha256(ROOT / 'scripts/run_forecast_bridge_frozen_replacement.py')))
    for name in ('annual.csv', 'config.json'):
        files[name] = sha256(dest / name)
    atomic_json(dest / 'complete.json', dict(files=files, original_observed_prediction_replayed=True,
        original_observed_features_replayed=True, evaluation_arrays_loaded=False))
    print(f'[FROZEN REPLACEMENT] {crop} {origin}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        for origin in ORIGINS:
            for crop in CROPS:
                run(crop, origin)
