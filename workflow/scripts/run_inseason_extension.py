"""Fixed-cutoff temporal confirmation, with no new fits or test calibration."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from inseason_extension_data import prepare
from run_inseason_ndvi_reuse import ROOT, CROPS, original_heads, run_root as screen_root
from run_crop_head_signal_match import references, expert_root, source_root
from run_ndvi_tail_replacement import forecast_prefix, yield_prediction
from inseason_ndvi_reuse import RATIOS, tail_mask, mix_trajectory, available_features
from inseason_calendar_audit import corrected_lead_times
from forecast_bridge_state import ForecastState
from run_forecast_bridge_state import run_root as state_root
from crop_signal_history_reference import IDENTITY
from run_crop_signal_screen import yearly_rmse
from run_ndvi_signal_permutation import check_files
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from summarize_forecast_bridge import verify

RESULT = ROOT/'benchmark/results/inseason_ndvi_extension_v1'
PLAN = ROOT/'Paper/task/季中主实验_固定起报位置扩年复核_20260907.md'
CODE = ('inseason_extension_data.py', 'run_inseason_extension.py', 'inseason_calendar_audit.py')


def run_root(crop):
    return RESULT/'pipelines'/crop/'origin_2012/seed_42'


def cached_prediction(file, labels, key, sources):
    with np.load(file) as f:
        for k in IDENTITY:
            np.testing.assert_array_equal(labels[k], f[k])
        prediction = f[key].astype(float)
    if not np.isfinite(prediction).all():
        raise ValueError('Invalid cached historical prediction')
    sources[str(file)] = sha256(file)
    return prediction


def run(crop):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA inference required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    dest = run_root(crop)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest/'complete.json').exists():
            verify(dest)
            return
        started = time.monotonic()
        raw, arrays, indices, encoders, features, support, scale, sources = prepare(crop)
        validation = {k: arrays['validation'][k] for k in IDENTITY}
        heads, epochs, head_sources = original_heads(crop, 2012, validation)
        sources.update(head_sources)
        replay = yield_prediction(heads, features['validation'], crop)
        expected = cached_prediction(source_root(crop, 2012)/'validation_predictions.npz', validation, 'prediction', sources)
        np.testing.assert_array_equal(replay, expected)
        labels = {k: arrays['test'][k] for k in IDENTITY}
        ex = expert_root(crop, 2012)
        ecfg = json.loads((ex/'config.json').read_text())
        if crop == 'soybean':
            base_source, key = Path(ecfg['source_directory']), 'component_prediction'
        else:
            base_source = ROOT/f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_2012/history_mlp/seed_42'
            key = 'prediction'
        base = cached_prediction(base_source/'test_predictions.npz', labels, key, sources)
        with np.load(ex/'validation_labels.npz') as f:
            validation_base = f['history_prediction']
        np.testing.assert_array_equal(cached_prediction(base_source/'validation_predictions.npz', validation, key, sources), validation_base)
        heads = [(model, cfg, arrays['test']['baseline'].astype(float) if branch == 'trend' else base)
                 for (model, cfg, _), (branch, _) in zip(heads, references(crop, 2012))]
        reference_file = screen_root(crop, 2012)/'history_reference.json'
        history_spec = json.loads(reference_file.read_text())
        sources[str(reference_file)] = sha256(reference_file)
        selected = history_spec['selected']
        if sha256(Path(selected['path'])/'validation_predictions.npz') != selected['prediction_sha256']:
            raise ValueError('Frozen history selection changed')
        strong = cached_prediction(Path(selected['path'])/'test_predictions.npz', labels, selected['key'], sources)
        original = yield_prediction(heads, features['test'], crop)
        deployed = .5*(heads[0][2]+heads[1][2]) if crop == 'maize' else base
        take, encoder = indices['test'], encoders['test']
        active, months = raw['relative_valid'][take] > 0, raw['source_month'][take]
        observed = raw['observed_ndvi'][take]
        state = state_root(crop, 'ndvi', 'biid', 2009, 42)
        marker = json.loads((state/'complete.json').read_text())
        check_files(state, marker['files'])
        if marker['full_fit_cutoff'] != 2009 or marker['smoke']:
            raise ValueError('Wrong frozen state model')
        cfg = json.loads((state/'config.json').read_text())
        for name, digest in cfg['code_sha256'].items():
            if sha256(ROOT/'scripts'/name) != digest:
                raise ValueError('State dependency changed')
        stats = json.loads((state/'normalization.json').read_text())
        model = ForecastState('biid').cuda().eval()
        model.load_state_dict(torch.load(state/'model.pt', map_location='cpu', weights_only=True))
        annual = forecast_prefix(model, raw, take, stats, observed, active, active)
        climo = -encoder.encode(np.full_like(annual, scale['mean']), active, True)[:, -18:-6]*scale['std']+scale['mean']
        for file in (state/'complete.json', state/'model.pt', state/'normalization.json', PLAN):
            sources[str(file)] = sha256(file)
        config = dict(crop=crop, seed=42, origin=2012, main_fraction=.1, years=[2013,2014,2015,2016],
            ratios=RATIOS, primary_interface='available_prefix', primary_completion='prefix_forecast',
            code_sha256={n:sha256(ROOT/'scripts'/n) for n in CODE}, sources=sources,
            fitted_until=2009, strong_history=selected, original_expert_epochs=epochs,
            new_fits=0, unchanged_validation_features=True, unchanged_validation_yield=True,
            future_actual_weather_given=True, future_remote_support_replaced=True,
            independent_test=False, primary_cutoff_selected_on_previous_nine_years=True)
        atomic_json(dest/'config.json', config)
        np.savez_compressed(dest/'labels.npz', **labels, active=active, source_month=months,
                            strong=strong, base=deployed, observed=original)
        reference = {k:yearly_rmse(labels['target'], p, labels['year']) for k,p in
                     dict(strong=strong, base=deployed, observed=original).items()}
        rows, files, prefix_cache = [], ['config.json', 'labels.npz'], {}
        for ratio in RATIOS:
            tail = tail_mask(active, ratio)
            leads = corrected_lead_times(labels['year'], months, active, tail)
            file = f'timeline_{round(ratio*100):03d}.npz'
            np.savez_compressed(dest/file, tail=tail, **leads)
            files.append(file)
            key = tuple(np.unique(np.column_stack((active.sum(1), tail.sum(1))), axis=0).ravel())
            if ratio in (0.,1.):
                prefix = annual
            elif key in prefix_cache:
                prefix = prefix_cache[key]
            else:
                prefix = forecast_prefix(model, raw, take, stats, observed, tail, active)
                prefix_cache[key] = prefix
            for mode, forecast in (('prefix_forecast', prefix), ('annual_forecast', annual), ('climatology', climo)):
                mixed = mix_trajectory(observed, forecast, tail)
                encoded = encoder.encode(mixed, tail, True)
                x = available_features(features['test'], encoded, tail, support)
                prediction = yield_prediction(heads, x, crop)
                if ratio == 0:
                    np.testing.assert_array_equal(prediction, original)
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite extension prediction')
                file = f'available_prefix_{mode}_{round(ratio*100):03d}.npz'
                np.savez_compressed(dest/file, prediction=prediction, mixed_ndvi=mixed)
                files.append(file)
                scores = yearly_rmse(labels['target'], prediction, labels['year'])
                for year, rmse in scores.items():
                    ix = labels['year'] == int(year)
                    valid = tail[ix] & np.isfinite(observed[ix])
                    errors = (forecast[ix]-observed[ix])[valid]**2
                    row = dict(crop=crop, origin=2012, seed=42, year=int(year), interface='available_prefix',
                        mode=mode, requested_fraction=ratio, rmse=rmse, samples=int(ix.sum()),
                        actual_fraction=float((tail[ix].sum(1)/active[ix].sum(1)).mean()),
                        mean_remaining_slots=float(tail[ix].sum(1).mean()),
                        mean_lead_days=float(leads['lead_days'][ix].mean()),
                        median_lead_days=float(np.median(leads['lead_days'][ix])),
                        future_ndvi_rmse=float(np.sqrt(errors.mean())) if len(errors) else np.nan)
                    for ref, values in reference.items():
                        row[f'{ref}_rmse'] = values[year]
                        row[f'gain_{ref}'] = 100*(1-rmse/values[year])
                    rows.append(row)
            print(f'[EXTENSION] {crop} ratio={ratio:.1f}', flush=True)
        pd.DataFrame(rows).to_csv(dest/'annual.csv', index=False)
        files.append('annual.csv')
        atomic_json(dest/'complete.json', dict(files={n:sha256(dest/n) for n in files},
            elapsed_seconds=time.monotonic()-started, new_fits=0, main_fraction=.1, logical_evaluations=33))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.crop)
