"""Reuse original successful crop heads, including their saved initialization choices."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from forecast_bridge_data import ROOT, CROPS, ORIGINS, load
from forecast_bridge_state import ForecastState
from run_forecast_bridge_state import run_root as state_root
from export_forecast_bridge import run_root as export_root
from crop_signal_screen_data import cache_root
from crop_signal_history_reference import OUT as STRONG, IDENTITY
from run_crop_head_signal_match import references, source_root, expert_root
from run_ndvi_signal_permutation import check_files
from run_ndvi_tail_replacement import forecast_prefix, yield_prediction
from run_crop_signal_screen import yearly_rmse
from summarize_forecast_bridge import verify
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from inseason_ndvi_reuse import (RATIOS, MODES, SUPPORTS, tail_mask, mix_trajectory,
                                TrajectoryEncoder, lead_times, historical_support, available_features)

RESULT = ROOT/'benchmark/results/inseason_ndvi_reuse_v1'
CODE = ('inseason_ndvi_reuse.py', 'run_inseason_ndvi_reuse.py', 'ndvi_tail_replacement.py',
        'run_ndvi_tail_replacement.py', 'run_crop_head_signal_match.py', 'run_ndvi_signal_permutation.py',
        'forecast_bridge_state.py', 'token_retention_state.py', 'dual_remote_state.py',
        'observed_remote_anomaly.py', 'observed_remote_benchmark.py')


def run_root(crop, origin):
    return RESULT/'pipelines'/crop/f'origin_{origin}/seed_42'


def original_heads(crop, origin, labels):
    ex = expert_root(crop, origin)
    audit = json.loads((ex/'audit.json').read_text())
    check_files(ex, audit['files'])
    if not audit['full_train_validation_replay']:
        raise ValueError('Historical cache replay is missing')
    ecfg = json.loads((ex/'config.json').read_text())
    anchor = ROOT/f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_{origin}/history_mlp/seed_42'
    epochs = dict(historical_mlp=json.loads((anchor/'metrics.json').read_text())['selected_epoch'])
    original_weight = (Path(ecfg['source_directory'])/'model_best.pt' if crop == 'soybean'
                       else anchor/'model_best.pt')
    if sha256(original_weight) != ecfg['weight_sha256']:
        raise ValueError('Historical expert weight changed')
    if crop == 'soybean':
        epochs['tabm'] = json.loads((original_weight.parent/'metrics.json').read_text())['selected_epoch']
    files = {str(p): sha256(p) for p in (ex/'audit.json', ex/'config.json', original_weight)}
    heads = []
    for branch, parent in references(crop, origin):
        check_files(parent, json.loads((parent/'audit.json').read_text())['files'])
        cfg = json.loads((parent/'config.json').read_text())
        model = joblib.load(parent/'model.joblib')
        if model.booster_.current_iteration() < 1:
            raise ValueError('Missing fitted original yield tree')
        file = cache_root(crop, origin)/'validation_labels.npz' if branch == 'trend' else ex/'validation_labels.npz'
        with np.load(file) as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(labels[k], f[k])
            base = f['baseline' if branch == 'trend' else 'history_prediction'].astype(float)
        heads.append((model, cfg, base))
        for p in (parent/'model.joblib', parent/'config.json', file):
            files[str(p)] = sha256(p)
    return heads, epochs, files


def strong_reference(crop, origin, labels):
    """Keep all existing history candidates, including epoch zero, without refitting."""
    registered = ROOT/f'benchmark/results/matched_joint_world_v1/history/{crop}/origin_{origin}/selection.json'
    selection = json.loads(registered.read_text())
    paths = {x['path'] for x in selection['candidates']}
    for rejected in selection['rejected']:
        path = Path(rejected['path'])
        if (path/'metrics.json').exists() and (path/'validation_predictions.npz').exists():
            paths.add(str(path))
    candidates, chosen = [], None
    for path in sorted(paths):
        directory = Path(path)
        file = directory/'validation_predictions.npz'
        with np.load(file) as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(labels[k], f[k])
            key = 'component_prediction' if 'component_prediction' in f.files else 'prediction'
            prediction = f[key].astype(float)
        score = float(np.mean(list(yearly_rmse(labels['target'], prediction, labels['year']).values())))
        metrics = json.loads((directory/'metrics.json').read_text()) if (directory/'metrics.json').exists() else {}
        record = dict(path=path, score=score, selected_epoch=metrics.get('selected_epoch'),
                      key=key, prediction_sha256=sha256(file))
        candidates.append(record)
        if chosen is None or score < chosen[0]['score']:
            chosen = (record, prediction)
    candidates.sort(key=lambda x: (x['score'], x['path']))
    return chosen[1], dict(selected=chosen[0], candidates=candidates,
        epoch_zero_candidates_allowed=True, prediction_calibration=False,
        selection='Existing development mean annual RMSE, fixed before temporal replacement',
        independent_test=False)


def run(crop, origin):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for prefix-state inference')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    dest = run_root(crop, origin)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest/'complete.json').exists():
            verify(dest)
            return
        started = time.monotonic()
        raw, _ = load(crop)
        fit = np.flatnonzero(raw['year'] <= origin-3)
        take = np.flatnonzero((raw['year'] > origin-3) & (raw['year'] <= origin))
        labels = {k: raw[k][take] for k in IDENTITY}
        active, months = raw['relative_valid'][take] > 0, raw['source_month'][take]
        if np.any(active.sum(1) == 0):
            raise ValueError('Empty activity calendar')
        for row in np.unique(np.concatenate((active, months), 1), axis=0):
            if np.any(np.diff(row[12:][row[:12].astype(bool)]) <= 0):
                raise ValueError('Unordered activity months')
        observed = raw['observed_ndvi'][take]
        export = export_root(crop, origin, 42)
        verify(export)
        with np.load(export/'predictions.npz') as f:
            ix = np.flatnonzero(f['year'] > origin-3)
            for k in IDENTITY:
                np.testing.assert_array_equal(labels[k], f[k][ix])
            annual_forecast = f['predicted_state'][ix]
        state = state_root(crop, 'ndvi', 'biid', origin-3, 42)
        state_marker = json.loads((state/'complete.json').read_text())
        check_files(state, state_marker['files'])
        if state_marker['full_fit_cutoff'] != origin-3 or state_marker['smoke']:
            raise ValueError('Wrong pretrained state cutoff')
        cfg_state = json.loads((state/'config.json').read_text())
        for name, digest in cfg_state['code_sha256'].items():
            if sha256(ROOT/'scripts'/name) != digest:
                raise ValueError('State implementation changed')
        stats = json.loads((state/'normalization.json').read_text())
        model = ForecastState('biid').cuda().eval()
        model.load_state_dict(torch.load(state/'model.pt', map_location='cpu', weights_only=True))
        replay = forecast_prefix(model, raw, take, stats, observed, active, active)
        np.testing.assert_allclose(replay, annual_forecast, atol=1e-6, rtol=0)
        replay_error = float(np.max(np.abs(replay-annual_forecast)))
        cache = cache_root(crop, origin)
        cm = json.loads((cache/'manifest.json').read_text())
        names = [f'{k}_validation.npy' for k in ('history', 'metadata', 'weather', 'ndvi')]
        check_files(cache, {n:cm['files'][n] for n in (*names, 'metadata_train.npy', 'train_labels.npz')})
        with np.load(cache/'train_labels.npz') as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(f[k], raw[k][fit])
        x = np.concatenate([np.load(cache/n) for n in names], 1)
        scale = cm['spec']['upstream']['upstream']['ndvi_normalization']
        encoder = TrajectoryEncoder(raw, fit, take, scale)
        np.testing.assert_array_equal(encoder.encode(observed), x[:, -36:])
        support = historical_support(raw, fit, take, np.load(cache/'metadata_train.npy', mmap_mode='r'))
        heads, epoch_record, sources = original_heads(crop, origin, labels)
        original = yield_prediction(heads, x, crop)
        original_file = source_root(crop, origin)/'validation_predictions.npz'
        with np.load(original_file) as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(labels[k], f[k])
            np.testing.assert_array_equal(original, f['prediction'])
        climo = -encoder.encode(np.full_like(annual_forecast, scale['mean']), active, True)[:, -18:-6]*scale['std']+scale['mean']
        strong, history_audit = strong_reference(crop, origin, labels)
        atomic_json(dest/'history_reference.json', history_audit)
        legacy_file = STRONG/f'{crop}_{origin}_validation.npz'
        legacy_manifest = json.loads((STRONG/'manifest.json').read_text())
        record = next(r for r in legacy_manifest['records'] if r['crop'] == crop and r['origin'] == origin)
        if sha256(legacy_file) != record['output_sha256']:
            raise ValueError('Legacy strong reference changed')
        with np.load(legacy_file) as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(labels[k], f[k])
            legacy = f['prediction']
        deployed_base = np.mean([b for _, _, b in heads], 0) if crop == 'maize' else heads[0][2]
        reference = dict(strong=strong, legacy=legacy, base=deployed_base, observed=original)
        np.savez_compressed(dest/'labels.npz', **labels, active=active, source_month=months, **reference)
        for file in (state/'complete.json', state/'model.pt', export/'complete.json', cache/'manifest.json',
                     original_file, legacy_file):
            sources[str(file)] = sha256(file)
        for candidate in history_audit['candidates']:
            sources[str(Path(candidate['path'])/'validation_predictions.npz')] = candidate['prediction_sha256']
        spec = dict(crop=crop, origin=origin, seed=42, ratios=RATIOS, modes=MODES, support_interfaces=SUPPORTS,
            original_expert_epochs=epoch_record, epoch_zero_policy='Preserve exact original assets as requested',
            heads_retrained=False, state_retrained=False, original_observed_replayed=True,
            original_observed_rmse=yearly_rmse(labels['target'], original, labels['year']),
            sources=sources, code_sha256={n:sha256(ROOT/'scripts'/n) for n in CODE},
            state_replay_max_error=replay_error, future_actual_weather_supplied=True,
            future_remote_values_enter_prefix=False,
            future_remote_support_retained_only_in_legacy_interface=True,
            availability_imputation='Train-only grid/month average of six remote support fields; predicted suffix is valid',
            lead_time='Days before final supplied active-month end; not validated harvest dates or product release latency',
            calendar='Same-year ascending active slots, with possible gaps; no cross-year repair',
            evaluation_years=np.unique(labels['year']).tolist(), independent_test=False,
            operational_inseason_forecast=False, dimensions=x.shape[1], new_fits=0)
        atomic_json(dest/'config.json', spec)
        ref_rmse = {k:yearly_rmse(labels['target'], v, labels['year']) for k,v in reference.items()}
        rows, ratio_records = [], []
        saved = ['labels.npz', 'config.json', 'history_reference.json']
        prefix_cache = {}
        for ratio in RATIOS:
            tail = tail_mask(active, ratio)
            counts, total = tail.sum(1), active.sum(1)
            leads = lead_times(labels['year'], months, active, tail)
            timeline_name = f'timeline_{round(100*ratio):03d}.npz'
            np.savez_compressed(dest/timeline_name, tail=tail, **leads)
            saved.append(timeline_name)
            key = tuple(np.unique(np.column_stack((total, counts)), axis=0).ravel().tolist())
            if ratio in (0., 1.):
                prefix = annual_forecast
            elif key in prefix_cache:
                prefix = prefix_cache[key]
            else:
                prefix = forecast_prefix(model, raw, take, stats, observed, tail, active)
                prefix_cache[key] = prefix
            for mode, forecast in [('annual_forecast', annual_forecast), ('prefix_forecast', prefix), ('climatology', climo)]:
                mixed = mix_trajectory(observed, forecast, tail)
                name = f'trajectory_{mode}_{round(100*ratio):03d}.npz'
                np.savez_compressed(dest/name, mixed_ndvi=mixed)
                saved.append(name)
                for interface in SUPPORTS:
                    available = interface == 'available_prefix'
                    encoded = encoder.encode(mixed, tail, available)
                    features = available_features(x, encoded, tail, support) if available else np.concatenate((x[:, :-36], encoded), 1)
                    prediction = yield_prediction(heads, features, crop)
                    if ratio == 0:
                        np.testing.assert_array_equal(prediction, original)
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite yield output')
                    name = f'{interface}_{mode}_{round(100*ratio):03d}.npz'
                    np.savez_compressed(dest/name, prediction=prediction)
                    saved.append(name)
                    scores = yearly_rmse(labels['target'], prediction, labels['year'])
                    for year, rmse in scores.items():
                        selected = labels['year'] == int(year)
                        valid = tail[selected] & np.isfinite(observed[selected])
                        state_error = ((forecast[selected]-observed[selected])[valid])**2
                        record = dict(crop=crop, origin=origin, year=int(year), seed=42, mode=mode, interface=interface,
                            requested_fraction=ratio, rmse=rmse, samples=int(selected.sum()),
                            replaced_slots=int(counts[selected].sum()), active_slots=int(total[selected].sum()),
                            actual_fraction=float(np.mean(counts[selected]/total[selected])),
                            mean_lead_days=float(leads['lead_days'][selected].mean()),
                            median_lead_days=float(np.median(leads['lead_days'][selected])),
                            mean_remaining_slots=float(counts[selected].mean()),
                            future_ndvi_rmse=float(np.sqrt(state_error.mean())) if len(state_error) else np.nan)
                        for ref, values in ref_rmse.items():
                            record[f'{ref}_rmse'] = values[year]
                            record[f'gain_{ref}'] = 100*(1-rmse/values[year])
                        rows.append(record)
            ratio_records.append(dict(requested_fraction=ratio, samples=len(total),
                actual_fraction_mean=float(np.mean(counts/total)), lead_days_mean=float(leads['lead_days'].mean()),
                lead_days_quantiles=np.quantile(leads['lead_days'], [0, .1, .5, .9, 1]).tolist(),
                remaining_slots_mean=float(counts.mean()),
                count_pairs=[dict(active=int(m), replaced=int(n), samples=int(num))
                    for (m,n),num in zip(*np.unique(np.column_stack((total,counts)), axis=0, return_counts=True))]))
            pd.DataFrame(rows).to_csv(dest/'annual.csv', index=False)
            atomic_json(dest/'ratio_counts.json', ratio_records)
            print(f'[INSEASON] {crop} {origin} tail={ratio:.0%} actual={ratio_records[-1]["actual_fraction_mean"]:.1%}', flush=True)
        for interface in SUPPORTS:
            with np.load(dest/f'{interface}_annual_forecast_100.npz') as a, np.load(dest/f'{interface}_prefix_forecast_100.npz') as b:
                np.testing.assert_array_equal(a['prediction'], b['prediction'])
        for path, digest in sources.items():
            if sha256(Path(path)) != digest:
                raise ValueError('Source asset changed during temporal sweep')
        saved += ['annual.csv', 'ratio_counts.json']
        atomic_json(dest/'complete.json', dict(seconds=time.monotonic()-started, cases=len(MODES)*len(SUPPORTS)*len(RATIOS),
            new_fits=0, original_observed_replayed=True, original_expert_epochs=epoch_record,
            prefix_has_no_future_remote_values=True, state_replay_max_error=replay_error,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            files={name:sha256(dest/name) for name in saved}))
        print(f'[INSEASON COMPLETE] {crop} {origin}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', choices=ORIGINS, type=int, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.crop, args.origin)
