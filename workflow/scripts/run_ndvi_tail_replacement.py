"""Freeze the successful observed heads and replace chronological NDVI tails."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from forecast_bridge_data import ROOT, CROPS, ORIGINS, load
from forecast_bridge_state import ForecastState, batch_arrays
from run_forecast_bridge_state import run_root as state_root, tensors
from export_forecast_bridge import run_root as export_root
from crop_signal_screen_data import cache_root, load as signal_load
from crop_signal_history_reference import OUT as STRONG, IDENTITY
from run_crop_head_signal_match import references, source_root, expert_root
from run_forecast_bridge_frozen_replacement import encode
from run_ndvi_signal_permutation import compose, check_files
from run_crop_signal_screen import yearly_rmse, year_weights
from summarize_forecast_bridge import verify
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from ndvi_tail_replacement import RATIOS, MODES, tail_mask, mix_trajectory, prefix_values, prefix_rollout

RESULT = ROOT/'benchmark/results/ndvi_tail_replacement_v1'
CODE = ('ndvi_tail_replacement.py', 'run_ndvi_tail_replacement.py',
        'run_forecast_bridge_frozen_replacement.py', 'run_crop_head_signal_match.py',
        'forecast_bridge_state.py', 'token_retention_state.py', 'dual_remote_state.py')


def run_root(crop, origin):
    return RESULT/'pipelines'/crop/f'origin_{origin}/seed_42'


def check_mlp(crop, origin):
    src = ROOT/f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_{origin}/history_mlp/seed_42'
    cfg = json.loads((src/'config.json').read_text())
    metric = json.loads((src/'metrics.json').read_text())
    if metric['selected_epoch'] < 1:
        raise ValueError('Untrained original historical MLP is forbidden')
    for name, digest in cfg['code_hashes'].items():
        if sha256(ROOT/'scripts'/name) != digest:
            raise ValueError('Original MLP implementation changed')
    return dict(source=str(src), selected_epoch=metric['selected_epoch'],
                weight_sha256=sha256(src/'model_best.pt'))


def prepare_heads(crop, origin, dest, labels):
    """Only the two zero-epoch soybean adapters are removed and their trees refit."""
    historical = check_mlp(crop, origin)
    ex = expert_root(crop, origin)
    ex_audit = json.loads((ex/'audit.json').read_text())
    check_files(ex, ex_audit['files'])
    ex_cfg = json.loads((ex/'config.json').read_text())
    claimed_mlp = (ex_cfg['anchor']['weight_sha256'] if crop == 'soybean'
                   else ex_cfg['weight_sha256'])
    if claimed_mlp != historical['weight_sha256']:
        raise ValueError('Expert cache refers to a different historical MLP')
    repair = False
    if crop == 'soybean':
        tabm_src = Path(ex_cfg['source_directory'])
        tabm_metric = json.loads((tabm_src/'metrics.json').read_text())
        repair = tabm_metric['selected_epoch'] == 0
        if sha256(tabm_src/'model_best.pt') != ex_cfg['weight_sha256']:
            raise ValueError('Original TabM asset changed')
        historical.update(original_tabm_epoch=tabm_metric['selected_epoch'], removed_untrained_tabm=repair)
    historical['expert_cache_config_sha256'] = sha256(ex/'config.json')
    pieces, sources = [], {str(ex/'audit.json'):sha256(ex/'audit.json')}
    for branch, parent in references(crop, origin):
        audit = json.loads((parent/'audit.json').read_text())
        check_files(parent, audit['files'])
        cfg = json.loads((parent/'config.json').read_text())
        model = joblib.load(parent/'model.joblib')
        if model.booster_.current_iteration() < 1:
            raise ValueError('Unfitted observed yield tree')
        file = cache_root(crop, origin)/'validation_labels.npz' if branch == 'trend' else ex/'validation_labels.npz'
        key = 'baseline' if branch == 'trend' else 'mlp_prediction' if repair else 'history_prediction'
        with np.load(file) as f:
            for name in IDENTITY:
                np.testing.assert_array_equal(labels[name], f[name])
            base = f[key].astype(float)
        for p in (parent/'model.joblib', parent/'config.json', file):
            sources[str(p)] = sha256(p)
        if repair:
            repair_dir = dest/'trained_mlp_repair'
            repair_dir.mkdir(parents=True, exist_ok=True)
            params = model.get_params()
            params.update(n_estimators=int(model.best_iteration_ or model.booster_.current_iteration()), n_jobs=4)
            specification = dict(crop=crop, origin=origin, seed=42, parameters=params,
                removed_untrained_tabm=True, original_tabm_epoch=0,
                preserved_trained_mlp=historical, original_tree_sha256=sha256(parent/'model.joblib'),
                selection='Reuse original tree count; no new validation early stopping or ratio-dependent fitting')
            spec_path = repair_dir/'registration.json'
            if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
                raise ValueError('Soybean repair registration changed')
            atomic_json(spec_path, specification)
            if (repair_dir/'complete.json').exists():
                verify(repair_dir)
                model = joblib.load(repair_dir/'model.joblib')
                cfg = json.loads((repair_dir/'config.json').read_text())
            else:
                x, a, _ = signal_load(crop, origin, 'ndvi')
                with np.load(ex/'train_labels.npz') as f:
                    for name in IDENTITY:
                        np.testing.assert_array_equal(a['train'][name], f[name])
                    base_fit = f['mlp_prediction'].astype(float)
                residual = a['train']['target'].astype(float)-base_fit
                center, scale = float(residual.mean()), max(float(residual.std()), 1e-6)
                model = lgb.LGBMRegressor(**params)
                model.fit(x['train'], (residual-center)/scale, sample_weight=year_weights(a['train']['year']))
                cfg = dict(head='trained_mlp_lightgbm_repair', normalization=dict(center=center, scale=scale),
                           parameters=params, frozen_across_all_replacement_ratios=True)
                joblib.dump(model, repair_dir/'model.joblib')
                atomic_json(repair_dir/'config.json', cfg)
                np.testing.assert_array_equal(model.predict(x['validation']),
                                               joblib.load(repair_dir/'model.joblib').predict(x['validation']))
                atomic_json(repair_dir/'complete.json', dict(new_tree_fits=1, trained_mlp_changed=False,
                    removed_zero_epoch_network=True, replay_error=0.,
                    files={n:sha256(repair_dir/n) for n in ('registration.json','model.joblib','config.json')}))
            sources[str(repair_dir/'complete.json')] = sha256(repair_dir/'complete.json')
        pieces.append((model, cfg, base))
    return pieces, historical, sources, repair


def yield_prediction(heads, features, crop):
    values = [compose(cfg, base, model.predict(features)) for model, cfg, base in heads]
    return .5*values[0]+.5*values[1] if crop == 'maize' else values[0]


def forecast_prefix(model, raw, take, stats, observed, tail, active):
    prefix, known = prefix_values(observed, active, tail, stats['ndvi']['mean'], stats['ndvi']['std'])
    outputs = []
    for start in range(0, len(take), 256):
        end = min(start+256, len(take))
        batch = tensors(batch_arrays(raw, take[start:end], stats, 'ndvi'), 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = prefix_rollout(model, batch, torch.as_tensor(prefix[start:end], device='cuda'),
                               torch.as_tensor(known[start:end], device='cuda'))
        outputs.append(p.float().cpu().numpy())
    return np.concatenate(outputs)*stats['ndvi']['std']+stats['ndvi']['mean']


def run(crop, origin):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for frozen prefix inference')
    torch.backends.cuda.matmul.allow_tf32 = False
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
        labels = {k:raw[k][take] for k in IDENTITY}
        active = raw['relative_valid'][take] > 0
        months = raw['source_month'][take]
        for row in np.unique(np.concatenate((active, months), axis=1), axis=0):
            act, mon = row[:12].astype(bool), row[12:]
            if np.any(np.diff(mon[act]) <= 0):
                raise ValueError('Expected ascending unique active calendar months')
        observed = raw['observed_ndvi'][take]
        export = export_root(crop, origin, 42)
        verify(export)
        with np.load(export/'predictions.npz') as f:
            ix = np.flatnonzero(f['year'] > origin-3)
            for key in IDENTITY:
                np.testing.assert_array_equal(labels[key], f[key][ix])
            annual = f['predicted_state'][ix]
        state = state_root(crop, 'ndvi', 'biid', origin-3, 42)
        marker = json.loads((state/'complete.json').read_text())
        if marker['smoke'] or marker['selected_epochs'] < 1 or marker['full_fit_cutoff'] != origin-3:
            raise ValueError('Invalid or untrained vegetation model')
        check_files(state, marker['files'])
        state_cfg = json.loads((state/'config.json').read_text())
        for name, digest in state_cfg['code_sha256'].items():
            if sha256(ROOT/'scripts'/name) != digest:
                raise ValueError('State source code changed')
        stats = json.loads((state/'normalization.json').read_text())
        model = ForecastState('biid').cuda().eval()
        model.load_state_dict(torch.load(state/'model.pt', map_location='cpu', weights_only=True))
        reproduced = forecast_prefix(model, raw, take, stats, observed, active, active)
        np.testing.assert_allclose(reproduced, annual, rtol=0, atol=1e-6)
        replay_error = float(np.max(np.abs(reproduced-annual)))
        cache = cache_root(crop, origin)
        cache_meta = json.loads((cache/'manifest.json').read_text())
        files_x = [f'{k}_validation.npy' for k in ('history','metadata','weather','ndvi')]
        check_files(cache, {n:cache_meta['files'][n] for n in files_x})
        x = np.concatenate([np.load(cache/n) for n in files_x], 1)
        scale = cache_meta['spec']['upstream']['upstream']['ndvi_normalization']
        np.testing.assert_array_equal(encode(raw, fit, take, observed, scale), x[:, -36:])
        heads, history_audit, source_files, repaired = prepare_heads(crop, origin, dest, labels)
        original_score = yield_prediction(heads, x, crop)
        if not repaired:
            with np.load(source_root(crop, origin)/'validation_predictions.npz') as f:
                np.testing.assert_array_equal(original_score, f['prediction'])
        zero = np.full_like(annual, scale['mean'])
        climo = -encode(raw, fit, take, zero, scale)[:, -18:-6]*scale['std']+scale['mean']
        reference_file = STRONG/f'{crop}_{origin}_validation.npz'
        reference_manifest = json.loads((STRONG/'manifest.json').read_text())
        ref_meta = next(r for r in reference_manifest['records'] if r['crop'] == crop and r['origin'] == origin)
        if sha256(reference_file) != ref_meta['output_sha256']:
            raise ValueError('Frozen strong history reference changed')
        with np.load(reference_file) as f:
            for key in IDENTITY:
                np.testing.assert_array_equal(labels[key], f[key])
            strong = f['prediction']
            raw_strong = f['raw_prediction']
        ref_metric_path = Path(ref_meta['directory'])/'metrics.json'
        ref_metrics = json.loads(ref_metric_path.read_text()) if ref_metric_path.exists() else {}
        ref_epoch = ref_metrics.get('selected_epoch')
        base_values = [b for _, _, b in heads]
        deployed_base = .5*base_values[0]+.5*base_values[1] if crop == 'maize' else base_values[0]
        np.savez_compressed(dest/'labels.npz', **labels, active=active, source_month=months,
                            legacy_strong=strong, legacy_raw=raw_strong, deployed_base=deployed_base)
        source_files.update({str(state/'complete.json'):sha256(state/'complete.json'),
                             str(export/'complete.json'):sha256(export/'complete.json'),
                             str(reference_file):sha256(reference_file)})
        spec = dict(crop=crop, origin=origin, seed=42, ratios=RATIOS, modes=MODES,
            history_audit=history_audit, soybean_adapter_repaired=repaired,
            source_files=source_files, code_sha256={n:sha256(ROOT/'scripts'/n) for n in CODE},
            target_quality_retained=True, future_actual_weather_retained=True,
            calendar='Annual ascending active-month sequence; not repaired harvest-season chronology',
            operational_inseason_forecast=False, state_retrained=False,
            prefix_update='Existing feedback only; no new parameters, no prefix-specific fine-tuning',
            legacy_reference_selected_epoch=ref_epoch, legacy_reference_has_epoch0=(ref_epoch == 0),
            legacy_reference_is_input=False, legacy_reference_used_development_selection_and_loo_calibration=True,
            state_replay_max_error=replay_error, dimensions=x.shape[1], evaluation_arrays_loaded=False)
        atomic_json(dest/'config.json', spec)
        references_rmse = {key:yearly_rmse(labels['target'], pred, labels['year']) for key,pred in
                           [('strong',strong),('raw_strong',raw_strong),('base',deployed_base),('observed',original_score)]}
        records, ratio_records, saved = [], [], ['labels.npz','config.json']
        cached_prefix = {}
        for ratio in RATIOS:
            tail = tail_mask(active, ratio)
            counts, total = tail.sum(1), active.sum(1)
            key = tuple(np.unique(np.column_stack((total, counts)), axis=0).ravel().tolist())
            if ratio in (0.,1.):
                prefix = annual
            elif key in cached_prefix:
                prefix = cached_prefix[key]
            else:
                prefix = forecast_prefix(model, raw, take, stats, observed, tail, active)
                cached_prefix[key] = prefix
            for mode, forecast in [('annual_forecast',annual),('prefix_forecast',prefix),('climatology',climo)]:
                mixed = mix_trajectory(observed, forecast, tail)
                features = np.concatenate((x[:, :-36], encode(raw, fit, take, mixed, scale)), 1)
                prediction = yield_prediction(heads, features, crop)
                if ratio == 0:
                    np.testing.assert_array_equal(prediction, original_score)
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite yield predictions')
                name = f'{mode}_tail_{round(100*ratio):03d}.npz'
                np.savez_compressed(dest/name, prediction=prediction, mixed_ndvi=mixed, tail=tail)
                saved.append(name)
                scores = yearly_rmse(labels['target'], prediction, labels['year'])
                for year, rmse in scores.items():
                    rows = labels['year'] == int(year)
                    valid_future = tail[rows] & np.isfinite(observed[rows])
                    state_error = ((forecast[rows]-observed[rows])[valid_future])**2
                    record = dict(crop=crop, origin=origin, year=int(year), seed=42, mode=mode,
                        requested_fraction=ratio, rmse=rmse, replaced_slots=int(counts[rows].sum()),
                        active_slots=int(total[rows].sum()), samples=int(rows.sum()),
                        actual_fraction=float(np.mean(np.divide(counts[rows],total[rows],
                            out=np.zeros_like(counts[rows],dtype=float),where=total[rows]>0))),
                        future_ndvi_rmse=float(np.sqrt(state_error.mean())) if len(state_error) else np.nan)
                    for ref,values in references_rmse.items():
                        record[f'{ref}_rmse'] = values[year]
                        record[f'gain_{ref}'] = 100*(1-rmse/values[year])
                    records.append(record)
            ratio_records.append(dict(requested_fraction=ratio, active_slots_mean=float(total.mean()),
                actual_fraction_mean=float(np.mean(np.divide(counts,total,out=np.zeros_like(counts,dtype=float),where=total>0))),
                empty_active_samples=int((total==0).sum()), count_pairs=[dict(active=int(m),replaced=int(n),samples=int(num))
                    for (m,n),num in zip(*np.unique(np.column_stack((total,counts)),axis=0,return_counts=True))]))
            pd.DataFrame(records).to_csv(dest/'annual.csv',index=False)
            print(f'[TAIL] {crop} {origin} requested={ratio:.0%} actual={ratio_records[-1]["actual_fraction_mean"]:.1%}',flush=True)
        atomic_json(dest/'ratio_counts.json',ratio_records)
        saved += ['annual.csv','ratio_counts.json']
        if sha256(state/'model.pt') != marker['files']['model.pt']:
            raise ValueError('Frozen state weights changed')
        atomic_json(dest/'complete.json',dict(seconds=time.monotonic()-started, cases=len(MODES)*len(RATIOS),
            state_retrained=False, removed_untrained_adapter=repaired,
            terminal_refits=int(repaired), original_observed_replayed=not repaired,
            state_replay_max_error=replay_error, prefix_has_no_future_ndvi_values=True,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            files={n:sha256(dest/n) for n in saved}))
        print(f'[TAIL COMPLETE] {crop} {origin}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--crop',choices=CROPS,required=True)
    parser.add_argument('--origin',type=int,choices=ORIGINS,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.crop,args.origin)
