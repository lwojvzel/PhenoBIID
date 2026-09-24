"""Rebuild regional state normalization/inputs and independently replay both weights."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from cybench_inseason_model_data import ROOT, RESULT, load_identity, state_population, load_seasons
from forecast_bridge_state import ForecastState
from run_cybench_inseason_state import root_for, verify_completed
from run_cybench_inseason_state_queue import jobs, GPUS
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from shared_gpu_queue_extended import run_extended_stage

SCRIPT = Path(__file__).resolve()
OUTPUT = RESULT / 'state_verification.json'


def manual_stats(known, target, product):
    active = known['active_mask']
    weather = np.where(active[..., None] & (known['weather_finite_grid_area_fraction'] > 0)
        & np.isfinite(known['weather']), known['weather'], np.nan).astype(np.float64)[active]
    j = ('ndvi', 'gpp').index(product)
    valid = active & (target['vegetation_support'][..., j] > 0) & np.isfinite(target['vegetation'][..., j])
    values = target['vegetation'][..., j][valid].astype(np.float64)
    return dict(weather_mean=np.nanmean(weather, axis=0).tolist(),
        weather_std=np.maximum(np.nanstd(weather, axis=0), 1e-6).tolist(),
        **{product: dict(mean=float(values.mean()), std=max(float(values.std()), 1e-6))})


def compare_stats(actual, expected, product):
    if set(actual) != set(expected) or set(actual[product]) != {'mean', 'std'}:
        raise ValueError('State normalization schema changed')
    for key in ('weather_mean', 'weather_std'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    for key in ('mean', 'std'):
        np.testing.assert_allclose(actual[product][key], expected[product][key], rtol=1e-12, atol=1e-12)


def manual_batch(known, scale, product):
    active = known['active_mask']
    j = ('ndvi', 'gpp').index(product)
    prior = known['previous_vegetation'][..., j]
    quality = known['previous_vegetation_support'][..., j]
    valid = active & known['previous_valid'][..., j] & (quality > 0) & np.isfinite(prior)
    previous = np.where(valid, (prior-scale[product]['mean'])/scale[product]['std'], 0).astype(np.float32)
    weather = np.where(active[..., None] & (known['weather_finite_grid_area_fraction'] > 0), known['weather'], np.nan)
    weather = (weather-np.asarray(scale['weather_mean'], np.float32))/np.asarray(scale['weather_std'], np.float32)
    c = known['context']
    context = np.column_stack((c[:, 0]/90, c[:, 1]/180, (known['year']-2000)/40, c[:, 2]/366, c[:, 3]/366)).astype(np.float32)
    return dict(weather=np.nan_to_num(weather, nan=0., posinf=0., neginf=0.).astype(np.float32),
        previous=previous, previous_valid=valid.astype(np.float32),
        previous_quality=np.where(valid, quality, 0).astype(np.float32),
        relative_valid=active.astype(np.float32), context=context)


@torch.no_grad()
def forward(weight, batch, stats, product):
    model = ForecastState('biid').cuda()
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
    model.eval()
    pieces = []
    for start in range(0, len(batch['context']), 256):
        inputs = {key: torch.as_tensor(value[start:start+256], device='cuda') for key, value in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(inputs)
        pieces.append(p.float().cpu().numpy())
    del model
    return np.concatenate(pieces)*stats[product]['std']+stats[product]['mean']


def verify_record():
    record = json.loads(OUTPUT.read_text())
    if not record['passed'] or record['models'] != 27 or record['verifier_sha256'] != sha256(SCRIPT):
        raise ValueError('Incomplete or changed regional state audit')
    for path, digest in record['sources'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Regional state audit source changed')
    return record


def worker():
    gate = RESULT / 'state_full_queue_complete.json'
    completion = json.loads(gate.read_text())
    if not completion['passed'] or completion['final_models'] != 27:
        raise ValueError('All 27 registered regional states required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for like-precision independent state replay')
    torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sources = {str(gate): sha256(gate)}
    rows, replayed = [], 0
    for job in jobs():
        record = verify_completed(**job)
        root = root_for(**job)
        ids, history, parts, source = load_identity(job['crop'], job['cutoff'], with_history=False)
        assert history is None
        population = state_population(ids, parts, job['cutoff'])
        ix = population['full_fit']
        known, target = load_seasons(job['crop'], ids, ix, source, maximum_year=job['cutoff'])
        sources.update(source)
        with np.load(root / 'partition_indices.npz') as saved:
            for key in population:
                np.testing.assert_array_equal(saved[key], population[key])
        csv = pd.read_csv(root / 'state_identities.csv', dtype={'country': str, 'adm_id': str})
        pd.testing.assert_frame_equal(csv, ids.iloc[ix].reset_index(drop=True), check_dtype=False)
        fit = np.flatnonzero(known['year'] <= job['cutoff']-2)
        val = np.flatnonzero(known['year'] >= job['cutoff']-1)
        inner = json.loads((root / 'inner_normalization.json').read_text())
        full = json.loads((root / 'normalization.json').read_text())
        compare_stats(inner, manual_stats({k: v[fit] for k, v in known.items()}, {k: v[fit] for k, v in target.items()}, job['product']), job['product'])
        compare_stats(full, manual_stats(known, target, job['product']), job['product'])
        p = forward(root / 'inner_best.pt', manual_batch({k: v[val] for k, v in known.items()}, inner, job['product']), inner, job['product'])
        j = ('ndvi', 'gpp').index(job['product'])
        valid = known['active_mask'][val] & np.isfinite(target['vegetation'][val, :, j]) & (target['vegetation_support'][val, :, j] > 0)
        truth = np.where(valid, target['vegetation'][val, :, j], np.nan).astype(np.float32)
        with np.load(root / 'inner_validation_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['sample_index'], ix[val])
            np.testing.assert_array_equal(saved['target'], truth)
            np.testing.assert_array_equal(saved['prediction'], p)
        annual = [float(np.sqrt(((p-truth)**2)[(known['year'][val] == year)[:, None] & valid].mean()))
                  for year in (job['cutoff']-1, job['cutoff'])]
        np.testing.assert_equal(float(np.mean(annual)), record['inner_state_rmse'])
        trace = json.loads((root / 'training_history.json').read_text())
        selected = [row for row in trace if row['stage'] == 'inner_selection']
        full_trace = [row for row in trace if row['stage'] == 'full_refit']
        assert min(selected, key=lambda row: row['score'])['epoch'] == record['selected_epochs']
        assert [row['epoch'] for row in full_trace] == list(range(1, record['selected_epochs']+1))
        assert all(np.isfinite(row['loss']) for row in trace)
        with np.load(root / 'replay_inputs.npz') as saved:
            local = np.searchsorted(ix, saved['sample_index'])
            np.testing.assert_array_equal(ix[local], saved['sample_index'])
            batch = manual_batch({k: v[local] for k, v in known.items()}, full, job['product'])
            for key, value in batch.items():
                np.testing.assert_array_equal(saved[key], value)
            out = forward(root / 'model.pt', batch, full, job['product'])
            np.testing.assert_array_equal(out, saved['prediction'])
        torch.manual_seed(job['seed'])
        initial = ForecastState('biid').state_dict()
        weights = torch.load(root / 'model.pt', map_location='cpu', weights_only=True)
        assert all(torch.isfinite(value).all() for value in weights.values())
        changed = sum(not torch.equal(value, initial[key]) for key, value in weights.items())
        assert changed == record['changed_parameter_tensors'] and changed > 0
        for name, digest in record['files'].items():
            sources[str(root / name)] = digest
        sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
        replayed += len(val)+len(out)
        rows.append(dict(**job, selected_epochs=record['selected_epochs'],
            full_training_rows=len(ix), validation_rows=len(val), inner_state_rmse=record['inner_state_rmse'],
            maximum_input_reconstruction_error=0., maximum_weight_replay_error=0.))
        print(f'[REGIONAL STATE VERIFIED] {job} rows={len(val)+len(out)}', flush=True)
        del weights, initial
        torch.cuda.empty_cache()
    inventory = RESULT / 'state_training_inventory.csv'
    pd.DataFrame(rows).to_csv(inventory, index=False)
    sources[str(inventory)] = sha256(inventory)
    for path, digest in sources.items():
        if sha256(Path(path)) != digest:
            raise ValueError('Regional state source changed during verification')
    atomic_json(OUTPUT, dict(passed=True, timestamp=datetime.now().astimezone().isoformat(), models=len(rows),
        state_rows_replayed=replayed, full_model_input_rows=27*32,
        maximum_input_reconstruction_error=0., maximum_weight_replay_error=0.,
        train_only_normalization_independently_rebuilt=True, changed_parameters_verified=True,
        selected_epoch_verified=True, numeric_yield_arrays_loaded=False,
        external_yield_performance_evaluated=False, sources=sources, verifier_sha256=sha256(SCRIPT)))
    print(f'[REGIONAL STATE AUDIT COMPLETE] models={len(rows)} replay_rows={replayed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        if not OUTPUT.exists():
            job = dict(key='cybench_full_state_independent_audit', marker=str(OUTPUT),
                command=[sys.executable, '-u', str(SCRIPT), '--worker'])
            run_extended_stage([job], 'regional_state_audit', GPUS, RESULT,
                ROOT / 'benchmark/logs/cybench_inseason_models_v1/state_audit', 0)
        verify_record()
