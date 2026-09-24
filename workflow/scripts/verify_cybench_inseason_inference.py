"""Independent prefix rollout and frozen yield replay for regional evaluation."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from cybench_inseason_model_data import ROOT, RESULT, RECIPES, feature_matrix, regional_climatology
from cybench_seasonal_inputs import issue_view
from cybench_inseason_yield_models import load_data, subset
from forecast_bridge_state import ForecastState
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_cybench_inseason_inference import root_for, verify_completed, conditions
from run_cybench_inseason_inference_queue import jobs
from run_cybench_inseason_state_queue import GPUS, process_identity
from verify_cybench_inseason_states import manual_batch
from verify_cybench_inseason_history import country_year_score, matrix, replay
from shared_gpu_queue_extended import run_extended_stage

SCRIPT = Path(__file__).resolve()


def output_for(smoke=False):
    return RESULT / ('inference_smoke_verification.json' if smoke else 'inference_verification.json')


def manual_masks(active, percent):
    counts = active.sum(1)
    count_hidden = (counts*percent+99)//100
    hidden = active & (active.cumsum(1) > (counts-count_hidden)[:, None])
    return active & ~hidden, hidden


def manual_prefix(known, targets, percent, scale, product):
    j = ('ndvi', 'gpp').index(product)
    visible, hidden = manual_masks(known['active_mask'], percent)
    truth = targets['vegetation'][..., j]
    valid = visible & np.isfinite(truth) & (targets['vegetation_support'][..., j] > 0)
    values = np.where(valid, (truth-scale[product]['mean'])/scale[product]['std'], 0).astype(np.float32)
    return manual_batch(known, scale, product), values, valid


def manual_step(model, batch, values, known):
    d = model.dynamics
    zeros = torch.zeros_like(batch['previous'])
    inputs = dict(weather=batch['weather'], previous_lai=zeros, previous_lai_valid=zeros,
        previous_ndvi=batch['previous'], previous_ndvi_valid=batch['previous_valid'],
        previous_ndvi_quality=batch['previous_quality'], relative_valid=batch['relative_valid'],
        context=batch['context'])
    states = d.initial_states(inputs)
    assert set(states) == {'ndvi'}
    state = states['ndvi']
    forcing = d.weather(batch['weather'], batch['context'])
    output = []
    # Deliberately do not call the registered prefix_rollout implementation.
    for slot in range(12):
        active = batch['relative_valid'][:, slot, None, None].bool()
        evolved = d.transitions['ndvi'](state, forcing[:, slot])
        state = torch.where(active, evolved, state)
        baseline = torch.where(batch['previous_valid'][:, slot].bool(), batch['previous'][:, slot], zeros[:, slot])
        estimated = baseline+d.heads['ndvi'](state.mean(1)).squeeze(-1)
        output.append(estimated)
        feedback = torch.where(known[:, slot], values[:, slot], estimated)
        message = d.feedback['ndvi'](feedback[:, None])[:, None]
        state = torch.where(active, d.norms['ndvi'](state+message), state)
    return torch.stack(output, 1)


@torch.no_grad()
def manual_forward(weight, batch, values, mask, scale, product):
    model = ForecastState('biid').cuda()
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
    model.eval()
    parts = []
    for start in range(0, len(values), 256):
        stop = start+256
        x = {k: torch.as_tensor(v[start:stop], device='cuda') for k, v in batch.items()}
        v = torch.as_tensor(values[start:stop], device='cuda')
        m = torch.as_tensor(mask[start:stop], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = manual_step(model, x, v, m)
        parts.append(p.float().cpu().numpy())
    del model
    return np.concatenate(parts)*scale[product]['std']+scale[product]['mean']


def verify_record(smoke=False):
    record = json.loads(output_for(smoke).read_text())
    if not record['passed'] or record['jobs'] != jobs(smoke) or record['verifier_sha256'] != sha256(SCRIPT):
        raise ValueError('Frozen regional inference audit changed')
    for filename, digest in record['sources'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Verified regional inference source changed')
    return record


def worker(smoke=False):
    with (RESULT / f'inference_{"smoke_" if smoke else ""}verification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output_for(smoke).exists():
            return verify_record(smoke)
        gate = RESULT / f'{"inference_smoke" if smoke else "inference_full"}_queue_complete.json'
        queue = json.loads(gate.read_text())
        assert queue['passed'] and queue['jobs'] == jobs(smoke)
        sources = {str(gate): sha256(gate)}
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        annual_rows, state_rows = [], []
        yield_count = state_count = poison_checks = 0
        for job in jobs(smoke):
            root = root_for(**job)
            completed = verify_completed(**job)
            config = json.loads((root / 'config.json').read_text())
            data = load_data(job['crop'], job['cutoff'], smoke, vegetation=True)
            ev = data['parts']['evaluation']
            known, truth = subset(data['known'], ev), subset(data['state_targets'], ev)
            ix, y, history = data['sample_index'][ev], data['target'][ev], data['history'][ev]
            ids = data['identities'].iloc[ev].reset_index(drop=True)
            saved_ids = pd.read_csv(root / 'identities.csv', dtype={'country': str, 'adm_id': str})
            pd.testing.assert_frame_equal(ids, saved_ids, check_dtype=False)
            with np.load(root / 'targets.npz') as saved:
                np.testing.assert_array_equal(saved['sample_index'], ix)
                np.testing.assert_array_equal(saved['yield_target'], y)
                for key in truth:
                    np.testing.assert_array_equal(saved[key], truth[key])
            head = Path(config['terminal_root'])
            norm = json.loads((head / 'normalization.json').read_text())
            expert = config['historical_expert']
            hn = json.loads(Path(expert['full_normalization']).read_text())
            anchor = replay(Path(expert['full_weight']), expert['model'], matrix(history, hn), data['trend'][ev], hn)
            metrics = {(row['mode'], row['percent']): row for row in json.loads((root / 'metrics.json').read_text())}
            assert set(metrics) == set(conditions())
            for percent in (0, 10, 30, 50):
                view = issue_view(known, truth, percent)
                visible, hidden = manual_masks(known['active_mask'], percent)
                np.testing.assert_array_equal(view['visible_mask'], visible)
                np.testing.assert_array_equal(view['hidden_mask'], hidden)
                with np.load(root / f'issued/tail_{percent:03d}.npz') as saved:
                    np.testing.assert_array_equal(saved['sample_index'], ix)
                    for key in saved.files:
                        if key != 'sample_index':
                            np.testing.assert_array_equal(saved[key], view[key])
                forecast = {}
                for product, asset in config['state_assets'].items():
                    if not percent:
                        continue
                    scale = json.loads(Path(asset['normalization']).read_text())
                    batch, values, mask = manual_prefix(known, truth, percent, scale, product)
                    p = manual_forward(asset['weight'], batch, values, mask, scale, product)
                    with np.load(root / f'forecasts/{product}_{percent:03d}.npz') as saved:
                        np.testing.assert_array_equal(saved['sample_index'], ix)
                        np.testing.assert_array_equal(saved['prefix_values'], values)
                        np.testing.assert_array_equal(saved['known'], mask)
                        np.testing.assert_array_equal(saved['prediction'], p)
                    forecast[product] = p
                    state_count += len(ix)
                changed = {key: value.copy() for key, value in truth.items()}
                changed['vegetation'][hidden] = 6789.
                changed['vegetation_support'][hidden] = .413
                poisoned_view = issue_view(known, changed, percent)
                if percent:
                    for product, asset in config['state_assets'].items():
                        scale = json.loads(Path(asset['normalization']).read_text())
                        before = manual_prefix(known, truth, percent, scale, product)
                        after = manual_prefix(known, changed, percent, scale, product)
                        for key in before[0]:
                            np.testing.assert_array_equal(before[0][key], after[0][key])
                        np.testing.assert_array_equal(before[1], after[1])
                        np.testing.assert_array_equal(before[2], after[2])
                    poison_checks += 1
                shared = feature_matrix(history, view, norm['features'], RECIPES[job['crop']])
                for mode in (('observed',) if not percent else ('biid', 'climatology')):
                    trajectory = {}
                    for product in RECIPES[job['crop']]:
                        j = ('ndvi', 'gpp').index(product)
                        suffix = (forecast[product] if mode == 'biid' else regional_climatology(known, norm['features'], product)) if percent else view['observed_vegetation'][..., j]
                        trajectory[product] = np.where(hidden, suffix, view['observed_vegetation'][..., j]).astype(np.float32)
                    tag = f'{mode}_{percent:03d}'
                    with np.load(root / f'trajectories/{tag}.npz') as saved:
                        np.testing.assert_array_equal(saved['sample_index'], ix)
                        for product, values in trajectory.items():
                            np.testing.assert_array_equal(saved[product], values)
                    x = feature_matrix(history, view, norm['features'], RECIPES[job['crop']], completed=trajectory)
                    np.testing.assert_array_equal(x[:, :578+60*len(trajectory)], shared[:, :578+60*len(trajectory)])
                    poisoned_x = feature_matrix(history, poisoned_view, norm['features'], RECIPES[job['crop']], completed=trajectory)
                    np.testing.assert_array_equal(x, poisoned_x)
                    p = replay(head / 'model.joblib', 'lightgbm', x, anchor, norm)
                    with np.load(root / f'predictions/{tag}.npz') as saved:
                        np.testing.assert_array_equal(saved['sample_index'], ix)
                        np.testing.assert_array_equal(saved['prediction'], p)
                        np.testing.assert_array_equal(saved['anchor'], anchor)
                        np.testing.assert_array_equal(saved['replay_features'], x[:32])
                    score, annual = country_year_score(y, p, ids)
                    metric = metrics[(mode, percent)]
                    np.testing.assert_allclose(score, metric['score'], rtol=0, atol=1e-12)
                    a, b = pd.DataFrame(annual), pd.DataFrame(metric['annual'])
                    pd.testing.assert_frame_equal(a, b[a.columns], check_exact=False, rtol=0, atol=1e-12)
                    measured_state = []
                    for product, values in trajectory.items():
                        j = ('ndvi', 'gpp').index(product)
                        for row in annual:
                            rows = ((ids.country == row['country']) & (ids.year == row['year'])).to_numpy()
                            gt = truth['vegetation'][rows, :, j].astype(np.float64)
                            valid = hidden[rows] & np.isfinite(gt) & (truth['vegetation_support'][rows, :, j] > 0)
                            if valid.any():
                                error = (values[rows].astype(np.float64)-gt)[valid]
                                measured_state.append(dict(product=product, country=row['country'], year=row['year'],
                                    valid_region_slots=int(valid.sum()), regions=int(valid.any(1).sum()),
                                    rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.abs(error).mean())))
                    assert measured_state == metric['state_errors']
                    base = dict(crop=job['crop'], cutoff=job['cutoff'], seed=job['seed'], mode=mode, percent=percent)
                    annual_rows.extend(dict(**base, **row) for row in annual)
                    state_rows.extend(dict(**base, **row) for row in measured_state)
                    yield_count += len(ix)
            sources.update(data['sources'])
            sources.update(config['sources'])
            sources.update({str(root / name): digest for name, digest in completed['files'].items()})
            sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
            print(f'[REGIONAL STATE-TO-YIELD VERIFIED] {job} rows={len(ix)}', flush=True)
        destination = RESULT / ('inference_smoke_summary' if smoke else 'inference_summary')
        destination.mkdir(exist_ok=True)
        pd.DataFrame(annual_rows).to_csv(destination / 'annual.csv', index=False)
        pd.DataFrame(state_rows).to_csv(destination / 'state_errors.csv', index=False)
        for name in ('annual.csv', 'state_errors.csv'):
            sources[str(destination / name)] = sha256(destination / name)
        for filename, digest in sources.items():
            if sha256(Path(filename)) != digest:
                raise ValueError('Inference source changed during independent replay')
        record = dict(passed=True, timestamp=datetime.now().astimezone().isoformat(), jobs=jobs(smoke),
            groups=len(jobs(smoke)), conditions=7*len(jobs(smoke)), new_fits=0,
            yield_prediction_rows_replayed=yield_count, state_prediction_rows_replayed=state_count,
            maximum_state_replay_error=0., maximum_yield_replay_error=0.,
            independent_state_input_and_feedback_reconstruction=True,
            feature_reconstruction_uses_registered_builder=True, hidden_value_and_support_checks=poison_checks,
            sources=sources, verifier_sha256=sha256(SCRIPT))
        atomic_json(output_for(smoke), record)
        return record


def run(smoke=False, wait=False):
    if output_for(smoke).exists():
        verify_record(smoke)
        print(f'[REGIONAL INFERENCE AUDIT REUSE] {output_for(smoke)}', flush=True)
        return
    gate = RESULT / f'{"inference_smoke" if smoke else "inference_full"}_queue_complete.json'
    while not gate.exists():
        if not wait or smoke:
            raise RuntimeError('Regional inference queue is incomplete')
        launcher = json.loads((RESULT / 'inference_launcher.json').read_text())
        if process_identity(launcher['pid']) != launcher['process']:
            if gate.exists():
                break
            raise RuntimeError('Regional inference worker is not live')
        print(f'[WAIT VERIFIED INFERENCE PROCESS] pid={launcher["pid"]}', flush=True)
        time.sleep(10)
    stage = 'inference_smoke_audit' if smoke else 'inference_audit'
    command = [sys.executable, '-u', str(SCRIPT), '--worker'] + (['--smoke'] if smoke else [])
    run_extended_stage([dict(key=f'cybench_{stage}', marker=str(output_for(smoke)), command=command)],
        stage, GPUS, RESULT, ROOT / f'benchmark/logs/cybench_inseason_models_v1/{stage}', 1)
    result = verify_record(smoke)
    print(f'[REGIONAL INFERENCE AUDIT COMPLETE] groups={result["groups"]} conditions={result["conditions"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.worker:
            worker(args.smoke)
        else:
            run(args.smoke, args.wait)
