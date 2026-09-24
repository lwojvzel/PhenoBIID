"""Frozen regional state-to-yield evaluation under supplied weather conditions."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from cybench_inseason_model_data import (ROOT, RESULT, RECIPES, SEEDS, BLOCKS, PERCENTS,
    VIEW_FIELDS, feature_matrix, regional_climatology, regional_score, state_arrays)
from cybench_seasonal_inputs import KNOWN_FIELDS, PRODUCTS, issue_view
from cybench_inseason_yield_models import hashes, load_data, subset, prediction
from forecast_bridge_state import ForecastState, batch_arrays
from inseason_nested_common import register, finish, verify
from ndvi_tail_replacement import prefix_values, prefix_rollout
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_cybench_inseason_readout import (root_for as head_root, verify_completed as verify_head,
    history_expert, historical_prediction)
from run_cybench_inseason_state import (root_for as state_root, verify_completed as verify_state,
    code_hashes as state_code_hashes)

CODE = ('run_cybench_inseason_inference.py', 'run_cybench_inseason_readout.py',
    'run_cybench_inseason_history.py', 'ndvi_tail_replacement.py',
    'verify_cybench_inseason_readouts.py', 'verify_cybench_inseason_states.py')
BATCH_SIZE = 256


def code_hashes():
    return {**state_code_hashes(), **hashes(CODE)}


def validate_job(crop, cutoff, seed, smoke=False):
    if crop not in RECIPES or cutoff not in BLOCKS or seed not in SEEDS:
        raise ValueError('Unregistered regional inference group')
    if smoke and (cutoff != 2001 or seed != 42):
        raise ValueError('Smoke inference uses the first cutoff and seed')


def root_for(crop, cutoff, seed, smoke=False):
    return RESULT / ('inference_smoke' if smoke else 'inference') / crop / f'cutoff_{cutoff}/seed_{seed}'


def conditions():
    return [('observed', 0)] + [(mode, p) for p in PERCENTS for mode in ('biid', 'climatology')]


def audited_assets(crop, cutoff, seed, sources, smoke=False):
    state_gate = RESULT / 'state_verification.json'
    head_gate = RESULT / ('readout_smoke_all_verification.json' if smoke else 'readout_terminal_verification.json')
    state_record, head_record = [json.loads(path.read_text()) for path in (state_gate, head_gate)]
    if not state_record['passed'] or state_record['models'] != 27:
        raise ValueError('All formal regional states must have independent replay evidence')
    if not head_record['passed'] or head_record['models'] != (14 if smoke else 18):
        raise ValueError('Independently verified regional terminal heads required')
    for path, record, filename in [(state_gate, state_record, 'verify_cybench_inseason_states.py'),
                                  (head_gate, head_record, 'verify_cybench_inseason_readouts.py')]:
        if record['verifier_sha256'] != sha256(ROOT / 'scripts' / filename):
            raise ValueError('Independent component verifier changed')
        sources[str(path)] = sha256(path)
    head_job = dict(route='terminal', crop=crop, cutoff=cutoff, seed=seed,
        model='lightgbm', percent=0, smoke=smoke)
    if head_job not in head_record['jobs']:
        raise ValueError('Head audit does not cover the same crop, cutoff and seed')
    verify_head(**head_job)
    head = head_root(**head_job)
    for name in ('complete.json', 'config.json', 'model.joblib', 'normalization.json',
                 'evaluation_predictions.npz', 'history_anchors.npz'):
        path = head / name
        if sha256(path) != head_record['sources'][str(path)]:
            raise ValueError('Terminal asset differs from its independently verified version')
        sources[str(path)] = sha256(path)
    states = {}
    for product in RECIPES[crop]:
        verify_state(crop, product, cutoff, seed)
        folder = state_root(crop, product, cutoff, seed)
        for name in ('model.pt', 'normalization.json', 'complete.json'):
            path = folder / name
            if sha256(path) != state_record['sources'][str(path)]:
                raise ValueError('State asset differs from its independently verified version')
            sources[str(path)] = sha256(path)
        states[product] = dict(weight=str(folder / 'model.pt'),
            normalization=str(folder / 'normalization.json'))
    return head, states


def state_inputs(view, stats, product):
    if set(view) != VIEW_FIELDS or product not in PRODUCTS:
        raise ValueError('State inference accepts the registered issued view only')
    hidden = view['hidden_mask']
    if np.isfinite(view['observed_vegetation'][hidden]).any() or np.any(view['observed_vegetation_support'][hidden]):
        raise ValueError('Unobserved vegetation or support entered issued state inputs')
    known = {key: view[key] for key in KNOWN_FIELDS}
    batch = batch_arrays(state_arrays(known), np.arange(len(hidden)), stats, product)
    values, mask = prefix_values(view['observed_vegetation'][..., PRODUCTS.index(product)],
        view['active_mask'], hidden, stats[product]['mean'], stats[product]['std'])
    np.testing.assert_array_equal(mask, view['observed_valid'][..., PRODUCTS.index(product)])
    return batch, values, mask


@torch.no_grad()
def forward_prefix(model, batch, values, mask, device='cuda'):
    model.eval()
    parts = []
    for start in range(0, len(values), BATCH_SIZE):
        stop = start+BATCH_SIZE
        inputs = {key: torch.as_tensor(value[start:stop], device=device) for key, value in batch.items()}
        prefix = torch.as_tensor(values[start:stop], device=device)
        visible = torch.as_tensor(mask[start:stop], device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            output = prefix_rollout(model, inputs, prefix, visible)
        parts.append(output.float().cpu().numpy())
    result = np.concatenate(parts)
    if result.shape != values.shape or not np.isfinite(result).all():
        raise FloatingPointError('Invalid normalized state prediction')
    return result


def complete_trajectory(view, values, product):
    prefix = view['observed_vegetation'][..., PRODUCTS.index(product)]
    values = np.asarray(values)
    if values.shape != prefix.shape or not np.isfinite(values[view['hidden_mask']]).all():
        raise ValueError('Wrong or nonfinite hidden trajectory')
    result = np.where(view['hidden_mask'], values, prefix).astype(np.float32)
    np.testing.assert_array_equal(result[~view['hidden_mask']], prefix[~view['hidden_mask']])
    return result


def state_errors(targets, hidden, completed, identities):
    rows = []
    for product, values in completed.items():
        j = PRODUCTS.index(product)
        truth = targets['vegetation'][..., j].astype(np.float64)
        support = targets['vegetation_support'][..., j]
        valid = hidden & np.isfinite(truth) & (support > 0)
        for (country, year), indices in identities.groupby(['country', 'year'], sort=True).indices.items():
            local = valid[indices]
            if not local.any():
                continue
            error = (np.asarray(values, np.float64)[indices]-truth[indices])[local]
            rows.append(dict(product=product, country=country, year=int(year),
                valid_region_slots=int(len(error)), regions=int(local.any(1).sum()),
                rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.abs(error).mean())))
    return rows


def verify_completed(crop, cutoff, seed, smoke=False):
    validate_job(crop, cutoff, seed, smoke)
    root = root_for(crop, cutoff, seed, smoke)
    record = verify(root, code_hashes())
    job = dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke)
    config = json.loads((root / 'config.json').read_text())
    if record['job'] != job or config['job'] != job or record['conditions'] != 7 or record['new_fits'] != 0:
        raise ValueError('Invalid frozen inference provenance')
    for path, digest in config['sources'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Regional inference source changed')
    return record


def run(crop, cutoff, seed, smoke=False):
    validate_job(crop, cutoff, seed, smoke)
    job = dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke)
    root = root_for(**job)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            return verify_completed(**job)
        if not torch.cuda.is_available():
            raise RuntimeError('Like-precision regional inference requires CUDA')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        data = load_data(crop, cutoff, smoke, vegetation=True)
        sources = data['sources']
        head, state_assets = audited_assets(crop, cutoff, seed, sources, smoke)
        expert = history_expert(crop, cutoff, seed, sources, smoke)
        ev = data['parts']['evaluation']
        ids = data['identities'].iloc[ev].reset_index(drop=True)
        ix, target, history = data['sample_index'][ev], data['target'][ev], data['history'][ev]
        known, state_targets = subset(data['known'], ev), subset(data['state_targets'], ev)
        with threadpool_limits(limits=2):
            anchor = historical_prediction(data, expert, 'full')[ev]
        with np.load(head / 'history_anchors.npz') as saved:
            np.testing.assert_array_equal(saved['sample_index'], data['sample_index'])
            np.testing.assert_array_equal(saved['full'][ev], anchor)
        norm = json.loads((head / 'normalization.json').read_text())
        if norm['features']['cutoff'] != cutoff:
            raise ValueError('Terminal normalization has the wrong training cutoff')
        fitted = joblib.load(head / 'model.joblib')
        code = code_hashes()
        register(root, dict(schema=1, job=job, code_sha256=code, sources=sources,
            products=list(RECIPES[crop]), state_assets=state_assets, terminal_root=str(head),
            historical_expert=expert, conditions=[dict(mode=m, percent=p) for m, p in conditions()],
            state_weights_are_formal=True, smoke_uses_small_head_and_formal_states=smoke,
            evaluation_rows=len(ev), input_width=578+96*len(RECIPES[crop]),
            state_batch_size=BATCH_SIZE, state_precision='CUDA BF16 autocast; physical FP32 trajectory',
            whole_future_weather_supplied=True, model_optimization=False, new_fits=0,
            crop_area_weighted=False, label_source='CY-Bench independent regional statistical yields',
            zero_shot_transfer=False, state_outputs_clipped=False, yield_clipped_nonnegative=True,
            observed_zero_is_diagnostic=True, all_registered_seeds_retained=True))
        start = time.monotonic()
        ids.to_csv(root / 'identities.csv', index=False)
        np.savez_compressed(root / 'targets.npz', sample_index=ix, yield_target=target, **state_targets)
        for folder in ('predictions', 'trajectories', 'forecasts', 'issued'):
            (root / folder).mkdir(exist_ok=True)
        all_metrics = []
        for percent in (0, *PERCENTS):
            view = issue_view(known, state_targets, percent)
            base_features = feature_matrix(history, view, norm['features'], RECIPES[crop])
            np.savez_compressed(root / f'issued/tail_{percent:03d}.npz', sample_index=ix,
                **{key: view[key] for key in ('observed_vegetation', 'observed_vegetation_support',
                    'observed_valid', 'visible_mask', 'hidden_mask', 'issue_day', 'days_to_supplied_eos')})
            trajectories = {'observed': {p: view['observed_vegetation'][..., PRODUCTS.index(p)] for p in RECIPES[crop]}} if percent == 0 else {'biid': {}, 'climatology': {}}
            if percent:
                for product, asset in state_assets.items():
                    scale = json.loads(Path(asset['normalization']).read_text())
                    batch, values, mask = state_inputs(view, scale, product)
                    model = ForecastState('biid').cuda()
                    model.load_state_dict(torch.load(asset['weight'], map_location='cpu', weights_only=True))
                    physical = forward_prefix(model, batch, values, mask)*scale[product]['std']+scale[product]['mean']
                    del model
                    trajectories['biid'][product] = complete_trajectory(view, physical, product)
                    climo = regional_climatology(known, norm['features'], product)
                    trajectories['climatology'][product] = complete_trajectory(view, climo, product)
                    np.savez_compressed(root / f'forecasts/{product}_{percent:03d}.npz', sample_index=ix,
                        prediction=physical, prefix_values=values, known=mask)
            for mode, completed in trajectories.items():
                features = feature_matrix(history, view, norm['features'], RECIPES[crop], completed=completed)
                metadata_width = 578+60*len(RECIPES[crop])
                np.testing.assert_array_equal(features[:, :metadata_width], base_features[:, :metadata_width])
                with threadpool_limits(limits=2):
                    p = prediction(fitted, features, anchor, norm['residual'], 'lightgbm')
                if not percent:
                    with np.load(head / 'evaluation_predictions.npz') as saved:
                        np.testing.assert_array_equal(saved['sample_index'], ix)
                        np.testing.assert_array_equal(saved['target'], target)
                        np.testing.assert_array_equal(saved['prediction'], p)
                tag = f'{mode}_{percent:03d}'
                np.savez_compressed(root / f'predictions/{tag}.npz', sample_index=ix, prediction=p,
                    anchor=anchor, replay_features=features[:32], year=ids.year.to_numpy())
                np.savez_compressed(root / f'trajectories/{tag}.npz', sample_index=ix, **completed)
                score, annual = regional_score(target, p, ids)
                all_metrics.append(dict(mode=mode, percent=percent, score=score, annual=annual,
                    state_errors=state_errors(state_targets, view['hidden_mask'], completed, ids)))
                print(f'[REGIONAL INFERENCE] {job} {tag} score={score:.6f}', flush=True)
        atomic_json(root / 'metrics.json', all_metrics)
        for path, digest in sources.items():
            if sha256(Path(path)) != digest:
                raise ValueError('Component changed during frozen inference')
        if code_hashes() != code:
            raise ValueError('Frozen inference implementation changed')
        finish(root, code, job=job, conditions=len(all_metrics), new_fits=0,
            evaluation_rows=len(ev), prediction_rows=len(ev)*len(all_metrics),
            seconds=time.monotonic()-start, maximum_observed_replay_error=0.,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        return verify_completed(**job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=RECIPES, required=True)
    parser.add_argument('--cutoff', type=int, choices=BLOCKS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--smoke', action='store_true')
    run(**vars(parser.parse_args()))
