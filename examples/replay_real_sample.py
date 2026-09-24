"""Replay the bundled real-data sample without the original data workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
from phenobiid.forecast_bridge_state import ForecastState, INPUTS
from phenobiid.ndvi_tail_replacement import prefix_rollout, prefix_values, mix_trajectory, tail_mask
from phenobiid.observed_remote_benchmark import trajectory_features
from phenobiid.run_ndvi_signal_permutation import compose
from phenobiid.task_aligned_world import TaskAlignedWorld
from phenobiid.numeric_embedding_readout import NeuralReadout


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        file = root / name
        if file.is_symlink() or root.resolve() not in file.resolve().parents:
            raise ValueError('Bundle paths must remain inside the bundle')
        if digest(file) != expected:
            raise ValueError(f'Bundle file changed: {name}')
    return manifest


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def encode(physical, active, scale, climatology):
    valid = active & np.isfinite(physical)
    values = np.where(valid, (physical - scale['mean']) / scale['std'], 0).astype(np.float32)
    anomalies = np.where(valid, values - climatology, 0).astype(np.float32)
    return np.concatenate((trajectory_features(values, valid), trajectory_features(anomalies, valid)), 1)


def contract(inputs, meta):
    products = meta['products']
    allowed = {'common', 'active', 'tail', 'history_x', 'trend'}
    if meta['crop'] == 'soybean':
        allowed.add('tabm_x')
    for product in products:
        allowed.update(f'{product}_{k}' for k in (*INPUTS, 'prefix', 'climatology'))
    if set(inputs) != allowed:
        raise ValueError('Unexpected inputs; labels and reference predictions are not model inputs')
    n = len(inputs['trend'])
    if inputs['common'].shape != (n, 465) or inputs['history_x'].shape != (n, 20):
        raise ValueError('Invalid historical/common feature dimensions')
    active, tail = inputs['active'], inputs['tail']
    if active.dtype != bool or tail.dtype != bool or active.shape != (n, 12):
        raise ValueError('Expected boolean twelve-slot calendars')
    if tail.shape != active.shape or np.any(tail & ~active) or np.any(active.sum(1) == 0):
        raise ValueError('Invalid hidden slots')
    if not np.array_equal(tail, tail_mask(active, meta['ratio'])):
        raise ValueError('Hidden slots do not match the declared cutoff')
    for product in products:
        prefix = inputs[f'{product}_prefix']
        if prefix.shape != active.shape or np.isfinite(prefix[tail | ~active]).any():
            raise ValueError('Future or inactive observed values entered the model')
        if not np.array_equal(inputs[f'{product}_relative_valid'] > 0, active):
            raise ValueError('State and terminal calendars differ')
        if inputs[f'{product}_weather'].shape != (n, 12, 13):
            raise ValueError('Invalid weather dimensions')
        if inputs[f'{product}_context'].shape != (n, 5):
            raise ValueError('Invalid state context dimensions')
        for key in ('previous', 'previous_valid', 'previous_quality', 'relative_valid', 'climatology'):
            if inputs[f'{product}_{key}'].shape != (n, 12):
                raise ValueError(f'Invalid state feature dimensions: {key}')
    if 'tabm_x' in inputs and inputs['tabm_x'].shape != (n, 20):
        raise ValueError('Invalid soybean historical feature dimensions')
    for name, array in inputs.items():
        if not name.endswith('_prefix') and not np.isfinite(array).all():
            raise ValueError(f'Nonfinite model input: {name}')


@torch.no_grad()
def history(root, inputs, meta):
    model = TaskAlignedWorld('history_mlp').eval()
    model.load_state_dict(torch.load(root / 'history.pt', map_location='cpu', weights_only=True))
    residual = model.yield_head(torch.from_numpy(inputs['history_x'])).squeeze(-1).numpy().astype(float)
    norm = meta['history_normalization']
    value = inputs['trend'].astype(float) + residual * norm['residual_std'] + norm['residual_mean']
    if meta['crop'] == 'soybean':
        bins = torch.load(root / 'history_bins.pt', map_location='cpu', weights_only=True)
        tabm = NeuralReadout('tabm', 20, bins).eval()
        tabm.load_state_dict(torch.load(root / 'tabm.pt', map_location='cpu', weights_only=True))
        component = tabm(torch.from_numpy(inputs['tabm_x'])).mean(1).numpy().astype(float)
        value = value + meta['tabm_scale'] * component
    return [inputs['trend'].astype(float), value] if meta['crop'] == 'maize' else [value]


@torch.no_grad()
def state(root, inputs, meta, product):
    model = ForecastState('biid').eval()
    model.load_state_dict(torch.load(root / f'{product}.pt', map_location='cpu', weights_only=True))
    b = {k: torch.from_numpy(inputs[f'{product}_{k}']) for k in INPUTS}
    scale = meta['state_scales'][product]
    values, known = prefix_values(inputs[f'{product}_prefix'], inputs['active'], inputs['tail'],
                                  scale['mean'], scale['std'])
    prediction = prefix_rollout(model, b, torch.from_numpy(values), torch.from_numpy(known))
    physical = prediction.numpy() * scale['std'] + scale['mean']
    return mix_trajectory(inputs[f'{product}_prefix'], physical, inputs['tail'])


def terminal(root, inputs, meta, trajectories, anchors):
    chunks = [inputs['common']]
    for product in meta['products']:
        chunks.append(encode(trajectories[product], inputs['active'], meta['encoder_scales'][product],
                             inputs[f'{product}_climatology']))
    features = np.concatenate(chunks, 1)
    if features.shape[1] != 465 + 36 * len(meta['products']) or not np.isfinite(features).all():
        raise ValueError('Invalid final feature matrix')
    if len(anchors) != len(meta['heads']):
        raise ValueError('Historical branch count mismatch')
    outputs = []
    for k, (head, anchor) in enumerate(zip(meta['heads'], anchors)):
        model = lgb.Booster(model_file=str(root / f'head_{k}.txt'))
        component = model.predict(features, num_threads=2)
        outputs.append(compose(head, anchor, component))
    prediction = .5 * outputs[0] + .5 * outputs[1] if meta['crop'] == 'maize' else outputs[0]
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite yield prediction')
    return prediction, features


def cpu_forward(root, inputs, meta):
    contract(inputs, meta)
    anchors = history(root, inputs, meta)
    trajectories = {p: state(root, inputs, meta, p) for p in meta['products']}
    prediction, features = terminal(root, inputs, meta, trajectories, anchors)
    return prediction, trajectories, anchors, features


def run(root):
    manifest = verify(root)
    rows = []
    for case in manifest['cases']:
        folder = root / 'assets' / case['crop']
        meta = dict(json.loads((folder / 'metadata.json').read_text()), ratio=case['ratio'])
        name = f"cutoff_{round(case['ratio'] * 100):03d}"
        inputs = read_npz(folder / f'{name}_inputs.npz')
        # The prediction is produced before reference targets are opened.
        prediction, trajectories, anchors, features = cpu_forward(folder, inputs, meta)
        reference = read_npz(folder / f'{name}_reference.npz')
        archived = {p: reference[f'{p}_mixed'] for p in meta['products']}
        old_anchors = [reference[f'anchor_{k}'] for k in range(len(meta['heads']))]
        replay, replay_features = terminal(folder, inputs, meta, archived, old_anchors)
        np.testing.assert_array_equal(replay_features, reference['features'])
        np.testing.assert_allclose(replay, reference['prediction'], atol=1e-12, rtol=0)
        errors = {p: float(np.max(np.abs(trajectories[p][inputs['tail']] - archived[p][inputs['tail']])))
                  for p in meta['products']}
        row = dict(crop=case['crop'], ratio=case['ratio'], samples=len(prediction),
            feature_shape=list(features.shape), archived_replay_max_error=float(np.max(np.abs(replay - reference['prediction']))),
            cpu_fp32_yield_max_difference=float(np.max(np.abs(prediction - reference['prediction']))),
            cpu_fp32_state_max_difference=errors,
            cpu_fp32_anchor_max_difference=max(float(np.max(np.abs(x-y))) for x, y in zip(anchors, old_anchors)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(archived_replay_passed=True, cpu_fp32_finite=True, cases=rows,
        scope='Real-data small-sample frozen inference, not retraining or thirteen-year reproduction',
        precision='Archived CUDA BF16 versus recomputed CPU FP32; differences are reported, not substituted',
        torch_version=torch.__version__, numpy_version=np.__version__, lightgbm_version=lgb.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=REPOSITORY / 'data' / 'sample')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    result = run(args.root.resolve())
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    else:
        print(json.dumps(result, indent=2))
