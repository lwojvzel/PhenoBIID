"""Evaluate rebuilt states and readouts using rebuilt historical predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
CONDITIONS = [(0, 'observed')] + [(p, m) for p in (10, 30, 50) for m in ('biid', 'climatology')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    part = Path(name)
    value = (root / part).resolve()
    if part.is_absolute() or '..' in part.parts or not value.is_relative_to(root.resolve()):
        raise ValueError('Path escapes the reconstruction package')
    return value


def verified(root, manifest, name):
    path = safe_path(root, name)
    if manifest['files'].get(name) != digest(path):
        raise ValueError('Package input changed: ' + name)
    return path


def encode(physical, active, scale, climatology):
    from observed_remote_benchmark import trajectory_features
    valid = active & np.isfinite(physical)
    values = np.where(valid, (physical-scale['mean'])/scale['std'], 0).astype(np.float32)
    anomalies = np.where(valid, values-climatology, 0).astype(np.float32)
    return np.concatenate((trajectory_features(values, valid), trajectory_features(anomalies, valid)), 1)


def prefix_contract(prefix, active, tail):
    if (prefix.shape != active.shape or tail.shape != active.shape or active.dtype != bool
            or tail.dtype != bool or np.any(tail & ~active)
            or np.isfinite(prefix[tail | ~active]).any()):
        raise ValueError('Hidden or inactive observations entered the state interface')


def allow_input(root, path, stage):
    path = path.resolve()
    if not path.is_relative_to(root) or stage == 'check':
        return True
    if 'reference' in path.relative_to(root).parts:
        return False
    return not (stage == 'state' and path.name.startswith('observed_'))


def install_reference_guard(root, stage):
    def audit(event, values):
        if stage == 'check' or event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if not allow_input(root, path, stage):
            raise PermissionError('Reference or unmasked observations are unavailable in this stage')
    sys.addaudithook(audit)


def run(root, case, seed, stage):
    manifest = json.loads((root / 'manifest.json').read_text())
    if dict(case=case, seed=seed) not in manifest['jobs']:
        raise ValueError('Unregistered full-chain case')
    prefix = 'cases/' + case
    spec = json.loads(verified(root, manifest, prefix + '/case.json').read_text())
    recipe = json.loads(verified(root, manifest, prefix + f'/seed_{seed}/recipe.json').read_text())
    if recipe['seed'] != seed:
        raise ValueError('Cross-seed readout recipe')
    output = safe_path(root, f'reconstructed/{case}/seed_{seed}')
    output.mkdir(parents=True, exist_ok=True)
    marker = output / f'{stage}.json'
    if marker.exists():
        raise FileExistsError('Do not replace an existing chain-stage record')
    install_reference_guard(root, stage)

    def array(name):
        return np.load(verified(root, manifest, prefix + '/' + name), allow_pickle=False)

    if stage == 'state':
        import torch
        from forecast_bridge_state import ForecastState, INPUTS
        from ndvi_tail_replacement import tail_mask, prefix_values, prefix_rollout, mix_trajectory
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('Original state evaluation requires CUDA')
        torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
        models = {}
        for product in spec['products']:
            model = ForecastState('biid').cuda().eval()
            path = verified(root, manifest, prefix + f'/seed_{seed}/state_{product}.pt')
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            models[product] = model
        for split in spec['splits']:
            active = array(f'{split}/active.npy')
            for product, model in models.items():
                inputs = {k: array(f'{split}/{product}_{k}.npy') for k in INPUTS}
                scale = spec['state_scales'][product]
                for percent in (10, 30, 50):
                    tail = tail_mask(active, percent/100)
                    observed = array(f'{split}/prefix_{product}_{percent:03d}.npy')
                    prefix_contract(observed, active, tail)
                    values, known = prefix_values(observed, active, tail, scale['mean'], scale['std'])
                    pieces = []
                    for start in range(0, len(active), 256):
                        batch = {k: torch.as_tensor(v[start:start+256], device='cuda') for k, v in inputs.items()}
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = prefix_rollout(model, batch,
                                torch.as_tensor(values[start:start+256], device='cuda'),
                                torch.as_tensor(known[start:start+256], device='cuda'))
                        pieces.append(result.float().cpu().numpy())
                    physical = np.concatenate(pieces)*scale['std']+scale['mean']
                    mixed = mix_trajectory(observed, physical, tail)
                    if not np.isfinite(mixed[tail]).all():
                        raise ValueError('Nonfinite reconstructed suffix')
                    np.save(output / f'{split}_{product}_{percent:03d}.npy', mixed)
                    print(f'[NEW STATE FORWARD] {case} {seed} {split} {product} {percent}', flush=True)
        files = {p.name:digest(p) for p in output.glob('*.npy')}
        result = dict(passed=True, products=spec['products'], rows=sum(spec['splits'].values()),
            reference_loaded=False, newly_reconstructed_weights=True, files=files)
    elif stage == 'readout':
        import lightgbm as lgb
        from chain_math import tail_mask, feature_matrix, compose
        state = json.loads((output / 'state.json').read_text())
        for name, checksum in state['files'].items():
            if digest(output / name) != checksum:
                raise ValueError('Newly predicted state changed')
        heads = [lgb.Booster(model_file=str(verified(root, manifest,
                    prefix + f'/seed_{seed}/head_{i}.txt'))) for i in range(len(recipe['branches']))]
        predictions = {key:[] for key in CONDITIONS}
        for split in spec['splits']:
            active = array(f'{split}/active.npy')
            observed = {p:array(f'{split}/observed_{p}.npy') for p in spec['products']}
            climate = {p:array(f'{split}/climatology_{p}.npy') for p in spec['products']}
            scales = spec['encoder_scales']
            constant = {p:-encode(np.full_like(observed[p], scales[p]['mean']), active,
                        scales[p], climate[p])[:, -18:-6]*scales[p]['std']+scales[p]['mean'] for p in spec['products']}
            anchor = array(f'seed_{seed}/history_{split}.npy')
            trend = array(f'{split}/trend.npy')
            common, support = array(f'{split}/common.npy'), array(f'{split}/support.npy')
            for percent, mode in CONDITIONS:
                tail = tail_mask(active, percent/100)
                if mode == 'observed':
                    values = observed
                elif mode == 'climatology':
                    values = {p:np.where(tail, constant[p], observed[p]) for p in spec['products']}
                else:
                    values = {p:np.load(output / f'{split}_{p}_{percent:03d}.npy') for p in spec['products']}
                    for p in values:
                        np.testing.assert_array_equal(values[p][~tail], observed[p][~tail])
                encoded = {p:encode(values[p], active, scales[p], climate[p]) for p in spec['products']}
                x = feature_matrix(common, encoded, tail, support, spec['recipe'])
                if not np.isfinite(x).all() or x.shape[1] != spec['feature_width']:
                    raise ValueError('Invalid terminal encoding')
                parts = [compose(branch, trend if branch['name'] == 'trend' else anchor,
                         model.predict(x, num_threads=4)) for branch, model in zip(recipe['branches'], heads)]
                prediction = .5*parts[0]+.5*parts[1] if spec['crop'] == 'maize' else parts[0]
                predictions[percent, mode].append(prediction)
            print(f'[NEW CHAIN READOUT] {case} {seed} {split}', flush=True)
        files = {}
        for (percent, mode), parts in predictions.items():
            path = output / f'tail_{percent:03d}_{mode}.npy'
            np.save(path, np.concatenate(parts))
            files[path.name] = digest(path)
        result = dict(passed=True, rows=sum(spec['splits'].values()), output_conditions=7,
            reference_loaded=False, historical_outputs_from_reconstructed_model=True,
            terminal_weights_from_reconstruction=True, state_record_sha256=digest(output/'state.json'), files=files)
    else:
        readout = json.loads((output / 'readout.json').read_text())
        maximum, rows, state_error = 0., 0, 0.
        for name, checksum in readout['files'].items():
            if digest(output / name) != checksum:
                raise ValueError('Chain output changed before reference check')
            actual = np.load(output / name)
            expected = array(f'seed_{seed}/reference/{name}')
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            rows += len(actual)
        for split in spec['splits']:
            for product in spec['products']:
                for percent in (10, 30, 50):
                    name = f'{split}_{product}_{percent:03d}.npy'
                    actual = np.load(output / name)
                    expected = array(f'seed_{seed}/reference/{name}')
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=True)
                    difference = np.abs(actual-expected)
                    state_error = max(state_error, float(np.max(difference[np.isfinite(difference)])))
        result = dict(passed=True, prediction_rows=rows, maximum_yield_error=maximum,
            maximum_state_error=state_error, complete_new_component_evaluation=True,
            raw_processing_reproduced=False, independent_recipe_confirmation=False,
            scientific_results_replaced=False, new_scientific_comparisons=0,
            readout_record_sha256=digest(output/'readout.json'))
    result.update(case=case, seed=seed, manifest_sha256=digest(root/'manifest.json'))
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), required=True)
    parser.add_argument('--stage', choices=('state', 'readout', 'check'), required=True)
    args = parser.parse_args()
    run(ROOT, args.case, args.seed, args.stage)
