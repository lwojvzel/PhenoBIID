"""Rebuild seed42 history, then realize the same fixed recipe at seeds45/48."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_nested_common import LABELS, hashes, register, finish, verify
from inseason_pipeline_seed_history import fit_fixed, residual_target
from numeric_embedding_readout import load as tabm_load, load_bins, NeuralReadout, predict
from review_revision_data import sha256
from run_crop_head_signal_match import expert_root
from run_inseason_direct_baselines import score
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from run_soybean_expert_seed_recheck import mlp_predict
from task_aligned_world import TaskAlignedWorld

OUT = ROOT / 'benchmark/results/inseason_pipeline_seeds_v1'
CODE = ('inseason_pipeline_seed_history.py', 'run_inseason_pipeline_seed_history.py',
        'task_aligned_world.py', 'numeric_embedding_readout.py', 'neural_process_readout.py',
        'multimodal_baseline.py', 'run_soybean_expert_seed_recheck.py')


def root_for(crop, cutoff, seed, component, smoke=False):
    return OUT / ('history_smoke' if smoke else 'history') / crop / f'cutoff_{cutoff}/seed_{seed}' / component


def load_mlp(crop, cutoff):
    origin = cutoff + 3
    source = ROOT / f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_{origin}/history_mlp/seed_42'
    config = json.loads((source / 'config.json').read_text())
    metrics = json.loads((source / 'metrics.json').read_text())
    if metrics['selected_epoch'] < 1 or metrics['weight_sha256'] != sha256(source / 'model_best.pt'):
        raise ValueError('Original MLP source is not a verified trained checkpoint')
    for name, digest in config['code_hashes'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError('Original MLP implementation changed')
    if (config['batch'], config['micro_batch']) != (2048, 256):
        raise ValueError('Original MLP batching differs')
    x, arrays, sources = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        file = ROOT / f'benchmark/cache/task_aligned_world_v1/{crop}/origin_{origin}/{split}.npz'
        if sha256(file) != config['input_manifest']['files'][split]:
            raise ValueError('Historical task inputs changed')
        sources[str(file)] = sha256(file)
        with np.load(file) as saved:
            x[split] = np.concatenate((saved['history'], saved['context']), 1)
            arrays[split] = {k: saved[k] for k in (*LABELS, 'baseline', 'target_residual')}
    if arrays['train']['year'].max() != cutoff:
        raise ValueError('Unexpected historical training cutoff')
    for file in (source / 'config.json', source / 'metrics.json', source / 'model_best.pt'):
        sources[str(file)] = sha256(file)
    return x, arrays, config['input_manifest']['normalization'], metrics['selected_epoch'], source, sources


def run(crop, cutoff, seed, component, smoke=False):
    if component == 'tabm' and crop != 'soybean':
        raise ValueError('Only soybean has the additional TabM historical component')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    root = root_for(crop, cutoff, seed, component, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(CODE)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        if seed != 42 and not smoke:
            replay = verify(root_for(crop, cutoff, 42, component), code)
            if not replay['original_predictions_replayed']:
                raise ValueError('Original seed42 history rebuild must succeed first')
        started = time.monotonic()
        bins = None
        if component == 'mlp':
            x, arrays, norm, epochs, source, sources = load_mlp(crop, cutoff)
            target = arrays['train']['target_residual']
        else:
            x, arrays, meta = tabm_load(crop, cutoff + 3, 'history')
            source = ROOT / f'benchmark/results/numeric_embedding_readout_v1/pipelines/soybean/origin_{cutoff + 3}/seed_42/tabm/history'
            cfg = json.loads((source / 'config.json').read_text())
            audit = json.loads((source / 'audit.json').read_text())
            if audit['smoke'] or sha256(source / 'model_best.pt') != audit['weight_sha256']:
                raise ValueError('Unverified original TabM')
            for name, digest in cfg['code_hashes'].items():
                if sha256(ROOT / 'scripts' / name) != digest:
                    raise ValueError('Original TabM code changed')
            epochs = json.loads((source / 'metrics.json').read_text())['selected_epoch']
            norm = dict(scale=meta['normalization']['residual_std'])
            bins = load_bins(crop, cutoff + 3, 'history')
            parent = root_for(crop, cutoff, seed, 'mlp')
            verify(parent, code)
            sources = {str(p): sha256(p) for p in (source / 'config.json', source / 'metrics.json',
                source / 'model_best.pt', parent / 'complete.json',
                ROOT / f'benchmark/cache/numeric_embedding_readout_v1/{crop}/origin_{cutoff + 3}/manifest.json')}
            for split in arrays:
                with np.load(parent / f'{split}_predictions.npz') as saved:
                    for key in LABELS:
                        np.testing.assert_array_equal(arrays[split][key], saved[key])
                    arrays[split]['history_prediction'] = saved['prediction']
            target = residual_target(arrays['train']['target'], arrays['train']['history_prediction'], norm['scale'])
        original_epochs = epochs
        if smoke:
            take = {s: np.concatenate([np.flatnonzero(a['year'] == y)[:16] for y in np.unique(a['year'])]) for s, a in arrays.items()}
            target = target[take['train']]
            x = {s: v[take[s]] for s, v in x.items()}
            arrays = {s: {k: v[take[s]] for k, v in a.items()} for s, a in arrays.items()}
            epochs = min(epochs, 1)
        register(root, dict(crop=crop, cutoff=cutoff, seed=seed, component=component, smoke=smoke,
            source=str(source), sources=sources, normalization=norm, selected_fixed_epochs=epochs,
            original_selected_epochs=original_epochs, code_sha256=code,
            selection='Fixed seed42 development capacity; no new validation or evaluation selection',
            initialization_component=epochs == 0, history_predictions_in_sample=True,
            training_rows=len(x['train']), training_years=np.unique(arrays['train']['year']).tolist()))
        trace = []

        def report(row):
            trace.append(row)
            atomic_json(root / 'training.json', trace)
            print(f'[FIXED HISTORY] {crop} {cutoff} {seed} {component} epoch={row["epoch"]}/{epochs}', flush=True)

        model, steps = fit_fixed(x['train'], target, component, epochs, seed, bins, report)
        atomic_json(root / 'training.json', trace)
        weight = root / 'model.pt'
        torch.save(model.state_dict(), weight)
        restored = (TaskAlignedWorld('history_mlp') if component == 'mlp' else NeuralReadout('tabm', 20, bins)).cuda()
        restored.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
        source_model = None
        if seed == 42 and not smoke:
            source_model = (TaskAlignedWorld('history_mlp') if component == 'mlp' else NeuralReadout('tabm', 20, bins)).cuda()
            source_model.load_state_dict(torch.load(source / 'model_best.pt', map_location='cpu', weights_only=True))
        errors, metrics = {}, {}
        for split, features in x.items():
            a = arrays[split]
            tensor = torch.from_numpy(features)

            def physical(m):
                if component == 'mlp':
                    return mlp_predict(m, tensor, a['baseline'], norm)
                return a['history_prediction'] + norm['scale'] * predict(m, tensor)

            prediction = physical(model)
            np.testing.assert_array_equal(prediction, physical(restored))
            if source_model is not None:
                reference = physical(source_model)
                errors[split] = float(np.max(np.abs(reference - prediction)))
                np.testing.assert_allclose(prediction, reference, atol=1e-7, rtol=0)
                if split != 'train':
                    with np.load(source / f'{split}_predictions.npz') as saved:
                        for key in LABELS:
                            np.testing.assert_array_equal(a[key], saved[key])
                        np.testing.assert_allclose(prediction, saved['prediction' if component == 'mlp' else 'component_prediction'], atol=1e-7, rtol=0)
                if component == 'tabm' and split in ('train', 'validation'):
                    old = expert_root(crop, cutoff + 3)
                    check_files(old, json.loads((old / 'audit.json').read_text())['files'])
                    with np.load(old / f'{split}_labels.npz') as saved:
                        np.testing.assert_allclose(prediction, saved['history_prediction'], atol=1e-7, rtol=0)
            np.savez_compressed(root / f'{split}_predictions.npz', prediction=prediction, **{k: a[k] for k in LABELS})
            metrics[split] = score(a['target'], prediction, a['year'])
        for name, digest in sources.items():
            if sha256(Path(name)) != digest:
                raise ValueError('Historical source changed during fixed-epoch fitting')
        atomic_json(root / 'metrics.json', dict(scores=metrics, original_replay_error=errors))
        finish(root, code, crop=crop, cutoff=cutoff, seed=seed, component=component, smoke=smoke,
            optimizer_steps=steps, fixed_epochs=epochs, initialization_component=epochs == 0,
            original_predictions_replayed=seed == 42 and not smoke,
            seconds=time.monotonic() - started, full_weight_replay=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=tuple(RECIPES), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), required=True)
    parser.add_argument('--component', choices=('mlp', 'tabm'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.crop, args.cutoff, args.seed, args.component, args.smoke)
