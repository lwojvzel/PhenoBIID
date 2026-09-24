"""Frozen historical predictions and zero-start residual world components."""
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_stable_remote import read_model
from gru_replication_results import history_selection, history_root, original_direct_root, world_root
from gru_task_priority import RecurrentWorld
from task_aligned_world import TaskAlignedWorld
from task_aligned_data import load
from run_task_aligned_world import tensors, predict, physical_yield
from run_staged_recurrent_world import run_root as staged_root, configure, train_mode
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/anchored_residual_world_v1'
CACHE = ROOT / 'benchmark/cache/anchored_residual_world_v1'
MODES = ('history', 'direct', 'world_frozen', 'world_joint')


def run_root(crop, origin, seed, mode, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_{seed}' / mode


def anchor_arrays(crop, origin, seed, arrays, meta):
    identity = history_selection()[crop, origin]
    source = history_root(crop, origin, seed, identity)
    metric = json.loads((source / 'metrics.json').read_text())
    if sha256(Path(metric['weight'])) != metric['weight_sha256']:
        raise ValueError('Historical source weight changed')
    root = CACHE / crop / f'origin_{origin}/seed_{seed}'; root.mkdir(parents=True, exist_ok=True)
    spec = dict(identity=identity, source=str(source), weight=metric['weight'], weight_sha256=metric['weight_sha256'],
        input_files=meta['files'], normalization=meta['normalization'], code_sha256=sha256(Path(__file__)),
        scope='Training anchor predictions are in-sample fixed-base residual fitting; validation/test predictions are replayed against saved historical arrays.')
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'manifest.json').exists():
            saved = json.loads((root / 'manifest.json').read_text())
            if saved['spec'] != spec:
                raise ValueError('Historical anchor cache changed')
            outputs = {}
            for split in arrays:
                path = root / f'{split}.npy'
                if sha256(path) != saved['files'][split]:
                    raise ValueError('Anchor array changed')
                outputs[split] = np.load(path)
            return outputs, saved
        if identity == 'history_mlp':
            model = TaskAlignedWorld('history_mlp').cuda()
            model.load_state_dict(torch.load(metric['weight'], map_location='cuda', weights_only=True))
        else:
            model = read_model(identity, metric['weight'])
        outputs = {}
        for split, a in arrays.items():
            if identity == 'history_mlp':
                residual, _ = predict(model, tensors(a), size=256)
            else:
                residual = model.predict(np.concatenate((a['history'], a['context']), 1).astype(np.float32))
            outputs[split] = physical_yield(a, residual, meta['normalization'])
            if split != 'train':
                with np.load(source / f'{split}_predictions.npz') as old:
                    for key in ('target', 'source_indices', 'row', 'col', 'year'):
                        np.testing.assert_array_equal(old[key], a[key])
                    np.testing.assert_allclose(outputs[split], old['prediction'], atol=1e-7, rtol=0)
            np.save(root / f'{split}.npy', outputs[split])
        del model; torch.cuda.empty_cache()
        saved = dict(spec=spec, files={s: sha256(root / f'{s}.npy') for s in arrays})
        atomic_json(root / 'manifest.json', saved)
        return outputs, saved


def initial_model(crop, origin, seed, mode):
    if mode.startswith('world'):
        candidates = [world_root(crop, origin, seed), staged_root(crop, origin, seed, 'state')]
        records = [(json.loads((p / 'metrics.json').read_text()), p) for p in candidates]
        records.sort(key=lambda a: (a[0]['state_scores']['validation']['rmse'], str(a[1])))
        record, source = records[0]; model = RecurrentWorld()
        selection = [dict(directory=str(p), validation_lai_rmse=m['state_scores']['validation']['rmse']) for m, p in records]
    else:
        source = history_root(crop, origin, seed, 'history_mlp') if mode == 'history' else original_direct_root(crop, origin, seed)
        record = json.loads((source / 'metrics.json').read_text())
        model = TaskAlignedWorld('history_mlp' if mode == 'history' else 'direct_gru')
        selection = None
    if sha256(Path(record['weight'])) != record['weight_sha256']:
        raise ValueError('Residual initialization changed')
    model.load_state_dict(torch.load(record['weight'], map_location='cpu', weights_only=True))
    nn.init.zeros_(model.yield_head[-1].weight); nn.init.zeros_(model.yield_head[-1].bias)
    return model, dict(directory=str(source), weight=record['weight'], weight_sha256=record['weight_sha256'], state_validation_selection=selection)


def optimizer_groups(model, mode):
    if mode == 'world_frozen':
        return configure(model, 'frozen')
    if mode == 'world_joint':
        return configure(model, 'joint')
    return [dict(params=model.parameters(), lr=1e-4)]


def set_training(model, mode):
    if mode.startswith('world'):
        train_mode(model, 'frozen' if mode == 'world_frozen' else 'joint')
    else:
        model.train()


def compose(anchor, residual, normalization):
    return anchor.astype(float) + residual.astype(float)*normalization['residual_std']
