"""Cached matched state representations and small nonlinear terminal heads."""
import argparse
import fcntl
import importlib.metadata
import json

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tabm import TabM
from torch import nn
from threadpoolctl import threadpool_limits

from water_response_state import context, fit_response, water_features, RESULT as WATER
from linear_state_yield import yield_features
from yield_sensitive_state import CACHE as STATE_CACHE
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/neural_process_readout_v1'
CACHE = ROOT / 'benchmark/cache/neural_process_readout_v1'
CONDITIONS = ('history', 'known', 'lai', 'predicted', 'previous', 'climatology', 'constant', 'observed', 'direct')
HEADS = ('mlp', 'tabm')
DIMS = dict(history=20, known=92, lai=104, predicted=182, previous=182,
            climatology=182, constant=182, observed=182, direct=560)
CODE = ('neural_process_readout.py', 'water_response_state.py', 'linear_state_yield.py')
SPLITS = ('train', 'validation', 'test')
LABELS = ('target', 'row', 'col', 'year', 'source_indices')


def versions():
    return {k: importlib.metadata.version(k) for k in ('torch', 'tabm', 'numpy', 'scikit-learn')}


def cache_root(crop, origin):
    return CACHE / crop / f'origin_{origin}'


def run_root(crop, origin, head, condition, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42' / head / condition


def features(a, predicted, difference, norm, response, condition):
    if condition == 'lai':
        x = yield_features(a, None, {'state_innovation': predicted}, 'state_innovation')
    elif condition in ('history', 'known', 'direct'):
        name = {'history': 'history', 'known': 'known_state', 'direct': 'direct_paired'}[condition]
        x = yield_features(a, difference, {}, name)
    else:
        x = water_features(a, predicted, norm, response, condition)
    if x.shape != (len(predicted), DIMS[condition]) or not np.isfinite(x).all():
        raise ValueError('Invalid neural readout inputs')
    return x


def source(crop, origin):
    arrays, meta, anchors, states, spec = context(crop, origin)
    previous = WATER / crop / f'origin_{origin}'
    if json.loads((previous / 'config.json').read_text()) != spec:
        raise ValueError('W17 source or code changed')
    checked = json.loads((ROOT / f'visualize/paper_experiments/water_response_state_v1/audit/{crop}_{origin}.json').read_text())
    if checked['fits'] != 25 or checked['maximum_replay_error'] != 0:
        raise ValueError('W17 full replay incomplete')
    differences = {}
    for split in SPLITS:
        with np.load(STATE_CACHE / crop / f'origin_{origin}/{split}.npz') as saved:
            for key in ('row', 'col', 'year', 'source_indices'):
                np.testing.assert_array_equal(saved[key], arrays[split][key])
            differences[split] = saved['weather_difference']
    return arrays, meta, anchors, states, differences, spec


def prepare(crop, origin, audit=False):
    root = cache_root(crop, origin); root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arrays, meta, anchors, states, differences, upstream = source(crop, origin)
        norm = meta['normalization']; response = fit_response(arrays['train'], norm)
        spec = dict(crop=crop, origin=origin, upstream=upstream, conditions=list(CONDITIONS), dimensions=DIMS,
            code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE}, versions=versions(),
            scope='Frozen in-sample free rollouts, not out-of-fold; observed condition is diagnostic only.')
        path = root / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['spec'] != spec:
                raise ValueError('Neural process cache specification changed')
            if not audit:
                return
        elif audit:
            raise ValueError('Missing cache for independent audit')
        files = {}
        for split, a in arrays.items():
            data = dict(history_prediction=anchors[split], **{k: a[k] for k in LABELS})
            file = root / f'{split}_labels.npz'
            if audit:
                with np.load(file) as old:
                    for key, value in data.items():
                        np.testing.assert_array_equal(old[key], value)
            else:
                np.savez(file, **data)
            files[file.name] = sha256(file)
        for condition in CONDITIONS:
            scaler = None
            for split in SPLITS:
                x = features(arrays[split], states[split]['state_only'], differences[split], norm, response, condition)
                if split == 'train':
                    scaler = StandardScaler().fit(x)
                    file = root / f'{condition}_scaler.joblib'
                    if audit:
                        old = joblib.load(file)
                        for key in ('mean_', 'scale_', 'var_'):
                            np.testing.assert_array_equal(getattr(old, key), getattr(scaler, key))
                    else:
                        joblib.dump(scaler, file)
                    files[file.name] = sha256(file)
                standardized = scaler.transform(x).astype(np.float32)
                file = root / f'{condition}_{split}.npy'
                if audit:
                    np.testing.assert_array_equal(np.load(file, mmap_mode='r'), standardized)
                else:
                    np.save(file, standardized)
                files[file.name] = sha256(file)
            print(f'[NEURAL CACHE] {crop} {origin} {condition} audit={audit}', flush=True)
        if audit:
            if files != manifest['files']:
                raise ValueError('Neural cache hash changed')
            atomic_json(root / 'audit.json', dict(full_array_replay=True, maximum_replay_error=0.,
                training_scalers_verified=True, source_units_and_alignment_verified=True, files=files))
        else:
            atomic_json(path, dict(spec=spec, normalization=norm, files=files))


def load(crop, origin, condition):
    root = cache_root(crop, origin)
    meta = json.loads((root / 'manifest.json').read_text())
    checked = json.loads((root / 'audit.json').read_text())
    if checked['files'] != meta['files'] or checked['maximum_replay_error'] != 0:
        raise ValueError('Unaudited neural input cache')
    if meta['spec']['versions'] != versions() or meta['spec']['code_hashes'] != {
            k: sha256(ROOT / 'scripts' / k) for k in CODE}:
        raise ValueError('Neural source/runtime changed')
    data, labels = {}, {}
    for split in SPLITS:
        for name in (f'{split}_labels.npz', f'{condition}_{split}.npy', f'{condition}_scaler.joblib'):
            if sha256(root / name) != meta['files'][name]:
                raise ValueError(f'Neural readout cache changed: {name}')
        data[split] = np.load(root / f'{condition}_{split}.npy')
        with np.load(root / f'{split}_labels.npz') as f:
            labels[split] = {k: f[k] for k in f.files}
    return data, labels, meta


class NeuralReadout(nn.Module):
    def __init__(self, head, dimensions):
        super().__init__()
        self.head = head
        if head == 'tabm':
            self.network = TabM.make(n_num_features=dimensions, d_out=1, k=16,
                n_blocks=3, d_block=128, dropout=.1)
        elif head == 'mlp':
            blocks = []
            for width in (dimensions, 128, 128):
                blocks.extend((nn.Linear(width, 128), nn.ReLU(), nn.Dropout(.1)))
            self.network = nn.Sequential(*blocks, nn.Linear(128, 1))
        else:
            raise ValueError('Unknown head')

    def forward(self, x):
        value = self.network(x)
        return value[..., 0] if self.head == 'tabm' else value


@torch.no_grad()
def predict(model, data, device='cuda'):
    model.eval()
    blocks = [model(data[start:start+512].to(device)).mean(1).cpu().numpy()
              for start in range(0, len(data), 512)]
    return np.concatenate(blocks).astype(float)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--audit', action='store_true'); a = p.parse_args()
    with threadpool_limits(limits=2):
        prepare(a.crop, a.origin, a.audit)
