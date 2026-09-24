"""Training-only numerical embeddings on the unchanged W18 feature arrays."""
import argparse
import fcntl
import importlib.metadata
import json
import warnings

import numpy as np
import torch
from torch import nn
from tabm import TabM
from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings

from neural_process_readout import (ROOT, CROPS, CONDITIONS, DIMS, SPLITS, predict,
    versions as old_versions, load as old_load)
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/numeric_embedding_readout_v1'
CACHE = ROOT / 'benchmark/cache/numeric_embedding_readout_v1'
HEADS = ('tabm',)
CODE = ('numeric_embedding_readout.py', 'neural_process_readout.py')


def versions():
    return dict(old_versions(), rtdl_num_embeddings=importlib.metadata.version('rtdl-num-embeddings'))


def run_root(crop, origin, head, condition, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42' / head / condition


def training_bins(x):
    value = torch.as_tensor(x, dtype=torch.float32, device='cpu')
    if value.ndim != 2 or len(value) <= 16 or not torch.isfinite(value).all():
        raise ValueError('Expected finite full training features')
    constant = (value == value[0]).all(0)
    bins = [None]*value.shape[1]
    columns = torch.where(~constant)[0]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The .*feature has just two bin edges')
        for group in columns.split(32):
            if len(group):
                for column, edges in zip(group.tolist(), compute_bins(value[:, group], n_bins=16)):
                    bins[column] = edges
    for column in torch.where(constant)[0].tolist():
        center = value[0, column]
        bins[column] = torch.stack((center-1., center+1.))
    if any(not torch.all(b[1:] > b[:-1]) or not torch.isfinite(b).all() for b in bins):
        raise ValueError('Nonfinite or degenerate numerical bins')
    return bins, torch.where(constant)[0].tolist()


def prepare(crop, origin, audit=False):
    root = CACHE / crop / f'origin_{origin}'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root / 'manifest.json'; previous = json.loads(marker.read_text()) if marker.exists() else None
        if audit and previous is None:
            raise ValueError('No bins to audit')
        files, constants, original = {}, {}, None
        for condition in CONDITIONS:
            x, _, meta = old_load(crop, origin, condition)
            if original is None:
                original = meta
            elif original != meta:
                raise ValueError('W18 conditions do not share a source manifest')
            bins, columns = training_bins(x['train'])
            constants[condition] = columns
            path = root / f'{condition}_bins.pt'
            if audit:
                old = torch.load(path, map_location='cpu', weights_only=True)
                if len(old) != len(bins):
                    raise ValueError('Bin feature count changed')
                for a, b in zip(old, bins):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            elif not path.exists():
                torch.save(bins, path)
            else:
                old = torch.load(path, map_location='cpu', weights_only=True)
                if len(old) != len(bins):
                    raise ValueError('Existing bin feature count changed')
                for a, b in zip(old, bins):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            files[path.name] = sha256(path)
            print(f'[NUMERIC BINS] {crop} {origin} {condition} constants={len(columns)} audit={audit}', flush=True)
        spec = dict(crop=crop, origin=origin, original_cache=original, conditions=list(CONDITIONS),
            dimensions=DIMS, versions=versions(), code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            embedding=dict(version='B', activation=False, dimensions=12, quantile_bins=16,
                constant_columns=constants, constant_edges='Training standardized value plus/minus one.'),
            scope='Only training features determine bins; original states, labels, cohorts and scaling unchanged.')
        manifest = dict(spec=spec, normalization=original['normalization'], files=files)
        if previous and manifest != previous:
            raise ValueError('Numerical bin recipe changed')
        if audit:
            atomic_json(root / 'audit.json', dict(full_array_replay=True, maximum_replay_error=0.,
                files=files, training_only_bins=True))
        else:
            atomic_json(marker, manifest)


def load_bins(crop, origin, condition):
    return torch.load(CACHE / crop / f'origin_{origin}/{condition}_bins.pt', weights_only=True, map_location='cpu')


def load(crop, origin, condition):
    root = CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    if not audit['training_only_bins'] or audit['maximum_replay_error'] != 0 or audit['files'] != meta['files']:
        raise ValueError('Unaudited numerical bins')
    if meta['spec']['versions'] != versions() or meta['spec']['code_hashes'] != {
            k: sha256(ROOT / 'scripts' / k) for k in CODE}:
        raise ValueError('Numerical source/runtime changed')
    file = root / f'{condition}_bins.pt'
    if sha256(file) != meta['files'][file.name]:
        raise ValueError('Numerical bin boundaries changed')
    x, labels, original = old_load(crop, origin, condition)
    if original != meta['spec']['original_cache']:
        raise ValueError('Original numerical inputs changed')
    return x, labels, meta


class NeuralReadout(nn.Module):
    def __init__(self, head, dimensions, bins):
        super().__init__()
        if head != 'tabm' or len(bins) != dimensions:
            raise ValueError('Expected TabM and one bin array per numerical feature')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The .*feature has just two bin edges')
            embedding = PiecewiseLinearEmbeddings(bins, 12, activation=False, version='B')
        self.network = TabM.make(n_num_features=dimensions, d_out=1, k=16,
            n_blocks=3, d_block=128, dropout=.1, num_embeddings=embedding)

    def forward(self, x):
        return self.network(x)[..., 0]


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--audit', action='store_true'); a = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    prepare(a.crop, a.origin, a.audit)
