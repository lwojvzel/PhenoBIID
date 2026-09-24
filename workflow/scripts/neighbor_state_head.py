"""Use the pinned official ModernNCA model with auditable inference memory."""
import json
from pathlib import Path
import sys

import torch

from review_revision_data import ROOT, sha256

VENDOR = ROOT / 'benchmark/vendor/modern_nca'
REFERENCE = ROOT / 'benchmark/external_code/modern_nca_reference'


def verify_source():
    records = json.loads((REFERENCE / 'manifest.json').read_text())
    if records['commit'] != '1b973adffa4203c3f62c4949957b1ba3efbe60d3':
        raise ValueError('Unregistered ModernNCA revision')
    for row in records['files']:
        path = Path(row['path'])
        if sha256(path) != row['sha256']:
            raise ValueError('Original ModernNCA source changed')
        relative = path.relative_to(REFERENCE)
        expected = path.read_bytes()
        if str(relative) == 'TALENT/model/lib/tabr/utils.py':
            expected = b'from __future__ import annotations\n\n' + expected
            if not expected.endswith(b'\n'):
                expected += b'\n'
        if (VENDOR / relative).read_bytes() != expected:
            raise ValueError('Unregistered ModernNCA adaptation')
    return records


verify_source()
sys.path.insert(0, str(VENDOR))
from TALENT.model.models.modernNCA import ModernNCA


def make(dimension, kind='modern', sample_rate=1.):
    if kind not in ('linear', 'modern'):
        raise ValueError('Unknown neighborhood head')
    embedding = dict(type='PLREmbeddings', n_frequencies=8, frequency_scale=1.,
                     d_embedding=8, lite=True) if kind == 'modern' else None
    return ModernNCA(d_in=dimension, d_num=dimension, d_out=1, dim=128,
        dropout=.1, d_block=256, n_blocks=int(kind == 'modern'),
        num_embeddings=embedding, temperature=1., sample_rate=sample_rate)


def candidate_indices(size, queries, device=None):
    queries = torch.as_tensor(queries, dtype=torch.long, device=device)
    if queries.ndim != 1 or len(queries) == 0 or (queries < 0).any() or (queries >= size).any():
        raise ValueError('Invalid training identities')
    if len(queries.unique()) != len(queries):
        raise ValueError('Duplicate query identities')
    keep = torch.ones(size, dtype=torch.bool, device=queries.device)
    keep[queries] = False
    return torch.arange(size, device=queries.device)[keep]


def encode(model, x):
    if model.num_embeddings is not None and model.d_num > 0:
        x = torch.cat((model.num_embeddings(x[:, :model.d_num]).flatten(1),
                       x[:, model.d_num:]), dim=-1)
    x = model.encoder(x)
    return model.post_encoder(x) if model.n_blocks > 0 else x


@torch.no_grad()
def encode_memory(model, x, batch_size=1024):
    if model.training or batch_size < 1:
        raise ValueError('Memory must be constructed in eval mode with a positive batch size')
    return torch.cat([encode(model, x[i:i+batch_size]) for i in range(0, len(x), batch_size)])


@torch.no_grad()
def from_memory(model, x, keys, targets):
    if model.training:
        raise ValueError('Inference memory cannot be used during training')
    if len(keys) != len(targets) or not len(keys) or targets.ndim != 1:
        raise ValueError('Invalid inference memory identities')
    distances = torch.cdist(encode(model, x), keys, p=2)/model.T
    return torch.softmax(-distances, dim=-1) @ targets
