"""Matched full-window inputs and deterministic training identities for W27."""
import hashlib
import importlib.metadata
import json

import numpy as np
import torch

from foundation_state_data import ROOT, CACHE, CROPS, CONDITIONS, DIMS, load as original_load
from neighbor_state_head import make, candidate_indices, encode_memory, from_memory, verify_source
from review_revision_data import sha256

RESULT = ROOT / 'benchmark/results/neighbor_state_readout_v1'
HEADS = ('linear', 'modern')


def run_root(crop, origin, head, condition, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42' / head / condition


def versions():
    return {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'scikit-learn')}


def load(crop, origin, condition):
    x, arrays, meta, unused_context_indices = original_load(crop, origin, condition)
    for split, years in (('train', range(1982, origin-2)),
                         ('validation', range(origin-2, origin+1)), ('test', range(origin+1, origin+5))):
        if not np.isin(arrays[split]['year'], list(years)).all():
            raise ValueError('Neighbor library or query has a wrong time window')
        if x[split].shape != (len(arrays[split]['target']), DIMS[condition]) or not np.isfinite(x[split]).all():
            raise ValueError('Invalid full-window retrieval input')
    return x, arrays, meta


def batches(order, batch_size=512):
    if order.ndim != 1 or len(order) < 2 or batch_size < 2:
        raise ValueError('Training requires at least two rows and a nontrivial batch')
    parts = list(order.split(batch_size))
    if len(parts[-1]) == 1:
        last = parts.pop()
        parts[-1] = torch.cat((parts[-1], last))
    return parts


def sample_neighbors(size, queries, generator, limit=4096):
    if limit < 1:
        raise ValueError('Nonpositive neighbor sample limit')
    candidates = candidate_indices(size, queries, 'cpu')
    if not len(candidates):
        raise ValueError('No external training neighbors after excluding queries')
    return candidates[torch.randperm(len(candidates), generator=generator)[:limit]]


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode('ascii'))
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def library_identity(x, arrays, normalized_target):
    return dict(features=array_digest(x), normalized_target=array_digest(normalized_target),
        rows=len(x), fields={k: array_digest(arrays[k])
            for k in ('target', 'row', 'col', 'year', 'source_indices')},
        first_year=int(arrays['year'].min()), last_year=int(arrays['year'].max()))


def resource_gate():
    from probe_neighbor_state_head import RESULT as probe, CASES
    verify_source()
    checks = {}
    for kind, n, d in CASES:
        path = probe / f'{kind}_{n}_{d}/audit.json'
        a = json.loads(path.read_text())
        if a['spec']['code_hashes'] != {f: sha256(ROOT / 'scripts' / f)
                for f in ('neighbor_state_head.py', 'probe_neighbor_state_head.py')}:
            raise ValueError('Retrieval resource implementation changed')
        if any(a['errors'][k] != 0 for k in ('repeat', 'checkpoint_replay', 'memory_rebuild')):
            raise ValueError('Retrieval exact replay failed')
        if a['errors']['chunk'] > 1e-4 or a['errors'].get('official', 0) > 1e-4 or a['peak_allocated_mib'] > 3000:
            raise ValueError('Retrieval resource or numerical threshold exceeded')
        checks[path.parent.name] = a
    return checks


@torch.no_grad()
def predict(model, train, target, query):
    model.eval()
    keys = encode_memory(model, train, 1024)
    value = np.concatenate([from_memory(model, part.cuda(), keys, target).cpu().numpy()
                            for part in query.split(256)]).astype(float)
    if value.shape != (len(query),) or not np.isfinite(value).all():
        raise ValueError('Invalid retrieval prediction')
    return value
