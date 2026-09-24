"""Identical training-year-stratified contexts for all W26 input conditions."""
import argparse
import fcntl
import importlib.metadata
import json

import numpy as np

import neural_process_readout as scalar
import latent_memory_readout as memory
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/foundation_state_readout_v1'
CACHE = ROOT / 'benchmark/cache/foundation_state_readout_v1'
CONDITIONS = (*scalar.CONDITIONS, 'memory_full', 'memory_random', 'memory_no_feedback', 'memory_zero')
DIMS = dict(scalar.DIMS, **{'memory_'+k: memory.DIMS[k]
                          for k in ('full', 'random', 'no_feedback', 'zero')})
SPLITS = scalar.SPLITS
MAX_CONTEXT = 32768
SEED = 42


def versions():
    return {k: importlib.metadata.version(k) for k in ('tabicl', 'torch', 'numpy', 'scikit-learn')}


def context_indices(years, limit=MAX_CONTEXT, seed=SEED):
    years = np.asarray(years)
    if years.ndim != 1 or len(years) == 0 or limit < 1 or not np.isfinite(years).all():
        raise ValueError('Context years must be a nonempty finite vector')
    if len(years) <= limit:
        return np.arange(len(years), dtype=np.int64)
    unique, counts = np.unique(years, return_counts=True)
    quota = np.zeros(len(unique), dtype=np.int64)
    remaining = limit
    while remaining:
        eligible = np.flatnonzero(quota < counts)
        added = np.minimum(counts[eligible]-quota[eligible], remaining//len(eligible))
        if not np.any(added):
            quota[eligible[:remaining]] += 1
            break
        quota[eligible] += added; remaining -= int(added.sum())
    rng = np.random.default_rng(seed)
    take = np.sort(np.concatenate([rng.choice(np.flatnonzero(years == year), int(n), replace=False)
                                   for year, n in zip(unique, quota)])).astype(np.int64)
    if len(take) != limit or len(np.unique(take)) != limit:
        raise ValueError('Invalid stratified context identity')
    return take


def source(crop, origin, condition):
    module, name = (memory, condition[7:]) if condition.startswith('memory_') else (scalar, condition)
    return module.load(crop, origin, name)


def run_root(crop, origin, condition, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42' / condition


def prepare(crop, origin, audit=False):
    root = CACHE / crop / f'origin_{origin}'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _, labels, _ = scalar.load(crop, origin, 'history')
        if not (labels['train']['year'] <= origin-3).all():
            raise ValueError('Context contains a non-training year')
        take = context_indices(labels['train']['year'])
        spec = dict(crop=crop, origin=origin, seed=SEED, max_context=MAX_CONTEXT,
            conditions=list(CONDITIONS), dimensions=DIMS,
            source_sha256=sha256(ROOT / 'scripts/foundation_state_data.py'), versions=versions(),
            sampling='Equal training-year quota with ascending-year remainders; no target-dependent selection.',
            upstream='Full-window frozen historical anchor, scalar state, and input scalers; no full-pipeline cross-fitting.')
        marker = root / 'manifest.json'
        old = json.loads(marker.read_text()) if marker.exists() else None
        if old is not None and old['spec'] != spec:
            raise ValueError('Foundation context recipe changed')
        if old is not None and not audit:
            return
        if audit and old is None:
            raise ValueError('No prepared foundation context')
        index_file = root / 'context_indices.npy'
        if audit:
            np.testing.assert_array_equal(take, np.load(index_file))
        else:
            np.save(index_file, take)
        files = {index_file.name: sha256(index_file)}
        sources = {}
        for condition in CONDITIONS:
            x, current, meta = source(crop, origin, condition)
            for split in SPLITS:
                for key in labels[split]:
                    np.testing.assert_array_equal(labels[split][key], current[split][key])
                if x[split].shape != (len(labels[split]['target']), DIMS[condition]):
                    raise ValueError('Wrong foundation feature dimensions')
            selected = x['train'][take]
            if not np.isfinite(selected).all():
                raise ValueError('Nonfinite training context')
            path = root / f'{condition}_context.npy'
            if audit:
                np.testing.assert_array_equal(selected, np.load(path))
            else:
                np.save(path, selected)
            files[path.name] = sha256(path); sources[condition] = meta
            print(f'[FOUNDATION CONTEXT] {crop} {origin} {condition} audit={audit}', flush=True)
        year, count = np.unique(labels['train']['year'][take], return_counts=True)
        value = dict(spec=spec, files=files, sources=sources,
            rows=int(len(take)), year_counts={str(y): int(n) for y, n in zip(year, count)})
        if audit:
            if value != old:
                raise ValueError('Foundation context reconstruction differs')
            atomic_json(root / 'audit.json', dict(maximum_replay_error=0.,
                identical_rows_all_conditions=True, training_years_only=True,
                manifest_sha256=sha256(marker), files=files))
        else:
            atomic_json(marker, value)


def load(crop, origin, condition):
    root = CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    if audit['manifest_sha256'] != sha256(root / 'manifest.json') or audit['maximum_replay_error'] != 0:
        raise ValueError('Unaudited context manifest')
    if meta['spec']['source_sha256'] != sha256(ROOT / 'scripts/foundation_state_data.py') or meta['spec']['versions'] != versions():
        raise ValueError('Context source or runtime changed')
    x, labels, source_meta = source(crop, origin, condition)
    if source_meta != meta['sources'][condition]:
        raise ValueError('Upstream state inputs changed')
    for name in ('context_indices.npy', f'{condition}_context.npy'):
        if sha256(root / name) != meta['files'][name]:
            raise ValueError('Context file changed')
    take = np.load(root / 'context_indices.npy')
    np.testing.assert_array_equal(x['train'][take], np.load(root / f'{condition}_context.npy'))
    return x, labels, meta, take


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', choices=(2004, 2008, 2012), type=int, required=True)
    p.add_argument('--audit', action='store_true'); args = p.parse_args()
    prepare(args.crop, args.origin, args.audit)
