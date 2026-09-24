"""Frozen protocol and verification helpers for nested seasonal readout refits."""
import json

from inseason_13year_data import ROOT, BLOCKS, RECIPES
from review_revision_data import sha256
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_nested_world_v1'
RATIOS = (.1, .3, .5)
LABELS = ('source_indices', 'year', 'row', 'col', 'target')
CODE = ('inseason_nested_common.py', 'inseason_13year_data.py',
        'forecast_bridge_data.py', 'inseason_complete_inputs.py',
        'inseason_ndvi_reuse.py', 'ndvi_tail_replacement.py')


def folder(stage, crop, cutoff, seed=42, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_{seed}' / stage


def hashes(extra=()):
    return {name: sha256(ROOT / 'scripts' / name) for name in (*CODE, *extra)}


def verify(root, expected=None):
    record = json.loads((root / 'complete.json').read_text())
    check_files(root, record['files'])
    for name, digest in record['code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError(f'Changed completed nested source: {name}')
    if expected is not None and record['code_sha256'] != expected:
        raise ValueError('Changed nested protocol')
    return record


def finish(root, code, **metadata):
    files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
             if p.is_file() and p.name not in ('run.lock', 'complete.json')}
    atomic_json(root / 'complete.json', dict(files=files, code_sha256=code, **metadata))


def register(root, spec):
    spec = json.loads(json.dumps(spec))
    path = root / 'config.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Changed nested registration')
    atomic_json(path, spec)


def branches(crop):
    if crop == 'maize':
        return (('trend', .5), ('mlp', .5))
    return (('tabm' if crop == 'soybean' else 'mlp', 1.),)
