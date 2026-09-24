"""Shared physical cohort for the registered thirteen-year rolling evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path

import numpy as np

from forecast_bridge_data import ROOT, identity_hash
from inseason_complete_inputs import complete_inputs, RECIPES
from review_revision_data import sha256
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from run_yield_only_classical_baselines import _exponential_smoothing_predictions

CACHE = ROOT / 'benchmark/cache/inseason_13year_v1'
OUT = ROOT / 'benchmark/results/inseason_13year_baselines_v1'
BLOCKS = {2001: (2002, 2003, 2004), 2005: (2006, 2007, 2008), 2009: tuple(range(2010, 2017))}
YEARS = tuple(y for group in BLOCKS.values() for y in group)
ALPHAS = (.1, .2, .35, .5, .65, .8, 1.)


def partition(a, cutoff):
    if cutoff not in BLOCKS:
        raise ValueError('Unregistered temporal block')
    y = a['year']
    return dict(inner_fit=np.flatnonzero(y <= cutoff-2),
        inner_validation=np.flatnonzero((y > cutoff-2) & (y <= cutoff)),
        full_fit=np.flatnonzero(y <= cutoff),
        evaluation=np.flatnonzero(np.isin(y, BLOCKS[cutoff])))


def prepare(crop):
    root = CACHE / crop
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'manifest.json').exists():
            load(crop)
            return
        raw, _, metadata, _, _, _, provenance = complete_inputs(crop)
        raw['metadata'] = metadata
        for name, values in raw.items():
            np.save(root / f'{name}.npy', values)
        original = ROOT / 'benchmark/cache/multimodal_main' / crop
        population = {k: np.load(original / f'{k}.npy', mmap_mode='r')
                      for k in ('year', 'row', 'col', 'target')}
        for alpha in ALPHAS:
            pred = _exponential_smoothing_predictions(population, alpha, 0.)
            np.save(root / f'smoothing_{alpha:g}.npy', pred[raw['source_indices']])
        sources = {}
        for end, years in BLOCKS.items():
            origin = end+3
            old = ROOT / f'benchmark/results/inseason_signal_match_v1/pipelines/{crop}/origin_{origin}/seed_42'
            marker = json.loads((old / 'complete.json').read_text())
            check_files(old, marker['files'])
            sources[str(old / 'complete.json')] = sha256(old / 'complete.json')
            parts = ('validation', 'test') if end == 2009 else ('validation',)
            gathered = {}
            for split in parts:
                with np.load(old / f'{split}_labels.npz') as f:
                    ix = np.searchsorted(raw['source_indices'], f['source_indices'])
                    for key in ('target', 'year', 'row', 'col', 'source_indices'):
                        np.testing.assert_array_equal(raw[key][ix], f[key])
                    gathered[split] = ix
            ix = np.concatenate(list(gathered.values()))
            np.testing.assert_array_equal(ix, partition(raw, end)['evaluation'])
            np.savez_compressed(root / f'block_{end}_identities.npz', indices=ix,
                **{k: raw[k][ix] for k in ('target', 'year', 'row', 'col', 'source_indices')})
            for recipe in ('ndvi', 'gpp', 'ndvi_gpp'):
                for mode in ('biid', 'climatology'):
                    for percent in range(0, 101, 10):
                        values = []
                        for split in parts:
                            path = old / f'{split}_{recipe}_{mode}_{percent:03d}.npz'
                            if not path.exists():
                                continue
                            sources[str(path)] = sha256(path)
                            with np.load(path) as f:
                                pred = f['prediction']
                            if len(pred) != len(gathered[split]):
                                raise ValueError('Frozen prediction shape mismatch')
                            values.append(pred)
                        if len(values) == len(parts):
                            np.save(root / f'world_{end}_{recipe}_{mode}_{percent:03d}.npy', np.concatenate(values))
            path = root / f'world_{end}_{RECIPES[crop]}_biid_010.npy'
            if not path.exists():
                raise ValueError('Missing registered Figure 4 world-model result')
        atomic_json(root / 'provenance.json', dict(raw=provenance, world=sources,
            world_weights_refitted=False, current_code_sha256=sha256(Path(__file__))))
        files = {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'prepare.lock'}
        atomic_json(root / 'manifest.json', dict(crop=crop, files=files, years=YEARS,
            blocks=BLOCKS, identities=identity_hash(raw['source_indices']),
            physical=True, inherited_selection_records_preserved=True,
            code_sha256=sha256(Path(__file__))))
        print(f'[13 YEAR INPUT READY] {crop} rows={len(raw["year"])}', flush=True)


def load(crop, verify=True):
    root = CACHE / crop
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['code_sha256'] != sha256(Path(__file__)):
        raise ValueError('Changed thirteen-year cohort implementation')
    if verify:
        check_files(root, manifest['files'])
    raw = {path.stem: np.load(path, mmap_mode='r') for path in root.glob('*.npy')
           if not path.stem.startswith(('world_', 'smoothing_'))}
    if tuple(manifest['years']) != YEARS:
        raise ValueError('Changed evaluation years')
    return raw, manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES)+('all',), required=True)
    args = p.parse_args()
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(prepare, c) for c in RECIPES]:
                future.result()
    else:
        prepare(args.crop)
