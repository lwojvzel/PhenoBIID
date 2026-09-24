"""Reuse causal prefix states and fill the precise inner/outer block gaps."""
import argparse
import fcntl
import json
import time

import numpy as np
import torch

from forecast_bridge_state import ForecastState
from inseason_13year_data import CACHE, load, partition
from inseason_nested_common import (ROOT, BLOCKS, RECIPES, RATIOS, LABELS,
                                    folder, hashes, verify, finish, register)
from inseason_signal_matching import remap_product
from ndvi_tail_replacement import tail_mask
from review_revision_data import sha256
from run_forecast_bridge_state import run_root as state_root
from run_inseason_matched_readout import verify_done
from run_ndvi_signal_permutation import check_files
from run_ndvi_tail_replacement import forecast_prefix
from run_review_revision_parallel import atomic_json

EXTRA = ('run_inseason_nested_states.py', 'forecast_bridge_state.py',
         'run_forecast_bridge_state.py', 'run_ndvi_tail_replacement.py',
         'run_inseason_matched_readout.py', 'inseason_signal_matching.py')


def forward_cache(crop, raw):
    root = ROOT / f'benchmark/results/inseason_matched_readout_v1/pipelines/{crop}/fit_2009/seed_42/forward_prefix'
    verify_done(root)
    with np.load(root / 'identities.npz') as saved:
        ix = np.searchsorted(raw['source_indices'], saved['source_indices'])
        for name in ('source_indices', 'year', 'row', 'col'):
            np.testing.assert_array_equal(raw[name][ix], saved[name])
        ends = saved['cutoff']
    if np.any(ends >= raw['year'][ix]):
        raise ValueError('Forward state cache contains nonpast training')
    return root, ix, ends


def available_cutoff(crop, product, upper):
    path = state_root(crop, product, 'biid', upper, 42).parent.parent
    choices = [int(p.parent.parent.name.removeprefix('cutoff_'))
               for p in path.glob('cutoff_*/seed_42/complete.json')]
    valid = [x for x in choices if x <= upper]
    if not valid:
        raise ValueError('No strictly preceding fitted state')
    return max(valid)


def run(crop, cutoff, seed=42, smoke=False):
    if seed != 42:
        raise ValueError('Cached state seed is fixed at 42')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = folder('states', crop, cutoff, seed, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(EXTRA)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        raw, _ = load(crop)
        rows = partition(raw, cutoff)
        source, source_ix, ends = forward_cache(crop, raw)
        provenance = {str(source / 'complete.json'): sha256(source / 'complete.json')}
        register(root, dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke,
            cache_sha256=sha256(CACHE / crop / 'manifest.json'), code_sha256=code,
            reuse=str(source), ratios=RATIOS, recipe=RECIPES[crop],
            state_policy='Inner: latest saved cutoff <= inner fit end; outer: exact block cutoff',
            forward_training='Unchanged causal 1993+ prefix cache; no state retraining'))
        started = time.monotonic()
        checks = []
        for stage, group, upper in (('inner', 'inner_validation', cutoff-2),
                                    ('full', 'evaluation', cutoff)):
            take = rows[group]
            if smoke:
                take = np.concatenate([take[raw['year'][take] == y][:12] for y in np.unique(raw['year'][take])])
            np.savez_compressed(root / f'{stage}_identities.npz', raw_indices=take,
                                **{k: raw[k][take] for k in LABELS})
            positions = np.searchsorted(source_ix, take)
            np.testing.assert_array_equal(source_ix[positions], take)
            outputs = {r: {} for r in RATIOS}
            for product in RECIPES[crop].split('_'):
                end = available_cutoff(crop, product, upper)
                if end >= int(raw['year'][take].min()) or (stage == 'full' and end != cutoff):
                    raise ValueError('Wrong inner/outer state cutoff')
                state = state_root(crop, product, 'biid', end, 42)
                marker = json.loads((state / 'complete.json').read_text())
                check_files(state, marker['files'])
                if marker['smoke'] or marker['selected_epochs'] < 1 or marker['full_fit_cutoff'] != end:
                    raise ValueError('Wrong or untrained state')
                cfg = json.loads((state / 'config.json').read_text())
                for name, digest in cfg['code_sha256'].items():
                    if sha256(ROOT / 'scripts' / name) != digest:
                        raise ValueError('Changed state source')
                norm = json.loads((state / 'normalization.json').read_text())
                if max(norm['fit_years']) > end:
                    raise ValueError('Future state statistics')
                for name in ('complete.json', 'model.pt', 'normalization.json'):
                    provenance[str(state / name)] = sha256(state / name)
                reused = ends[positions] == end
                missing = np.flatnonzero(~reused)
                model = None
                if len(missing):
                    model = ForecastState('biid').cuda().eval()
                    model.load_state_dict(torch.load(state / 'model.pt', weights_only=True, map_location='cpu'))
                for ratio in RATIOS:
                    file = source / f'prefix_{round(100*ratio):02d}.npz'
                    with np.load(file) as saved:
                        output = np.array(saved[product][positions], copy=True)
                    if len(missing):
                        indices = take[missing]
                        active = raw['relative_valid'][indices] > 0
                        output[missing] = forecast_prefix(model, remap_product(raw, product), indices,
                            dict(norm, ndvi=norm[product]), raw[f'observed_{product}'][indices],
                            tail_mask(active, ratio), active)
                    if not np.isfinite(output).all():
                        raise ValueError('Nonfinite exported state')
                    outputs[ratio][product] = output
                checks.append(dict(stage=stage, product=product, actual_cutoff=end,
                    expected_maximum_cutoff=upper, reused_rows=int(reused.sum()), computed_rows=len(missing)))
                del model
                torch.cuda.empty_cache()
                print(f'[NESTED STATES] {crop} {cutoff} {stage} {product} actual={end} recomputed={len(missing)}', flush=True)
            for ratio, output in outputs.items():
                np.savez_compressed(root / f'{stage}_prefix_{round(100*ratio):02d}.npz', **output)
        atomic_json(root / 'provenance.json', dict(sources=provenance, checks=checks,
                    forward_training_cache=str(source), all_state_cutoffs_precede_targets=True))
        finish(root, code, smoke=smoke, seconds=time.monotonic()-started,
               state_weights_retrained=False, fixed_state_seed=42)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES), required=True)
    p.add_argument('--cutoff', choices=tuple(BLOCKS), type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.crop, args.cutoff, args.seed, args.smoke)
