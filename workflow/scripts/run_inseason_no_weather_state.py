"""Preceding-validation refit of the matched BIID without state weather."""
import argparse
import fcntl
import json
import time

import numpy as np
import torch

from forecast_bridge_data import load, root as data_root, fit_stats, identity_hash
from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_no_weather_state import NoWeatherState
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_forecast_bridge_state import (CODE as BASE_CODE, train_epoch, predict,
    state_metrics, sample_by_year)
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_no_weather_v1'
CODE = (*BASE_CODE, 'inseason_no_weather_state.py', 'run_inseason_no_weather_state.py')


def state_root(crop, product, cutoff, smoke=False):
    return OUT / ('smoke' if smoke else 'states') / crop / product / f'cutoff_{cutoff}/seed_42'


def verify_state(crop, product, cutoff, smoke=False):
    root = state_root(crop, product, cutoff, smoke)
    marker = json.loads((root / 'complete.json').read_text())
    check_files(root, marker['files'])
    config = json.loads((root / 'config.json').read_text())
    code = {name:sha256(ROOT / 'scripts' / name) for name in CODE}
    if config['code_sha256'] != code or config['state_weather'] != 'constant_zero_standardized':
        raise ValueError('Changed no-weather state specification')
    if marker['smoke'] != smoke or marker['selected_epochs'] < 1 or marker['full_fit_cutoff'] != cutoff:
        raise ValueError('Invalid no-weather checkpoint')
    if config['selection_years'] != [cutoff-1, cutoff]:
        raise ValueError('Incorrect no-weather selection window')
    norm = json.loads((root / 'normalization.json').read_text())
    if max(norm['fit_years']) != cutoff:
        raise ValueError('Incorrect no-weather normalization')
    return root, norm


def run(crop, product, cutoff, smoke=False):
    if product not in RECIPES[crop].split('_'):
        raise ValueError('Product is not in the fixed crop recipe')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = state_root(crop, product, cutoff, smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify_state(crop, product, cutoff, smoke)
            return
        a, meta = load(crop)
        fit = np.flatnonzero(a['year'] <= cutoff-2)
        validation = np.flatnonzero((a['year'] > cutoff-2) & (a['year'] <= cutoff))
        full = np.flatnonzero(a['year'] <= cutoff)
        if smoke:
            fit, validation, full = [sample_by_year(a, ix, 16) for ix in (fit, validation, full)]
        spec = dict(crop=crop, product=product, cutoff=cutoff, seed=42, smoke=smoke,
            architecture='biid', state_weather='constant_zero_standardized',
            terminal_weather='unchanged in later fixed-head evaluation',
            maximum_epochs=30, patience=5, batch=256, lr=.0003, weight_decay=.0001,
            data_sha256=sha256(data_root(crop) / 'manifest.json'),
            code_sha256={name:sha256(ROOT / 'scripts' / name) for name in CODE},
            fit_identity=identity_hash(a['source_indices'][fit]),
            inner_validation_identity=identity_hash(a['source_indices'][validation]),
            full_identity=identity_hash(a['source_indices'][full]),
            selection_years=np.unique(a['year'][validation]).tolist(),
            protocol=meta['protocol'], target_observations_in_training_forward=False,
            evaluation_used_for_selection=False, parameters_and_budget_matched=True)
        config = root / 'config.json'
        if config.exists() and json.loads(config.read_text()) != spec:
            raise ValueError('Frozen no-weather registration changed')
        atomic_json(config, spec)
        started = time.monotonic()
        stats = fit_stats(a, fit)
        set_seed(42)
        model = NoWeatherState().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        rng = np.random.default_rng(42)
        best, selected, stale, trace = float('inf'), 0, 0, []
        for epoch in range(1, (1 if smoke else 30)+1):
            loss = train_epoch(model, a, fit, stats, product, optimizer, rng, 256)
            errors = state_metrics(a, validation, predict(model, a, validation, stats, product), product)
            value = float(np.mean(list(errors.values())))
            trace.append(dict(stage='inner_selection', epoch=epoch, loss=loss, rmse=errors))
            if value < best:
                best, selected, stale = value, epoch, 0
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, root / 'inner_best.pt')
            else:
                stale += 1
            atomic_json(root / 'training_history.json', trace)
            print(f'[NO WEATHER] {crop} {product} {cutoff} epoch={epoch} inner={value:.6f}', flush=True)
            if stale >= 5:
                break
        if selected < 1:
            raise ValueError('No trained finite checkpoint')
        del optimizer, model
        torch.cuda.empty_cache()
        norm = fit_stats(a, full)
        set_seed(42)
        model = NoWeatherState().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        rng = np.random.default_rng(42)
        for epoch in range(1, selected+1):
            loss = train_epoch(model, a, full, norm, product, optimizer, rng, 256)
            trace.append(dict(stage='full_refit', epoch=epoch, loss=loss))
            atomic_json(root / 'training_history.json', trace)
            print(f'[NO WEATHER REFIT] {crop} {product} {cutoff} {epoch}/{selected}', flush=True)
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, root / 'model.pt')
        atomic_json(root / 'normalization.json', norm)
        ix = full[:32]
        prediction = predict(model, a, ix, norm, product)
        restored = NoWeatherState().cuda()
        restored.load_state_dict(torch.load(root / 'model.pt', map_location='cpu', weights_only=True))
        np.testing.assert_array_equal(prediction, predict(restored, a, ix, norm, product))
        files = {name:sha256(root / name) for name in ('config.json','model.pt','inner_best.pt',
            'normalization.json','training_history.json')}
        atomic_json(root / 'complete.json', dict(files=files, smoke=smoke, selected_epochs=selected,
            full_fit_cutoff=cutoff, full_training_rows=len(full), inner_state_rmse=best,
            maximum_replay_error=0., seconds=time.monotonic()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        print(f'[NO WEATHER COMPLETE] {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=tuple(RECIPES), required=True)
    parser.add_argument('--product', choices=('ndvi','gpp'), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.crop, args.product, args.cutoff, args.smoke)
