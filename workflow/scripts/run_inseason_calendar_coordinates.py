"""Matched natural-month coordinate refits using unchanged seasonal models."""
import argparse
import fcntl
import json
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from inseason_13year_data import ROOT, CACHE, RECIPES, BLOCKS, load, partition
from inseason_calendar_coordinates import to_calendar
from inseason_nested_common import LABELS, hashes, register, finish, verify
from run_inseason_13year_direct import (make_design, neural_fit,
    root_for as transformer_root, CODE as DIRECT_CODE)
from run_inseason_direct_extension import (train_cnn, root_for as cnn_root,
    verify as verify_cnn, CODE as CNN_CODE)
from inseason_direct_extension_models import SeasonalCNNHistoryLSTM
from run_input_matched_direct_yield import make_neural_model
from run_inseason_complete_direct import predict
from run_inseason_direct_baselines import score
from run_crop_signal_screen import year_weights
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from review_revision_data import sha256

OUT = ROOT / 'benchmark/results/inseason_calendar_coordinates_v1'
MODELS = ('cnn_rnn', 'transformer')
CODE = tuple(dict.fromkeys((*DIRECT_CODE, *CNN_CODE, 'inseason_calendar_coordinates.py',
    'run_inseason_calendar_coordinates.py')))


def root_for(crop, cutoff, model, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_42' / model


def reference_root(crop, cutoff, model):
    return (cnn_root(crop, cutoff, model) if model == 'cnn_rnn'
            else transformer_root(crop, cutoff, model))


def fit_model(name, x, residual, weights, years, dest, smoke, epochs=None):
    if name == 'cnn_rnn':
        output = train_cnn(x, residual, weights, years, dest, smoke, epochs)
        return output['selected_epochs'] if epochs is None else output
    return neural_fit(name, x, residual, weights, years, dest, smoke, epochs)


def run(crop, cutoff, model, smoke=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = model == 'cnn_rnn'
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = root_for(crop, cutoff, model, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(CODE)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        started = time.monotonic()
        raw, _ = load(crop)
        groups = partition(raw, cutoff)
        reference = reference_root(crop, cutoff, model)
        if model == 'cnn_rnn':
            verify_cnn(reference)
        else:
            marker = json.loads((reference / 'complete.json').read_text())
            check_files(reference, marker['files'])
            for name, digest in marker['code_sha256'].items():
                if sha256(ROOT / 'scripts' / name) != digest:
                    raise ValueError('Changed original coordinate model implementation')
        original = reference / 'tail_10'
        saved_metrics = json.loads((original / 'metrics.json').read_text())
        if smoke:
            groups = {s: np.concatenate([ix[raw['year'][ix] == y][:16]
                for y in np.unique(raw['year'][ix])]) for s, ix in groups.items()}
        register(root, dict(crop=crop, cutoff=cutoff, model=model, seed=42, smoke=smoke,
            ratio=.1, recipe=RECIPES[crop], code_sha256=code,
            original_root=str(original), reference_marker_sha256=sha256(reference / 'complete.json'),
            cache_sha256=sha256(CACHE / crop / 'manifest.json'),
            coordinates='Same active tokens moved to natural-month positions; intermediate padding allowed',
            calendar_features_and_static_summaries_unchanged=True,
            observation_months_and_information_unchanged=True,
            selection='Same preceding validation and full refit; independent trained epoch selection',
            training_function='Unmodified train_cnn/neural_fit from registered direct controls',
            periods={s: np.unique(raw['year'][ix]).tolist() for s, ix in groups.items()}))
        x, residual, norm = make_design(raw, groups['inner_fit'], groups['inner_validation'], RECIPES[crop], .1)
        if not smoke:
            expected = json.loads((original / 'inner_normalization.json').read_text())
            if norm != expected:
                raise ValueError('Coordinate experiment changed inner normalization')
        x = {s: to_calendar(value, raw['source_month'][groups[k]])
            for s, k, value in [('train', 'inner_fit', x['train']), ('other', 'inner_validation', x['other'])]}
        atomic_json(root / 'inner_normalization.json', norm)
        with threadpool_limits(limits=2):
            selected = fit_model(model, x, residual, year_weights(raw['year'][groups['inner_fit']]),
                raw['year'][groups['inner_validation']], root, smoke)
        if selected < 1:
            raise ValueError('Untrained calendar candidate')
        del x, residual
        x, residual, norm = make_design(raw, groups['full_fit'], groups['evaluation'], RECIPES[crop], .1)
        if not smoke and norm != saved_metrics['full_normalization']:
            raise ValueError('Coordinate experiment changed full normalization')
        if not smoke:
            shape, width = x['other']['sequence'].shape[-1], x['other']['static'].shape[-1]
            original_model = (SeasonalCNNHistoryLSTM(shape, width) if model == 'cnn_rnn'
                else make_neural_model(model, shape, width, .1)).cuda()
            original_model.load_state_dict(torch.load(original / 'model.pt', map_location='cpu', weights_only=True))
            replay = x['other']['anchor']+norm['center']+norm['scale']*predict(original_model, x['other'])
            with np.load(original / 'evaluation_predictions.npz') as saved:
                for key in LABELS:
                    np.testing.assert_array_equal(raw[key][groups['evaluation']], saved[key])
                np.testing.assert_array_equal(replay, saved['prediction'])
            del original_model
            torch.cuda.empty_cache()
        x = {s: to_calendar(value, raw['source_month'][groups[k]])
            for s, k, value in [('train', 'full_fit', x['train']), ('other', 'evaluation', x['other'])]}
        with threadpool_limits(limits=2):
            output = fit_model(model, x, residual, year_weights(raw['year'][groups['full_fit']]),
                None, root, smoke, selected)
        prediction = x['other']['anchor']+norm['center']+norm['scale']*output
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite calendar-coordinate prediction')
        labels = {k: raw[k][groups['evaluation']] for k in LABELS}
        np.savez_compressed(root / 'evaluation_predictions.npz', prediction=prediction, **labels)
        measured = score(labels['target'], prediction, labels['year'])
        atomic_json(root / 'metrics.json', dict(scores=measured, full_normalization=norm,
            selected_epochs=selected, original_epochs=saved_metrics['selection']['selected_epochs'],
            sequence_shape=list(x['other']['sequence'].shape[1:]), static_width=x['other']['static'].shape[1],
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        finish(root, code, smoke=smoke, seconds=time.monotonic()-started,
            original_prediction_replayed=not smoke, normalization_matched=not smoke,
            reversible_coordinate_transform=True, final_models=1)
        print(f'[COORDINATES] {crop} {cutoff} {model} annual={measured["mean_annual_rmse"]:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=tuple(RECIPES), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS), required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.crop, args.cutoff, args.model, args.smoke)
