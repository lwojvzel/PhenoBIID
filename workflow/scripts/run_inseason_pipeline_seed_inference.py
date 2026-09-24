"""Evaluate matching-seed components with the unchanged seasonal forward path."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from forecast_bridge_state import ForecastState
from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_13year_extended_history import extended_predictions
from inseason_nested_common import LABELS, hashes, register, finish, verify
from inseason_ndvi_reuse import TrajectoryEncoder
from inseason_pipeline_seed_inference import (RATIOS, MODES, component_identity,
    state_training, preserve_prefix, reference_map)
from inseason_pipeline_seed_terminal import paired_prediction, combine
from inseason_signal_matching import load_group, remap_product, feature_matrix
from ndvi_tail_replacement import tail_mask, mix_trajectory
from review_revision_data import sha256
from run_forecast_bridge_state import run_root as state_root
from run_inseason_direct_baselines import score
from run_inseason_pipeline_seed_terminal import root_for as terminal_root, CODE as TERMINAL_CODE
from run_ndvi_signal_permutation import check_files, compose
from run_ndvi_tail_replacement import forecast_prefix
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_pipeline_seeds_v1'
REFERENCES = ROOT / 'visualize/paper_experiments/inseason_13year_v1/comparisons.csv'
CODE = (*TERMINAL_CODE, 'forecast_bridge_state.py', 'token_retention_state.py',
        'dual_remote_state.py', 'run_forecast_bridge_state.py',
        'inseason_13year_extended_history.py', 'inseason_baseline_tables.py',
        'inseason_pipeline_seed_inference.py', 'run_inseason_pipeline_seed_inference.py')


def root_for(crop, cutoff, seed, smoke=False):
    return OUT / ('inference_smoke' if smoke else 'inference') / crop / f'cutoff_{cutoff}/seed_{seed}'


def load_states(crop, cutoff, seed, products, sources):
    models, normalizations, records = {}, {}, {}
    for product in products:
        folder = state_root(crop, product, 'biid', cutoff, seed)
        config = json.loads((folder / 'config.json').read_text())
        marker = json.loads((folder / 'complete.json').read_text())
        component_identity(config, crop, cutoff, seed, product)
        check_files(folder, marker['files'])
        for name, digest in config['code_sha256'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise ValueError('State source implementation changed')
        normalization = json.loads((folder / 'normalization.json').read_text())
        original = state_root(crop, product, 'biid', cutoff, 42) / 'normalization.json'
        selected = state_training(config, marker,
            json.loads((folder / 'training_history.json').read_text()),
            normalization, json.loads(original.read_text()))
        model = ForecastState('biid').cuda().eval()
        model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
        models[product] = model
        normalizations[product] = dict(normalization, ndvi=normalization[product])
        records[product] = dict(seed=seed, selected_epochs=selected, root=str(folder))
        for file in [folder / 'complete.json', *[folder / n for n in marker['files']], original]:
            sources[str(file)] = sha256(file)
    return models, normalizations, records


def run(crop, cutoff, seed, smoke=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Registered prefix replay requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = root_for(crop, cutoff, seed, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = hashes(CODE)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root, code)
            return
        if seed != 42 and not smoke:
            original = verify(root_for(crop, cutoff, 42), code)
            if not original['original_predictions_replayed']:
                raise ValueError('Original full-pipeline replay must pass first')
        started = time.monotonic()
        with threadpool_limits(limits=4):
            raw, groups, scales, _, sources = load_group(crop, cutoff+3)
            recipe, products = RECIPES[crop], RECIPES[crop].split('_')
            all_labels = {k: np.concatenate([g['labels'][k] for g in groups.values()]) for k in LABELS}
            if tuple(np.unique(all_labels['year'])) != BLOCKS[cutoff]:
                raise ValueError('Incomplete registered evaluation years')
            terminal = terminal_root(crop, cutoff, seed)
            terminal_marker = verify(terminal, hashes(TERMINAL_CODE))
            terminal_config = json.loads((terminal / 'config.json').read_text())
            component_identity(terminal_config, crop, cutoff, seed)
            if terminal_config['recipe'] != recipe or terminal_marker['complete_pipeline']:
                raise ValueError('Wrong terminal recipe or phase')
            with np.load(terminal / 'evaluation_anchor.npz') as f:
                anchor_archive = {k: f[k] for k in (*LABELS, 'prediction')}
            anchor_full = paired_prediction(anchor_archive, all_labels)
            heads = [joblib.load(terminal / b['name'] / 'model.joblib') for b in terminal_config['branches']]
            expected_names = ['trend', 'mlp'] if crop == 'maize' else ['expert']
            if [b['name'] for b in terminal_config['branches']] != expected_names:
                raise ValueError('Crop readout branch order changed')
            for file in [terminal / 'complete.json', *[terminal / n for n in terminal_marker['files']]]:
                sources[str(file)] = sha256(file)
            with REFERENCES.open() as stream:
                reference = reference_map(list(csv.DictReader(stream)))[crop]
            sources[str(REFERENCES)] = sha256(REFERENCES)
            candidates = extended_predictions(crop, cutoff, all_labels, sources)
            strong_archive = dict(all_labels, prediction=candidates[reference['baseline']])
            del candidates
            models, state_stats, state_records = load_states(crop, cutoff, seed, products, sources)
            original_root = ROOT / f'benchmark/results/inseason_signal_match_v1/pipelines/{crop}/origin_{cutoff+3}/seed_42'
            original_marker = json.loads((original_root / 'complete.json').read_text())
            check_files(original_root, original_marker['files'])
            sources[str(original_root / 'complete.json')] = sha256(original_root / 'complete.json')
            for split in groups:
                names = [f'{split}_{recipe}_biid_000.npz']
                names += [f'{split}_{recipe}_{m}_{round(r*100):03d}.npz' for r in RATIOS for m in MODES]
                names += [f'{split}_trajectories_{round(r*100):03d}.npz' for r in RATIOS]
                for name in names:
                    sources[str(original_root / name)] = original_marker['files'][name]
            register(root, dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke, recipe=recipe,
                ratios=RATIOS, modes=MODES, code_sha256=code, sources=sources, state_records=state_records,
                terminal=str(terminal), historical=terminal_config['historical'],
                reference=reference, input='True valid prefix feedback; full actual weather condition',
                terminal_training='Original complete observations; fixed original capacity',
                original_development_selection_retained=True, new_training=False,
                same_seed_history_state_terminal=True, paired_direct_repeats_completed=False))
            predictions = {(0, 'observed'): [], **{(round(r*100), m): [] for r in RATIOS for m in MODES}}
            originals = {key: [] for key in predictions}
            label_parts, anchor_parts, base_parts, strong_parts, active_parts = [], [], [], [], []
            trajectories = {round(r*100): {p: [] for p in products} for r in RATIOS}
            masks = {round(r*100): [] for r in RATIOS}
            state_replay_errors = {p: 0. for p in products}
            fit = np.flatnonzero(raw['year'] <= cutoff)
            for split, group in groups.items():
                local = (np.concatenate([np.flatnonzero(group['labels']['year'] == y)[:16]
                         for y in np.unique(group['labels']['year'])]) if smoke else np.arange(len(group['take'])))
                take = group['take'][local]
                labels = {k: group['labels'][k][local] for k in LABELS}
                active = raw['relative_valid'][take] > 0
                observed = {p: raw[f'observed_{p}'][take] for p in products}
                encoders = ({p: TrajectoryEncoder(remap_product(raw, p), fit, take, scales[p]) for p in products}
                            if smoke else {p: group['encoders'][p] for p in products})
                common, support = group['common'][local], group['support'][local]
                anchor = paired_prediction(anchor_archive, labels)
                bases = [group['heads'][recipe][i][2][local] if b['name'] == 'trend' else anchor
                         for i, b in enumerate(terminal_config['branches'])]

                def readout(values, tail):
                    encodings = {p: encoders[p].encode(values[p], tail, True) for p in products}
                    matrix = feature_matrix(common, encodings, tail, support, recipe)
                    if not np.isfinite(matrix).all() or matrix.shape[1] != terminal_config['feature_width']:
                        raise ValueError('Terminal feature interface changed')
                    parts = [compose(b['config'], base, model.booster_.predict(matrix))
                             for b, base, model in zip(terminal_config['branches'], bases, heads)]
                    return combine(crop, parts)

                def original_prediction(percent, mode):
                    file = original_root / f'{split}_{recipe}_{mode}_{percent:03d}.npz'
                    with np.load(file) as f:
                        result = f['prediction'][local]
                    return result

                zeros = np.zeros_like(active)
                predictions[0, 'observed'].append(readout(observed, zeros))
                if seed == 42 and not smoke:
                    originals[0, 'observed'].append(original_prediction(0, 'biid'))
                climates = {p: -encoders[p].encode(np.full_like(observed[p], scales[p]['mean']), active, True)
                            [:, -18:-6]*scales[p]['std']+scales[p]['mean'] for p in products}
                for ratio in RATIOS:
                    percent = round(100*ratio)
                    tail = tail_mask(active, ratio)
                    mixed = {}
                    for p in products:
                        forecast = forecast_prefix(models[p], remap_product(raw, p), take, state_stats[p],
                                                   observed[p], tail, active)
                        if smoke:
                            changed = observed[p].copy()
                            changed[tail] = 1000000.
                            hidden = forecast_prefix(models[p], remap_product(raw, p), take, state_stats[p],
                                                     changed, tail, active)
                            np.testing.assert_array_equal(forecast, hidden)
                        mixed[p] = mix_trajectory(observed[p], forecast, tail)
                        preserve_prefix(observed[p], mixed[p], active, tail)
                        if seed == 42 and not smoke:
                            file = original_root / f'{split}_trajectories_{percent:03d}.npz'
                            with np.load(file) as f:
                                expected = f[p][local]
                            np.testing.assert_allclose(mixed[p], expected, rtol=0, atol=1e-6)
                            state_replay_errors[p] = max(state_replay_errors[p],
                                float(np.max(np.abs(mixed[p][tail]-expected[tail]))))
                        trajectories[percent][p].append(mixed[p])
                    for mode in MODES:
                        values = mixed if mode == 'biid' else {p: mix_trajectory(observed[p], climates[p], tail) for p in products}
                        predictions[percent, mode].append(readout(values, tail))
                        if seed == 42 and not smoke:
                            originals[percent, mode].append(original_prediction(percent, mode))
                    masks[percent].append(tail)
                    print(f'[PIPELINE INFERENCE] {crop} {cutoff} seed={seed} {split} suffix={percent}', flush=True)
                label_parts.append(labels)
                anchor_parts.append(anchor)
                base_parts.append(combine(crop, bases))
                strong_parts.append(paired_prediction(strong_archive, labels))
                active_parts.append(active)
            labels = {k: np.concatenate([a[k] for a in label_parts]) for k in LABELS}
            strong = np.concatenate(strong_parts)
            if not smoke:
                for k in LABELS:
                    np.testing.assert_array_equal(labels[k], all_labels[k])
                np.testing.assert_array_equal(np.concatenate(anchor_parts), anchor_full)
            np.savez_compressed(root / 'labels.npz', **labels, strong=strong,
                history_anchor=np.concatenate(anchor_parts), deployed_base=np.concatenate(base_parts),
                active=np.concatenate(active_parts))
            reference_score = score(labels['target'], strong, labels['year'])
            records, metrics, max_replay = [], [], 0.
            for (percent, mode), parts in predictions.items():
                prediction = np.concatenate(parts)
                if seed == 42 and not smoke:
                    expected = np.concatenate(originals[percent, mode])
                    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
                    max_replay = max(max_replay, float(np.max(np.abs(prediction-expected))))
                measured = score(labels['target'], prediction, labels['year'])
                np.savez_compressed(root / f'tail_{percent:03d}_{mode}.npz', prediction=prediction, **labels)
                metrics.append(dict(percent=percent, mode=mode, **measured))
                for year, rmse in measured['per_year_rmse'].items():
                    ref = reference_score['per_year_rmse'][year]
                    records.append(dict(crop=crop, cutoff=cutoff, seed=seed, year=int(year),
                        recipe=recipe, percent=percent, mode=mode, rmse=rmse, reference_rmse=ref,
                        reference=reference['baseline'], gain=100*(1-rmse/ref),
                        samples=int((labels['year'] == int(year)).sum())))
            for percent, values in trajectories.items():
                np.savez_compressed(root / f'trajectories_{percent:03d}.npz',
                    **{p: np.concatenate(v) for p, v in values.items()}, tail=np.concatenate(masks[percent]))
            pd.DataFrame(records).to_csv(root / 'annual.csv', index=False)
            atomic_json(root / 'metrics.json', dict(scores=metrics, reference=reference_score,
                maximum_original_yield_error=max_replay if seed == 42 and not smoke else None,
                maximum_original_state_errors=state_replay_errors if seed == 42 and not smoke else None))
            for name, digest in sources.items():
                if sha256(Path(name)) != digest:
                    raise ValueError(f'Changed inference dependency: {name}')
            finish(root, code, crop=crop, cutoff=cutoff, seed=seed, smoke=smoke,
                original_predictions_replayed=seed == 42 and not smoke, complete_pipeline=not smoke,
                state_history_terminal_seed=seed, seconds=time.monotonic()-started,
                evaluation_rows=len(labels['target']), output_conditions=len(predictions),
                maximum_original_yield_error=max_replay if seed == 42 and not smoke else None,
                hidden_observation_perturbation_checked=smoke,
                paired_direct_repeats_completed=False, new_training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=tuple(RECIPES), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    run(args.crop, args.cutoff, args.seed, args.smoke)
