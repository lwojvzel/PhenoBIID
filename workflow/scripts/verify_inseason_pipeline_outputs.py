"""Reconstruct readouts from saved trajectories and audit paired annual scores.

State inference is not rerun here. Its weights and training provenance are
checked; every yield readout is independently replayed from the saved trajectory.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES, load, partition
from inseason_13year_extended_history import extended_predictions
from inseason_nested_common import LABELS, hashes, verify
from inseason_signal_matching import load_group, feature_matrix
from review_revision_data import sha256
from run_inseason_pipeline_seed_inference import CODE, OUT, REFERENCES, root_for
from run_inseason_pipeline_seed_terminal import root_for as terminal_root, CODE as TERMINAL_CODE
from run_ndvi_signal_permutation import compose, check_files
from run_review_revision_parallel import atomic_json

CONDITIONS = [(0, 'observed')] + [(p, m) for p in (10, 30, 50) for m in ('biid', 'climatology')]


def independent_tail(active, percent):
    count = active.sum(axis=1)
    hidden = (count*percent+99)//100
    return active & (np.cumsum(active, axis=1) > (count-hidden)[:, None])


def measure(truth, prediction, years):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.ndim != 1 or truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid yield prediction vector')
    per_year = {str(int(y)): float(np.sqrt(np.mean((truth[years == y]-prediction[years == y])**2)))
                for y in np.unique(years)}
    return dict(pooled_rmse=float(np.sqrt(np.mean((truth-prediction)**2))),
                mean_annual_rmse=float(np.mean(list(per_year.values()))), per_year_rmse=per_year)


def equal_metrics(actual, expected):
    if set(actual['per_year_rmse']) != set(expected['per_year_rmse']):
        raise ValueError('Annual score years differ')
    for key in ('pooled_rmse', 'mean_annual_rmse'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    for year in expected['per_year_rmse']:
        np.testing.assert_allclose(actual['per_year_rmse'][year], expected['per_year_rmse'][year], rtol=0, atol=1e-12)


def audit(seeds):
    with REFERENCES.open() as stream:
        references = {r['crop']: r for r in csv.DictReader(stream)}
    sources = {str(REFERENCES): sha256(REFERENCES)}
    rows, directories, replay_rows, original_rows = [], [], 0, 0
    maximum_head_error, maximum_original_error = 0., 0.
    for crop, recipe in RECIPES.items():
        physical, _ = load(crop)
        sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
        products = recipe.split('_')
        for cutoff in BLOCKS:
            ix = partition(physical, cutoff)['evaluation']
            expected_labels = {k: physical[k][ix] for k in LABELS}
            raw, groups, scales, _, original_sources = load_group(crop, cutoff+3)
            sources.update(original_sources)
            reference = extended_predictions(crop, cutoff, expected_labels, sources)[references[crop]['baseline']]
            for seed in seeds:
                folder = root_for(crop, cutoff, seed)
                marker = verify(folder, hashes(CODE))
                config = json.loads((folder / 'config.json').read_text())
                if (any(config[k] != v or marker[k] != v for k, v in
                        dict(crop=crop, cutoff=cutoff, seed=seed, smoke=False).items()) or
                        not marker['complete_pipeline'] or marker['new_training']):
                    raise ValueError('Wrong pipeline identity or scope')
                if config['reference'] != references[crop] or config['recipe'] != recipe:
                    raise ValueError('Reference or recipe changed by seed')
                terminal = terminal_root(crop, cutoff, seed)
                verify(terminal, hashes(TERMINAL_CODE))
                terminal_config = json.loads((terminal / 'config.json').read_text())
                if config['terminal'] != str(terminal) or config['historical'] != terminal_config['historical']:
                    raise ValueError('Pipeline did not use matching terminal/history provenance')
                with np.load(folder / 'labels.npz') as f:
                    saved = {k: f[k] for k in f.files}
                for k in LABELS:
                    np.testing.assert_array_equal(saved[k], expected_labels[k])
                np.testing.assert_array_equal(saved['strong'], reference)
                np.testing.assert_array_equal(saved['active'], physical['relative_valid'][ix] > 0)
                with np.load(terminal / 'evaluation_anchor.npz') as f:
                    for k in LABELS:
                        np.testing.assert_array_equal(saved[k], f[k])
                    np.testing.assert_array_equal(saved['history_anchor'], f['prediction'])
                if set(config['state_records']) != set(products):
                    raise ValueError('Missing or extra crop state product')
                for product in products:
                    state = ROOT / f'benchmark/results/forecast_state_bridge_v1/states/pipelines/{crop}/{product}/biid/cutoff_{cutoff}/seed_{seed}'
                    record = config['state_records'][product]
                    if record['root'] != str(state) or record['seed'] != seed:
                        raise ValueError('State from another seed was used')
                    state_marker = json.loads((state / 'complete.json').read_text())
                    check_files(state, state_marker['files'])
                    if state_marker['selected_epochs'] != record['selected_epochs']:
                        raise ValueError('State epoch provenance differs')
                trajectories = {}
                for percent in (10, 30, 50):
                    with np.load(folder / f'trajectories_{percent:03d}.npz') as f:
                        if set(f.files) != {*products, 'tail'}:
                            raise ValueError('Trajectory schema differs')
                        trajectories[percent] = {k: f[k] for k in f.files}
                    np.testing.assert_array_equal(trajectories[percent]['tail'], independent_tail(saved['active'], percent))
                models = [joblib.load(terminal / b['name'] / 'model.joblib') for b in terminal_config['branches']]
                predictions = {key: [] for key in CONDITIONS}
                base_parts, offset = [], 0
                for split, group in groups.items():
                    take = group['take']
                    span = slice(offset, offset+len(take))
                    for k in LABELS:
                        np.testing.assert_array_equal(saved[k][span], group['labels'][k])
                    active = raw['relative_valid'][take] > 0
                    observed = {p: raw[f'observed_{p}'][take] for p in products}
                    anchor = saved['history_anchor'][span]
                    bases = [group['heads'][recipe][i][2] if b['name'] == 'trend' else anchor
                             for i, b in enumerate(terminal_config['branches'])]
                    base_parts.append(.5*bases[0]+.5*bases[1] if crop == 'maize' else bases[0])
                    climates = {p: -group['encoders'][p].encode(np.full_like(observed[p], scales[p]['mean']), active, True)
                                [:, -18:-6]*scales[p]['std']+scales[p]['mean'] for p in products}
                    for percent, mode in CONDITIONS:
                        tail = independent_tail(active, percent)
                        if mode == 'observed':
                            values = observed
                        elif mode == 'climatology':
                            values = {p: np.where(tail, climates[p], observed[p]) for p in products}
                        else:
                            values = {p: trajectories[percent][p][span] for p in products}
                            for p in products:
                                np.testing.assert_array_equal(values[p][~tail], observed[p][~tail])
                                if not np.isfinite(values[p][tail]).all():
                                    raise ValueError('Nonfinite hidden state predictions')
                        encoded = {p: group['encoders'][p].encode(values[p], tail, True) for p in products}
                        x = feature_matrix(group['common'], encoded, tail, group['support'], recipe)
                        parts = [compose(b['config'], base, model.booster_.predict(x))
                                 for b, base, model in zip(terminal_config['branches'], bases, models)]
                        predictions[percent, mode].append(.5*parts[0]+.5*parts[1] if crop == 'maize' else parts[0])
                    offset += len(take)
                if offset != len(ix):
                    raise ValueError('Incomplete pipeline cohort')
                np.testing.assert_array_equal(saved['deployed_base'], np.concatenate(base_parts))
                metrics = json.loads((folder / 'metrics.json').read_text())
                stored_scores = {(r['percent'], r['mode']): r for r in metrics['scores']}
                if set(stored_scores) != set(CONDITIONS) or len(metrics['scores']) != len(CONDITIONS):
                    raise ValueError('Missing or repeated output condition')
                ref_score = measure(saved['target'], reference, saved['year'])
                equal_metrics(metrics['reference'], ref_score)
                with (folder / 'annual.csv').open() as stream:
                    annual_rows = list(csv.DictReader(stream))
                indexed = {(int(r['percent']), r['mode'], int(r['year'])): r for r in annual_rows}
                if len(indexed) != len(annual_rows) or len(indexed) != len(CONDITIONS)*len(BLOCKS[cutoff]):
                    raise ValueError('Annual records missing or duplicated')
                for (percent, mode), pieces in predictions.items():
                    predicted = np.concatenate(pieces)
                    file = folder / f'tail_{percent:03d}_{mode}.npz'
                    with np.load(file) as f:
                        if set(f.files) != {*LABELS, 'prediction'}:
                            raise ValueError('Prediction schema differs')
                        for key in LABELS:
                            np.testing.assert_array_equal(f[key], saved[key])
                        np.testing.assert_allclose(f['prediction'], predicted, rtol=0, atol=1e-12)
                        maximum_head_error = max(maximum_head_error, float(np.max(np.abs(predicted-f['prediction']))))
                    replay_rows += len(predicted)
                    if seed == 42:
                        original_file = CACHE / crop / f'world_{cutoff}_{recipe}_{"biid" if mode == "observed" else mode}_{percent:03d}.npy'
                        original = np.load(original_file)
                        np.testing.assert_allclose(predicted, original, rtol=0, atol=1e-12)
                        maximum_original_error = max(maximum_original_error, float(np.max(np.abs(predicted-original))))
                        original_rows += len(original)
                        sources[str(original_file)] = sha256(original_file)
                    measured = measure(saved['target'], predicted, saved['year'])
                    equal_metrics(stored_scores[percent, mode], measured)
                    for year, rmse in measured['per_year_rmse'].items():
                        ref_rmse = ref_score['per_year_rmse'][year]
                        expected = dict(crop=crop, cutoff=cutoff, seed=seed, year=int(year), recipe=recipe,
                            percent=percent, mode=mode, rmse=rmse, reference_rmse=ref_rmse,
                            reference=references[crop]['baseline'], gain=100*(1-rmse/ref_rmse),
                            samples=int((saved['year'] == int(year)).sum()))
                        row = indexed[percent, mode, int(year)]
                        for key, value in expected.items():
                            if isinstance(value, float):
                                np.testing.assert_allclose(float(row[key]), value, rtol=0, atol=1e-12)
                            elif str(row[key]) != str(value):
                                raise ValueError(f'Annual metadata differs: {key}')
                        rows.append(expected)
                for name, digest in config['sources'].items():
                    if name in sources and sources[name] != digest:
                        raise ValueError('Conflicting pipeline source versions')
                    sources[name] = digest
                sources[str(folder / 'complete.json')] = sha256(folder / 'complete.json')
                directories.append(str(folder))
                print(f'[PIPELINE VERIFIED] {crop} {cutoff} seed={seed}', flush=True)
    for name, digest in sources.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Pipeline source changed during verification: {name}')
    frame = pd.DataFrame(rows)
    suffix = 'replay' if tuple(seeds) == (42,) else 'all'
    csv_path = OUT / f'inference_{suffix}_verified_annual.csv'
    frame.to_csv(csv_path, index=False)
    if len(directories) != 12*len(seeds) or len(frame) != 4*13*7*len(seeds):
        raise ValueError('Incomplete full-pipeline matrix')
    for crop in RECIPES:
        selected = frame[(frame.crop == crop) & (frame.seed == 42) & (frame.percent == 10) & (frame['mode'] == 'biid')]
        np.testing.assert_allclose(selected.rmse.mean(), float(references[crop]['world_rmse']), rtol=0, atol=1e-12)
        np.testing.assert_allclose(selected.reference_rmse.mean(), float(references[crop]['baseline_rmse']), rtol=0, atol=1e-12)
    result = dict(passed=True, timestamp=datetime.now().astimezone().isoformat(), seeds=list(seeds),
        groups=len(directories), output_conditions=len(directories)*7, annual_rows=len(frame),
        head_prediction_rows_replayed=replay_rows, original_prediction_rows_replayed=original_rows,
        maximum_readout_replay_error=maximum_head_error, maximum_original_prediction_error=maximum_original_error,
        same_seed_components=True, original_main_table_reproduced=True, sources=sources,
        annual_csv=str(csv_path), annual_sha256=sha256(csv_path),
        verifier_sha256=sha256(Path(__file__)), pipeline_code_sha256=hashes(CODE),
        state_forward_rerun_in_verifier=False, readouts_replayed_from_saved_trajectories=True,
        paired_direct_repeats_completed=False)
    atomic_json(OUT / f'inference_{suffix}_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'pipeline_code_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('replay', 'all'), default='replay')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        audit((42,) if args.stage == 'replay' else (42, 45, 48))
