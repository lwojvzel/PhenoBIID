"""Replay direct weights on the canonical cohort and verify the original table."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from inseason_13year_data import ROOT, CACHE, BLOCKS, YEARS, RECIPES, load, partition
from inseason_complete_inputs import flat_features
from inseason_direct_seed_protocol import (OUT, MODELS, PERCENTS, SEEDS, root_for,
    identity, original, selection_contract, normalization_contract, compare_labels, physical_prediction)
from inseason_nested_common import LABELS, hashes, verify
from review_revision_data import sha256
from run_inseason_13year_direct import make_design
from run_inseason_direct_seed import CODE
from run_review_revision_parallel import atomic_json
from run_yield_only_classical_baselines import build_model
from verify_inseason_pipeline_outputs import measure, equal_metrics


def audit(stage):
    seeds = (42,) if stage == 'original' else SEEDS
    percents = (10,) if stage == 'primary' else PERCENTS
    annual, summary, registry, sources, completed = [], [], [], {}, {}
    chunks = {}
    replayed_rows, original_rows = 0, 0
    for crop in RECIPES:
        raw, _ = load(crop)
        sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
        for cutoff in BLOCKS:
            indices = partition(raw, cutoff)
            labels = {k: raw[k][indices['evaluation']] for k in LABELS}
            inner_labels = {k: raw[k][indices['inner_validation']] for k in LABELS}
            for percent in percents:
                x, residual, norm = make_design(raw, indices['full_fit'], indices['evaluation'], RECIPES[crop], percent/100)
                features = flat_features(x['other'])
                inner, inner_residual, inner_norm = make_design(raw, indices['inner_fit'], indices['inner_validation'], RECIPES[crop], percent/100)
                for model in MODELS:
                    old_root, old_dest, old_config, old_metrics, _ = original(crop, cutoff, model, percent)
                    normalization_contract(norm, old_metrics['full_normalization'])
                    for seed in seeds:
                        key = identity(crop, cutoff, seed, model, percent)
                        root = root_for(crop, cutoff, seed, model, percent)
                        marker = verify(root, hashes(CODE))
                        config = json.loads((root / 'config.json').read_text())
                        metrics = json.loads((root / 'metrics.json').read_text())
                        for name, value in key.items():
                            if marker[name] != value or config[name] != value:
                                raise ValueError('Direct seed checkpoint identity mismatch')
                        if (config['smoke'] or marker['smoke'] or config['original_weight_reused'] != (seed == 42) or
                                marker['new_final_fits'] != int(seed != 42) or config['periods'] != old_config['periods'] or
                                config['counts'] != {name: len(ix) for name, ix in indices.items()}):
                            raise ValueError('Wrong direct seed training or evaluation scope')
                        for filename, expected in config['source_sha256'].items():
                            if sha256(ROOT / filename) != expected:
                                raise ValueError('A direct seed source changed')
                            sources[str(ROOT / filename)] = expected
                        normalization_contract(norm, metrics['full_normalization'])
                        if (metrics['sequence_shape'] != list(x['train']['sequence'].shape[1:]) or
                                metrics['static_width'] != x['train']['static'].shape[1] or metrics['flat_width'] != features.shape[1]):
                            raise ValueError('Feature representation changed')
                        weight = ROOT / metrics['weight']
                        if sha256(weight) != metrics['weight_sha256']:
                            raise ValueError('Direct weight changed')
                        fitted = joblib.load(weight)
                        if fitted.get_params() != metrics['final_parameters']:
                            raise ValueError('Serialized parameters differ')
                        if model == 'lightgbm':
                            parameters = selection_contract(metrics['selection'], seed)
                            normalization_contract(inner_norm, json.loads((root / 'inner_normalization.json').read_text()))
                            inner_weight = (old_dest if seed == 42 else root) / 'inner/model.joblib'
                            expected_inner = joblib.load(inner_weight).booster_.predict(flat_features(inner['other']))
                            with np.load(root / 'inner_predictions.npz') as saved:
                                compare_labels(saved, inner_labels)
                                np.testing.assert_array_equal(saved['standardized_target'], inner_residual['other'])
                                np.testing.assert_array_equal(saved['prediction'], expected_inner)
                            chosen = metrics['selection']['candidates'][metrics['selection']['selected_candidate']]
                            value = measure(inner_residual['other'], expected_inner, inner_labels['year'])['mean_annual_rmse']
                            np.testing.assert_allclose(value, chosen['score'], rtol=0, atol=1e-12)
                        else:
                            parameters = build_model(model, seed, 1).get_params()
                            if metrics['selection'] != dict(fixed_parameters=True) or len(fitted.estimators_) != 128:
                                raise ValueError('Forest capacity or selection changed')
                        if any(fitted.get_params()[k] != v for k, v in parameters.items()):
                            raise ValueError('Final parameters did not follow original selection protocol')
                        prediction = physical_prediction(fitted, features, x['other']['anchor'], norm, model)
                        with np.load(root / 'evaluation_predictions.npz') as saved:
                            compare_labels(saved, labels)
                            np.testing.assert_array_equal(prediction, saved['prediction'])
                        with np.load(ROOT / config['paired_world_prediction']) as world:
                            compare_labels(world, labels)
                        if seed == 42:
                            with np.load(old_dest / 'evaluation_predictions.npz') as saved:
                                compare_labels(saved, labels)
                                np.testing.assert_array_equal(prediction, saved['prediction'])
                            original_rows += len(prediction)
                        measured = measure(labels['target'], prediction, labels['year'])
                        equal_metrics(measured, metrics['scores'])
                        annual.extend(dict(crop=crop, cutoff=cutoff, seed=seed, model=model,
                            percent=percent, year=int(year), rmse=value) for year, value in measured['per_year_rmse'].items())
                        chunks.setdefault((crop, seed, model, percent), []).append((labels['target'], prediction, labels['year']))
                        completed[str((root / 'complete.json').relative_to(ROOT))] = sha256(root / 'complete.json')
                        registry.append(dict(**key, weight=str(weight), weight_sha256=sha256(weight),
                            predictions=str(root / 'evaluation_predictions.npz'), complete=str(root / 'complete.json')))
                        replayed_rows += len(prediction)
                        print(f'[DIRECT VERIFIED] {crop} {cutoff} {model} seed={seed} suffix={percent}', flush=True)
                del x, residual, inner, inner_residual
    for (crop, seed, model, percent), parts in chunks.items():
        target, prediction, year = [np.concatenate([part[i] for part in parts]) for i in range(3)]
        if tuple(np.unique(year)) != YEARS:
            raise ValueError('Incomplete annual coverage')
        measured = measure(target, prediction, year)
        summary.append(dict(crop=crop, seed=seed, model=model, percent=percent,
            pooled_rmse=measured['pooled_rmse'], mean_annual_rmse=measured['mean_annual_rmse']))
    expected_conditions = 4*3*len(seeds)*2*len(percents)
    frame, scores = pd.DataFrame(annual), pd.DataFrame(summary)
    if len(registry) != expected_conditions or len(frame) != 4*len(seeds)*2*len(percents)*13:
        raise ValueError('Incomplete direct seed matrix')
    published_root = ROOT / 'visualize/paper_experiments/inseason_direct_extension_v1'
    published = pd.read_csv(published_root / 'scores.csv').set_index(['crop', 'model', 'percent'])
    published_annual = pd.read_csv(published_root / 'annual.csv').set_index(['crop', 'model', 'percent', 'year'])
    for row in scores[scores.seed == 42].itertuples():
        expected = published.loc[(row.crop, row.model, row.percent)]
        np.testing.assert_allclose([row.pooled_rmse, row.mean_annual_rmse],
            [expected.pooled_rmse, expected.mean_annual_rmse], rtol=0, atol=1e-12)
    for row in frame[frame.seed == 42].itertuples():
        np.testing.assert_allclose(row.rmse, published_annual.loc[(row.crop, row.model, row.percent, row.year), 'rmse'], rtol=0, atol=1e-12)
    sources.update({str(published_root / name): sha256(published_root / name) for name in ('scores.csv', 'annual.csv')})
    for filename, expected in sources.items():
        if sha256(Path(filename)) != expected:
            raise ValueError('Source changed during direct seed audit')
    for name, value in [('annual', frame), ('scores', scores), ('checkpoints', pd.DataFrame(registry))]:
        value.to_csv(OUT / f'{stage}_verified_{name}.csv', index=False)
    report = dict(passed=True, timestamp=datetime.now().astimezone().isoformat(), stage=stage,
        seeds=seeds, percents=percents, conditions=len(registry), annual_rows=len(frame), summary_rows=len(scores),
        prediction_rows_replayed=replayed_rows, original_prediction_rows=original_rows, maximum_original_error=0.,
        maximum_weight_replay_error=0., original_main_scores_reproduced=True, same_seed_world_labels_matched=True,
        training_code_sha256=hashes(CODE), verifier_sha256=sha256(Path(__file__)), completed_sources=completed,
        source_sha256=sources, output_sha256={str((OUT / f'{stage}_verified_{name}.csv').relative_to(ROOT)):
            sha256(OUT / f'{stage}_verified_{name}.csv') for name in ('annual', 'scores', 'checkpoints')})
    atomic_json(OUT / ('original_replay_verification.json' if stage == 'original' else f'{stage}_verification.json'), report)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('sha256') and k != 'completed_sources'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('original', 'primary', 'all'), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        audit(args.stage)
