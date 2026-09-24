"""Independently inspect seed readout weights, anchors, and original predictions."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np

from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_nested_common import LABELS, hashes, verify
from review_revision_data import sha256
from run_inseason_pipeline_seed_terminal import CODE, OUT, root_for
from run_inseason_pipeline_seed_history import root_for as history_root, CODE as HISTORY_CODE
from run_review_revision_parallel import atomic_json


def compare_trees(actual, reference):
    left, right = actual.booster_.dump_model(), reference.booster_.dump_model()
    for key in ('tree_info', 'feature_names', 'objective', 'average_output', 'num_tree_per_iteration'):
        if left[key] != right[key]:
            raise ValueError(f'Original terminal reconstruction differs: {key}')
    return len(left['tree_info'])


def same_capacity(config, original):
    for key in ('crop', 'cutoff', 'smoke', 'recipe', 'features', 'training_input', 'selection',
                'original_development_selection_retained', 'same_seed_historical_anchor',
                'training_rows', 'training_years', 'evaluation_years', 'feature_width', 'code_sha256'):
        if config[key] != original[key]:
            raise ValueError(f'Terminal seed capacity or inputs differ: {key}')
    if len(config['branches']) != len(original['branches']):
        raise ValueError('Seed branch count differs')
    for branch, base in zip(config['branches'], original['branches']):
        expected = dict(base, parameters=dict(base['parameters'], random_state=config['seed']))
        if branch != expected:
            raise ValueError('Seed changed original branch capacity or normalization')


def audit(seeds):
    sources, records, replay_rows, maximum_error = {}, [], 0, 0.
    for crop in RECIPES:
        for cutoff in BLOCKS:
            original_root = root_for(crop, cutoff, 42)
            original_config = json.loads((original_root / 'config.json').read_text())
            for seed in seeds:
                root = root_for(crop, cutoff, seed)
                marker = verify(root, hashes(CODE))
                config = json.loads((root / 'config.json').read_text())
                fits = json.loads((root / 'fits.json').read_text())
                same_capacity(config, original_config)
                expected = dict(crop=crop, cutoff=cutoff, seed=seed, smoke=False)
                if any(config[k] != v or marker[k] != v for k, v in expected.items()):
                    raise ValueError('Unexpected terminal run identity')
                if marker['complete_pipeline'] or fits['corresponding_seed_world_predictions_generated']:
                    raise ValueError('Readout weights alone are not a completed world-model evaluation')
                if marker['original_predictions_replayed'] != (seed == 42):
                    raise ValueError('Original reconstruction flag is wrong')
                branch_count = 2 if crop == 'maize' else 1
                if (len(config['branches']) != branch_count or len(fits['branches']) != branch_count or
                        marker['terminal_branch_fits'] != branch_count):
                    raise ValueError('Incomplete crop-specific terminal branches')
                component = 'tabm' if crop == 'soybean' else 'mlp'
                parent = history_root(crop, cutoff, seed, component)
                parent_record = verify(parent, hashes(HISTORY_CODE))
                if config['historical']['root'] != str(parent):
                    raise ValueError('Terminal readout used another seed historical anchor')
                if config['historical']['initialization_component'] != parent_record['initialization_component']:
                    raise ValueError('Initialization provenance was lost')
                for stage, splits in [('train', ('train',)),
                                      ('evaluation', ('validation', 'test') if cutoff == 2009 else ('validation',))]:
                    arrays = []
                    for split in splits:
                        with np.load(parent / f'{split}_predictions.npz') as f:
                            arrays.append({k: f[k] for k in (*LABELS, 'prediction')})
                    with np.load(root / f'{stage}_anchor.npz') as saved:
                        if set(saved.files) != {*LABELS, 'prediction'}:
                            raise ValueError('Anchor schema changed')
                        for key in (*LABELS, 'prediction'):
                            np.testing.assert_array_equal(saved[key], np.concatenate([a[key] for a in arrays]))
                for b, fit in zip(config['branches'], fits['branches']):
                    weight = root / b['name'] / 'model.joblib'
                    model = joblib.load(weight)
                    if model.get_params() != b['parameters']:
                        raise ValueError('Saved tree estimator parameters differ from registration')
                    tree_count = int(model.booster_.current_iteration())
                    if (fit['branch'] != b['name'] or tree_count != fit['fitted_trees'] or
                            fit['selected_trees'] != b['original_trees'] or
                            not 1 <= tree_count <= fit['selected_trees']):
                        raise ValueError('Selected/fitted tree accounting differs')
                    if seed == 42:
                        old = ROOT / f'benchmark/results/inseason_terminal_paths_v1/pipelines/{crop}/cutoff_{cutoff}/seed_42/full'
                        old_marker = verify(old)
                        if not old_marker['full_rebuild_verified']:
                            raise ValueError('Original terminal reconstruction reference not verified')
                        old_weight = old / b['name'] / 'model.joblib'
                        compare_trees(model, joblib.load(old_weight))
                        sources[str(old / 'complete.json')] = sha256(old / 'complete.json')
                        sources[str(old_weight)] = sha256(old_weight)
                    records.append(dict(crop=crop, cutoff=cutoff, seed=seed, recipe=RECIPES[crop],
                        branch=b['name'], selected_trees=fit['selected_trees'], fitted_trees=tree_count,
                        weight=str(weight), weight_sha256=sha256(weight), history=str(parent),
                        marker_sha256=sha256(root / 'complete.json')))
                if seed == 42:
                    reference = ROOT / f'benchmark/results/inseason_signal_match_v1/pipelines/{crop}/origin_{cutoff+3}/seed_42'
                    splits = ('validation', 'test') if cutoff == 2009 else ('validation',)
                    for percent in (10, 30, 50):
                        for mode in ('biid', 'climatology'):
                            labels, predictions = [], []
                            for split in splits:
                                label_file = reference / f'{split}_labels.npz'
                                file = reference / f'{split}_{RECIPES[crop]}_{mode}_{percent:03d}.npz'
                                with np.load(label_file) as f:
                                    labels.append({k: f[k] for k in LABELS})
                                with np.load(file) as f:
                                    predictions.append(f['prediction'])
                                sources[str(file)] = sha256(file)
                                sources[str(label_file)] = sha256(label_file)
                            with np.load(root / f'replay_{percent:03d}_{mode}.npz') as f:
                                for key in LABELS:
                                    np.testing.assert_array_equal(f[key], np.concatenate([a[key] for a in labels]))
                                original = np.concatenate(predictions)
                                np.testing.assert_array_equal(f['original'], original)
                                np.testing.assert_allclose(f['prediction'], original, rtol=0, atol=1e-12)
                                maximum_error = max(maximum_error, float(np.max(np.abs(f['prediction']-original))))
                                replay_rows += len(original)
                for name, digest in config['sources'].items():
                    if name in sources and sources[name] != digest:
                        raise ValueError('Conflicting terminal input versions')
                    sources[name] = digest
                sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
    for name, digest in sources.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Terminal source changed: {name}')
    if len(records) != 15 * len(seeds):
        raise ValueError('Expected fifteen branch fits per seed')
    suffix = 'replay' if tuple(seeds) == (42,) else 'seeds'
    registry = OUT / f'terminal_{suffix}_checkpoints.csv'
    with registry.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(passed=True, timestamp=datetime.now().astimezone().isoformat(), seeds=list(seeds),
        terminal_groups=12*len(seeds), terminal_branch_fits=len(records),
        original_tree_structures_exact=True, original_prediction_rows_checked=replay_rows,
        maximum_original_prediction_error=maximum_error, same_seed_historical_anchors=True,
        complete_pipeline=False, sources=sources, registry=str(registry), registry_sha256=sha256(registry),
        verifier_sha256=sha256(Path(__file__)), training_code_sha256=hashes(CODE),
        scope='Fixed original terminal weights and anchors; corresponding-seed states not yet evaluated')
    atomic_json(OUT / f'terminal_{suffix}_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'training_code_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('replay', 'all'), default='replay')
    args = parser.parse_args()
    audit((42,) if args.stage == 'replay' else (42, 45, 48))
