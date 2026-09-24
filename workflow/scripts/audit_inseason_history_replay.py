"""Verify exact archived history weights and full saved prediction identities."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_nested_common import LABELS, hashes, verify
from review_revision_data import sha256
from run_inseason_pipeline_seed_history import CODE, OUT, root_for
from run_review_revision_parallel import atomic_json


def compare_weights(actual, reference):
    if actual.keys() != reference.keys():
        raise ValueError('Historical checkpoint tensor names differ')
    for name, value in actual.items():
        other = reference[name]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
            raise ValueError(f'Historical checkpoint tensor differs: {name}')
    return sum(value.numel() for value in actual.values())


def audit(component):
    expected_code = hashes(CODE)
    jobs = [(crop, cutoff, 'mlp') for crop in RECIPES for cutoff in BLOCKS]
    if component == 'all':
        jobs.extend(('soybean', cutoff, 'tabm') for cutoff in BLOCKS)
    records = []
    for crop, cutoff, kind in jobs:
        folder = root_for(crop, cutoff, 42, kind)
        marker = verify(folder, expected_code)
        config = json.loads((folder / 'config.json').read_text())
        source = (ROOT / f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_{cutoff+3}/history_mlp/seed_42'
                  if kind == 'mlp' else ROOT / f'benchmark/results/numeric_embedding_readout_v1/pipelines/soybean/origin_{cutoff+3}/seed_42/tabm/history')
        expected = dict(crop=crop, cutoff=cutoff, seed=42, component=kind, smoke=False)
        if any(marker.get(k) != v or config.get(k) != v for k, v in expected.items()):
            raise ValueError('Incorrect component identity or smoke output')
        if config['source'] != str(source) or not marker['original_predictions_replayed']:
            raise ValueError('Original history source was not replayed')
        original_epochs = json.loads((source / 'metrics.json').read_text())['selected_epoch']
        epochs = config['selected_fixed_epochs']
        if epochs != original_epochs or marker['fixed_epochs'] != epochs:
            raise ValueError('Reconstruction changed original training duration')
        expected_steps = epochs * ((config['training_rows'] + (2047 if kind == 'mlp' else 1023))
                                   // (2048 if kind == 'mlp' else 1024))
        if marker['optimizer_steps'] != expected_steps or marker['initialization_component'] != (epochs == 0):
            raise ValueError('Incorrect training-step accounting')
        if kind == 'mlp' and epochs < 1:
            raise ValueError('MLP history must have been trained')
        for name, digest in config['sources'].items():
            if sha256(Path(name)) != digest:
                raise ValueError(f'History dependency changed: {name}')
        weights = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        original = torch.load(source / 'model_best.pt', map_location='cpu', weights_only=True)
        values = compare_weights(weights, original)
        replay = json.loads((folder / 'metrics.json').read_text())['original_replay_error']
        if set(replay) != {'train', 'validation', 'test'} or any(v != 0 for v in replay.values()):
            raise ValueError('Original forward reconstruction was not exact')
        if kind == 'mlp':
            training_archive = ROOT / f'benchmark/cache/task_aligned_world_v1/{crop}/origin_{cutoff+3}/train.npz'
            training_digest = config['sources'][str(training_archive)]
        else:
            training_archive = ROOT / f'benchmark/cache/neural_process_readout_v1/{crop}/origin_{cutoff+3}/train_labels.npz'
            original_config = json.loads((source / 'config.json').read_text())
            training_digest = original_config['input_manifest']['spec']['original_cache']['files']['train_labels.npz']
        if sha256(training_archive) != training_digest:
            raise ValueError('Original training labels changed')
        counts = {}
        for split in ('train', 'validation', 'test'):
            with np.load(folder / f'{split}_predictions.npz', allow_pickle=False) as saved:
                if set(saved.files) != {*LABELS, 'prediction'}:
                    raise ValueError('Unexpected prediction fields')
                count = len(saved['prediction'])
                if not np.isfinite(saved['prediction']).all():
                    raise ValueError('Nonfinite historical predictions')
                if any(saved[k].shape != (count,) for k in saved.files):
                    raise ValueError('Prediction or identity shape mismatch')
                counts[split] = count
                if split == 'train':
                    if count != config['training_rows'] or saved['year'].max() != cutoff:
                        raise ValueError('Training identity count or end differs')
                    with np.load(training_archive, allow_pickle=False) as old:
                        for key in LABELS:
                            np.testing.assert_array_equal(saved[key], old[key])
                else:
                    with np.load(source / f'{split}_predictions.npz', allow_pickle=False) as old:
                        for key in LABELS:
                            np.testing.assert_array_equal(saved[key], old[key])
                        column = 'prediction' if kind == 'mlp' else 'component_prediction'
                        np.testing.assert_array_equal(saved['prediction'], old[column])
        records.append(dict(crop=crop, cutoff=cutoff, seed=42, component=kind,
            fixed_epochs=epochs, optimizer_steps=expected_steps, initialization_component=epochs == 0,
            weight_values_verified=values, weights_exactly_equal=True,
            train_rows=counts['train'], validation_rows=counts['validation'], test_rows=counts['test'],
            original_replay_max_error=0., source=str(source / 'model_best.pt'),
            source_sha256=sha256(source / 'model_best.pt'), rebuilt=str(folder / 'model.pt'),
            rebuilt_sha256=sha256(folder / 'model.pt'), marker_sha256=sha256(folder / 'complete.json'),
            training_identity_source=str(training_archive), training_identity_sha256=training_digest))
        print(f'[HISTORY AUDIT] {crop} {cutoff} {kind}: weights and predictions exact', flush=True)
    stem = 'history_replay' if component == 'all' else 'mlp_replay'
    path = OUT / f'{stem}_checkpoints.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(passed=True, timestamp=datetime.now().astimezone().isoformat(),
        scope='Archived seed42 history reconstruction, not new model selection or multi-seed yield results',
        components=len(records), gradient_trained_components=sum(r['fixed_epochs'] > 0 for r in records),
        initialization_components=sum(r['initialization_component'] for r in records),
        weight_values_verified=sum(r['weight_values_verified'] for r in records),
        prediction_rows=sum(r[k] for r in records for k in ('train_rows', 'validation_rows', 'test_rows')),
        archived_validation_test_rows=sum(r[k] for r in records for k in ('validation_rows', 'test_rows')),
        maximum_prediction_error=0., all_weight_values_exact=True,
        original_development_selection_retained=True, complete_pipeline_repeated=False,
        sources=records, registry=str(path), registry_sha256=sha256(path),
        verifier_sha256=sha256(Path(__file__)), training_code_sha256=expected_code)
    atomic_json(OUT / f'{stem}_verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=('mlp', 'all'), default='all')
    args = parser.parse_args()
    result = audit(args.component)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'training_code_sha256')}, indent=2))
