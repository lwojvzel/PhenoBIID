"""Verify fixed-capacity historical components and paired identities at all seeds."""
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from inseason_13year_data import ROOT, BLOCKS, RECIPES
from inseason_nested_common import LABELS, hashes, verify
from review_revision_data import sha256
from run_inseason_pipeline_seed_history import CODE, OUT, root_for
from run_review_revision_parallel import atomic_json

CAPACITY_KEYS = ('crop', 'cutoff', 'component', 'smoke', 'source', 'normalization',
                 'selected_fixed_epochs', 'original_selected_epochs',
                 'initialization_component', 'history_predictions_in_sample',
                 'training_rows', 'training_years', 'selection', 'code_sha256')


def same_capacity(config, reference):
    for key in CAPACITY_KEYS:
        if key not in config or config[key] != reference[key]:
            raise ValueError(f'Historical seed changed fixed capacity or inputs: {key}')


def main():
    original_path = OUT / 'history_replay_verification.json'
    original = json.loads(original_path.read_text())
    if (not original['passed'] or original['components'] != 15 or
            original['verifier_sha256'] != sha256(ROOT / 'scripts/audit_inseason_history_replay.py')):
        raise ValueError('Complete original history reconstruction is required')
    code = hashes(CODE)
    if original['training_code_sha256'] != code:
        raise ValueError('Original reconstruction used different training code')
    source_files = {str(original_path): sha256(original_path)}
    for record in original['sources']:
        source_files[record['source']] = record['source_sha256']
        source_files[record['rebuilt']] = record['rebuilt_sha256']
        parent = Path(record['rebuilt']).parent / 'complete.json'
        source_files[str(parent)] = record['marker_sha256']
    records = []
    for crop in RECIPES:
        for cutoff in BLOCKS:
            for component in (('mlp', 'tabm') if crop == 'soybean' else ('mlp',)):
                original_root = root_for(crop, cutoff, 42, component)
                reference = json.loads((original_root / 'config.json').read_text())
                for seed in (42, 45, 48):
                    folder = root_for(crop, cutoff, seed, component)
                    marker = verify(folder, code)
                    config = json.loads((folder / 'config.json').read_text())
                    same_capacity(config, reference)
                    if any(config[k] != v or marker[k] != v for k, v in
                           dict(crop=crop, cutoff=cutoff, seed=seed, component=component, smoke=False).items()):
                        raise ValueError('Incorrect historical component identity')
                    if marker['original_predictions_replayed'] != (seed == 42):
                        raise ValueError('Repetition mislabeled as original replay')
                    epochs = config['selected_fixed_epochs']
                    batch = 2048 if component == 'mlp' else 1024
                    steps = epochs * ((config['training_rows'] + batch - 1) // batch)
                    if (marker['optimizer_steps'] != steps or marker['fixed_epochs'] != epochs or
                            marker['initialization_component'] != (epochs == 0)):
                        raise ValueError('Optimizer-step or initialization accounting differs')
                    if component == 'mlp' and epochs < 1:
                        raise ValueError('Historical MLP did not train')
                    if component == 'tabm':
                        parent = root_for(crop, cutoff, seed, 'mlp') / 'complete.json'
                        if config['sources'].get(str(parent)) != sha256(parent):
                            raise ValueError('TabM used the wrong seed historical parent')
                    for name, digest in config['sources'].items():
                        if name in source_files and source_files[name] != digest:
                            raise ValueError('Historical repeats disagree on a source version')
                        source_files[name] = digest
                    counts = {}
                    for split in ('train', 'validation', 'test'):
                        with np.load(folder / f'{split}_predictions.npz', allow_pickle=False) as saved:
                            with np.load(original_root / f'{split}_predictions.npz', allow_pickle=False) as base:
                                if set(saved.files) != {*LABELS, 'prediction'}:
                                    raise ValueError('Unexpected historical prediction fields')
                                for key in LABELS:
                                    np.testing.assert_array_equal(saved[key], base[key])
                                if saved['prediction'].shape != base['prediction'].shape or not np.isfinite(saved['prediction']).all():
                                    raise ValueError('Incomplete paired historical predictions')
                                counts[split] = len(saved['prediction'])
                    records.append(dict(crop=crop, cutoff=cutoff, seed=seed, component=component,
                        fixed_epochs=epochs, optimizer_steps=steps, initialization_component=epochs == 0,
                        training_rows=counts['train'], validation_rows=counts['validation'], test_rows=counts['test'],
                        paired_identities=True, weight=str(folder / 'model.pt'),
                        weight_sha256=sha256(folder / 'model.pt'),
                        complete_marker=str(folder / 'complete.json'),
                        complete_marker_sha256=sha256(folder / 'complete.json')))
                    print(f'[HISTORY SEEDS] {crop} {cutoff} {component} seed={seed} verified', flush=True)
    if len(records) != 45:
        raise ValueError('Expected 36 MLP and 9 soybean TabM components')
    for name, digest in source_files.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Historical source changed: {name}')
    registry = OUT / 'history_seed_checkpoints.csv'
    with registry.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(passed=True, timestamp=datetime.now().astimezone().isoformat(),
        scope='Historical components only; no completed multi-seed world-model yield table',
        seeds=[42, 45, 48], components=len(records),
        gradient_trained_components=sum(r['optimizer_steps'] > 0 for r in records),
        initialization_components=sum(r['initialization_component'] for r in records),
        paired_identities_across_seeds=True, fixed_original_capacity=True,
        original_development_selection_retained=True, complete_pipeline_repeated=False,
        source_files=source_files, registry=str(registry), registry_sha256=sha256(registry),
        verifier_sha256=sha256(Path(__file__)), training_code_sha256=code)
    atomic_json(OUT / 'history_seeds_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_files', 'training_code_sha256')}, indent=2))


if __name__ == '__main__':
    main()
