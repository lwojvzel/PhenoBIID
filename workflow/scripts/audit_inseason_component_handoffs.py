"""Verify full-row handoffs between independently reconstructed components."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from inseason_pipeline_seed_terminal import residual_labels
from review_revision_data import sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'benchmark/results/inseason_reproducibility_v1'
RECIPES = dict(maize=('gpp',), rice=('ndvi',), soybean=('ndvi',), wheat=('ndvi', 'gpp'))
LABELS = ('source_indices', 'year', 'row', 'col', 'target')


def matched_identity(left, right):
    if not set(LABELS).issubset(left) or not set(LABELS).issubset(right):
        raise ValueError('Missing handoff identities')
    n = len(left['source_indices'])
    if len(np.unique(left['source_indices'])) != n:
        raise ValueError('Duplicate row identities')
    for key in LABELS:
        if left[key].shape != (n,) or right[key].shape != (n,):
            raise ValueError('Handoff row shape changed')
        np.testing.assert_array_equal(left[key], right[key])
    return n


def checked_residuals(labels, new_anchor, old_anchor, recipe):
    if new_anchor.shape != old_anchor.shape or new_anchor.shape != labels['target'].shape:
        raise ValueError('Historical anchor shape changed')
    if not np.isfinite(new_anchor).all():
        raise ValueError('Nonfinite historical output')
    np.testing.assert_array_equal(new_anchor, old_anchor)
    result = []
    for branch in recipe['branches']:
        target = (labels['trend_target_residual'] if branch['name'] == 'trend' else
                  residual_labels(labels['target'], new_anchor, branch['normalization']))
        checksum = hashlib.sha256(target.tobytes()).hexdigest()
        if checksum != branch['residual_bytes_sha256'] or str(target.dtype) != branch['residual_dtype']:
            raise ValueError('New history does not reproduce the terminal training target')
        result.append(dict(branch=branch['name'], rows=len(target), dtype=str(target.dtype), sha256=checksum))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not replace component-handoff evidence')
    sources = {}

    def read_json(path):
        sources[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def read_npz(path):
        sources[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            return {key:saved[key] for key in saved.files}

    def read_npy(path):
        sources[str(path)] = sha256(path)
        return np.load(path, mmap_mode='r', allow_pickle=False)

    completed, bundles = {}, {}
    for stage in ('history', 'state', 'terminal'):
        folder = EVIDENCE / f'{stage}_retrain_20260909'
        complete = read_json(folder / 'complete.json')
        audit = read_json(folder / 'original_weight_audit.json')
        if not complete['passed'] or not audit['passed'] or complete['new_scientific_comparisons'] != 0:
            raise ValueError('An independently reconstructed stage is incomplete')
        bundle = Path(complete['relocated']).resolve()
        if bundle.is_relative_to(ROOT):
            raise ValueError('Not a relocated reconstruction')
        manifest = read_json(bundle / 'manifest.json')
        for name, digest in manifest['files'].items():
            if sha256(bundle / name) != digest:
                raise ValueError('Relocated input changed')
        for case in complete['cases']:
            if stage == 'terminal':
                output = Path(case['results'])
                files = case['result_files']
            else:
                output = Path(case['reconstructed'])
                files = case['files']
            for name, digest in files.items():
                if sha256(output / name) != digest:
                    raise ValueError('Reconstructed output changed')
        completed[stage], bundles[stage] = complete, bundle
    states = {(r['job']['case'], r['job']['product'], r['job']['seed'])
              for r in completed['state']['cases']}
    histories = {(r['job']['case'], r['job']['component'], r['job']['seed'])
                 for r in completed['history']['cases']}
    terminals = {(r['job']['case'], r['job']['seed']) for r in completed['terminal']['cases']}
    if (len(states), len(histories), len(terminals)) != (45, 45, 36):
        raise ValueError('Incomplete registered component matrix')
    records, cohorts = [], []
    for crop, products in RECIPES.items():
        for cutoff in (2001, 2005, 2009):
            case = f'{crop}/cutoff_{cutoff}'
            history_case = bundles['history'] / 'cases' / case
            terminal_case = bundles['terminal'] / 'cases' / case
            state_case = bundles['state'] / 'cases' / case
            training = read_npz(history_case / 'mlp/train_labels.npz')
            terminal_labels = read_npz(terminal_case / 'train_labels.npz')
            state_labels = {key:read_npy(state_case / f'data/{key}.npy') for key in LABELS}
            rows = matched_identity(training, terminal_labels)
            matched_identity(training, state_labels)
            if training['year'].max() != cutoff:
                raise ValueError('Wrong complete training cutoff')
            if crop == 'soybean':
                matched_identity(training, read_npz(history_case / 'tabm/train_labels.npz'))
            component = 'tabm' if crop == 'soybean' else 'mlp'
            parts = ('validation', 'test') if cutoff == 2009 else ('validation',)
            evaluation_parts = [read_npz(history_case / component / f'reference/{split}_labels.npz')
                                for split in parts]
            evaluation = {key:np.concatenate([part[key] for part in evaluation_parts]) for key in LABELS}
            if set(evaluation['year']) != set(range(cutoff+1, 2017 if cutoff == 2009 else cutoff+4)):
                raise ValueError('Incorrect thirteen-year block')
            cohorts.append(dict(case=case, training_rows=rows, training_identities_exact=True,
                                evaluation_rows=len(evaluation['year']), evaluation_years=sorted(set(evaluation['year'].tolist()))))
            for seed in (42, 45, 48):
                if ((case, seed) not in terminals or (case, component, seed) not in histories
                        or any((case, product, seed) not in states for product in products)):
                    raise ValueError('Cross-seed or missing component connection')
                parent = bundles['history'] / f'reconstructed/{case}/seed_{seed}/{component}'
                new_anchor = read_npy(parent / 'train_prediction.npy')
                old_anchor = read_npy(terminal_case / f'seed_{seed}/anchor.npy')
                recipe = read_json(terminal_case / f'seed_{seed}/recipe.json')
                branches = checked_residuals(terminal_labels, new_anchor, old_anchor, recipe)
                inference = ROOT / f'benchmark/results/inseason_pipeline_seeds_v1/inference/{case}/seed_{seed}'
                reference = read_npz(inference / 'labels.npz')
                matched_identity(evaluation, reference)
                new_evaluation = np.concatenate([read_npy(parent / f'checked_{split}_prediction.npy') for split in parts])
                np.testing.assert_array_equal(new_evaluation, reference['history_anchor'])
                records.append(dict(case=case, seed=seed, history_component=component, products=products,
                    training_anchor_rows=len(new_anchor), evaluation_anchor_rows=len(new_evaluation),
                    maximum_training_anchor_error=0, maximum_evaluation_anchor_error=0,
                    residual_targets=branches, same_seed_components=True))
    for filename, digest in sources.items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Handoff dependency changed while being checked')
    result = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(), systems=len(records),
        physical_cohorts=cohorts, cases=records,
        training_anchor_rows=sum(r['training_anchor_rows'] for r in records),
        evaluation_anchor_rows=sum(r['evaluation_anchor_rows'] for r in records),
        residual_targets_checked=sum(len(r['residual_targets']) for r in records),
        all_training_identities_exact=True, all_new_historical_anchors_exact=True,
        new_state_forward_executed=False, terminal_refit_with_new_history_executed=False,
        complete_new_chain_executed=False, raw_processing_reproduced=False,
        independent_recipe_confirmation=False, scientific_results_replaced=False,
        scope='Full-row identity, reconstructed-history anchor, and terminal residual handoffs; not a newly executed full pipeline',
        sources=sources, verifier_sha256=sha256(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('cases','sources','physical_cohorts')}, indent=2))


if __name__ == '__main__':
    main()
