"""Export all fixed-capacity terminal recipes and complete prepared fit inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from crop_signal_screen_data import load as signal_load, cache_root, select_components
from inseason_13year_data import ROOT, RECIPES, BLOCKS
from inseason_nested_common import LABELS, hashes, verify
from inseason_pipeline_seed_terminal import residual_labels, validate_feature_schema
from run_inseason_pipeline_seed_terminal import CODE, root_for
from run_crop_signal_screen import year_weights
from review_revision_data import sha256

TEMPLATE = ROOT / 'Paper/iclr2027/reproducibility/terminal_retrain'
SEEDS = (42, 45, 48)
MATH = {'run_crop_signal_screen.py': ('year_weights',),
        'inseason_pipeline_seed_terminal.py': ('residual_labels',)}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def export_math(destination):
    chunks, sources = [], {}
    for filename, names in MATH.items():
        path = ROOT / 'scripts' / filename
        source = path.read_text()
        lines = source.splitlines()
        found = set()
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                chunks.append('\n'.join(lines[node.lineno-1:node.end_lineno]))
                found.add(node.name)
        if found != set(names):
            raise ValueError('Missing original math definition')
        sources[str(path.relative_to(ROOT))] = sha256(path)
    (destination / 'terminal_math.py').write_text(
        '"""Unmodified arithmetic definitions from registered training source."""\nimport numpy as np\n\n' +
        '\n\n'.join(chunks) + '\n')
    return sources


def export_case(destination, crop, cutoff):
    origin, recipe = cutoff + 3, RECIPES[crop]
    x, labels, manifest = signal_load(crop, origin, recipe.split('_')[0])
    if recipe == 'ndvi_gpp':
        other, identities, _ = signal_load(crop, origin, 'gpp')
        for split in x:
            for key in LABELS:
                np.testing.assert_array_equal(labels[split][key], identities[split][key])
            np.testing.assert_array_equal(x[split][:, :465], other[split][:, :465])
            x[split] = np.concatenate((x[split], other[split][:, -36:]), 1)
    validate_feature_schema(x['train'], manifest['spec']['names'])
    case = destination / f'cases/{crop}/cutoff_{cutoff}'
    case.mkdir(parents=True)
    np.save(case / 'train.npy', x['train'], allow_pickle=False)
    np.savez(case / 'train_labels.npz', **{k: labels['train'][k] for k in LABELS},
        trend_target_residual=labels['train']['target_residual'])
    (case / 'reference').mkdir()
    # Deterministic row positions, never prediction errors, determine probes.
    probes = []
    for split in ('train', 'validation'):
        indices = np.linspace(0, len(x[split])-1, min(1024, len(x[split])), dtype=int)
        probes.append(x[split][indices])
    probe = np.concatenate(probes)
    np.save(case / 'reference/probe.npy', probe, allow_pickle=False)
    cache = cache_root(crop, origin)
    components = set().union(*(set(select_components(p)) for p in recipe.split('_')))
    source_names = ['manifest.json', 'audit.json'] + [f'{s}_labels.npz' for s in labels]
    source_names += [f'{component}_{split}.npy' for component in components for split in labels]
    sources = {str((cache / name).relative_to(ROOT)): sha256(cache / name) for name in source_names}
    spec = dict(schema=1, crop=crop, cutoff=cutoff, recipe=recipe,
        training_rows=len(x['train']), feature_width=x['train'].shape[1],
        training_years=np.unique(labels['train']['year']).tolist(),
        sample_weight_bytes_sha256=hashlib.sha256(year_weights(labels['train']['year']).tobytes()).hexdigest(),
        feature_names=manifest['spec']['names'], training_input='Complete original observed-trajectory features',
        probe_scope='Arithmetic probes only: evenly spaced train and full-observed validation rows; not new seasonal scores')
    if spec['training_years'][-1] != cutoff:
        raise ValueError('Unexpected training cutoff')
    write_json(case / 'case.json', spec)
    for seed in SEEDS:
        root = root_for(crop, cutoff, seed)
        marker = verify(root, hashes(CODE))
        config = json.loads((root / 'config.json').read_text())
        if marker['smoke'] or (config['crop'], config['cutoff'], config['seed']) != (crop, cutoff, seed):
            raise ValueError('Wrong original terminal identity')
        if (config['training_rows'], config['feature_width']) != x['train'].shape:
            raise ValueError('Training matrix differs from registered terminal inputs')
        with np.load(root / 'train_anchor.npz') as saved:
            for key in LABELS:
                np.testing.assert_array_equal(labels['train'][key], saved[key])
            anchor = saved['prediction'].astype(float)
        output = case / f'seed_{seed}'
        (output / 'reference').mkdir(parents=True)
        np.save(output / 'anchor.npy', anchor, allow_pickle=False)
        branches = []
        for index, branch in enumerate(config['branches']):
            model_path = root / branch['name'] / 'model.joblib'
            model = joblib.load(model_path)
            if model.get_params() != branch['parameters']:
                raise ValueError('Saved model parameters differ from recipe')
            residual = (labels['train']['target_residual'] if branch['name'] == 'trend' else
                        residual_labels(labels['train']['target'], anchor, branch['config']['normalization']))
            branches.append(dict(name=branch['name'], parameters=branch['parameters'],
                normalization=branch['config']['normalization'], head=branch['config']['head'],
                original_fitted_trees=model.booster_.current_iteration(),
                residual_dtype=str(residual.dtype),
                residual_bytes_sha256=hashlib.sha256(residual.tobytes()).hexdigest()))
            write_json(output / f'reference/trees_{index}.json', model.booster_.dump_model())
            np.save(output / f'reference/probe_{index}.npy', model.booster_.predict(probe, num_threads=4), allow_pickle=False)
        history = {k: v for k, v in config['historical'].items() if k != 'root'}
        write_json(output / 'recipe.json', dict(seed=seed, branches=branches, historical=history,
            historical_frozen=True, original_development_capacity_retained=True,
            normalization_refitted=False, early_stopping=False))
        for name, checksum in marker['files'].items():
            sources[str((root / name).relative_to(ROOT))] = checksum
        sources[str((root / 'complete.json').relative_to(ROOT))] = sha256(root / 'complete.json')
        for name, checksum in marker['code_sha256'].items():
            sources['scripts/' + name] = checksum
        # Cache manifests must also match the sources recorded at original fit time.
        for name in ('manifest.json', 'audit.json'):
            original = config['sources'][str(cache / name)]
            if sources[str((cache / name).relative_to(ROOT))] != original:
                raise ValueError('Original fitted feature source differs')
    print(f'[EXPORTED] {crop} cutoff={cutoff} rows={len(x["train"])} width={x["train"].shape[1]} seeds=3', flush=True)
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError('Choose a new export destination')
    args.destination.mkdir(parents=True)
    sources = export_math(args.destination)
    for name in ('run.py', 'requirements.txt', 'README.md'):
        shutil.copy2(TEMPLATE / name, args.destination / name)
        sources[str((TEMPLATE / name).relative_to(ROOT))] = sha256(TEMPLATE / name)
    jobs = []
    for crop in RECIPES:
        for cutoff in BLOCKS:
            sources.update(export_case(args.destination, crop, cutoff))
            jobs.extend(dict(case=f'{crop}/cutoff_{cutoff}', seed=seed) for seed in SEEDS)
    for relative, checksum in sources.items():
        if sha256(ROOT / relative) != checksum:
            raise ValueError(f'Original asset changed during export: {relative}')
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    write_json(args.destination / 'provenance.json', dict(sources=sources,
        source_paths='Relative to the original project root; no absolute personal paths',
        definitions_extracted_unchanged=MATH, new_performance_search=False,
        contains_full_terminal_training_data=True, contains_full_raw_processing=False,
        contains_historical_or_state_retraining=False, public_release_ready=False))
    files = {str(p.relative_to(args.destination)): sha256(p) for p in sorted(args.destination.rglob('*')) if p.is_file()}
    write_json(args.destination / 'manifest.json', dict(schema=1, files=files, jobs=jobs,
        systems=len(jobs), branch_fits=45, full_terminal_training_rows=True,
        full_world_model_retraining=False, scientific_results_replaced=False, public_release_ready=False))
    print(f'[PACKAGE READY] systems={len(jobs)} branches=45 files={len(files)}', flush=True)


if __name__ == '__main__':
    main()
