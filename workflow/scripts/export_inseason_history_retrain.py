"""Export complete frozen history training inputs and unmodified definitions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from build_inseason_cpu_replay import export_symbols
from inseason_13year_data import ROOT, RECIPES, BLOCKS
from inseason_nested_common import LABELS, hashes, verify
from inseason_pipeline_seed_history import residual_target
from numeric_embedding_readout import load as tabm_load
from review_revision_data import sha256
from run_inseason_pipeline_seed_history import load_mlp, root_for, CODE

TEMPLATE = ROOT / 'Paper/iclr2027/reproducibility/history_retrain'
SEEDS = (42, 45, 48)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def definitions(destination):
    (destination / 'core').mkdir()
    (destination / 'source_snapshot').mkdir()
    specifications = (
        ('biid_world_model.py', ('FeedForward', 'RelevanceModulation', 'BIIDLayer', 'BIIDStack',
         'HistoryEncoder', 'PreviousLAIStateEncoder', 'WeatherTokenizer', 'CropWorldDynamics'),
         'from __future__ import annotations\nimport math\nimport torch\nfrom torch import nn'),
        ('numeric_embedding_readout.py', ('NeuralReadout',),
         'import warnings\nimport torch\nfrom torch import nn\nfrom tabm import TabM\nfrom rtdl_num_embeddings import PiecewiseLinearEmbeddings'),
        ('multimodal_baseline.py', ('set_seed',), 'import random\nimport numpy as np\nimport torch'),
        ('run_soybean_expert_seed_recheck.py', ('mlp_predict',), 'import numpy as np\nimport torch'),
        ('neural_process_readout.py', ('predict',), 'import numpy as np\nimport torch'),
    )
    records = [export_symbols(destination, *item) for item in specifications]
    for name in ('task_aligned_world.py', 'inseason_pipeline_seed_history.py'):
        for folder in ('core', 'source_snapshot'):
            shutil.copy2(ROOT / 'scripts' / name, destination / folder / name)
        records.append(dict(file=name, full_source_sha256=sha256(ROOT / 'scripts' / name), copied_whole=True))
    return records


def export_component(destination, crop, cutoff, component, x, arrays, sources):
    case = destination / f'cases/{crop}/cutoff_{cutoff}/{component}'
    (case / 'reference').mkdir(parents=True)
    if set(x) != {'train', 'validation', 'test'} or x['train'].shape[1] != 20:
        raise ValueError('Incomplete historical feature partitions')
    for split in x:
        parent = case if split == 'train' else case / 'reference'
        np.save(parent / f'{split}_x.npy', x[split], allow_pickle=False)
        keys = (*LABELS, 'baseline', 'target_residual') if component == 'mlp' else LABELS
        np.savez(parent / f'{split}_labels.npz', **{k: arrays[split][k] for k in keys})
    bin_file = ROOT / f'benchmark/cache/numeric_embedding_readout_v1/{crop}/origin_{cutoff+3}/history_bins.pt'
    if component == 'tabm':
        shutil.copy2(bin_file, case / 'bins.pt')
        sources[str(bin_file)] = sha256(bin_file)
    write_json(case / 'case.json', dict(crop=crop, cutoff=cutoff, component=component, dimensions=20,
        splits={s: dict(rows=len(x[s]), years=np.unique(arrays[s]['year']).tolist()) for s in x},
        full_training_rows=True, features_sha256=sha256(case / 'train_x.npy')))
    for seed in SEEDS:
        original = root_for(crop, cutoff, seed, component)
        marker = verify(original, hashes(CODE))
        config = json.loads((original / 'config.json').read_text())
        if (config['smoke'] or (config['crop'], config['cutoff'], config['seed'], config['component'])
                != (crop, cutoff, seed, component) or config['training_rows'] != len(x['train'])):
            raise ValueError('Original component does not match the complete input cohort')
        folder = case / f'seed_{seed}'
        (folder / 'reference').mkdir(parents=True)
        for split in x:
            path = original / f'{split}_predictions.npz'
            with np.load(path) as saved:
                for key in LABELS:
                    np.testing.assert_array_equal(saved[key], arrays[split][key])
            shutil.copy2(path, folder / 'reference' / path.name)
        shutil.copy2(original / 'model.pt', folder / 'reference/model.pt')
        if component == 'mlp':
            target = arrays['train']['target_residual']
        else:
            mlp = root_for(crop, cutoff, seed, 'mlp') / 'train_predictions.npz'
            with np.load(mlp) as saved:
                for key in LABELS:
                    np.testing.assert_array_equal(saved[key], arrays['train'][key])
                target = residual_target(arrays['train']['target'], saved['prediction'], config['normalization']['scale'])
            sources[str(mlp)] = sha256(mlp)
        write_json(folder / 'recipe.json', dict(seed=seed, epochs=config['selected_fixed_epochs'],
            component=component, normalization=config['normalization'],
            initialization_component=config['initialization_component'], optimizer_steps=marker['optimizer_steps'],
            target_dtype=str(target.dtype), target_bytes_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
            mlp_anchor='Reconstructed same-seed MLP' if component == 'tabm' else 'Causal trend',
            model_selection=False, all_training_rows=True))
        sources.update(config['sources'])
        sources.update({str(original / name): checksum for name, checksum in marker['files'].items()})
        sources[str(original / 'complete.json')] = sha256(original / 'complete.json')
        sources.update({str(ROOT / 'scripts' / name): checksum for name, checksum in marker['code_sha256'].items()})
    print(f'[HISTORY EXPORTED] {crop}/{cutoff}/{component} rows={len(x["train"])}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError('Do not overwrite an existing reconstruction package')
    args.destination.mkdir(parents=True)
    code = definitions(args.destination)
    sources = {str(ROOT / 'scripts' / r['file']): r['full_source_sha256'] for r in code}
    sources[str(ROOT / 'scripts/build_inseason_cpu_replay.py')] = sha256(ROOT / 'scripts/build_inseason_cpu_replay.py')
    for name in ('README.md', 'requirements.txt', 'run.py'):
        shutil.copy2(TEMPLATE / name, args.destination / name)
        sources[str(TEMPLATE / name)] = sha256(TEMPLATE / name)
    jobs = []
    for crop in RECIPES:
        for cutoff in BLOCKS:
            x, arrays, _, _, _, lineage = load_mlp(crop, cutoff)
            sources.update(lineage)
            export_component(args.destination, crop, cutoff, 'mlp', x, arrays, sources)
            jobs.extend(dict(case=f'{crop}/cutoff_{cutoff}', seed=s, component='mlp') for s in SEEDS)
            if crop == 'soybean':
                tx, ta, meta = tabm_load(crop, cutoff+3, 'history')
                for split in arrays:
                    for key in LABELS:
                        np.testing.assert_array_equal(arrays[split][key], ta[split][key])
                original_cache = ROOT / f'benchmark/cache/neural_process_readout_v1/{crop}/origin_{cutoff+3}'
                for name, checksum in meta['spec']['original_cache']['files'].items():
                    sources[str(original_cache / name)] = checksum
                export_component(args.destination, crop, cutoff, 'tabm', tx, ta, sources)
                jobs.extend(dict(case=f'{crop}/cutoff_{cutoff}', seed=s, component='tabm') for s in SEEDS)
    for name, checksum in sources.items():
        if sha256(Path(name)) != checksum:
            raise ValueError(f'Original input changed: {name}')
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    write_json(args.destination / 'provenance.json', dict(
        sources={str(Path(k).relative_to(ROOT)): v for k, v in sources.items()},
        extracted_definitions=code, prepared_training_only=True,
        raw_processing_reproduced=False, scientific_results_replaced=False, public_release_ready=False))
    files = {str(p.relative_to(args.destination)): sha256(p) for p in sorted(args.destination.rglob('*')) if p.is_file()}
    write_json(args.destination / 'manifest.json', dict(schema=1, files=files, jobs=jobs,
        components=45, gradient_trained_components=39, initialization_components=6,
        original_selection_retained=True, full_world_model_retraining=False))
    print(f'[HISTORY PACKAGE READY] components={len(jobs)} files={len(files)}', flush=True)


if __name__ == '__main__':
    main()
