"""Package full original physical state inputs and fixed refit configurations."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from build_inseason_cpu_replay import export_symbols
from forecast_bridge_data import ROOT, load, root as data_root, fit_stats, identity_hash
from forecast_bridge_state import batch_arrays
from inseason_13year_data import RECIPES, BLOCKS
from review_revision_data import sha256
from run_forecast_bridge_state import run_root, CODE
from run_ndvi_signal_permutation import check_files

TEMPLATE = ROOT / 'Paper/iclr2027/reproducibility/state_retrain'
KEYS = ('source_indices', 'year', 'row', 'col', 'target', 'weather', 'context', 'relative_valid',
        'previous_ndvi', 'previous_gpp', 'previous_ndvi_quality', 'previous_gpp_quality',
        'observed_lai', 'observed_ndvi', 'observed_gpp')
SEEDS = (42, 45, 48)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def export_code(destination):
    (destination / 'core').mkdir()
    (destination / 'source_snapshot').mkdir()
    specifications = (
        ('biid_world_model.py', ('FeedForward', 'RelevanceModulation', 'BIIDLayer', 'BIIDStack',
         'HistoryEncoder', 'PreviousLAIStateEncoder', 'WeatherTokenizer', 'CropWorldDynamics'),
         'from __future__ import annotations\nimport math\nimport torch\nfrom torch import nn'),
        ('multimodal_baseline.py', ('set_seed',), 'import random\nimport numpy as np\nimport torch'),
        ('forecast_bridge_data.py', ('PRODUCTS', 'identity_hash', 'fit_stats'), 'import numpy as np'),
        ('run_forecast_bridge_state.py', ('tensors', 'predict', 'train_epoch'),
         'import numpy as np\nimport torch\nfrom forecast_bridge_state import batch_arrays, masked_loss'),
    )
    records = [export_symbols(destination, *s) for s in specifications]
    for name in ('dual_remote_state.py', 'token_retention_state.py', 'forecast_bridge_state.py'):
        for directory in ('core', 'source_snapshot'):
            shutil.copy2(ROOT / 'scripts' / name, destination / directory / name)
        records.append(dict(file=name, full_source_sha256=sha256(ROOT / 'scripts' / name), copied_whole=True))
    return records


def subset_original(raw, cutoff):
    take = np.flatnonzero(raw['year'] <= cutoff)
    if not len(take):
        raise ValueError('No complete training window')
    values = {key: np.asarray(raw[key][take]) for key in KEYS}
    if values['year'].max() != cutoff or len(np.unique(values['source_indices'])) != len(take):
        raise ValueError('Wrong temporal boundary or repeated training identities')
    return take, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError('Choose a new state export directory')
    args.destination.mkdir(parents=True)
    records = export_code(args.destination)
    sources = {str(ROOT / 'scripts' / r['file']): r['full_source_sha256'] for r in records}
    sources[str(ROOT / 'scripts/build_inseason_cpu_replay.py')] = sha256(ROOT / 'scripts/build_inseason_cpu_replay.py')
    for name in ('README.md', 'requirements.txt', 'run.py'):
        shutil.copy2(TEMPLATE / name, args.destination / name)
        sources[str(TEMPLATE / name)] = sha256(TEMPLATE / name)
    jobs = []
    for crop, products in RECIPES.items():
        raw, meta = load(crop, verify=True)
        sources[str(data_root(crop) / 'manifest.json')] = sha256(data_root(crop) / 'manifest.json')
        for key in KEYS:
            sources[str(data_root(crop) / f'{key}.npy')] = meta['files'][f'{key}.npy']
        for cutoff in BLOCKS:
            original_take, values = subset_original(raw, cutoff)
            take = np.arange(len(original_take))
            stats = fit_stats(values, take)
            if stats != fit_stats(raw, original_take):
                raise ValueError('Compact physical arrays changed normalization')
            case = args.destination / f'cases/{crop}/cutoff_{cutoff}'
            (case / 'data').mkdir(parents=True)
            for key, value in values.items():
                np.save(case / 'data' / f'{key}.npy', value, allow_pickle=False)
            write_json(case / 'case.json', dict(crop=crop, cutoff=cutoff, keys=KEYS, products=products.split('_'),
                rows=len(take), source_identity=identity_hash(values['source_indices']),
                years=np.unique(values['year']).tolist(), statistics=stats,
                physical_shapes={k: list(v.shape) for k, v in values.items()}, no_post_cutoff_rows=True))
            for product in products.split('_'):
                for seed in SEEDS:
                    original = run_root(crop, product, 'biid', cutoff, seed)
                    marker = json.loads((original / 'complete.json').read_text())
                    check_files(original, marker['files'])
                    config = json.loads((original / 'config.json').read_text())
                    expected = dict(crop=crop, product=product, cutoff=cutoff, seed=seed,
                                    architecture='biid', epochs=30, patience=5, batch=256, smoke=False)
                    if any(config.get(k) != v for k, v in expected.items()):
                        raise ValueError('Changed registered state recipe')
                    if (marker['smoke'] or marker['full_training_rows'] != len(take)
                            or marker['selected_epochs'] < 1 or config['full_identity'] != identity_hash(values['source_indices'])):
                        raise ValueError('Wrong original full-refit cohort')
                    for name in CODE:
                        file = ROOT / 'scripts' / name
                        if sha256(file) != config['code_sha256'][name]:
                            raise ValueError('Changed original state code')
                        sources[str(file)] = config['code_sha256'][name]
                    if stats != json.loads((original / 'normalization.json').read_text()):
                        raise ValueError('Original full-refit normalization differs')
                    # Permuting compact positions must preserve every original batch value.
                    compact_order = np.random.default_rng(seed).permutation(take)
                    original_order = np.random.default_rng(seed).permutation(original_take)
                    np.testing.assert_array_equal(values['source_indices'][compact_order], raw['source_indices'][original_order])
                    for start in (0, max(0, len(take)-256)):
                        a = batch_arrays(values, compact_order[start:start+256], stats, product)
                        b = batch_arrays(raw, original_order[start:start+256], stats, product)
                        for key in a:
                            np.testing.assert_array_equal(a[key], b[key])
                    folder = case / product / f'seed_{seed}'
                    (folder / 'reference').mkdir(parents=True)
                    shutil.copy2(original / 'model.pt', folder / 'reference/model.pt')
                    shutil.copy2(original / 'training_history.json', folder / 'reference/training_history.json')
                    write_json(folder / 'recipe.json', dict(product=product, seed=seed,
                        epochs=marker['selected_epochs'], batch=config['batch'], optimizer=config['optimizer'],
                        selection_repeated=False, full_training_rows=len(take),
                        source_identity=config['full_identity'], full_refit_only=True))
                    sources.update({str(original / name): value for name, value in marker['files'].items()})
                    sources[str(original / 'complete.json')] = sha256(original / 'complete.json')
                    jobs.append(dict(case=f'{crop}/cutoff_{cutoff}', product=product, seed=seed,
                                     work=int(marker['selected_epochs'])*len(take)))
            print(f'[STATE INPUT EXPORTED] {crop}/{cutoff} rows={len(take)} products={products}', flush=True)
    for name, checksum in sources.items():
        if sha256(Path(name)) != checksum:
            raise ValueError(f'Original state source changed: {name}')
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    write_json(args.destination / 'provenance.json', dict(
        sources={str(Path(k).relative_to(ROOT)): v for k, v in sources.items()}, code=records,
        compact_row_order_preserved=True, no_post_cutoff_training_rows=True,
        original_raw_preprocessing_repeated=False, early_stopping_selection_repeated=False,
        public_release_ready=False))
    files = {str(p.relative_to(args.destination)): sha256(p) for p in sorted(args.destination.rglob('*')) if p.is_file()}
    write_json(args.destination / 'manifest.json', dict(schema=1, files=files,
        jobs=sorted(jobs, key=lambda j:(j['work'], j['case'], j['product'], j['seed'])),
        components=45, pilot_order='Full training row count times selected epochs, never performance',
        original_model_selection_retained=True, scientific_results_replaced=False))
    print(f'[STATE PACKAGE READY] components={len(jobs)} files={len(files)}', flush=True)


if __name__ == '__main__':
    main()
