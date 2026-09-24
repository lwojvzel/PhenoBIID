"""Recreate full base/world cohorts and original rolling input encodings."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import run_raw_foundation_reproduction as foundation
from run_raw_foundation_reproduction import ROOT, compare_array, digest, write_json
from run_raw_remote_reproduction import selected_nodes


EVIDENCE = ROOT / 'benchmark/results/inseason_reproducibility_v1/fresh_cohort_20260909'
ENV = ROOT / 'Paper/release/inseason_fresh_cohort_env'
BASE = ROOT / 'benchmark/results/inseason_reproducibility_v1'
CROPS = ('maize', 'rice', 'soybean', 'wheat')
ORIGINS = (2004, 2008, 2012)
EXPORTS = {
    'multimodal_baseline.py': (
        'import csv, json, math\nfrom dataclasses import dataclass\nfrom pathlib import Path\n'
        'from typing import Any\nimport numpy as np\nfrom numpy.lib.format import open_memmap\n',
        ('PROJECT_ROOT', 'PROCESSED_ROOT', 'ERA5_ROOT', 'LAI_ROOT', 'CACHE_ROOT', 'CROPS', 'MIRCA_YEARS',
         'ERA5_VARIABLES', 'MODE_MODALITIES', 'CACHE_VERSION', 'GRID_WIDTH', 'nearest_mirca_year',
         'save_json', 'write_csv', 'load_coordinates', '_crop_paths', '_valid_sample_mask',
         '_cache_is_current', 'build_crop_cache', 'load_cache', 'NormalizationStats',
         '_stream_mean_std', 'compute_normalization')),
    'run_history_multimodal_baselines.py': (
        'import numpy as np\nfrom multimodal_baseline import GRID_WIDTH\n',
        ('HISTORY_FEATURE_NAMES', 'build_causal_history_features')),
    'biid_world_model.py': (
        'import json, math\nfrom dataclasses import dataclass, asdict\nfrom pathlib import Path\n'
        'import numpy as np\nfrom numpy.lib.format import open_memmap\n'
        'from multimodal_baseline import (CACHE_ROOT as MULTIMODAL_CACHE_ROOT, CROPS, LAI_ROOT, '
        'PROCESSED_ROOT, compute_normalization, load_cache, nearest_mirca_year, save_json)\n'
        'from run_history_multimodal_baselines import build_causal_history_features\n',
        ('PROJECT_ROOT', 'WORLD_CACHE_ROOT', 'LAI_REL_ROOT', 'WORLD_CACHE_VERSION', 'WorldNormalizationStats',
         'world_cache_dir', '_world_cache_is_current', 'build_world_cache', 'load_world_cache',
         'compute_world_stats', 'load_or_compute_world_stats', 'make_world_arrays')),
    'audit_crop_area_fraction.py': (
        'from pathlib import Path\nimport numpy as np\n',
        ('ROOT', 'MIRCA_ROOT', 'MIRCA_YEARS', 'nearest_mirca_year', 'grid_area_hectares',
         'maximum_monthly_growing_area', 'sample_fraction')),
    'review_revision_data.py': ('from pathlib import Path\nimport hashlib\n', ('ROOT', 'CROPS', 'SPLITS', 'sha256')),
    'run_review_revision_parallel.py': ('import json\n', ('atomic_json',)),
    'prepare_pku_ndvi.py': ('from review_revision_data import ROOT\n', ('OUT',)),
    'prepare_reclue_monthly_gpp.py': ('import numpy as np\nfrom review_revision_data import ROOT\n', ('OUT', 'crop_support')),
    'dual_remote_data.py': ('import numpy as np\nfrom prepare_pku_ndvi import OUT\n', ('extract',)),
    'forward_protocol_revision.py': (
        'import numpy as np\nimport hashlib\n'
        'from biid_world_model import WorldNormalizationStats, load_world_cache, make_world_arrays\n'
        'from multimodal_baseline import load_cache\nfrom review_revision_data import load_shared\n'
        'from run_history_multimodal_baselines import build_causal_history_features\n',
        ('index_hash', 'safe_std', 'RawInputs')),
    'observed_remote_benchmark.py': ('', ('INPUTS',)),
    'stable_remote_data.py': (
        'from dataclasses import asdict\nimport fcntl, json\nfrom pathlib import Path\nimport numpy as np\n'
        'from dual_remote_data import extract\nfrom forward_protocol_revision import RawInputs, index_hash\n'
        'from observed_remote_benchmark import INPUTS\nfrom prepare_pku_ndvi import OUT as NDVI_ROOT\n'
        'from review_revision_data import ROOT, CROPS, SPLITS, sha256\nfrom run_review_revision_parallel import atomic_json\n',
        ('CACHE', 'ORIGINS', 'split_rows', 'cache_root', 'prepare', 'load')),
}


def export_definitions(source_path, destination, imports, names):
    source = source_path.read_text()
    nodes = selected_nodes(source, names)
    lines = source.splitlines(keepends=True)
    definitions = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [item.lineno for item in getattr(node, 'decorator_list', [])])
        definitions.append(''.join(lines[start-1:node.end_lineno]).rstrip())
    output = 'from __future__ import annotations\n' + imports + '\n\n'.join(definitions) + '\n'
    generated = selected_nodes(output, names)
    for name in names:
        assert ast.dump(nodes[name]) == ast.dump(generated[name]), name
    destination.write_text(output)
    return {'symbols': list(names), 'source_sha256': digest(source_path),
            'exported_sha256': digest(destination), 'unchanged_symbol_ast': True}


def compare_npz(generated, reference):
    rows = {}
    with np.load(generated, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        assert set(a.files) == set(b.files), generated
        for name in a.files:
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f'Encoding contract changed: {generated}:{name}')
            np.testing.assert_array_equal(x, y, err_msg=f'{generated}:{name}')
            if x.tobytes() != y.tobytes():
                raise ValueError(f'Encoding bytes changed: {generated}:{name}')
            rows[name] = {'shape': list(x.shape), 'dtype': str(x.dtype), 'elements': int(x.size),
                          'exact_values_missingness_and_bytes': True}
    return {'arrays': rows, 'generated_sha256': digest(generated), 'reference_sha256': digest(reference)}


def source_inventory():
    sources, links, proofs = {}, {}, []
    for run, groups in (
        ('raw_foundation_20260909_attempt2', [('crop_slots', 'Data/processed/crop_yield_growing_season'),
                                            ('era5_split', 'Data/era5land/monthly_npy_lon180_0p5deg_by_var')]),
        ('raw_remote_20260909', [('lai', 'Data/processed/glass_lai_avhrr_005d'),
                               ('ndvi', 'Data/processed/pku_gimms_ndvi_v1p2'),
                               ('gpp', 'Data/processed/reclue_monthly_gpp_v1')]),
    ):
        root = BASE / run
        complete = json.loads((root / 'complete.json').read_text())
        audit = json.loads((root / 'original_source_output_audit.json').read_text())
        assert complete['passed'] and audit['passed']
        proofs.extend([root / 'complete.json', root / 'original_source_output_audit.json'])
        for path, expected in audit['sources'].items():
            assert digest(path) == expected, path
        workspace = Path(complete['workspace'])
        for group, relative in groups:
            details_path = root / f'{group}_verification.json'
            proofs.append(details_path)
            details = json.loads(details_path.read_text())['arrays_detail']
            links[relative] = workspace / relative
            for name, item in details.items():
                path = workspace / relative / name
                assert digest(path) == item['generated_sha256'], path
                sources[str(path)] = item['generated_sha256']
        if run == 'raw_remote_20260909':
            path = workspace / 'Data/processed/pku_gimms_ndvi_v1p2/manifest.json'
            sources[str(path)] = digest(path)
    return sources, links, proofs


def main():
    global EVIDENCE, ENV
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not args.run or args.attempt < 1:
        raise SystemExit('Use --run with a positive attempt identifier')
    if args.attempt > 1:
        EVIDENCE = EVIDENCE.with_name(EVIDENCE.name + f'_attempt{args.attempt}')
        ENV = ENV.with_name(ENV.name + f'_attempt{args.attempt}')
    if EVIDENCE.exists() or ENV.exists():
        raise FileExistsError('Preserve completed or failed attempts')
    sources, links, proofs = source_inventory()
    EVIDENCE.mkdir(parents=True)
    foundation.EVIDENCE = EVIDENCE
    workspace = Path(tempfile.mkdtemp(prefix='fresh-cohort-', dir=ROOT.parent / 'AgroClimate_reproduction_runs'))
    (workspace / 'scripts').mkdir()
    (workspace / 'original_scripts').mkdir()
    exports = {}
    for name, (imports, names) in EXPORTS.items():
        source = ROOT / 'scripts' / name
        shutil.copy2(source, workspace / 'original_scripts' / name)
        exports[name] = export_definitions(source, workspace / 'scripts' / name, imports, names)
    runner = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_cohort_guarded.py'
    shutil.copy2(runner, workspace / runner.name)
    for relative, source in links.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    references = {}
    for crop in CROPS:
        for part in ('multimodal_main', 'biid_world_model'):
            root = ROOT / 'benchmark/cache' / part / crop
            for path in root.glob('*'):
                if path.suffix in ('.npy', '.json', '.csv'):
                    references[str(path.relative_to(ROOT))] = digest(path)
        for origin in ORIGINS:
            root = ROOT / 'benchmark/cache/stable_remote_v1' / crop / f'origin_{origin}__w0__v3'
            for name in ('train.npz', 'validation.npz', 'test.npz', 'manifest.json'):
                path = root / name
                references[str(path.relative_to(ROOT))] = digest(path)
        for name in ('source_indices.npy', 'crop_coverage.npy'):
            path = ROOT / 'benchmark/cache/inseason_13year_v1' / crop / name
            references[str(path.relative_to(ROOT))] = digest(path)
    for year in range(1982, 2017):
        path = ROOT / 'visualize/paper_experiments/reclue_monthly_gpp_alignment_v1' / f'{year}.json'
        references[str(path.relative_to(ROOT))] = digest(path)
    registration = {'workspace': str(workspace), 'original_project': str(ROOT), 'attempt': args.attempt,
        'input_files': sources, 'helper_exports': exports, 'runner_sha256': digest(runner),
        'driver_sha256': digest(Path(__file__)), 'packages': ['numpy==2.0.2'],
        'reference_inputs_used_for_generation': False, 'models_fitted': 0,
        'coverage_adapter': 'fresh MIRCA area using original pure area functions; no saved predictions',
        'remaining': ['physical_and_289_metadata', 'model_specific_encodings', 'full_new_input_evaluation', 'regional_inputs']}
    write_json(workspace / 'registration.json', registration)
    write_json(EVIDENCE / 'registration.json', registration)
    write_json(EVIDENCE / 'reference_files.json', references)
    write_json(EVIDENCE / 'provider_proofs.json', {str(p): digest(p) for p in proofs})
    print(f'[REGISTERED] {workspace}; {len(sources)} new source files', flush=True)
    foundation.run_logged([sys.executable, '-m', 'venv', ENV], 'environment_create', workspace)
    python = ENV / 'bin/python'
    foundation.run_logged([python, '-m', 'pip', 'install', '--disable-pip-version-check', 'numpy==2.0.2'],
                          'environment_install', workspace)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (EVIDENCE / 'pip_freeze.txt').write_text(freeze)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    foundation.run_logged([python, workspace / runner.name], 'generate', workspace, env)
    generated = json.loads((workspace / 'generation.json').read_text())
    assert generated['generated'] and not generated['old_predictions_loaded']
    array_checks, encoded_checks, meta_checks, coverage_checks = {}, {}, {}, {}
    for crop in CROPS:
        for part in ('multimodal_main', 'biid_world_model'):
            relative = Path('benchmark/cache') / part / crop
            produced = {p.name for p in (workspace / relative).glob('*.npy')}
            expected = {p.name for p in (ROOT / relative).glob('*.npy')}
            assert produced == expected and len(produced) == (12 if part == 'multimodal_main' else 10)
            for name in sorted(produced):
                key = str(relative / name)
                array_checks[key] = compare_array(workspace / key, ROOT / key)
                assert array_checks[key]['generated_sha256'] == array_checks[key]['reference_sha256']
            new = json.loads((workspace / relative / 'metadata.json').read_text())
            old = json.loads((ROOT / relative / 'metadata.json').read_text())
            if 'source_cache' in old:
                assert new.pop('source_cache') == str(workspace / 'benchmark/cache/multimodal_main' / crop)
                old.pop('source_cache')
            assert new == old, relative
            meta_checks[str(relative)] = True
        norm = Path('benchmark/cache/biid_world_model') / crop / 'normalization.json'
        assert json.loads((workspace / norm).read_text()) == json.loads((ROOT / norm).read_text())
        for name in ('source_indices.npy', 'crop_coverage.npy'):
            new = workspace / 'benchmark/cache/fresh_coverage' / crop / name
            old = ROOT / 'benchmark/cache/inseason_13year_v1' / crop / name
            coverage_checks[f'{crop}/{name}'] = compare_array(new, old)
        for origin in ORIGINS:
            relative = Path('benchmark/cache/stable_remote_v1') / crop / f'origin_{origin}__w0__v3'
            for split in ('train', 'validation', 'test'):
                key = str(relative / f'{split}.npz')
                encoded_checks[key] = compare_npz(workspace / key, ROOT / key)
            new, old = [json.loads((root / relative / 'manifest.json').read_text()) for root in (workspace, ROOT)]
            for key in ('normalization', 'ndvi_normalization', 'source_indices_hash'):
                assert new[key] == old[key], (relative, key)
            for split in ('train', 'validation', 'test'):
                for key in ('n', 'years', 'index_sha256'):
                    assert new['splits'][split][key] == old['splits'][split][key]
    support = json.loads((workspace / 'fresh_gpp_support.json').read_text())
    for year in range(1982, 2017):
        old = json.loads((ROOT / 'visualize/paper_experiments/reclue_monthly_gpp_alignment_v1' / f'{year}.json').read_text())
        for crop in CROPS:
            assert support[crop][str(year)] == old['crops'][crop]['original_benchmark_support'], (crop, year)
    write_json(EVIDENCE / 'arrays_verification.json', array_checks)
    write_json(EVIDENCE / 'encodings_verification.json', encoded_checks)
    write_json(EVIDENCE / 'coverage_verification.json', coverage_checks)
    write_json(EVIDENCE / 'fresh_gpp_support.json', support)
    for path, expected in sources.items():
        assert digest(path) == expected, path
    for relative, expected in references.items():
        assert digest(ROOT / relative) == expected, relative
    for name, value in exports.items():
        assert digest(ROOT / 'scripts' / name) == value['source_sha256']
        assert digest(workspace / 'original_scripts' / name) == value['source_sha256']
    result = {'passed': True, 'workspace': str(workspace), 'environment': str(ENV),
        'base_world_arrays': len(array_checks), 'rolling_npz_files': len(encoded_checks),
        'rolling_arrays': sum(len(x['arrays']) for x in encoded_checks.values()),
        'base_world_elements': sum(x['elements'] for x in array_checks.values()),
        'rolling_elements': sum(x['elements'] for x in encoded_checks.values() for x in x['arrays'].values()),
        'crop_rows': generated['crops'], 'coverage_arrays': len(coverage_checks), 'gpp_support_crop_years': 140,
        'all_original_inputs_and_scripts_unchanged': True, 'all_generated_values_equal': True,
        'fresh_base_world_and_rolling_inputs_complete': True, 'complete_model_feature_interface': False,
        'full_raw_to_evaluation_reproduced': False, 'independent_recipe_confirmation': False,
        'models_fitted': 0, 'new_scientific_comparisons': 0, 'scientific_results_replaced': False,
        'finished': time.time()}
    write_json(EVIDENCE / 'complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
