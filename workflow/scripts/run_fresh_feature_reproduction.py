"""Restore original physical trajectories, anomalies and complete metadata."""
import argparse
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
from run_fresh_cohort_reproduction import CROPS, ORIGINS, export_definitions, compare_npz


BASE = ROOT / 'benchmark/results/inseason_reproducibility_v1'
COHORT = BASE / 'fresh_cohort_20260909'
REMOTE = BASE / 'raw_remote_20260909'
EVIDENCE = BASE / 'fresh_features_20260909'
ENV = ROOT / 'Paper/release/inseason_fresh_features_env'
EXPORTS = {
    'observed_remote_benchmark.py': ('import numpy as np\n', ('INPUTS', 'trajectory_features')),
    'observed_remote_anomaly.py': ('import numpy as np\n', ('seasonal_anomalies',)),
    'prepare_reclue_monthly_gpp.py': ('from review_revision_data import ROOT\n', ('OUT', 'AUDIT')),
    'crop_signal_screen_data.py': (
        'import calendar, fcntl, json\nfrom pathlib import Path\nimport numpy as np\n'
        'from observed_remote_anomaly import seasonal_anomalies\n'
        'from observed_remote_benchmark import trajectory_features\n'
        'from prepare_pku_ndvi import OUT as NDVI\nfrom prepare_reclue_monthly_gpp import OUT as GPP, AUDIT as GPP_AUDIT\n'
        'from review_revision_data import ROOT, CROPS, sha256\nfrom run_review_revision_parallel import atomic_json\n'
        'from stable_remote_data import cache_root as old_cache_root\n',
        ('CACHE', 'ORIGINS', 'SPLITS', 'PRODUCTS', 'CONDITIONS', 'COMPONENTS', 'LABELS', 'INPUT_KEYS', 'CODE',
         'cache_root', 'active_slots', 'calendar_fields', 'normalize_values', 'construct_components',
         'source_inputs', 'prepare')),
    'forecast_bridge_data.py': (
        'import calendar, fcntl, json\nfrom pathlib import Path\nimport numpy as np\n'
        'from crop_signal_screen_data import source_inputs, active_slots, ORIGINS\n'
        'from dual_remote_data import extract\nfrom review_revision_data import ROOT, CROPS, sha256\n'
        'from run_review_revision_parallel import atomic_json\n'
        'from run_history_multimodal_baselines import build_causal_history_features\n'
        'from observed_remote_benchmark import trajectory_features\n',
        ('CACHE', 'PRODUCTS', 'MAX_YEAR', 'KEYS', 'root', 'identity_hash', 'physical_history', 'previous_gpp',
         'prepare', 'load', 'fit_stats', 'history_features', 'climatology', 'remote_features')),
    'inseason_signal_matching.py': (
        'import numpy as np\nfrom crop_signal_screen_data import GPP\nfrom review_revision_data import sha256\n',
        ('physical_gpp',)),
    'ndvi_tail_replacement.py': ('import numpy as np\n', ('tail_mask',)),
    'inseason_ndvi_reuse.py': (
        'import numpy as np\nfrom observed_remote_anomaly import seasonal_anomalies\n'
        'from observed_remote_benchmark import trajectory_features\n',
        ('TrajectoryEncoder', 'historical_support')),
    'inseason_direct_data.py': (
        'import json\nimport numpy as np\nfrom dual_remote_data import extract, OUT as NDVI_ROOT\n'
        'from forecast_bridge_data import ROOT, PRODUCTS, load\n'
        'from inseason_signal_matching import physical_gpp\nfrom review_revision_data import sha256\n'
        'from run_history_multimodal_baselines import build_causal_history_features\n'
        'from stable_remote_data import cache_root\n', ('RECIPES', 'physical_inputs', 'split_rows')),
    'inseason_extension_data.py': (
        'import json\nimport numpy as np\nfrom forecast_bridge_data import ROOT, load\n'
        'from crop_signal_screen_data import cache_root, calendar_fields, NDVI, GPP\n'
        'from stable_remote_data import cache_root as stable_root\nfrom dual_remote_data import extract\n'
        'from inseason_ndvi_reuse import TrajectoryEncoder, historical_support\n'
        'from review_revision_data import sha256\n', ('prepare',)),
    'inseason_complete_inputs.py': (
        'import json\nimport numpy as np\nfrom crop_signal_screen_data import cache_root, active_slots\n'
        'from forecast_bridge_data import ROOT, fit_stats, history_features, climatology, remote_features\n'
        'from inseason_direct_data import physical_inputs, split_rows, RECIPES\n'
        'from inseason_extension_data import prepare as extension_prepare\n'
        'from inseason_ndvi_reuse import historical_support\nfrom review_revision_data import sha256\n',
        ('complete_inputs',)),
}


def verify_source_proofs():
    paths = [COHORT / 'complete.json', COHORT / 'original_source_output_audit.json',
             REMOTE / 'complete.json', REMOTE / 'original_source_output_audit.json']
    for path in paths:
        record = json.loads(path.read_text())
        assert record['passed'], path
        for name, expected in record.get('sources', {}).items():
            assert digest(name) == expected, name
    previous = json.loads(paths[0].read_text())
    registry = json.loads((COHORT / 'registration.json').read_text())
    sources = dict(registry['input_files'])
    workspace = Path(previous['workspace'])
    for part in ('multimodal_main', 'biid_world_model', 'stable_remote_v1'):
        for path in (workspace / 'benchmark/cache' / part).rglob('*'):
            if path.is_file() and path.suffix in ('.npy', '.npz', '.json', '.csv'):
                sources[str(path)] = digest(path)
    for path, expected in sources.items():
        assert digest(path) == expected, path
    paths.extend([COHORT / 'registration.json', COHORT / 'fresh_gpp_support.json',
                  REMOTE / 'gpp_verification.json'])
    return workspace, registry, sources, {str(p): digest(p) for p in paths}


def new_gpp_gate(workspace, remote_workspace, remote_complete, cohort_support):
    if not remote_complete.get('passed') or remote_complete['stages']['gpp']['arrays'] != 527:
        raise ValueError('New GPP reconstruction has not completed')
    years = set(map(str, range(1982, 2017)))
    if set(cohort_support) != set(CROPS) or any(set(v) != years for v in cohort_support.values()):
        raise ValueError('Fresh sample support matrix is incomplete')
    if any(v['target_values_accessed'] for rows in cohort_support.values() for v in rows.values()):
        raise ValueError('Product support computation read yield targets')
    relative = Path('Data/processed/reclue_monthly_gpp_v1')
    source = remote_workspace / relative
    details = json.loads((REMOTE / 'gpp_verification.json').read_text())['arrays_detail']
    folder = workspace / relative
    folder.mkdir(parents=True)
    for name in ('monthly_0p5', 'crops', 'lat.npy', 'lon.npy'):
        (folder / name).symlink_to(source / name, target_is_directory=(source / name).is_dir())
    files = {}
    for name, item in details.items():
        path = folder / name
        if digest(path) != item['generated_sha256']:
            raise ValueError('GPP values differ from the independently audited reconstruction')
        files[str(path)] = item['generated_sha256']
    manifest = folder / 'manifest.json'
    write_json(manifest, {'kind': 'Fresh reconstruction lineage, not an original provider manifest',
                         'raster_years': sorted(map(int, years)), 'arrays': 527,
                         'remote_complete_sha256': digest(REMOTE / 'complete.json'),
                         'remote_audit_sha256': digest(REMOTE / 'original_source_output_audit.json'),
                         'cohort_support_sha256': digest(COHORT / 'fresh_gpp_support.json'),
                         'numerical_processing_changed': False, 'files': files})
    gate = workspace / 'visualize/paper_experiments/reclue_monthly_gpp_alignment_v1/full_period_ready.json'
    gate.parent.mkdir(parents=True)
    write_json(gate, {'complete': True, 'all_relative_arrays_rebuilt': True, 'yield_targets_read': False,
                     'scope': 'GPP raster reconstruction and support diagnostic, not complete model input or training',
                     'old_readiness_record_copied': False, 'fresh_support_rows': 140,
                     'manifest_sha256': digest(manifest), 'files': files})
    return manifest, gate


def compare_outputs(workspace, evidence):
    arrays, labels, checks, statistics = {}, {}, {}, {}
    for crop in CROPS:
        for origin in ORIGINS:
            relative = Path('benchmark/cache/crop_signal_screen_v1') / crop / f'origin_{origin}'
            for split in ('train', 'validation'):
                key = str(relative / f'{split}_labels.npz')
                labels[key] = compare_npz(workspace / key, ROOT / key)
                for part in ('history', 'metadata', 'weather', 'lai', 'ndvi', 'gpp'):
                    key = str(relative / f'{part}_{split}.npy')
                    arrays[key] = compare_array(workspace / key, ROOT / key)
            a, b = [json.loads((root / relative / 'manifest.json').read_text())['spec'] for root in (workspace, ROOT)]
            for key in ('crop', 'origin', 'conditions', 'names'):
                assert a[key] == b[key], (relative, key)
            assert a['upstream']['gpp_training_normalization'] == b['upstream']['gpp_training_normalization']
            for key in ('normalization', 'ndvi_normalization', 'source_indices_hash'):
                assert a['upstream']['upstream'][key] == b['upstream']['upstream'][key]
            checks[str(relative)] = True
        relative = Path('benchmark/cache/forecast_state_bridge_v1/raw') / crop
        produced = {p.name for p in (workspace / relative).glob('*.npy')}
        assert produced == {p.name for p in (ROOT / relative).glob('*.npy')} and len(produced) == 20
        for name in sorted(produced):
            key = str(relative / name)
            arrays[key] = compare_array(workspace / key, ROOT / key)
        a, b = [json.loads((root / relative / 'manifest.json').read_text()) for root in (workspace, ROOT)]
        for key in ('rows', 'years', 'coverage', 'disjoint_month_row_fraction', 'split_identity_checks'):
            assert a[key] == b[key], (relative, key)
        complete = workspace / 'benchmark/cache/fresh_complete_inputs' / crop
        for name in (*sorted(produced), 'metadata.npy'):
            path = complete / name
            reference = ROOT / 'benchmark/cache/inseason_13year_v1' / crop / name
            key = str(path.relative_to(workspace))
            arrays[key] = compare_array(path, reference)
        stats = json.loads((complete / 'statistics.json').read_text())
        for cutoff in (2001, 2005, 2009):
            ref = ROOT / f'Paper/release/inseason_state_retrain_v1/cases/{crop}/cutoff_{cutoff}/case.json'
            old = json.loads(ref.read_text())['statistics']
            assert stats[str(cutoff)] == old, (crop, cutoff, 'state statistics')
            statistics[f'{crop}/{cutoff}'] = {'passed': True, 'reference': str(ref), 'sha256': digest(ref)}
        print(f'[PHYSICAL AND METADATA VERIFIED] {crop}', flush=True)
    assert all(v['generated_sha256'] == v['reference_sha256'] for v in arrays.values())
    write_json(evidence / 'arrays_verification.json', arrays)
    write_json(evidence / 'labels_verification.json', labels)
    write_json(evidence / 'statistics_verification.json', statistics)
    write_json(evidence / 'metadata_verification.json', checks)
    return arrays, labels, statistics


def main():
    global EVIDENCE, ENV
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not args.run or args.attempt < 1:
        raise SystemExit('Use --run and a positive attempt identifier')
    if args.attempt > 1:
        EVIDENCE = EVIDENCE.with_name(EVIDENCE.name + f'_attempt{args.attempt}')
        ENV = ENV.with_name(ENV.name + f'_attempt{args.attempt}')
    if EVIDENCE.exists() or ENV.exists():
        raise FileExistsError('Preserve previous attempts')
    cohort_workspace, cohort_registry, inputs, proofs = verify_source_proofs()
    remote_complete = json.loads((REMOTE / 'complete.json').read_text())
    remote_workspace = Path(remote_complete['workspace'])
    EVIDENCE.mkdir(parents=True)
    foundation.EVIDENCE = EVIDENCE
    workspace = Path(tempfile.mkdtemp(prefix='fresh-features-', dir=ROOT.parent / 'AgroClimate_reproduction_runs'))
    (workspace / 'scripts').mkdir()
    (workspace / 'original_scripts').mkdir()
    inherited, exports = {}, {}
    for name, item in cohort_registry['helper_exports'].items():
        if name not in EXPORTS:
            for directory in ('scripts', 'original_scripts'):
                shutil.copy2(cohort_workspace / directory / name, workspace / directory / name)
            inherited[name] = item
    for name, (imports, symbols) in EXPORTS.items():
        original = ROOT / 'scripts' / name
        shutil.copy2(original, workspace / 'original_scripts' / name)
        exports[name] = export_definitions(original, workspace / 'scripts' / name, imports, symbols)
    for part in ('multimodal_main', 'biid_world_model', 'stable_remote_v1'):
        path = workspace / 'benchmark/cache' / part
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(cohort_workspace / 'benchmark/cache' / part, target_is_directory=True)
    for part in ('crop_yield_growing_season', 'glass_lai_avhrr_005d', 'pku_gimms_ndvi_v1p2'):
        path = workspace / 'Data/processed' / part
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to((cohort_workspace / 'Data/processed' / part).resolve(), target_is_directory=True)
    runner = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_features_guarded.py'
    guard = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_cohort_guarded.py'
    shutil.copy2(runner, workspace / runner.name)
    shutil.copy2(guard, workspace / 'cohort_guard.py')
    support = json.loads((COHORT / 'fresh_gpp_support.json').read_text())
    manifest, gate = new_gpp_gate(workspace, remote_workspace, remote_complete, support)
    references = {}
    for crop in CROPS:
        folders = [ROOT / 'benchmark/cache/crop_signal_screen_v1' / crop / f'origin_{o}' for o in ORIGINS]
        folders.append(ROOT / 'benchmark/cache/forecast_state_bridge_v1/raw' / crop)
        for folder in folders:
            for p in folder.iterdir():
                if p.suffix in ('.npy', '.npz', '.json'):
                    references[str(p)] = digest(p)
        for name in ('source_indices', 'year', 'row', 'col', 'target', 'history', 'baseline', 'context',
                     'crop_coverage', 'relative_valid', 'source_month', 'weather', 'previous_ndvi_quality',
                     'previous_gpp_quality', 'observed_lai', 'previous_lai', 'observed_ndvi', 'previous_ndvi',
                     'observed_gpp', 'previous_gpp', 'metadata'):
            p = ROOT / 'benchmark/cache/inseason_13year_v1' / crop / f'{name}.npy'
            references[str(p)] = digest(p)
        for cutoff in (2001, 2005, 2009):
            p = ROOT / f'Paper/release/inseason_state_retrain_v1/cases/{crop}/cutoff_{cutoff}/case.json'
            references[str(p)] = digest(p)
    registry = {'original_project': str(ROOT), 'workspace': str(workspace), 'environment': str(ENV),
                'input_files': inputs, 'inherited_exports': inherited, 'new_exports': exports,
                'runner_sha256': digest(runner), 'guard_sha256': digest(guard), 'driver_sha256': digest(Path(__file__)),
                'packages': ['numpy==2.0.2'], 'models_fitted': 0, 'reference_arrays_used_for_generation': False,
                'new_gpp_gate': {str(p): digest(p) for p in (manifest, gate)},
                'remaining': ['model_specific_encodings', 'full_new_input_evaluation', 'regional_inputs']}
    write_json(workspace / 'registration.json', registry)
    write_json(EVIDENCE / 'registration.json', registry)
    write_json(EVIDENCE / 'provider_proofs.json', proofs)
    write_json(EVIDENCE / 'reference_files.json', references)
    print(f'[REGISTERED] {workspace}; {len(inputs)} fresh input files', flush=True)
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
    arrays, labels, statistics = compare_outputs(workspace, EVIDENCE)
    for p, expected in inputs.items():
        assert digest(p) == expected, p
    for p, expected in references.items():
        assert digest(p) == expected, p
    for name, item in {**inherited, **exports}.items():
        assert digest(ROOT / 'scripts' / name) == digest(workspace / 'original_scripts' / name) == item['source_sha256']
        assert digest(workspace / 'scripts' / name) == item['exported_sha256']
    generated = json.loads((workspace / 'generation.json').read_text())
    result = {'passed': True, 'workspace': str(workspace), 'environment': str(ENV),
              'arrays': len(arrays), 'elements': sum(v['elements'] for v in arrays.values()),
              'label_npz_files': len(labels), 'label_arrays': sum(len(v['arrays']) for v in labels.values()),
              'statistics_cases': len(statistics), 'physical_rows': sum(v['rows'] for v in generated['crops'].values()),
              'fresh_physical_metadata_complete': True, 'complete_model_feature_interface': False,
              'full_raw_to_evaluation_reproduced': False, 'independent_recipe_confirmation': False,
              'original_inputs_and_scripts_unchanged': True, 'models_fitted': 0,
              'new_scientific_comparisons': 0, 'scientific_results_replaced': False, 'finished': time.time()}
    write_json(EVIDENCE / 'complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
