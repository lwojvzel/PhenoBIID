"""Rebuild all global main-model encodings from audited fresh input arrays."""
import argparse
import ast
import hashlib
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
from run_raw_foundation_reproduction import ROOT, digest, compare_array, write_json
from run_fresh_cohort_reproduction import export_definitions, compare_npz


BASE = ROOT / 'benchmark/results/inseason_reproducibility_v1'
PROVIDER = BASE / 'fresh_features_20260909'
EVIDENCE = BASE / 'fresh_interfaces_20260909'
ENV = ROOT / 'Paper/release/inseason_history_cuda_env'
RECIPES = dict(maize='gpp', rice='ndvi', soybean='ndvi', wheat='ndvi_gpp')
CUTOFFS = (2001, 2005, 2009)
LABELS = ('target', 'row', 'col', 'year', 'source_indices')
STATE_KEYS = ('source_indices', 'year', 'row', 'col', 'target', 'weather', 'context', 'relative_valid',
              'previous_ndvi', 'previous_gpp', 'previous_ndvi_quality', 'previous_gpp_quality',
              'observed_lai', 'observed_ndvi', 'observed_gpp')
VERSIONS = {'numpy': '2.0.2', 'torch': '2.5.1+cu124', 'scikit-learn': '1.6.1', 'scipy': '1.13.1',
            'joblib': '1.5.3', 'threadpoolctl': '3.6.0', 'rtdl-num-embeddings': '0.0.12'}
EXPORTS = {
    'review_revision_data.py': ('from pathlib import Path\nimport hashlib\n', ('ROOT', 'sha256')),
    'prepare_reclue_monthly_gpp.py': ('from review_revision_data import ROOT\n', ('OUT', 'AUDIT')),
    'observed_remote_benchmark.py': ('import numpy as np\n', ('trajectory_features',)),
    'observed_remote_anomaly.py': ('import numpy as np\n', ('seasonal_anomalies',)),
    'inseason_ndvi_reuse.py': ('import numpy as np\nfrom observed_remote_anomaly import seasonal_anomalies\n'
                             'from observed_remote_benchmark import trajectory_features\n',
                             ('TrajectoryEncoder', 'historical_support')),
    'forecast_bridge_data.py': ('import numpy as np\n', ('PRODUCTS', 'identity_hash', 'fit_stats')),
    'forecast_bridge_state.py': ('import numpy as np\n', ('INPUTS', 'batch_arrays')),
    'inseason_signal_matching.py': ('import numpy as np\nfrom prepare_reclue_monthly_gpp import OUT as GPP\n'
                                   'from review_revision_data import sha256\n',
                                   ('RECIPES', 'physical_gpp', 'remap_product', 'feature_matrix')),
    'ndvi_tail_replacement.py': ('import numpy as np\n', ('tail_mask',)),
    'build_inseason_cpu_replay.py': ('import numpy as np\n', ('selected_climatology',)),
    'linear_state_yield.py': ('import numpy as np\n', ('CONDITIONS', 'yield_features')),
    'numeric_embedding_readout.py': ('import warnings\nimport torch\nfrom rtdl_num_embeddings import compute_bins\n', ('training_bins',)),
    'run_crop_signal_screen.py': ('import numpy as np\n', ('year_weights',)),
}


def original_definitions(path, names, **scope):
    from run_raw_remote_reproduction import selected_nodes
    nodes = selected_nodes(Path(path).read_text(), names)
    module = ast.Module(body=list(nodes.values()), type_ignores=[])
    namespace = dict(np=np, **scope)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def reference_files():
    result = {}
    def add(path):
        result[str(path)] = digest(path)
    for crop in RECIPES:
        for cutoff in CUTOFFS:
            name = f'{crop}/cutoff_{cutoff}'
            for stage in ('history', 'state', 'terminal', 'reconstructed_chain'):
                directory = ROOT / f'Paper/release/inseason_{stage}_retrain_v1/cases' / name
                if stage == 'reconstructed_chain':
                    directory = ROOT / 'Paper/release/inseason_reconstructed_chain_v1/cases' / name
                for path in directory.rglob('*'):
                    relative = path.relative_to(directory)
                    if not path.is_file() or any(p.startswith('seed_') for p in relative.parts):
                        continue
                    if path.suffix in ('.npy', '.npz', '.json', '.pt'):
                        add(path)
            add(ROOT / f'Paper/release/inseason_history_retrain_v1/cases/{name}/mlp/seed_42/recipe.json')
            if crop == 'soybean':
                add(ROOT / f'benchmark/cache/neural_process_readout_v1/soybean/origin_{cutoff+3}/history_scaler.joblib')
    add(ROOT / 'Paper/iclr2027/reproducibility/reconstructed_chain/run.py')
    for name in ('neural_process_readout.py', 'run_inseason_pipeline_seed_history.py',
                 'export_inseason_history_retrain.py', 'export_inseason_state_retrain.py',
                 'export_inseason_terminal_retrain.py', 'export_inseason_reconstructed_chain.py'):
        add(ROOT / 'scripts' / name)
    return result


def compare_interfaces(workspace, evidence):
    import joblib
    import torch
    arrays, labels, metadata, bins_result = {}, {}, {}, {}
    state_math = original_definitions(ROOT / 'scripts/forecast_bridge_state.py', ('batch_arrays',))['batch_arrays']
    feature_fn = original_definitions(ROOT / 'scripts/observed_remote_benchmark.py', ('trajectory_features',))['trajectory_features']
    weight_fn = original_definitions(ROOT / 'scripts/run_crop_signal_screen.py', ('year_weights',))['year_weights']
    encode_scope = original_definitions(ROOT / 'Paper/iclr2027/reproducibility/reconstructed_chain/run.py', ('encode',))
    # The original encode imports the same trajectory function at call time.
    import types
    temporary = types.ModuleType('observed_remote_benchmark')
    temporary.trajectory_features = feature_fn
    previous_module = sys.modules.get('observed_remote_benchmark')
    sys.modules['observed_remote_benchmark'] = temporary
    feature_matrix = original_definitions(ROOT / 'scripts/inseason_signal_matching.py', ('RECIPES', 'feature_matrix'))['feature_matrix']
    cases = {}

    def compare(path, reference=None, expected=None):
        key = str(path.relative_to(workspace))
        if reference is not None:
            arrays[key] = compare_array(path, reference)
            assert arrays[key]['generated_sha256'] == arrays[key]['reference_sha256'], key
        else:
            actual = np.load(path, allow_pickle=False)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
            np.testing.assert_array_equal(actual, expected, err_msg=key)
            assert actual.tobytes() == expected.tobytes(), key
            arrays[key] = dict(elements=int(actual.size), shape=list(actual.shape), dtype=str(actual.dtype),
                               generated_sha256=digest(path), reference_calculated_after_generation=True,
                               expected_array_sha256=hashlib.sha256(expected.tobytes()).hexdigest())

    try:
        for crop, recipe in RECIPES.items():
            for cutoff in CUTOFFS:
                name = f'{crop}/cutoff_{cutoff}'
                h = workspace / 'history/cases' / name
                hr = ROOT / 'Paper/release/inseason_history_retrain_v1/cases' / name
                for component in ('mlp', 'tabm') if crop == 'soybean' else ('mlp',):
                    for split in ('train', 'validation', 'test'):
                        suffix = Path('') if split == 'train' else Path('reference')
                        local, ref = h / component / suffix, hr / component / suffix
                        compare(local / f'{split}_x.npy', ref / f'{split}_x.npy')
                        key = str((local / f'{split}_labels.npz').relative_to(workspace))
                        labels[key] = compare_npz(local / f'{split}_labels.npz', ref / f'{split}_labels.npz')
                norm = json.loads((h / 'normalization.json').read_text())
                assert norm == json.loads((hr / 'mlp/seed_42/recipe.json').read_text())['normalization']
                if crop == 'soybean':
                    scaler = joblib.load(ROOT / f'benchmark/cache/neural_process_readout_v1/soybean/origin_{cutoff+3}/history_scaler.joblib')
                    new_scaler = json.loads((h / 'tabm/scaler.json').read_text())
                    assert new_scaler == {k: getattr(scaler, k).tolist() for k in ('mean_', 'scale_', 'var_')}
                    a, b = [torch.load(p, map_location='cpu', weights_only=True) for p in (h / 'tabm/bins.pt', hr / 'tabm/bins.pt')]
                    assert len(a) == len(b) == 20
                    for x, y in zip(a, b):
                        assert x.dtype == y.dtype and x.shape == y.shape
                        torch.testing.assert_close(x, y, atol=0, rtol=0)
                    x = np.load(h / 'tabm/train_x.npy', mmap_mode='r')
                    constant = np.flatnonzero((x == x[0]).all(0)).tolist()
                    assert json.loads((h / 'tabm/bin_columns.json').read_text()) == dict(constant_columns=constant, training_only=True)
                    bins_result[name] = dict(arrays=20, elements=sum(x.numel() for x in a),
                                            constant_columns=constant, generated_sha256=digest(h / 'tabm/bins.pt'))
                state = workspace / 'state/cases' / name
                sr = ROOT / 'Paper/release/inseason_state_retrain_v1/cases' / name
                original = {k: np.load(sr / f'data/{k}.npy', mmap_mode='r') for k in STATE_KEYS}
                for key in STATE_KEYS:
                    compare(state / f'data/{key}.npy', sr / f'data/{key}.npy')
                stats = json.loads((state / 'statistics.json').read_text())
                assert stats == json.loads((sr / 'case.json').read_text())['statistics']
                for product in recipe.split('_'):
                    expected = state_math(original, np.arange(len(original['year'])), stats, product)
                    expected['target'] = (original[f'observed_{product}']-stats[product]['mean'])/stats[product]['std']
                    for key, value in expected.items():
                        compare(state / f'encoded/{product}_{key}.npy', expected=value)
                terminal = workspace / 'terminal/cases' / name
                tr = ROOT / 'Paper/release/inseason_terminal_retrain_v1/cases' / name
                for key in ('train.npy', 'reference/probe.npy'):
                    compare(terminal / key, tr / key)
                labels[str((terminal / 'train_labels.npz').relative_to(workspace))] = compare_npz(terminal / 'train_labels.npz', tr / 'train_labels.npz')
                with np.load(tr / 'train_labels.npz') as labels_old:
                    compare(terminal / 'sample_weights.npy', expected=weight_fn(labels_old['year']))
                chain = workspace / 'evaluation/cases' / name
                cr = ROOT / 'Paper/release/inseason_reconstructed_chain_v1/cases' / name
                a, b = [json.loads((p / 'case.json').read_text()) for p in (chain, cr)]
                assert a == b, name
                for split, n in a['splits'].items():
                    folder, ref = chain / split, cr / split
                    expected_files = list(ref.glob('*.npy'))
                    for file in expected_files:
                        compare(folder / file.name, file)
                    with np.load(hr / f'mlp/reference/{split}_labels.npz') as old, np.load(folder / 'labels.npz') as new:
                        assert set(new.files) == set(LABELS) and len(new['year']) == n
                        for key in LABELS:
                            np.testing.assert_array_equal(new[key], old[key])
                            assert new[key].dtype == old[key].dtype and new[key].tobytes() == old[key].tobytes()
                    labels[str((folder / 'labels.npz').relative_to(workspace))] = dict(arrays=list(LABELS), rows=n, exact=True)
                    active = np.load(ref / 'active.npy')
                    encoded = {}
                    for product in a['products']:
                        encoded[product] = encode_scope['encode'](np.load(ref / f'observed_{product}.npy'), active,
                            a['encoder_scales'][product], np.load(ref / f'climatology_{product}.npy'))
                        compare(folder / f'observed_encoding_{product}.npy', expected=encoded[product])
                    full = feature_matrix(np.load(ref / 'common.npy'), encoded, np.zeros_like(active),
                                          np.load(ref / 'support.npy'), recipe)
                    compare(folder / 'observed_features.npy', expected=full)
                metadata[name] = dict(history_normalization=True, state_statistics=True, evaluation_case=True,
                                      training_rows=len(original['year']), evaluation_rows=sum(a['splits'].values()))
                cases[name] = a
                print(f'[ALL INTERFACES VERIFIED] {name}', flush=True)
    finally:
        if previous_module is None:
            sys.modules.pop('observed_remote_benchmark', None)
        else:
            sys.modules['observed_remote_benchmark'] = previous_module
    actual = {str(p.relative_to(workspace)) for part in ('history', 'state', 'terminal', 'evaluation')
              for p in (workspace / part).rglob('*.npy')}
    assert actual == set(arrays) and len(cases) == 12 and len(bins_result) == 3
    assert len(arrays) == 686 and len(labels) == 73, (len(arrays), len(labels))
    for filename, value in [('arrays_verification.json', arrays), ('labels_verification.json', labels),
                            ('metadata_verification.json', metadata), ('bins_verification.json', bins_result)]:
        write_json(evidence / filename, value)
    return arrays, labels, metadata, bins_result


def main():
    global EVIDENCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not args.run or args.attempt < 1:
        raise SystemExit('Use --run and a positive attempt identifier')
    if args.attempt > 1:
        EVIDENCE = EVIDENCE.with_name(EVIDENCE.name + f'_attempt{args.attempt}')
    if EVIDENCE.exists():
        raise FileExistsError('Preserve existing interface attempts')
    complete = json.loads((PROVIDER / 'complete.json').read_text())
    audit = json.loads((PROVIDER / 'original_source_output_audit.json').read_text())
    assert complete['passed'] and audit['passed'] and complete['fresh_physical_metadata_complete']
    for path, expected in audit['sources'].items():
        assert digest(path) == expected, path
    previous = Path(complete['workspace'])
    registry = json.loads((PROVIDER / 'registration.json').read_text())
    inputs = dict(registry['input_files'])
    for relative in ('benchmark/cache/crop_signal_screen_v1', 'benchmark/cache/forecast_state_bridge_v1/raw',
                     'benchmark/cache/fresh_complete_inputs'):
        for p in (previous / relative).rglob('*'):
            if p.is_file() and p.suffix in ('.npy', '.npz', '.json'):
                inputs[str(p)] = digest(p)
    for p, checksum in inputs.items():
        assert digest(p) == checksum, p
    references = reference_files()
    python = ENV / 'bin/python'
    command = 'import importlib.metadata,json;print(json.dumps({p:importlib.metadata.version(p) for p in '+repr(tuple(VERSIONS))+'}))'
    assert json.loads(subprocess.check_output([python, '-c', command], text=True)) == VERSIONS
    assert 'include-system-site-packages = false' in (ENV / 'pyvenv.cfg').read_text()
    EVIDENCE.mkdir(parents=True)
    foundation.EVIDENCE = EVIDENCE
    workspace = Path(tempfile.mkdtemp(prefix='fresh-interfaces-', dir=ROOT.parent / 'AgroClimate_reproduction_runs'))
    for name in ('scripts', 'original_scripts'):
        (workspace / name).mkdir()
    exports = {}
    for name, (imports, symbols) in EXPORTS.items():
        source = ROOT / 'scripts' / name
        shutil.copy2(source, workspace / 'original_scripts' / name)
        exports[name] = export_definitions(source, workspace / 'scripts' / name, imports, symbols)
    (workspace / 'Data').symlink_to(previous / 'Data', target_is_directory=True)
    (workspace / 'benchmark/cache').mkdir(parents=True)
    for part in ('stable_remote_v1', 'crop_signal_screen_v1', 'forecast_state_bridge_v1', 'fresh_complete_inputs'):
        (workspace / 'benchmark/cache' / part).symlink_to(previous / 'benchmark/cache' / part, target_is_directory=True)
    runner = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_interfaces_guarded.py'
    guard = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_cohort_guarded.py'
    shutil.copy2(runner, workspace / runner.name)
    shutil.copy2(guard, workspace / 'cohort_guard.py')
    reg = dict(original_project=str(ROOT), workspace=str(workspace), environment=str(ENV),
               environment_reused='Audited isolated history reconstruction environment; CPU preprocessing only',
               input_files=inputs, exports=exports, versions=VERSIONS, driver_sha256=digest(Path(__file__)),
               runner_sha256=digest(runner), guard_sha256=digest(guard), reference_inputs_for_generation=False,
               models_fitted=0, scientific_results_replaced=False)
    for folder in (workspace, EVIDENCE):
        write_json(folder / 'registration.json', reg)
    write_json(EVIDENCE / 'reference_files.json', references)
    write_json(EVIDENCE / 'provider_proofs.json', {str(PROVIDER / n): digest(PROVIDER / n)
        for n in ('complete.json', 'original_source_output_audit.json', 'registration.json')})
    (EVIDENCE / 'pip_freeze.txt').write_text(subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)
    print(f'[INTERFACES REGISTERED] {workspace}; {len(inputs)} fresh inputs', flush=True)
    foundation.run_logged([python, workspace / runner.name], 'generate', workspace, env)
    arrays, labels, metadata, bins = compare_interfaces(workspace, EVIDENCE)
    for path, expected in {**inputs, **references}.items():
        assert digest(path) == expected, path
    generated = json.loads((workspace / 'generation.json').read_text())
    assert len(generated['cases']) == 12 and not generated['old_predictions_loaded'] and not generated['gpu_used']
    for name, item in exports.items():
        assert digest(ROOT / 'scripts' / name) == digest(workspace / 'original_scripts' / name) == item['source_sha256']
        assert digest(workspace / 'scripts' / name) == item['exported_sha256']
    result = dict(passed=True, workspace=str(workspace), environment=str(ENV), arrays=len(arrays),
                  elements=sum(v['elements'] for v in arrays.values()), label_files=len(labels), cases=len(metadata),
                  bin_cases=len(bins), bin_arrays=sum(v['arrays'] for v in bins.values()),
                  training_rows=sum(v['training_rows'] for v in metadata.values()),
                  evaluation_rows=sum(v['evaluation_rows'] for v in metadata.values()),
                  global_main_model_static_interfaces_complete=True, seed_specific_residual_targets_reproduced=False,
                  baseline_interfaces_reproduced=False,
                  full_raw_to_evaluation_reproduced=False, independent_recipe_confirmation=False,
                  models_fitted=0, new_scientific_comparisons=0, scientific_results_replaced=False,
                  original_inputs_and_scripts_unchanged=True, finished=time.time())
    write_json(EVIDENCE / 'complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
