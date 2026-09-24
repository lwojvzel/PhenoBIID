"""Recompute metrics and replay saved rolling or world-readout checkpoints."""
import argparse
import gc
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import torch
import xgboost as xgb
from catboost import CatBoostRegressor

from multimodal_baseline import regression_metrics
from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json
from stable_remote_data import RESULT, load, Features


def read_model(engine, path):
    if engine == 'catboost':
        model = CatBoostRegressor(); model.load_model(path)
    elif engine == 'xgb_smooth':
        model = xgb.XGBRegressor(); model.load_model(path); model.set_params(device='cpu')
    else:
        model = joblib.load(path)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world', action='store_true')
    p.add_argument('--minimal', action='store_true')
    p.add_argument('--weighted', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    warnings.filterwarnings('ignore', message='X does not have valid feature names, but LGBMRegressor was fitted with feature names')
    root = ROOT / 'benchmark/results/crop_weighted_diagnostic_v1' if args.weighted else ROOT / 'benchmark/results/minimal_state_readout_v1' if args.minimal else ROOT / 'benchmark/results/stable_world_readout_v1' if args.world else RESULT
    directory = ROOT / 'visualize/paper_experiments' / root.name
    directory.mkdir(parents=True, exist_ok=True)
    configs = sorted((root / 'pipelines').glob('*/*/*/*/*/config.json' if args.minimal else '*/*/*/*/config.json'))
    records = []
    for config_path in configs:
        config = json.loads(config_path.read_text())
        if args.weighted:
            from crop_weighted_diagnostic import load as load_weighted, Features as WeightedFeatures
            arrays, meta = load_weighted(config['crop'], config['origin'])
            assert meta == config['input_manifest']
            factory = WeightedFeatures(arrays)
        elif args.minimal:
            from minimal_state_readout import MinimalFeatures, load_data
            arrays, meta = load_data(config['track'], config['crop'], config['origin'], config['seed'])
            assert meta == config['input_manifest']
            factory = MinimalFeatures(arrays, config['track'])
        elif args.world:
            from stable_world_readout import WorldFeatures
            from run_yield_head_redesign import load_forward
            arrays, meta = load_forward(config['crop'], config['seed'], config['origin'])
            assert meta == config['manifest']
            factory = WorldFeatures(arrays)
        else:
            arrays, meta = load(config['crop'], config['origin'], config['window'])
            assert meta == config['input_manifest']
            factory = Features(arrays)
        for name, digest in config['code_hashes'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise AssertionError(f'Source changed: {name}')
        for path in sorted(config_path.parent.glob('*/metrics.json')):
            m = json.loads(path.read_text())
            if sha256(Path(m['weight'])) != m['weight_sha256']:
                raise AssertionError(f'Weight changed: {path}')
            features, names = factory.build(m['condition'])
            assert names == m['features']
            model = read_model(m['engine'], m['weight'])
            norm = meta['normalization']; differences = {}
            for split in ('validation', 'test'):
                a = arrays[split]
                with np.load(path.parent / f'{split}_predictions.npz') as saved:
                    for key in ('target', 'baseline', 'source_indices', 'row', 'col', 'year'):
                        np.testing.assert_array_equal(saved[key], a[key])
                    actual = regression_metrics(saved['target'], saved['prediction'])
                    for key in ('rmse', 'nrmse', 'mae', 'r2'):
                        np.testing.assert_allclose(actual[key], m['scores'][split][key], atol=1e-7, rtol=0)
                    replay = a['baseline'][:256].astype(float) + model.predict(features[split][:256]).astype(float) * norm['residual_std'] + norm['residual_mean']
                    error = float(np.max(np.abs(replay - saved['prediction'][:256])))
                    if error > 1e-7:
                        raise AssertionError(f'Checkpoint replay differs: {path} {split} {error}')
                    differences[split] = error
            records.append(dict(directory=str(path.parent), maximum_replay_error=max(differences.values())))
            del model, features; gc.collect()
        print(f'[AUDIT] {config_path.parent} total={len(records)}', flush=True)
        del arrays, factory; gc.collect()
    report = dict(complete_models=len(records), evaluated_splits_recomputed=2 * len(records),
                  replay_rows_per_split=256, maximum_replay_error=max(r['maximum_replay_error'] for r in records),
                  source_and_weight_hashes_verified=True, metadata_and_sample_alignment_verified=True,
                  models=records)
    atomic_json(directory / 'checkpoint_replay_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'models'}, indent=2))


if __name__ == '__main__':
    main()
