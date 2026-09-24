"""Restore full registered model interfaces without weights or predictions."""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from rtdl_num_embeddings import compute_bins

from cohort_guard import allowed


LABELS = ('target', 'row', 'col', 'year', 'source_indices')
STATE_KEYS = ('source_indices', 'year', 'row', 'col', 'target', 'weather', 'context', 'relative_valid',
              'previous_ndvi', 'previous_gpp', 'previous_ndvi_quality', 'previous_gpp_quality',
              'observed_lai', 'observed_ndvi', 'observed_gpp')
RECIPES = dict(maize='gpp', rice='ndvi', soybean='ndvi', wheat='ndvi_gpp')
CUTOFFS = (2001, 2005, 2009)
PACKAGES = ('numpy', 'torch', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'rtdl-num-embeddings')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def same_identity(a, b):
    for key in LABELS:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def standardized_history(arrays, feature_function):
    values = {s: feature_function(a, None, {}, 'history') for s, a in arrays.items()}
    if any(v.dtype != np.float64 or v.shape[1] != 20 for v in values.values()):
        raise ValueError('Original TabM history uses twenty float64 columns before scaling')
    scaler = StandardScaler().fit(values['train'])
    return {s: scaler.transform(v).astype(np.float32) for s, v in values.items()}, scaler


def main():
    workspace = Path(__file__).resolve().parent
    reg = json.loads((workspace / 'registration.json').read_text())
    if {p: importlib.metadata.version(p) for p in PACKAGES} != reg['versions']:
        raise ValueError('Preprocessing environment changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Feature reconstruction must not use a GPU')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    inputs = {Path(p).resolve() for p in reg['input_files']}
    project = Path(reg['original_project'])
    arguments = workspace, project, inputs, Path(sys.prefix).resolve(), workspace.parent
    accessed = set()

    def check(path, writing):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(path)).resolve()
        if not allowed(path, writing, *arguments):
            raise PermissionError('Only fresh inputs and local generated interfaces are allowed')
        if path in inputs:
            accessed.add(str(path))

    def guard(event, values):
        if event == 'open':
            mode, flags = values[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            check(values[0], writing)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            check(values[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(values[0], True)
            check(values[1], True)
        elif event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
            raise PermissionError('Interface generation needs no network or child process')

    sys.addaudithook(guard)
    try:
        open(project / 'Paper/release/inseason_history_retrain_v1/cases/maize/cutoff_2001/mlp/train_x.npy', 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('Reference input access was not blocked')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace / 'scripts'))
    os.chdir(workspace)
    from forecast_bridge_data import fit_stats
    from forecast_bridge_state import batch_arrays
    from inseason_signal_matching import physical_gpp, remap_product, feature_matrix
    from inseason_ndvi_reuse import TrajectoryEncoder, historical_support
    from ndvi_tail_replacement import tail_mask
    from build_inseason_cpu_replay import selected_climatology
    from linear_state_yield import yield_features
    from numeric_embedding_readout import training_bins
    from run_crop_signal_screen import year_weights
    from review_revision_data import sha256

    cases = {}
    for crop, recipe in RECIPES.items():
        products = recipe.split('_')
        physical = workspace / 'benchmark/cache/fresh_complete_inputs' / crop
        early = workspace / 'benchmark/cache/forecast_state_bridge_v1/raw' / crop
        early_raw = {p.stem: np.load(p, mmap_mode='r') for p in early.glob('*.npy')}
        full_raw = {p.stem: np.load(p, mmap_mode='r') for p in physical.glob('*.npy')
                    if p.stem not in ('metadata', 'support_fit_2009')}
        metadata = np.load(physical / 'metadata.npy', mmap_mode='r')
        for cutoff in CUTOFFS:
            print(f'[INTERFACE START] {crop}/{cutoff}', flush=True)
            origin, name = cutoff + 3, f'{crop}/cutoff_{cutoff}'
            stable = workspace / f'benchmark/cache/stable_remote_v1/{crop}/origin_{origin}__w0__v3'
            stable_meta = json.loads((stable / 'manifest.json').read_text())
            arrays, hx = {}, {}
            for split in ('train', 'validation', 'test'):
                with np.load(stable / f'{split}.npz') as saved:
                    arrays[split] = {k: saved[k] for k in (*LABELS, 'history', 'context', 'baseline', 'target_residual')}
                a = arrays[split]
                hx[split] = np.concatenate((a['history'], a['context']), 1)
                assert hx[split].dtype == np.float32 and hx[split].shape == (len(a['year']), 20)
                folder = workspace / 'history/cases' / name / 'mlp'
                if split != 'train':
                    folder /= 'reference'
                save_array(folder / f'{split}_x.npy', hx[split])
                np.savez(folder / f'{split}_labels.npz', **{k: a[k] for k in (*LABELS, 'baseline', 'target_residual')})
            assert arrays['train']['year'].max() == cutoff
            save_json(workspace / 'history/cases' / name / 'normalization.json', stable_meta['normalization'])
            if crop == 'soybean':
                tx, scaler = standardized_history(arrays, yield_features)
                base = workspace / 'history/cases' / name / 'tabm'
                for split, value in tx.items():
                    folder = base if split == 'train' else base / 'reference'
                    save_array(folder / f'{split}_x.npy', value)
                    np.savez(folder / f'{split}_labels.npz', **{k: arrays[split][k] for k in LABELS})
                save_json(base / 'scaler.json', {k: getattr(scaler, k).tolist() for k in ('mean_', 'scale_', 'var_')})
                bins, constants = training_bins(tx['train'])
                torch.save(bins, base / 'bins.pt')
                save_json(base / 'bin_columns.json', {'constant_columns': constants, 'training_only': True})
                del tx, bins, scaler

            # Training keeps the original early GPP roundtrip; evaluation reads physical monthly GPP.
            fit = np.flatnonzero(early_raw['year'] <= cutoff)
            train_raw = {k: early_raw[k][fit] for k in STATE_KEYS}
            stats = fit_stats(train_raw, np.arange(len(fit)))
            same_identity(train_raw, arrays['train'])
            state = workspace / 'state/cases' / name
            for k, value in train_raw.items():
                save_array(state / 'data' / f'{k}.npy', value)
            save_json(state / 'statistics.json', stats)
            for product in products:
                values = batch_arrays(train_raw, np.arange(len(fit)), stats, product)
                values['target'] = (train_raw[f'observed_{product}']-stats[product]['mean'])/stats[product]['std']
                for k, value in values.items():
                    save_array(state / 'encoded' / f'{product}_{k}.npy', value)
            del train_raw, values

            screen = workspace / f'benchmark/cache/crop_signal_screen_v1/{crop}/origin_{origin}'
            screen_meta = json.loads((screen / 'manifest.json').read_text())
            scales = dict(ndvi=screen_meta['spec']['upstream']['upstream']['ndvi_normalization'],
                          gpp=screen_meta['spec']['upstream']['gpp_training_normalization'])
            terminal = workspace / 'terminal/cases' / name
            terminal_features, common = {}, {}
            for split in ('train', 'validation'):
                parts = [np.load(screen / f'{p}_{split}.npy') for p in ('history', 'metadata', 'weather', *products)]
                common[split] = np.concatenate(parts[:3], 1)
                terminal_features[split] = np.concatenate(parts, 1)
                with np.load(screen / f'{split}_labels.npz') as labels:
                    same_identity(labels, arrays[split])
            save_array(terminal / 'train.npy', terminal_features['train'])
            np.savez(terminal / 'train_labels.npz', **{k: arrays['train'][k] for k in LABELS},
                     trend_target_residual=arrays['train']['target_residual'])
            save_array(terminal / 'sample_weights.npy', year_weights(arrays['train']['year']))
            probes = [x[np.linspace(0, len(x)-1, min(1024, len(x)), dtype=int)] for x in terminal_features.values()]
            save_array(terminal / 'reference/probe.npy', np.concatenate(probes))

            raw = dict(full_raw if cutoff == 2009 else early_raw)
            raw['observed_gpp'], _, _ = physical_gpp(raw)
            raw['previous_gpp'], raw['previous_gpp_quality'], _ = physical_gpp(raw, True)
            fit = np.flatnonzero(raw['year'] <= cutoff)
            splits = dict(validation=np.flatnonzero((raw['year'] > cutoff) & (raw['year'] <= origin)))
            if cutoff == 2009:
                splits['test'] = np.flatnonzero(raw['year'] > origin)
            support_train = np.load(screen / 'metadata_train.npy', mmap_mode='r')
            evaluation_sizes = {}
            for split, take in splits.items():
                a = arrays[split]
                same_identity({k: raw[k][take] for k in LABELS}, a)
                active = raw['relative_valid'][take] > 0
                if split == 'test':
                    norm = stable_meta['normalization']
                    weather = raw['weather'][take]
                    weather = np.where(np.isfinite(weather), (weather-np.array(norm['weather_mean'], np.float32)) /
                                       np.array(norm['weather_std'], np.float32), 0).astype(np.float32)
                    common[split] = np.concatenate((hx[split], metadata[take], weather.reshape(len(take), -1)), 1)
                support = historical_support(raw, fit, take, support_train)
                folder = workspace / 'evaluation/cases' / name / split
                for k, value in dict(active=active, common=common[split], support=support,
                                     trend=a['baseline'].astype(float)).items():
                    save_array(folder / f'{k}.npy', value)
                np.savez(folder / 'labels.npz', **{k: a[k] for k in LABELS})
                encoded = {}
                for product in products:
                    encoder = TrajectoryEncoder(remap_product(raw, product), fit, take, scales[product])
                    observed = raw[f'observed_{product}'][take]
                    assert not np.isfinite(observed[~active]).any()
                    encoded[product] = encoder.encode(observed)
                    if split == 'validation':
                        np.testing.assert_array_equal(encoded[product], np.load(screen / f'{product}_validation.npy'))
                    values = {f'{product}_{k}': v for k, v in batch_arrays(raw, take, stats, product).items()}
                    values.update({f'observed_{product}': observed,
                                   f'climatology_{product}': selected_climatology(encoder, np.arange(len(take))),
                                   f'observed_encoding_{product}': encoded[product]})
                    for percent in (10, 30, 50):
                        tail = tail_mask(active, percent/100)
                        values[f'prefix_{product}_{percent:03d}'] = np.where(active & ~tail, observed, np.nan).astype(np.float32)
                    for k, value in values.items():
                        save_array(folder / f'{k}.npy', value)
                    del encoder, values
                observed_x = feature_matrix(common[split], encoded, np.zeros_like(active), support, recipe)
                if split == 'validation':
                    np.testing.assert_array_equal(observed_x, terminal_features[split])
                save_array(folder / 'observed_features.npy', observed_x)
                evaluation_sizes[split] = len(take)
            case = dict(crop=crop, cutoff=cutoff, recipe=recipe, products=products, splits=evaluation_sizes,
                        evaluation_years=np.unique(np.concatenate([raw['year'][v] for v in splits.values()])).tolist(),
                        feature_width=465+36*len(products), encoder_scales={p: scales[p] for p in products},
                        state_scales={p: stats[p] for p in products})
            save_json(workspace / 'evaluation/cases' / name / 'case.json', case)
            cases[name] = dict(training_rows=len(fit), evaluation_rows=sum(evaluation_sizes.values()),
                               splits=evaluation_sizes, products=products)
            print(f'[INTERFACE FINISHED] {name}: {case["splits"]}', flush=True)
            del raw, arrays, hx, common, terminal_features, encoded, observed_x
            gc.collect()
    files = {str(p.relative_to(workspace)): sha256(p) for part in ('history', 'state', 'terminal', 'evaluation')
             for p in sorted((workspace / part).rglob('*')) if p.is_file()}
    save_json(workspace / 'generation.json', dict(generated=True, cases=cases, files=files,
        accessed_input_files=sorted(accessed), models_fitted=0, old_predictions_loaded=False,
        bins_recomputed_from_training=True, gpu_used=False, full_raw_to_evaluation_reproduced=False))
    print('[INTERFACE GENERATION COMPLETE]', flush=True)


if __name__ == '__main__':
    main()
