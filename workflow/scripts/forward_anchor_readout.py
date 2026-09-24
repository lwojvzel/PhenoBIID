"""Forward historical residuals; vegetation trajectories remain frozen in-sample."""
import argparse
import fcntl
import hashlib
import json

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from neural_process_readout import (ROOT, CROPS, CONDITIONS, HEADS, DIMS, NeuralReadout,
    predict, versions, load as original_load, cache_root as original_cache)
from task_aligned_data import CACHE as TASK_CACHE
from multimodal_baseline import CACHE_ROOT as RAW_CACHE
from run_history_multimodal_baselines import build_causal_history_features
from gru_replication_results import history_selection
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/forward_anchor_readout_v1'
CACHE = ROOT / 'benchmark/cache/forward_anchor_readout_v1'
TREATMENTS = ('in_sample', 'forward')
SPLITS = ('train', 'validation', 'test')
CODE = ('forward_anchor_readout.py', 'run_forward_history_anchor.py',
        'run_history_multimodal_baselines.py', 'stable_remote_models.py', 'task_aligned_world.py')


def windows(origin):
    return [(s-4, s-3, s-1, s, min(s+3, origin-3)) for s in range(1993, origin-2, 4)]


def fold_root(crop, origin, start, smoke=False):
    return RESULT / ('smoke_anchors' if smoke else 'anchors') / crop / f'origin_{origin}/start_{start}'


def cache_root(crop, origin):
    return CACHE / crop / f'origin_{origin}'


def run_root(crop, origin, head, condition, smoke=False, treatment='forward'):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42' / treatment / head / condition


def raw_source(crop, origin):
    root = TASK_CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text())
    if sha256(root / 'train.npz') != meta['files']['train']:
        raise ValueError('Original task training rows changed')
    names = ('history', 'context', 'target', 'baseline', 'row', 'col', 'year', 'source_indices')
    with np.load(root / 'train.npz') as f:
        a = {k: f[k] for k in names}
    if a['year'].max() != origin-3:
        raise ValueError('Unexpected outer training end')
    raw = {k: np.load(RAW_CACHE / crop / f'{k}.npy', mmap_mode='r') for k in ('target', 'row', 'col', 'year')}
    ids = np.flatnonzero(raw['year'] <= origin-3)
    source = {k: np.asarray(v[ids]) for k, v in raw.items()}
    ix = np.searchsorted(ids, a['source_indices'])
    np.testing.assert_array_equal(ids[ix], a['source_indices'])
    for key in ('target', 'row', 'col', 'year'):
        np.testing.assert_array_equal(source[key][ix], a[key])
    spec = dict(crop=crop, origin=origin, family=history_selection()[crop, origin],
        task_train_sha256=meta['files']['train'], raw_files={k: sha256(RAW_CACHE / crop / f'{k}.npy') for k in raw},
        code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE},
        raw_labels_used_through=origin-3, original_normalization=meta['normalization'])
    return a, source, ix, spec


def causal_history(source, mean, std):
    history, baseline = build_causal_history_features(source, mean, std)
    # Keep the original fixed 1981--2016 count denominator, not the truncated window length.
    span = max(int(source['year'].max()-source['year'].min()), 1)
    history[:, 14] = np.rint(history[:, 14].astype(float)*span)/35.
    return history, baseline


def fold_data(crop, origin, start, smoke=False):
    a, source, ix, spec = raw_source(crop, origin)
    window = next(w for w in windows(origin) if w[3] == start)
    fit_end, val_start, val_end, first, last = window
    years = a['year']
    indices = dict(fit=np.flatnonzero(years <= fit_end),
        validation=np.flatnonzero((years >= val_start) & (years <= val_end)),
        forward=np.flatnonzero((years >= first) & (years <= last)))
    if any(not len(v) for v in indices.values()):
        raise ValueError('Empty forward fold')
    if smoke:
        indices = {s: np.concatenate([v[years[v] == y][:128] for y in np.unique(years[v])]) for s, v in indices.items()}
    fit_target = a['target'][indices['fit']].astype(float)
    mean, std = float(fit_target.mean()), max(float(fit_target.std()), 1e-6)
    history, baseline = causal_history(source, mean, std)
    history, baseline = history[ix], baseline[ix]
    residual = a['target'].astype(float)-baseline.astype(float)
    rmean, rstd = float(residual[indices['fit']].mean()), max(float(residual[indices['fit']].std()), 1e-6)
    x = np.concatenate((history, a['context']), 1).astype(np.float32)
    y = ((residual-rmean)/rstd).astype(np.float32)
    labels = {s: dict(baseline=baseline[v], **{k: a[k][v] for k in ('target', 'row', 'col', 'year', 'source_indices')})
              for s, v in indices.items()}
    spec.update(start=start, window=list(window), smoke=smoke,
        normalization=dict(target_mean=mean, target_std=std, residual_mean=rmean, residual_std=rstd),
        rows={s: len(v) for s, v in indices.items()},
        row_hashes={s: hashlib.sha256(np.asarray(a['source_indices'][v], dtype='<i8').tobytes()).hexdigest()
                    for s, v in indices.items()},
        scope='Historical base only: train and inner validation precede every forward prediction year.')
    return {s: x[v] for s, v in indices.items()}, {s: y[v] for s, v in indices.items()}, labels, spec


def prepare(crop, origin, audit=False):
    root = cache_root(crop, origin); root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _, labels, original = original_load(crop, origin, 'history')
        a = labels['train']; take = np.flatnonzero(a['year'] >= 1993)
        forward = np.full(len(take), np.nan); lineage = []
        for _, _, _, first, _ in windows(origin):
            source = fold_root(crop, origin, first)
            checked = json.loads((source / 'audit.json').read_text())
            config = json.loads((source / 'config.json').read_text())
            if checked['maximum_replay_error'] != 0 or not checked['temporal_order_verified']:
                raise ValueError('Historical fold audit failed')
            if sha256(source / checked['weight_name']) != checked['weight_sha256']:
                raise ValueError('Historical fold weight changed')
            prediction_file = source / 'forward_predictions.npz'
            if sha256(prediction_file) != checked['predictions']['forward']:
                raise ValueError('Historical fold output changed')
            with np.load(prediction_file) as f:
                ix = np.searchsorted(a['source_indices'][take], f['source_indices'])
                for k in ('target', 'row', 'col', 'year', 'source_indices'):
                    np.testing.assert_array_equal(a[k][take][ix], f[k])
                if np.isfinite(forward[ix]).any():
                    raise ValueError('Duplicated forward rows')
                forward[ix] = f['prediction']
            lineage.append(dict(directory=str(source), config_sha256=sha256(source / 'config.json'), audit=checked))
            if config['code_hashes'] != {k: sha256(ROOT / 'scripts' / k) for k in CODE}:
                raise ValueError('Frozen historical fold code changed')
        if not np.isfinite(forward).all():
            raise ValueError('Missing historical forward predictions')
        spec = dict(crop=crop, origin=origin, source_manifest=original, lineage=lineage,
            first_terminal_training_year=1993, treatments=list(TREATMENTS),
            scope='Only historical residual targets are forward fitted. Vegetation remains in-sample; outer base unchanged.')
        path = root / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['spec'] != spec:
                raise ValueError('Forward terminal cache changed')
            if not audit:
                return
        elif audit:
            raise ValueError('Missing forward terminal cache')
        files = {}
        for split in SPLITS:
            item = {k: v[take] if split == 'train' else v for k, v in labels[split].items()}
            item['forward_history_prediction'] = forward if split == 'train' else item['history_prediction']
            file = root / f'{split}_labels.npz'
            if audit:
                with np.load(file) as old:
                    for k, v in item.items():
                        np.testing.assert_array_equal(old[k], v)
            else:
                np.savez(file, **item)
            files[file.name] = sha256(file)
        for condition in CONDITIONS:
            x, _, _ = original_load(crop, origin, condition)
            x['train'] = x['train'][take]
            scaler = StandardScaler().fit(x['train'])
            file = root / f'{condition}_scaler.joblib'
            if audit:
                old = joblib.load(file)
                for key in ('mean_', 'scale_', 'var_'):
                    np.testing.assert_array_equal(getattr(old, key), getattr(scaler, key))
            else:
                joblib.dump(scaler, file)
            files[file.name] = sha256(file)
            for split in SPLITS:
                value = scaler.transform(x[split]).astype(np.float32)
                file = root / f'{condition}_{split}.npy'
                if audit:
                    np.testing.assert_array_equal(np.load(file, mmap_mode='r'), value)
                else:
                    np.save(file, value)
                files[file.name] = sha256(file)
            print(f'[FORWARD CACHE] {crop} {origin} {condition} audit={audit}', flush=True)
        if audit:
            if manifest['files'] != files:
                raise ValueError('Forward terminal file hash changed')
            atomic_json(root / 'audit.json', dict(files=files, maximum_replay_error=0., training_scalers_verified=True,
                all_forward_rows_covered=True, outer_history_unchanged=True))
        else:
            atomic_json(path, dict(spec=spec, normalization=original['normalization'], files=files))


def load(crop, origin, condition, treatment='forward'):
    if treatment not in TREATMENTS:
        raise ValueError('Unknown historical treatment')
    root = cache_root(crop, origin)
    meta = json.loads((root / 'manifest.json').read_text())
    checked = json.loads((root / 'audit.json').read_text())
    if checked['files'] != meta['files'] or checked['maximum_replay_error'] != 0:
        raise ValueError('Unaudited forward terminal cache')
    data, labels = {}, {}
    for split in SPLITS:
        for name in (f'{condition}_{split}.npy', f'{split}_labels.npz', f'{condition}_scaler.joblib'):
            if sha256(root / name) != meta['files'][name]:
                raise ValueError('Forward terminal input changed')
        data[split] = np.load(root / f'{condition}_{split}.npy')
        with np.load(root / f'{split}_labels.npz') as f:
            labels[split] = {k: f[k] for k in f.files}
        if treatment == 'forward':
            labels[split]['history_prediction'] = labels[split]['forward_history_prediction']
    return data, labels, meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument('--audit', action='store_true'); args = p.parse_args()
    with threadpool_limits(limits=2):
        prepare(args.crop, args.origin, args.audit)
