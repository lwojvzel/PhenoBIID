"""Audited frozen recurrent memories for matched terminal readout experiments."""
import argparse
import fcntl
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from neural_process_readout import (ROOT, CROPS, versions, SPLITS, LABELS,
    CACHE as OLD_CACHE, NeuralReadout, predict, load as old_load)
from yield_sensitive_state import (STATE_INPUTS, CACHE as STATE_CACHE,
    run_root as state_root, load as state_load, YieldSensitiveState)
from export_yield_sensitive_state import export_root as old_export_root
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/latent_memory_readout_v1'
CACHE = ROOT / 'benchmark/cache/latent_memory_readout_v1'
CONDITIONS = ('full', 'mean', 'random', 'no_feedback', 'zero')
HEADS = ('tabm',)
DIMS = dict(full=488, mean=136, random=488, no_feedback=488, zero=488)
MODES = ('trained', 'random', 'no_feedback')
CODE = ('latent_memory_readout.py', 'yield_sensitive_state.py',
        'neural_process_readout.py', 'linear_state_yield.py')


def run_root(crop, origin, head, condition, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}/seed_42' / head / condition


def export_root(crop, origin):
    return RESULT / 'rollouts' / crop / f'origin_{origin}'


@torch.no_grad()
def rollout(model, a, feedback=True, device='cuda', size=256):
    model.eval()
    states, memories = [], []
    for start in range(0, len(a['previous_lai']), size):
        batch = {k: torch.from_numpy(a[k][start:start+size]).to(device) for k in STATE_INPUTS}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            lai, memory = model(batch, observation_feedback=feedback, return_latent=True)
        states.append(lai.float().cpu().numpy())
        memories.append(memory.float().cpu().numpy())
    return np.concatenate(states), np.concatenate(memories)


def checked_state(crop, origin):
    arrays, meta = state_load(crop, origin)
    source = STATE_CACHE / crop / f'origin_{origin}'
    for split, digest in meta['files'].items():
        if sha256(source / f'{split}.npz') != digest:
            raise ValueError('Original state input changed')
    root = state_root(crop, origin, 'state_only')
    config = json.loads((root / 'config.json').read_text())
    metric = json.loads((root / 'metrics.json').read_text())
    if config['input_manifest'] != meta or not metric['state_constraint_met']:
        raise ValueError('Unmatched or inadmissible state checkpoint')
    for name, digest in config['code_hashes'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError(f'Original state training source changed: {name}')
    if sha256(Path(metric['weight'])) != metric['weight_sha256']:
        raise ValueError('Frozen state weight changed')
    previous = old_export_root(crop, origin, 'state_only')
    exported = json.loads((previous / 'manifest.json').read_text())
    if exported['spec']['weight_sha256'] != metric['weight_sha256']:
        raise ValueError('Original rollout uses another state checkpoint')
    if exported['audit']['maximum_replay_error'] != 0:
        raise ValueError('Original rollout not audited')
    for split, digest in exported['files'].items():
        if sha256(previous / f'{split}.npy') != digest:
            raise ValueError('Original scalar rollout changed')
    return arrays, dict(input_manifest=meta, state_config=config, state_metrics=metric,
                       original_export=exported)


def export(crop, origin, audit=False):
    root = export_root(crop, origin); root.mkdir(parents=True, exist_ok=True)
    with (root / 'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arrays, source = checked_state(crop, origin)
        spec = dict(crop=crop, origin=origin, source=source, modes=list(MODES),
            code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE}, versions=versions(),
            inference='Frozen free rollout; BF16 batch256; no outcome or target state inputs.',
            random='Seed42 original initialization with the same learned W9 prior; no fitting.')
        marker = root / 'manifest.json'
        saved = json.loads(marker.read_text()) if marker.exists() else None
        if saved and saved['spec'] != spec:
            raise ValueError('Latent rollout recipe changed')
        if audit and saved is None:
            raise ValueError('No rollout to independently replay')
        if saved and not audit:
            return
        files = {}
        for mode in MODES:
            set_seed(42)
            model = YieldSensitiveState().cuda()
            if mode != 'random':
                model.load_state_dict(torch.load(source['state_metrics']['weight'], map_location='cuda', weights_only=True))
            else:
                weight = root / 'random_state.pt'
                if audit:
                    checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
                    for key, value in model.state_dict().items():
                        np.testing.assert_array_equal(value.cpu().numpy(), checkpoint[key].numpy())
                    model.load_state_dict(checkpoint)
                else:
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weight)
                files[weight.name] = sha256(weight)
            for split, a in arrays.items():
                lai, hidden = rollout(model, a, feedback=mode != 'no_feedback')
                if hidden.shape != (len(lai), 12, 32) or not np.isfinite(hidden).all():
                    raise ValueError('Invalid internal trajectory')
                if not np.all(hidden[a['relative_valid'] <= 0] == 0):
                    raise ValueError('Padded memory is not zero')
                if mode == 'trained':
                    np.testing.assert_array_equal(lai, np.load(old_export_root(crop, origin, 'state_only') / f'{split}.npy'))
                for kind, value in (('lai', lai), ('memory', hidden)):
                    file = root / f'{mode}_{split}_{kind}.npy'
                    if audit:
                        np.testing.assert_array_equal(value, np.load(file, mmap_mode='r'))
                    else:
                        np.save(file, value)
                    files[file.name] = sha256(file)
            del model; torch.cuda.empty_cache()
            print(f'[MEMORY EXPORT] {crop} {origin} {mode} audit={audit}', flush=True)
        if audit:
            if files != saved['files']:
                raise ValueError('Latent exports changed')
            atomic_json(root / 'audit.json', dict(files=files, full_array_replay=True,
                maximum_replay_error=0., original_scalar_predictions_identical=True))
        else:
            atomic_json(marker, dict(spec=spec, files=files))


def features(base, memories, active, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown memory condition')
    if base.shape != (len(active), 104) or active.shape[1:] != (12,):
        raise ValueError('Wrong common features or slot mask')
    if condition == 'zero':
        added = np.zeros((len(base), 384), dtype=np.float32)
    else:
        name = 'trained' if condition in ('full', 'mean') else condition
        hidden = memories[name]
        if hidden.shape != (len(base), 12, 32):
            raise ValueError('Wrong recurrent memory shape')
        hidden = np.where(active[..., None] > 0, hidden, 0)
        if condition == 'mean':
            added = hidden.astype(float).sum(1)/np.maximum((active > 0).sum(1, keepdims=True), 1)
        else:
            added = hidden.reshape(len(base), -1)
    value = np.concatenate((base, added), 1).astype(float)
    if value.shape != (len(base), DIMS[condition]) or not np.isfinite(value).all():
        raise ValueError('Invalid readout features')
    return value


def prepare(crop, origin, audit=False):
    root = CACHE / crop / f'origin_{origin}'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old_x, labels, old_meta = old_load(crop, origin, 'lai')
        arrays, source = checked_state(crop, origin)
        previous = export_root(crop, origin)
        exported = json.loads((previous / 'manifest.json').read_text())
        checked = json.loads((previous / 'audit.json').read_text())
        if checked['files'] != exported['files'] or checked['maximum_replay_error'] != 0:
            raise ValueError('Internal state export not audited')
        if exported['spec']['source'] != source:
            raise ValueError('State export inputs changed')
        for name, digest in exported['files'].items():
            if sha256(previous / name) != digest:
                raise ValueError(f'Internal state file changed: {name}')
        # Start from the original standardized prefix; do not round-trip its raw values.
        spec = dict(crop=crop, origin=origin, original_cache=old_meta, state_export=exported,
            dimensions=DIMS, conditions=list(CONDITIONS), versions=versions(),
            code_hashes={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            prefix='Original W18 standardized 104 columns, unchanged; only appended memory is train-standardized.',
            scope='In-sample frozen states, not full-pipeline cross-fitting; latent path is not scalar LAI mediation.')
        marker = root / 'manifest.json'; saved = json.loads(marker.read_text()) if marker.exists() else None
        if saved and saved['spec'] != spec:
            raise ValueError('Memory feature recipe changed')
        if saved and not audit:
            return
        if audit and not saved:
            raise ValueError('No memory cache to audit')
        files = {}
        for split in SPLITS:
            for key in ('row', 'col', 'year', 'source_indices'):
                np.testing.assert_array_equal(labels[split][key], arrays[split][key])
            file = root / f'{split}_labels.npz'
            if audit:
                with np.load(file) as prior:
                    for key, value in labels[split].items():
                        np.testing.assert_array_equal(prior[key], value)
            else:
                np.savez(file, **labels[split])
            files[file.name] = sha256(file)
        for condition in CONDITIONS:
            scaler = None
            for split in SPLITS:
                memories = {m: np.load(previous / f'{m}_{split}_memory.npy', mmap_mode='r') for m in MODES}
                x = features(old_x[split], memories, arrays[split]['relative_valid'], condition)
                if split == 'train':
                    scaler = StandardScaler().fit(x[:, 104:])
                    file = root / f'{condition}_scaler.joblib'
                    if audit:
                        prior = joblib.load(file)
                        for key in ('mean_', 'scale_', 'var_'):
                            np.testing.assert_array_equal(getattr(prior, key), getattr(scaler, key))
                    else:
                        joblib.dump(scaler, file)
                    files[file.name] = sha256(file)
                value = np.concatenate((old_x[split], scaler.transform(x[:, 104:])), 1).astype(np.float32)
                np.testing.assert_array_equal(value[:, :104], old_x[split])
                file = root / f'{condition}_{split}.npy'
                if audit:
                    np.testing.assert_array_equal(np.load(file, mmap_mode='r'), value)
                else:
                    np.save(file, value)
                files[file.name] = sha256(file)
            print(f'[MEMORY CACHE] {crop} {origin} {condition} audit={audit}', flush=True)
        if audit:
            if files != saved['files']:
                raise ValueError('Memory cache hashes changed')
            atomic_json(root / 'audit.json', dict(files=files, maximum_replay_error=0.,
                full_array_replay=True, training_scalers_verified=True, common_prefix_identical=True))
        else:
            atomic_json(marker, dict(spec=spec, normalization=old_meta['normalization'], files=files))


def load(crop, origin, condition):
    root = CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    if audit['maximum_replay_error'] != 0 or audit['files'] != meta['files']:
        raise ValueError('Unaudited memory input')
    if meta['spec']['versions'] != versions() or meta['spec']['code_hashes'] != {
            k: sha256(ROOT / 'scripts' / k) for k in CODE}:
        raise ValueError('Memory source/runtime changed')
    x, labels = {}, {}
    for split in SPLITS:
        for name in (f'{split}_labels.npz', f'{condition}_{split}.npy', f'{condition}_scaler.joblib'):
            if sha256(root / name) != meta['files'][name]:
                raise ValueError('Memory input changed')
        x[split] = np.load(root / f'{condition}_{split}.npy')
        with np.load(root / f'{split}_labels.npz') as f:
            labels[split] = {k: f[k] for k in f.files}
    return x, labels, meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', choices=(2004, 2008, 2012), type=int, required=True)
    p.add_argument('--export', action='store_true'); p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if args.export:
        torch.cuda.set_per_process_memory_fraction(2000*2**20/torch.cuda.get_device_properties(0).total_memory)
        export(args.crop, args.origin, args.audit)
    else:
        with threadpool_limits(limits=2):
            prepare(args.crop, args.origin, args.audit)
