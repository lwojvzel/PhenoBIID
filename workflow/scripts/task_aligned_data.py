"""Training-only local forcing statistics for task-aligned state experiments."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from biid_world_model import WorldNormalizationStats
from forward_protocol_revision import RawInputs
from run_history_multimodal_baselines import build_causal_history_features
from stable_remote_data import ROOT, CROPS, load as load_source
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

CACHE = ROOT / 'benchmark/cache/task_aligned_world_v1'


def local_moments(arrays, field, mask_key):
    def keys(a):
        return (a['row'].astype(np.int64) * 720 + a['col'])[:, None] * 12 + np.minimum(a['source_month'], 11)
    train = arrays['train']; values = train[field]
    if values.ndim == 2:
        values = values[..., None]
    mask = (train[mask_key] > 0) & (train['relative_valid'] > 0)
    index = keys(train)
    unique, inverse = np.unique(index[mask], return_inverse=True)
    channels = values.shape[-1]
    count = np.bincount(inverse, minlength=len(unique)).astype(float)
    sums = np.stack([np.bincount(inverse, weights=values[..., v][mask], minlength=len(unique)) for v in range(channels)], -1)
    square = np.stack([np.bincount(inverse, weights=values[..., v][mask] ** 2, minlength=len(unique)) for v in range(channels)], -1)
    months = np.minimum(train['source_month'], 11)
    month_count = np.bincount(months[mask], minlength=12).astype(float)
    month_sums = np.stack([np.bincount(months[mask], weights=values[..., v][mask], minlength=12) for v in range(channels)], -1)
    month_square = np.stack([np.bincount(months[mask], weights=values[..., v][mask] ** 2, minlength=12) for v in range(channels)], -1)
    output = {}
    for split, a in arrays.items():
        key = keys(a); month = np.minimum(a['source_month'], 11)
        ix = np.searchsorted(unique, key); safe = np.clip(ix, 0, len(unique) - 1)
        known = (ix < len(unique)) & (unique[safe] == key)
        n = np.where(known, count[safe], 0)
        total = np.where(known[..., None], sums[safe], 0)
        ss = np.where(known[..., None], square[safe], 0)
        mn, mt, ms = month_count[month].copy(), month_sums[month].copy(), month_square[month].copy()
        if split == 'train':
            own = np.where(mask[..., None], values, 0)
            n -= mask; total -= own; ss -= own ** 2
            mn -= mask; mt -= own; ms -= own ** 2
        fallback = np.divide(mt, mn[..., None], out=np.zeros_like(mt), where=mn[..., None] > 0)
        fallback_ss = np.divide(ms, mn[..., None], out=np.ones_like(ms), where=mn[..., None] > 0)
        mean = np.divide(total, n[..., None], out=fallback.copy(), where=n[..., None] > 0)
        second = np.divide(ss, n[..., None], out=fallback_ss.copy(), where=n[..., None] > 0)
        std = np.sqrt(np.maximum(second - mean ** 2, .05 ** 2))
        output[split] = (mean.astype(np.float32), std.astype(np.float32))
    return output


def prepare(crop, origin):
    root = CACHE / crop / f'origin_{origin}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'manifest.json').exists():
            return root
        source, meta = load_source(crop, origin, 0)
        raw = RawInputs(crop); stats = WorldNormalizationStats(**meta['normalization'])
        history, baseline = build_causal_history_features(raw.cache, stats.target_mean, stats.target_std)
        arrays = {}
        for split, a in source.items():
            ix = np.searchsorted(raw.source, a['source_indices'])
            np.testing.assert_array_equal(raw.source[ix], a['source_indices'])
            arrays[split] = raw.arrays(ix, (stats, history, baseline))
            for key in ('target', 'history', 'context', 'previous_lai', 'relative_valid', 'baseline'):
                np.testing.assert_array_equal(arrays[split][key], a[key])
        lai = local_moments(arrays, 'target_lai', 'target_lai_valid')
        weather = local_moments(arrays, 'weather', 'relative_valid')
        for split, a in arrays.items():
            a['lai_climo'] = lai[split][0][..., 0]
            a['lai_scale'] = lai[split][1][..., 0]
            a['weather_climo'] = weather[split][0]
            a['weather_anomaly'] = np.where(a['relative_valid'][..., None] > 0, a['weather'] - a['weather_climo'], 0)
            count = np.maximum(np.cumsum(a['relative_valid'], axis=1), 1)
            a['weather_cumulative'] = (np.cumsum(a['weather_anomaly'], axis=1) / np.sqrt(count)[..., None]).astype(np.float32)
            if not all(np.isfinite(a[k]).all() for k in ('lai_climo', 'lai_scale', 'weather_climo', 'weather_anomaly', 'weather_cumulative')):
                raise ValueError('Nonfinite local statistics')
            np.savez(root / f'{split}.npz', **a)
        a = arrays['train']; mask = (a['target_lai_valid'] > 0) & (a['relative_valid'] > 0)
        error = a['previous_lai'] - a['target_lai']
        loss_scales = dict(lai=float(max(np.mean(error[mask].astype(float) ** 2), 1e-4)),
                           local=float(max(np.mean((error / a['lai_scale'])[mask].astype(float) ** 2), 1e-4)))
        record = dict(source_manifest=meta, normalization=meta['normalization'], loss_scales=loss_scales,
                      protocol='Full training history; three validation years; four evaluation years. Joint supervised model, no frozen in-sample state-readout fitting.',
                      climate='Training-only grid/month climatology, standardized anomalies, cumulative anomaly divided by square root of active prefix length; statistical forcing summaries, not physical storage units.',
                      code_sha256=sha256(Path(__file__)), files={s: sha256(root / f'{s}.npz') for s in arrays})
        atomic_json(root / 'manifest.json', record)
        print(f'[TASK DATA] {crop} {origin} loss scales {loss_scales}', flush=True)
    return root


def load(crop, origin):
    root = CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text())
    arrays = {}
    for s in ('train', 'validation', 'test'):
        with np.load(root / f'{s}.npz') as f:
            arrays[s] = {k: f[k] for k in f.files}
    return arrays, meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), default=2012)
    a = p.parse_args(); prepare(a.crop, a.origin)
