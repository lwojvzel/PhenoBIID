from __future__ import annotations
import numpy as np
import hashlib
from biid_world_model import WorldNormalizationStats, load_world_cache, make_world_arrays
from multimodal_baseline import load_cache
from review_revision_data import load_shared
from run_history_multimodal_baselines import build_causal_history_features
def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()

def safe_std(values):
    return max(float(np.std(values, dtype=np.float64)), 1e-6)

class RawInputs:
    def __init__(self, crop):
        self.crop = crop
        self.cache, self.world = load_cache(crop), load_world_cache(crop)
        self.source = np.asarray(self.world["source_indices"])
        self.years = np.asarray(self.cache["year"])[self.source]
        # Coverage is deterministic source metadata; do not reuse fitted states,
        # history predictions or normalization from the old revision cache.
        shared, metadata = load_shared(crop, 42)
        ids = np.concatenate([a["source_indices"] for a in shared.values()])
        q = np.concatenate([a["crop_coverage"] for a in shared.values()])
        order = np.argsort(ids)
        if len(np.unique(ids)) != len(ids) or not np.array_equal(ids[order], self.source):
            raise ValueError("Coverage/sample universe differs from raw world cache")
        self.coverage = q[order]
        self.source_manifest = {"coverage_cache": metadata["sample_index_hash"],
                                "world_source_indices": index_hash(self.source)}

    def normalization(self, indices):
        total = np.zeros(13, dtype=np.float64)
        square = total.copy()
        count = total.copy()
        lai_total = lai_square = 0.0
        lai_count = 0
        for start in range(0, len(indices), 10000):
            ix = indices[start:start + 10000]
            valid = np.asarray(self.world["relative_valid"][ix], dtype=bool)
            weather = np.asarray(self.world["weather_rel"][ix], dtype=np.float64)
            mask = valid[..., None] & np.isfinite(weather)
            total += np.where(mask, weather, 0).sum((0, 1))
            square += np.where(mask, weather * weather, 0).sum((0, 1))
            count += mask.sum((0, 1))
            lai = np.asarray(self.world["target_lai_rel"][ix], dtype=np.float64)
            values = lai[valid & np.isfinite(lai)]
            lai_total += values.sum()
            lai_square += np.square(values).sum()
            lai_count += len(values)
        if not lai_count or np.any(count == 0):
            raise ValueError("Missing training observations in normalization")
        wm = total / count
        ws = np.sqrt(np.maximum(square / count - wm * wm, 1e-12))
        lm = float(lai_total / lai_count)
        ls = float(np.sqrt(max(lai_square / lai_count - lm * lm, 1e-12)))
        target = np.asarray(self.cache["target"])[self.source[indices]].astype(np.float64)
        mean, std = float(target.mean()), safe_std(target)
        history, baseline = build_causal_history_features(self.cache, mean, std)
        residual = target - baseline[self.source[indices]]
        stats = WorldNormalizationStats(wm.tolist(), ws.tolist(), lm, ls, mean, std,
                                        float(residual.mean()), safe_std(residual))
        return stats, history, baseline

    def arrays(self, indices, fitted):
        stats, history, baseline = fitted
        a = make_world_arrays(self.cache, self.world, stats, indices)
        source = a["source_indices"]
        a["history"] = history[source].copy()
        a["baseline"] = baseline[source].copy()
        a["target_residual"] = ((a["target"] - a["baseline"] - stats.residual_mean)
                                / stats.residual_std).astype(np.float32)
        for key in ("year", "row", "col"):
            a[key] = np.asarray(self.cache[key])[source].copy()
        a["crop_coverage"] = self.coverage[indices].copy()
        return a
