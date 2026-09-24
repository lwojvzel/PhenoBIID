"""Observed target-season RS diagnostic; never a future-state forecast input."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path

import numpy as np

from dual_remote_data import extract
from forward_protocol_revision import RawInputs, index_hash, split_indices
from prepare_pku_ndvi import OUT as NDVI
from review_revision_data import ROOT, CROPS, SPLITS, sha256
from run_review_revision_parallel import atomic_json

CACHE = ROOT / "benchmark/cache/observed_remote_benchmark_v1"
RESULT = ROOT / "benchmark/results/observed_remote_benchmark_v1"
LOGS = ROOT / "benchmark/logs/observed_remote_benchmark_v1"
VARIANTS = ("history", "metadata", "observed_lai", "observed_ndvi", "observed_both")
ENGINES = ("ridge", "hgb", "lightgbm", "xgboost", "mlp", "tabm", "gru", "diffusion")
INPUTS = ("history", "context", "crop_coverage", "relative_valid", "source_month",
          "observed_lai", "observed_lai_valid", "observed_ndvi", "observed_ndvi_valid")


def prepare(crop, origin=2012):
    root = CACHE / crop / f"origin_{origin}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "manifest.json").exists():
            return root
        raw = RawInputs(crop)
        indices = split_indices(raw.years, origin)
        fitted = raw.normalization(indices["train"])
        arrays, product_stats = {}, None
        for split in SPLITS:
            a = raw.arrays(indices[split], fitted)
            ndvi, _ = extract(a)
            valid = np.isfinite(ndvi) & (a["relative_valid"] > 0)
            if split == "train":
                product_stats = dict(mean=float(ndvi[valid].mean(dtype=np.float64)),
                                     std=float(ndvi[valid].std(dtype=np.float64)))
            a["observed_ndvi"] = np.where(valid, (ndvi-product_stats["mean"])/product_stats["std"], 0).astype(np.float32)
            a["observed_ndvi_valid"] = valid.astype(np.float32)
            a["observed_lai"] = a["target_lai"]
            a["observed_lai_valid"] = a["target_lai_valid"]
            arrays[split] = {k: a[k] for k in (*INPUTS, "target", "target_residual", "baseline",
                                              "source_indices", "row", "col", "year")}
            np.savez(root / f"{split}.npz", **arrays[split])
        manifest = dict(crop=crop, origin=origin, normalization=asdict(fitted[0]), ndvi_normalization=product_stats,
            protocol="Retrospective observed target-season RS; not a prospective forecast",
            sample_universe="Unchanged LAI-based world-cache cohort; all input conditions share rows",
            alignment="Existing ascending valid natural-month packing, not repaired cross-year crop seasons",
            history="Past-only lags/trend; evaluation rolls annually with preceding observed yields",
            upstream_models_used=False, target_weather_input=False,
            source_manifests={str(NDVI / "manifest.json"): sha256(NDVI / "manifest.json")},
            source_world_hash=index_hash(raw.source), implementation_sha256=sha256(Path(__file__)),
            splits={s: dict(n=len(a["target"]), years=np.unique(a["year"]).tolist(),
                indices_sha256=index_hash(a["source_indices"]), file_sha256=sha256(root / f"{s}.npz"),
                active_slots=int(a["relative_valid"].sum()),
                lai_valid_fraction=float(a["observed_lai_valid"].sum()/a["relative_valid"].sum()),
                ndvi_valid_fraction=float(a["observed_ndvi_valid"].sum()/a["relative_valid"].sum()))
                for s, a in arrays.items()})
        atomic_json(root / "manifest.json", manifest)
        print(f"[DATA] {crop} origin={origin} {[(s,len(a['target'])) for s,a in arrays.items()]}", flush=True)
    return root


def load(crop, origin=2012):
    root = CACHE / crop / f"origin_{origin}"
    meta = json.loads((root / "manifest.json").read_text())
    arrays = {}
    for s in SPLITS:
        with np.load(root / f"{s}.npz", allow_pickle=False) as f:
            arrays[s] = {k: f[k] for k in f.files}
    return arrays, meta


def trajectory_features(values, valid):
    """Keep every slot; summaries supplement, rather than replace, the trajectory."""
    valid = np.asarray(valid, bool)
    x = np.where(valid, values, 0).astype(np.float32)
    count = valid.sum(1)
    mean = x.sum(1)/np.maximum(count, 1)
    std = np.sqrt(np.where(valid, (x-mean[:, None])**2, 0).sum(1)/np.maximum(count, 1))
    maximum = np.where(count > 0, np.where(valid, x, -np.inf).max(1), 0)
    minimum = np.where(count > 0, np.where(valid, x, np.inf).min(1), 0)
    peak = np.where(count > 0, np.where(valid, x, -np.inf).argmax(1)/11., 0)
    return np.concatenate((x, np.stack((mean, std, maximum, minimum, x.sum(1), peak), 1)), 1).astype(np.float32)


def make_features(b, variant):
    if set(b) != set(INPUTS) or variant not in VARIANTS:
        raise ValueError("Observed-input contract mismatch; yield labels must stay outside the encoder")
    n = len(b["history"])
    history = np.concatenate((b["history"], b["context"]), 1).astype(np.float32)
    if history.shape != (n, 20):
        raise ValueError("Expected 15 historical features and 5 context features")
    seq = np.zeros((n, 12, 7), np.float32)
    names = [f"history_{i}" for i in range(15)] + [f"context_{i}" for i in range(5)]
    if variant == "history":
        return history, seq, names
    active = np.asarray(b["relative_valid"], bool)
    month = np.asarray(b["source_month"])
    if np.any(active & (month > 11)) or not np.all(active.sum(1) > 0):
        raise ValueError("Invalid active-slot calendar")
    seq[:, :, 0] = active
    seq[:, :, 1] = np.where(active, np.sin(2*np.pi*month/12), 0)
    seq[:, :, 2] = np.where(active, np.cos(2*np.pi*month/12), 0)
    for j, p in enumerate(("lai", "ndvi")):
        valid = active & (b[f"observed_{p}_valid"] > 0)
        seq[:, :, 3+j] = valid
        if variant in (f"observed_{p}", "observed_both"):
            if not np.isfinite(b[f"observed_{p}"][valid]).all():
                raise ValueError("Nonfinite valid remote observation")
            seq[:, :, 5+j] = np.where(valid, b[f"observed_{p}"], 0)
    meta = np.concatenate((b["crop_coverage"][:, None], seq[:, :, :5].reshape(n, -1)), 1)
    chunks = [history, meta]
    names += ["crop_area_fraction"] + [f"slot_{k}_{v}" for k in range(12)
                                      for v in ("active", "calendar_sin", "calendar_cos", "lai_valid", "ndvi_valid")]
    for j, p in enumerate(("lai", "ndvi")):
        if variant in (f"observed_{p}", "observed_both"):
            chunks.append(trajectory_features(seq[:, :, 5+j], seq[:, :, 3+j] > 0))
            names += [f"observed_{p}_slot_{k}" for k in range(12)]
            names += [f"observed_{p}_{v}" for v in ("mean", "std", "max", "min", "sum", "peak_slot")]
    x = np.concatenate(chunks, 1).astype(np.float32)
    if not np.isfinite(x).all() or len(names) != x.shape[1]:
        raise ValueError("Invalid feature matrix")
    return x, seq, names


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--origin", type=int, default=2012)
    a = p.parse_args()
    prepare(a.crop, a.origin)
