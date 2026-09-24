"""Shared, immutable inputs for the second reviewer-response experiment suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_crop_area_fraction import grid_area_hectares, sample_fraction
from biid_world_model import (
    CropWorldDynamics, load_or_compute_world_stats, load_world_cache,
    local_indices_for_split, make_world_arrays,
)
from multimodal_baseline import load_cache, load_coordinates, save_json, set_seed
from run_biid_world_model import make_loader, predict_dynamics
from run_biid_yield_fusion import (
    load_state, predict_hgb_history_base, predict_mlp_history_base,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "benchmark/cache/review_revision_v2"
RESULT_ROOT = ROOT / "benchmark/results/review_revision_v2"
CROPS = ("maize", "rice", "soybean", "wheat")
SEEDS = (42, 45, 48)
SPLITS = ("train", "validation", "test")
SCHEMA = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reliability(q: np.ndarray) -> np.ndarray:
    return 0.1 + 0.9 * np.clip(q, 0, 1) / (np.clip(q, 0, 1) + 0.05)


def training_climatology(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Fit grid/calendar means on train only, excluding self for train rows."""
    train = arrays["train"]
    def keys(a):
        return (a["row"].astype(np.int64) * 720 + a["col"])[:, None] * 12 + np.minimum(a["source_month"], 11)
    size = 360 * 720 * 12
    valid = train["target_lai_valid"] > 0
    values = train["target_lai"]
    k = keys(train)
    sums = np.bincount(k[valid], weights=values[valid], minlength=size)
    counts = np.bincount(k[valid], minlength=size)
    months = train["source_month"][valid]
    month_sum = np.bincount(months, weights=values[valid], minlength=12)
    month_count = np.bincount(months, minlength=12)
    result = {}
    for split, a in arrays.items():
        key = keys(a)
        month = np.minimum(a["source_month"], 11)
        numerator, denominator = sums[key].copy(), counts[key].astype(float)
        fallback_sum, fallback_count = month_sum[month].copy(), month_count[month].astype(float)
        if split == "train":
            numerator -= values * valid
            denominator -= valid
            fallback_sum -= values * valid
            fallback_count -= valid
        fallback = np.divide(fallback_sum, fallback_count, out=np.zeros_like(fallback_sum), where=fallback_count > 0)
        prediction = np.divide(numerator, denominator, out=fallback, where=denominator > 0)
        result[split] = (prediction * a["relative_valid"]).astype(np.float32)
    return result


def shuffled_coverage(a: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Shuffle within crop and 20-degree region, never conditioned on yield."""
    groups = (a["row"].astype(np.int64) // 40) * 18 + a["col"].astype(np.int64) // 40
    rng = np.random.default_rng(seed)
    result = a["crop_coverage"].copy()
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        result[idx] = result[rng.permutation(idx)]
    return result


def prepare(crop: str, seed: int, device: torch.device) -> None:
    destination = CACHE_ROOT / crop / f"seed_{seed}"
    metadata_path = destination / "manifest.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["schema"] != SCHEMA or any(not (destination / f"{s}.npz").exists() for s in SPLITS):
            raise RuntimeError(f"Incomplete/incompatible cache: {destination}")
        print(f"[CACHE EXISTS] {destination}", flush=True)
        return
    set_seed(seed)
    cache, world = load_cache(crop), load_world_cache(crop)
    stats = load_or_compute_world_stats(crop, cache, world)
    arrays = {}
    for split, split_id in zip(SPLITS, (0, 1, 2)):
        a = make_world_arrays(cache, world, stats, local_indices_for_split(cache, world, split_id))
        for key in ("year", "row", "col"):
            a[key] = np.asarray(cache[key][a["source_indices"]]).copy()
        arrays[split] = a
    climatology = training_climatology(arrays)
    lat, _ = load_coordinates()
    area = grid_area_hectares(lat)
    checkpoint = ROOT / "benchmark/results/biid_world_model" / crop / "biid_climate" / f"seed_{seed}" / "dynamics_best.pt"
    model = CropWorldDynamics(variant="biid_climate", dim=128, state_tokens=8, heads=4, biid_layers=2, dropout=0.1).to(device)
    incompatible = model.load_state_dict(load_state(checkpoint, device), strict=False)
    allowed = {"prior_history_norm.weight", "prior_history_norm.bias"}
    if set(incompatible.missing_keys) - allowed or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.eval()
    for split, a in arrays.items():
        print(f"[CACHE] {crop}/{seed}/{split}: {len(a['target'])} samples", flush=True)
        a["predicted_lai"] = predict_dynamics(model, make_loader(a, 4096, False, True), "biid_climate", device)
        a["climatology_lai"] = climatology[split]
        a["crop_coverage"] = np.clip(sample_fraction(crop, a["year"], a["row"], a["col"], area), 0, 1).astype(np.float32)
        a["shuffled_coverage"] = shuffled_coverage(a, 701 + SPLITS.index(split))
    del model
    torch.cuda.empty_cache()
    source = "mlp" if crop == "maize" else "hgb"
    if source == "mlp":
        history, history_path = predict_mlp_history_base(crop, seed, cache, arrays, stats, device, 4096)
    else:
        history, history_path = predict_hgb_history_base(crop, 42, cache, arrays, stats)
    destination.mkdir(parents=True, exist_ok=True)
    for split, a in arrays.items():
        a["history_base"] = history[split]
        for key in ("weather", "previous_lai", "history", "context", "predicted_lai", "history_base", "crop_coverage"):
            if not np.isfinite(a[key]).all():
                raise ValueError(f"Nonfinite {split}/{key}")
        temporary = destination / f"{split}.tmp.npz"
        np.savez(temporary, **a)
        temporary.replace(destination / f"{split}.npz")
    save_json({
        "schema": SCHEMA, "crop": crop, "seed": seed,
        "normalization": asdict(stats), "history_source": source,
        "history_checkpoint": str(history_path), "history_sha256": sha256(Path(history_path)),
        "dynamics_checkpoint": str(checkpoint), "dynamics_sha256": sha256(checkpoint),
        "split_sizes": {s: len(a["target"]) for s, a in arrays.items()},
        "sample_index_hash": {s: hashlib.sha256(a["source_indices"].tobytes()).hexdigest() for s, a in arrays.items()},
        "uniform_drop_probability": float((0.5 * (1 - reliability(arrays['train']['crop_coverage']))).mean()),
        "climatology": "train grid/natural-month mean; train leave-self-out; train-only month fallback",
        "state_training_predictions": "in-sample frozen dynamics, not cross-fitted",
        "protocol": "fixed development, not independent confirmation",
    }, metadata_path)
    print(f"[CACHE DONE] {metadata_path}", flush=True)


def load_shared(crop: str, seed: int):
    root = CACHE_ROOT / crop / f"seed_{seed}"
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata["schema"] != SCHEMA:
        raise ValueError("Unsupported input cache version")
    arrays = {}
    for split in SPLITS:
        with np.load(root / f"{split}.npz", allow_pickle=False) as data:
            arrays[split] = {key: data[key] for key in data.files}
    return arrays, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=CROPS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Cache inference requires an available GPU")
    prepare(args.crop, args.seed, torch.device("cuda"))
