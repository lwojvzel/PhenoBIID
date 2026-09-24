"""Index-preserving, training-only normalization of the added NDVI observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_pku_ndvi import OUT, digest, save_json
from review_revision_data import ROOT, CROPS, SPLITS, load_shared

CACHE = ROOT / "benchmark/cache/dual_remote_state_v1"


def extract(a, previous=False):
    n = len(a["year"])
    values = np.full((n, 12), np.nan, np.float32)
    quality = np.zeros_like(values)
    for year in np.unique(a["year"]):
        rows = np.flatnonzero(a["year"] == year)
        source_year = int(year) - int(previous)
        path = OUT / "monthly_0p5" / f"ndvi_monthly_{source_year}.npy"
        if source_year < 1982:
            continue
        grid = np.load(path, mmap_mode="r")
        qgrid = np.load(OUT / "monthly_0p5" / f"quality_fraction_monthly_{source_year}.npy", mmap_mode="r")
        month = np.minimum(a["source_month"][rows], 11)
        r, c = a["row"][rows, None], a["col"][rows, None]
        valid = (a["relative_valid"][rows] > 0) & (a["source_month"][rows] != 255)
        values[rows] = np.where(valid, grid[month, r, c], np.nan)
        quality[rows] = np.where(valid, qgrid[month, r, c], 0)
    return values, quality


def prepare(crop):
    destination = CACHE / crop
    marker = destination / "manifest.json"
    if marker.exists():
        return
    if not json.loads((OUT / "manifest.json").read_text())["complete"]:
        raise RuntimeError("Incomplete product processing")
    arrays, _ = load_shared(crop, 42)
    raw = {s: (extract(a), extract(a, True)) for s, a in arrays.items()}
    train = raw["train"][0][0]
    finite = np.isfinite(train)
    mean, std = float(train[finite].mean(dtype=np.float64)), float(train[finite].std(dtype=np.float64))
    if not np.isfinite(mean) or std < 1e-6:
        raise ValueError("NDVI training statistics are not usable")
    destination.mkdir(parents=True, exist_ok=True)
    coverage = []
    hashes = {}
    for split, ((target, target_q), (previous, previous_q)) in raw.items():
        a = arrays[split]
        valid, old_valid = np.isfinite(target), np.isfinite(previous)
        saved = dict(target_ndvi=np.where(valid, (target-mean)/std, 0).astype(np.float32),
                     target_ndvi_valid=valid.astype(np.float32),
                     previous_ndvi=np.where(old_valid, (previous-mean)/std, 0).astype(np.float32),
                     previous_ndvi_valid=old_valid.astype(np.float32),
                     previous_ndvi_quality=previous_q.astype(np.float32),
                     source_indices=a["source_indices"])
        path = destination / f"{split}.npz"
        np.savez(path, **saved)
        hashes[split] = digest(path)
        active = a["relative_valid"] > 0
        for period, selected in (("all", np.ones(len(active), bool)), ("pre_2003", a["year"] < 2003), ("from_2003", a["year"] >= 2003)):
            m = active & selected[:, None]
            if not m.any():
                continue
            lai_valid = a["target_lai_valid"] > 0
            coverage.append(dict(crop=crop, split=split, period=period, slots=int(m.sum()),
                                 lai_valid_fraction=float(lai_valid[m].mean()),
                                 ndvi_valid_fraction=float(valid[m].mean()),
                                 either_valid_fraction=float((lai_valid | valid)[m].mean()),
                                 ndvi_when_lai_missing=int((m & valid & ~lai_valid).sum()),
                                 previous_ndvi_valid_fraction=float(old_valid[m].mean()),
                                 ndvi_good_native_area_time_mean=float(target_q[m].mean()),
                                 crop_area_fraction_mean=float(a["crop_coverage"][selected].mean())))
    save_json(marker, {"crop": crop, "normalization": {"mean": mean, "std": std},
                       "fit": "training target NDVI only, valid quality-screened slots",
                       "source_manifest_sha256": digest(OUT / "manifest.json"),
                       "cache_sha256": hashes, "coverage": coverage,
                       "history_1981": "missing, not backfilled",
                       "sample_universe": "unchanged original LAI-based world-cache rows; not expansion to all yield cells"})
    print(f"[NDVI CACHE] {crop}: mean={mean:.5f} std={std:.5f}", flush=True)


def load(crop, seed, needs_ndvi=True):
    arrays, meta = load_shared(crop, seed)
    if not needs_ndvi:
        for a in arrays.values():
            for key in ("previous_ndvi", "previous_ndvi_valid", "previous_ndvi_quality", "target_ndvi", "target_ndvi_valid"):
                a[key] = np.zeros_like(a["previous_lai"])
        meta["ndvi"] = {"used": False, "reason": "LAI-only matched control; NDVI never enters forward or loss"}
        return arrays, meta
    extra = json.loads((CACHE / crop / "manifest.json").read_text())
    for split, a in arrays.items():
        with np.load(CACHE / crop / f"{split}.npz") as d:
            np.testing.assert_array_equal(d["source_indices"], a["source_indices"])
            a.update({k: d[k] for k in d.files if k != "source_indices"})
    meta["ndvi"] = extra
    return arrays, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS)
    args = p.parse_args()
    for crop in ((args.crop,) if args.crop else CROPS):
        prepare(crop)
