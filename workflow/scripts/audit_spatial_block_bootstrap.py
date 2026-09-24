#!/usr/bin/env python3
"""Paired spatial-block bootstrap for the fixed-split yield comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multimodal_baseline import CROPS


ROOT = Path(__file__).resolve().parents[1]
WORLD_ROOT = ROOT / "benchmark/results/biid_fixed_protocol_validation_ensemble_v1"
TABLE_ROOT = ROOT / "visualize/paper_experiments/review_revision_20260904"
BLOCK_DEGREES = (10, 20)
N_BOOTSTRAP = 5_000


def block_bootstrap(
    target: np.ndarray,
    baseline: np.ndarray,
    model: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    block_degrees: int,
    seed: int,
) -> tuple[float, float, float, int]:
    cells_per_block = block_degrees * 2
    lon_blocks = 360 // block_degrees
    group = (row.astype(np.int64) // cells_per_block) * lon_blocks + (
        col.astype(np.int64) // cells_per_block
    )
    unique, inverse = np.unique(group, return_inverse=True)
    baseline_sse = np.bincount(
        inverse, weights=np.square(baseline - target), minlength=unique.size
    )
    model_sse = np.bincount(
        inverse, weights=np.square(model - target), minlength=unique.size
    )
    count = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for start in range(0, N_BOOTSTRAP, 250):
        stop = min(start + 250, N_BOOTSTRAP)
        sampled = rng.integers(0, unique.size, size=(stop - start, unique.size))
        sampled_count = count[sampled].sum(axis=1)
        baseline_rmse = np.sqrt(baseline_sse[sampled].sum(axis=1) / sampled_count)
        model_rmse = np.sqrt(model_sse[sampled].sum(axis=1) / sampled_count)
        gains[start:stop] = 100.0 * (baseline_rmse - model_rmse) / baseline_rmse
    point_baseline = float(np.sqrt(np.mean(np.square(baseline - target))))
    point_model = float(np.sqrt(np.mean(np.square(model - target))))
    point = 100.0 * (point_baseline - point_model) / point_baseline
    low, high = np.percentile(gains, [2.5, 97.5])
    return point, float(low), float(high), int(unique.size)


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for crop_index, crop in enumerate(CROPS):
        with np.load(WORLD_ROOT / crop / "test_predictions.npz") as values:
            target = np.asarray(values["target"], dtype=np.float64)
            baseline = np.asarray(values["best_yield_only_prediction"], dtype=np.float64)
            model = np.asarray(values["prediction"], dtype=np.float64)
            grid_row = np.asarray(values["row"], dtype=np.int64)
            grid_col = np.asarray(values["col"], dtype=np.int64)
        for block_degrees in BLOCK_DEGREES:
            point, low, high, n_blocks = block_bootstrap(
                target,
                baseline,
                model,
                grid_row,
                grid_col,
                block_degrees,
                seed=20260904 + crop_index * 100 + block_degrees,
            )
            rows.append(
                {
                    "crop": crop,
                    "block_degrees": block_degrees,
                    "n_spatial_blocks": n_blocks,
                    "n_samples": int(target.size),
                    "gain_percent": point,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(low > 0.0),
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_ROOT / "table_spatial_block_bootstrap.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
