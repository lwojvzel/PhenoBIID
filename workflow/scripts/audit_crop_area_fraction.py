#!/usr/bin/env python3
"""Quantify target-crop growing-area share and stratify fixed-test results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multimodal_baseline import CROPS, load_coordinates, regression_metrics


ROOT = Path(__file__).resolve().parents[1]
MIRCA_ROOT = ROOT / "Data/processed/crop_yield_growing_season"
WORLD_ROOT = ROOT / "benchmark/results/biid_fixed_protocol_validation_ensemble_v1"
OUTPUT_ROOT = ROOT / "benchmark/results/review_revision_20260904/crop_area_fraction"
TABLE_ROOT = ROOT / "visualize/paper_experiments/review_revision_20260904"
MIRCA_YEARS = (2000, 2005, 2010, 2015)
THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.10, 0.20)


def nearest_mirca_year(year: int) -> int:
    return min(MIRCA_YEARS, key=lambda value: (abs(year - value), value))


def grid_area_hectares(latitude: np.ndarray) -> np.ndarray:
    radius_m = 6_371_008.8
    half = np.deg2rad(0.25)
    delta_lon = np.deg2rad(0.5)
    center = np.deg2rad(latitude)
    area_m2 = radius_m**2 * delta_lon * (
        np.sin(center + half) - np.sin(center - half)
    )
    return area_m2 / 10_000.0


def maximum_monthly_growing_area(crop: str, year: int) -> np.ndarray:
    path = (
        MIRCA_ROOT
        / crop
        / "mirca"
        / str(year)
        / "mirca_month_area_total_0p5.npy"
    )
    values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    if values.shape != (12, 360, 720):
        raise RuntimeError(f"Unexpected monthly MIRCA shape at {path}: {values.shape}")
    return np.max(values, axis=0)


def sample_fraction(
    crop: str, years: np.ndarray, rows: np.ndarray, cols: np.ndarray, cell_area: np.ndarray
) -> np.ndarray:
    output = np.zeros(years.size, dtype=np.float64)
    for mirca_year in MIRCA_YEARS:
        selected = np.asarray(
            [nearest_mirca_year(int(year)) == mirca_year for year in years], dtype=bool
        )
        if not np.any(selected):
            continue
        area = maximum_monthly_growing_area(crop, mirca_year)
        output[selected] = (
            area[rows[selected], cols[selected]] / cell_area[rows[selected]]
        )
    return output


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    latitude, _longitude = load_coordinates()
    cell_area = grid_area_hectares(latitude)
    distribution_rows: list[dict[str, float | int | str]] = []
    threshold_rows: list[dict[str, float | int | str]] = []
    for crop in CROPS:
        with np.load(WORLD_ROOT / crop / "test_predictions.npz") as values:
            years = np.asarray(values["year"], dtype=np.int64)
            rows = np.asarray(values["row"], dtype=np.int64)
            cols = np.asarray(values["col"], dtype=np.int64)
            target = np.asarray(values["target"], dtype=np.float64)
            world = np.asarray(values["prediction"], dtype=np.float64)
            baseline = np.asarray(values["best_yield_only_prediction"], dtype=np.float64)
        fraction = sample_fraction(crop, years, rows, cols, cell_area)
        percentiles = np.percentile(fraction, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
        distribution_rows.append(
            {
                "crop": crop,
                "n_samples": int(fraction.size),
                **{
                    f"fraction_p{label}": float(value)
                    for label, value in zip(
                        (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100), percentiles
                    )
                },
                "fraction_mean": float(fraction.mean()),
                "fraction_over_one_percent": float(np.mean(fraction >= 0.01)),
                "fraction_over_ten_percent": float(np.mean(fraction >= 0.10)),
                "growing_area_fraction_over_one": float(np.mean(fraction > 1.0)),
            }
        )
        for threshold in THRESHOLDS:
            selected = fraction >= threshold
            if not np.any(selected):
                continue
            baseline_metrics = regression_metrics(target[selected], baseline[selected])
            world_metrics = regression_metrics(target[selected], world[selected])
            threshold_rows.append(
                {
                    "crop": crop,
                    "minimum_maximum_monthly_growing_area_fraction": threshold,
                    "n_samples": int(selected.sum()),
                    "retained_percent": float(100.0 * selected.mean()),
                    "baseline_rmse": baseline_metrics["rmse"],
                    "world_model_rmse": world_metrics["rmse"],
                    "gain_percent": float(
                        100.0
                        * (baseline_metrics["rmse"] - world_metrics["rmse"])
                        / baseline_metrics["rmse"]
                    ),
                    "world_model_r2": world_metrics["r2"],
                }
            )
        crop_dir = OUTPUT_ROOT / crop
        crop_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            crop_dir / "test_crop_area_fraction.npz",
            year=years,
            row=rows,
            col=cols,
            maximum_monthly_growing_area_fraction=fraction.astype(np.float32),
        )
    distribution = pd.DataFrame(distribution_rows)
    thresholds = pd.DataFrame(threshold_rows)
    distribution.to_csv(TABLE_ROOT / "table_crop_area_fraction_distribution.csv", index=False)
    thresholds.to_csv(TABLE_ROOT / "table_crop_area_fraction_stratified_performance.csv", index=False)
    print(distribution.to_string(index=False))
    print("\nThreshold performance\n", thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
