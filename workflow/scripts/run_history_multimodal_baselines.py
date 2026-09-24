#!/usr/bin/env python3
"""Run nested historical-yield, weather, and LAI residual baselines."""

from __future__ import annotations

import argparse
import gc
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from multimodal_baseline import (
    CROPS,
    GRID_WIDTH,
    RESULTS_ROOT,
    build_features,
    choose_indices,
    compute_normalization,
    evaluation_context,
    load_cache,
    metrics_by_year,
    regression_metrics,
    save_json,
    save_predictions,
    write_csv,
)


INPUT_CONFIGS = {
    "history_yield": {
        "base_mode": None,
        "modalities": ("historical_yield",),
        "label": "Historical yield",
    },
    "history_yield_era5": {
        "base_mode": "C2",
        "modalities": ("historical_yield", "era5_climate"),
        "label": "Historical yield + ERA5 climate",
    },
    "history_yield_era5_lai": {
        "base_mode": "C4",
        "modalities": ("historical_yield", "era5_climate", "lai_remote_sensing"),
        "label": "Historical yield + ERA5 climate + LAI",
    },
}

HISTORY_FEATURE_NAMES = (
    "yield_lag_1_normalized",
    "yield_lag_2_normalized",
    "yield_lag_3_normalized",
    "yield_lag_4_normalized",
    "yield_lag_5_normalized",
    "yield_lag_1_observed",
    "yield_lag_2_observed",
    "yield_lag_3_observed",
    "yield_lag_4_observed",
    "yield_lag_5_observed",
    "historical_mean_normalized",
    "historical_trend_prediction_normalized",
    "historical_std_normalized",
    "historical_trend_slope_normalized",
    "historical_observation_count_fraction",
)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def build_causal_history_features(
    cache: dict[str, np.ndarray],
    target_mean: float,
    target_std: float,
    lag_years: int = 5,
    min_trend_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build features using only yields strictly before each sample's year."""
    if lag_years != 5:
        raise ValueError("The frozen feature schema currently expects five lag years.")

    years = np.asarray(cache["year"], dtype=np.int64)
    rows = np.asarray(cache["row"], dtype=np.int64)
    cols = np.asarray(cache["col"], dtype=np.int64)
    targets = np.asarray(cache["target"], dtype=np.float64)
    keys = rows * GRID_WIDTH + cols
    n_grid = 360 * GRID_WIDTH
    n_samples = targets.size

    features = np.zeros((n_samples, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    baseline = np.full(n_samples, target_mean, dtype=np.float32)

    count = np.zeros(n_grid, dtype=np.int16)
    sum_y = np.zeros(n_grid, dtype=np.float64)
    sum_yy = np.zeros(n_grid, dtype=np.float64)
    sum_t = np.zeros(n_grid, dtype=np.float64)
    sum_tt = np.zeros(n_grid, dtype=np.float64)
    sum_ty = np.zeros(n_grid, dtype=np.float64)
    lag_buffers = [np.full(n_grid, np.nan, dtype=np.float32) for _ in range(lag_years)]
    lag_buffer_years = np.full(lag_years, -1, dtype=np.int64)
    first_year = int(years.min())
    total_year_span = max(int(years.max()) - first_year, 1)

    for year in sorted(np.unique(years)):
        sample_idx = np.flatnonzero(years == year)
        sample_keys = keys[sample_idx]
        sample_count = count[sample_keys].astype(np.float64)

        for lag in range(1, lag_years + 1):
            expected_year = int(year) - lag
            slot = expected_year % lag_years
            if lag_buffer_years[slot] == expected_year:
                lag_values = lag_buffers[slot][sample_keys].astype(np.float64)
                observed = np.isfinite(lag_values)
                features[sample_idx, lag - 1] = np.where(
                    observed,
                    (lag_values - target_mean) / target_std,
                    0.0,
                )
                features[sample_idx, lag_years + lag - 1] = observed.astype(np.float32)

        historical_mean = np.divide(
            sum_y[sample_keys],
            sample_count,
            out=np.full(sample_idx.size, target_mean, dtype=np.float64),
            where=sample_count > 0,
        )
        denominator = (
            sample_count * sum_tt[sample_keys]
            - np.square(sum_t[sample_keys])
        )
        slope = np.divide(
            sample_count * sum_ty[sample_keys]
            - sum_t[sample_keys] * sum_y[sample_keys],
            denominator,
            out=np.zeros(sample_idx.size, dtype=np.float64),
            where=(
                (sample_count >= min_trend_points)
                & (np.abs(denominator) > 1.0e-12)
            ),
        )
        time_value = float(int(year) - first_year)
        intercept = np.divide(
            sum_y[sample_keys] - slope * sum_t[sample_keys],
            sample_count,
            out=np.full(sample_idx.size, target_mean, dtype=np.float64),
            where=sample_count > 0,
        )
        trend_prediction = np.where(
            sample_count >= min_trend_points,
            intercept + slope * time_value,
            historical_mean,
        )
        trend_prediction = np.maximum(trend_prediction, 0.0)
        historical_variance = np.divide(
            sum_yy[sample_keys],
            sample_count,
            out=np.zeros(sample_idx.size, dtype=np.float64),
            where=sample_count > 0,
        ) - np.square(historical_mean)
        historical_std = np.sqrt(np.maximum(historical_variance, 0.0))

        features[sample_idx, 10] = (historical_mean - target_mean) / target_std
        features[sample_idx, 11] = (trend_prediction - target_mean) / target_std
        features[sample_idx, 12] = historical_std / target_std
        features[sample_idx, 13] = slope / target_std
        features[sample_idx, 14] = sample_count / total_year_span
        baseline[sample_idx] = trend_prediction.astype(np.float32)

        current_time = float(int(year) - first_year)
        current_targets = targets[sample_idx]
        count[sample_keys] += 1
        sum_y[sample_keys] += current_targets
        sum_yy[sample_keys] += np.square(current_targets)
        sum_t[sample_keys] += current_time
        sum_tt[sample_keys] += current_time * current_time
        sum_ty[sample_keys] += current_time * current_targets

        current_slot = int(year) % lag_years
        lag_buffers[current_slot].fill(np.nan)
        lag_buffers[current_slot][sample_keys] = current_targets.astype(np.float32)
        lag_buffer_years[current_slot] = int(year)

    return features, baseline


def build_model_features(
    cache: dict[str, np.ndarray],
    stats: Any,
    history_features: np.ndarray,
    indices: np.ndarray,
    input_config: str,
) -> np.ndarray:
    config = INPUT_CONFIGS[input_config]
    parts = [
        np.asarray(history_features[indices], dtype=np.float32),
        np.asarray(cache["context"][indices], dtype=np.float32),
    ]
    base_mode = config["base_mode"]
    if base_mode is not None:
        dynamic, _static, _target = build_features(cache, stats, base_mode, indices)
        parts.append(dynamic.reshape(indices.size, -1))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def run_one(
    crop: str,
    input_config: str,
    seed: int,
    max_iter: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> dict[str, Any]:
    cache = load_cache(crop)
    stats = compute_normalization(cache)
    train_idx = choose_indices(cache, 0, max_train_samples)
    val_idx = choose_indices(cache, 1, max_eval_samples)
    test_idx = choose_indices(cache, 2, max_eval_samples)
    history_features, historical_baseline = build_causal_history_features(
        cache,
        target_mean=stats.target_mean,
        target_std=stats.target_std,
    )

    train_features = build_model_features(
        cache, stats, history_features, train_idx, input_config
    )
    val_features = build_model_features(
        cache, stats, history_features, val_idx, input_config
    )
    test_features = build_model_features(
        cache, stats, history_features, test_idx, input_config
    )

    train_true = np.asarray(cache["target"][train_idx], dtype=np.float64)
    val_true = np.asarray(cache["target"][val_idx], dtype=np.float64)
    test_true = np.asarray(cache["target"][test_idx], dtype=np.float64)
    train_base = historical_baseline[train_idx].astype(np.float64)
    val_base = historical_baseline[val_idx].astype(np.float64)
    test_base = historical_baseline[test_idx].astype(np.float64)
    train_residual = train_true - train_base
    residual_mean = float(train_residual.mean())
    residual_std = float(train_residual.std())
    if residual_std < 1.0e-6:
        residual_std = 1.0
    train_residual_norm = (train_residual - residual_mean) / residual_std

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
        verbose=1,
    )
    model.fit(train_features, train_residual_norm)
    val_prediction = (
        val_base + model.predict(val_features) * residual_std + residual_mean
    )
    test_prediction = (
        test_base + model.predict(test_features) * residual_std + residual_mean
    )

    val_rmse = float(np.sqrt(np.mean(np.square(val_prediction - val_true))))
    area_weight, latitude_weight = evaluation_context(cache, test_idx)
    metrics = regression_metrics(
        test_true,
        test_prediction,
        area_weight,
        latitude_weight,
    )
    metrics["validation_rmse"] = val_rmse

    run_dir = RESULTS_ROOT / crop / f"hgb_{input_config}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model_best.joblib", compress=3)
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "target_mean": stats.target_mean,
            "target_std": stats.target_std,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
        },
        run_dir / "normalization.json",
    )
    write_csv(
        metrics_by_year(cache, test_idx, test_true, test_prediction),
        run_dir / "test_metrics_by_year.csv",
    )
    save_predictions(
        run_dir / "test_predictions.npz",
        cache,
        test_idx,
        test_true,
        test_prediction,
    )
    save_json(
        {
            "crop": crop,
            "model": "hgb_history_residual",
            "mode": input_config,
            "input_label": INPUT_CONFIGS[input_config]["label"],
            "modalities": list(INPUT_CONFIGS[input_config]["modalities"]),
            "seed": seed,
            "forecast_protocol": (
                "rolling_one_year_ahead; every sample uses only yields from "
                "years strictly earlier than its target year"
            ),
            "history_lag_years": 5,
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
            "residual_baseline": "causal per-grid historical linear trend",
            "min_points_for_historical_trend": 5,
            "train_years": [1981, 2011],
            "validation_years": [2012, 2012],
            "test_years": [2013, 2016],
            "max_iter": max_iter,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "train_samples": int(train_idx.size),
            "validation_samples": int(val_idx.size),
            "test_samples": int(test_idx.size),
            "n_features": int(train_features.shape[1]),
        },
        run_dir / "config.json",
    )
    print(
        f"[HISTORY-HGB] crop={crop} input={input_config} "
        f"val_rmse={val_rmse:.6f} test_rmse={metrics['rmse']:.6f} "
        f"test_r2={metrics['r2']:.6f}",
        flush=True,
    )
    return {
        "crop": crop,
        "model": "hgb_history_residual",
        "mode": input_config,
        "seed": seed,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default="maize,rice,soybean,wheat")
    parser.add_argument(
        "--inputs",
        default="history_yield,history_yield_era5,history_yield_era5_lai",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    args = parser.parse_args()

    crops = parse_csv(args.crops)
    input_configs = parse_csv(args.inputs)
    invalid_crops = sorted(set(crops) - set(CROPS))
    invalid_inputs = sorted(set(input_configs) - set(INPUT_CONFIGS))
    if invalid_crops:
        raise SystemExit(f"Unsupported crops: {invalid_crops}")
    if invalid_inputs:
        raise SystemExit(f"Unsupported input configurations: {invalid_inputs}")
    max_train_samples = (
        None if args.max_train_samples <= 0 else args.max_train_samples
    )
    max_eval_samples = (
        None if args.max_eval_samples <= 0 else args.max_eval_samples
    )

    rows: list[dict[str, Any]] = []
    for crop in crops:
        for input_config in input_configs:
            rows.append(
                run_one(
                    crop=crop,
                    input_config=input_config,
                    seed=args.seed,
                    max_iter=args.max_iter,
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples,
                )
            )
            gc.collect()

    crop_tag = "-".join(crops)
    output_path = (
        RESULTS_ROOT / "summary" / f"history_multimodal_hgb_{crop_tag}.csv"
    )
    write_csv(rows, output_path)
    print(f"Nested history multimodal summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
