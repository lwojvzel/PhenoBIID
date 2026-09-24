#!/usr/bin/env python3
"""Run statistical and classical causal yield-only baselines."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import faiss
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from multimodal_baseline import (
    CROPS,
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
from run_history_multimodal_baselines import (
    HISTORY_FEATURE_NAMES,
    build_causal_history_features,
    parse_csv,
)
from rebuild_yield_only_summaries import rebuild_summary


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmark/results/yield_only_baselines"
STATISTICAL_METHODS = (
    "global_mean",
    "grid_climatology",
    "previous_year",
    "rolling_mean_3",
    "rolling_mean_5",
    "exponential_smoothing",
    "linear_trend",
)
LEARNED_METHODS = (
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "huber",
    "linear_svr",
    "approximate_knn",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
)
ALL_METHODS = STATISTICAL_METHODS + LEARNED_METHODS


def _history_values(
    history_features: np.ndarray,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = (
        history_features[:, :5].astype(np.float64) * target_std + target_mean
    )
    observed = history_features[:, 5:10].astype(bool)
    return values, observed


def _rolling_prediction(
    values: np.ndarray,
    observed: np.ndarray,
    fallback: np.ndarray,
    width: int,
) -> np.ndarray:
    selected_values = values[:, :width]
    selected_observed = observed[:, :width]
    count = selected_observed.sum(axis=1)
    total = np.where(selected_observed, selected_values, 0.0).sum(axis=1)
    return np.divide(
        total,
        count,
        out=fallback.astype(np.float64, copy=True),
        where=count > 0,
    )


def _exponential_smoothing_predictions(
    cache: dict[str, np.ndarray],
    alpha: float,
    default: float,
) -> np.ndarray:
    years = np.asarray(cache["year"], dtype=np.int64)
    rows = np.asarray(cache["row"], dtype=np.int64)
    cols = np.asarray(cache["col"], dtype=np.int64)
    target = np.asarray(cache["target"], dtype=np.float64)
    keys = rows * 720 + cols
    level = np.full(360 * 720, default, dtype=np.float64)
    seen = np.zeros(360 * 720, dtype=bool)
    prediction = np.full(target.size, default, dtype=np.float64)
    for year in sorted(np.unique(years)):
        idx = np.flatnonzero(years == year)
        key = keys[idx]
        prediction[idx] = level[key]
        old = level[key]
        level[key] = np.where(
            seen[key], alpha * target[idx] + (1.0 - alpha) * old, target[idx]
        )
        seen[key] = True
    return prediction


def build_statistical_predictions(
    cache: dict[str, np.ndarray],
    history_features: np.ndarray,
    trend_baseline: np.ndarray,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_idx = choose_indices(cache, 0)
    val_idx = choose_indices(cache, 1)
    train_mean = float(np.asarray(cache["target"][train_idx], dtype=np.float64).mean())
    values, observed = _history_values(history_features, target_mean, target_std)
    historical_mean = history_features[:, 10].astype(np.float64) * target_std + target_mean

    alpha_candidates = (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    alpha_predictions: dict[float, np.ndarray] = {}
    alpha_scores: dict[float, float] = {}
    val_true = np.asarray(cache["target"][val_idx], dtype=np.float64)
    for alpha in alpha_candidates:
        pred = _exponential_smoothing_predictions(cache, alpha, train_mean)
        alpha_predictions[alpha] = pred
        alpha_scores[alpha] = float(
            np.sqrt(np.mean(np.square(pred[val_idx] - val_true)))
        )
    best_alpha = min(alpha_scores, key=alpha_scores.get)

    predictions = {
        "global_mean": np.full(history_features.shape[0], train_mean, dtype=np.float64),
        "grid_climatology": historical_mean,
        "previous_year": _rolling_prediction(values, observed, historical_mean, 1),
        "rolling_mean_3": _rolling_prediction(values, observed, historical_mean, 3),
        "rolling_mean_5": _rolling_prediction(values, observed, historical_mean, 5),
        "exponential_smoothing": alpha_predictions[best_alpha],
        "linear_trend": trend_baseline.astype(np.float64),
    }
    metadata = {
        "exponential_smoothing_alpha": best_alpha,
        "exponential_smoothing_validation_rmse": alpha_scores[best_alpha],
        "exponential_smoothing_search": alpha_scores,
    }
    return predictions, metadata


def _linear_pipeline(model: RegressorMixin) -> RegressorMixin:
    return make_pipeline(StandardScaler(), model)


def build_model(method: str, seed: int, n_jobs: int) -> RegressorMixin:
    if method == "linear_regression":
        return _linear_pipeline(LinearRegression(n_jobs=n_jobs))
    if method == "ridge":
        return _linear_pipeline(Ridge(alpha=10.0))
    if method == "lasso":
        return _linear_pipeline(Lasso(alpha=1.0e-3, max_iter=5000, tol=1.0e-5))
    if method == "elastic_net":
        return _linear_pipeline(
            ElasticNet(alpha=1.0e-3, l1_ratio=0.5, max_iter=5000, tol=1.0e-5)
        )
    if method == "huber":
        return _linear_pipeline(
            HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=500, tol=1.0e-5)
        )
    if method == "linear_svr":
        return _linear_pipeline(
            LinearSVR(C=1.0, epsilon=0.0, loss="squared_epsilon_insensitive", dual="auto", max_iter=5000, random_state=seed)
        )
    if method == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=20, min_samples_leaf=16, random_state=seed
        )
    if method == "random_forest":
        return RandomForestRegressor(
            n_estimators=128,
            max_depth=20,
            min_samples_leaf=8,
            max_features=0.75,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if method == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=128,
            max_depth=20,
            min_samples_leaf=8,
            max_features=0.75,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if method == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    if method == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if method == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=8.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            early_stopping_rounds=60,
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unsupported learned method: {method}")


def _fit_model(
    model: RegressorMixin,
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> None:
    if method == "lightgbm":
        model.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            eval_metric="rmse",
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
        )
    elif method == "xgboost":
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    else:
        model.fit(train_x, train_y)


def _fit_predict_knn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    n_jobs: int,
    neighbors: int = 25,
) -> tuple[np.ndarray, np.ndarray, faiss.Index, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    train_scaled = np.ascontiguousarray((train_x - mean) / std, dtype=np.float32)
    val_scaled = np.ascontiguousarray((val_x - mean) / std, dtype=np.float32)
    test_scaled = np.ascontiguousarray((test_x - mean) / std, dtype=np.float32)
    faiss.omp_set_num_threads(n_jobs)
    index = faiss.IndexHNSWFlat(train_scaled.shape[1], 32)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 96
    index.add(train_scaled)

    def query(values: np.ndarray) -> np.ndarray:
        distance, neighbor_index = index.search(values, neighbors)
        weights = 1.0 / np.maximum(distance, 1.0e-6)
        return np.sum(weights * train_y[neighbor_index], axis=1) / weights.sum(axis=1)

    return query(val_scaled), query(test_scaled), index, mean, std


def _serializable_parameters(model: RegressorMixin) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for key, value in model.get_params().items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            parameters[key] = value
        elif isinstance(value, (list, tuple)):
            parameters[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else repr(item)
                for item in value
            ]
        else:
            parameters[key] = repr(value)
    return parameters


def _save_result(
    crop: str,
    method: str,
    seed_label: str,
    cache: dict[str, np.ndarray],
    test_idx: np.ndarray,
    test_true: np.ndarray,
    test_prediction: np.ndarray,
    validation_rmse: float,
    config: dict[str, Any],
    model: RegressorMixin | None = None,
) -> dict[str, Any]:
    run_dir = RESULTS_ROOT / crop / method / f"seed_{seed_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    area_weight, latitude_weight = evaluation_context(cache, test_idx)
    metrics = regression_metrics(
        test_true, np.maximum(test_prediction, 0.0), area_weight, latitude_weight
    )
    metrics["validation_rmse"] = validation_rmse
    if model is not None:
        joblib.dump(model, run_dir / "model_best.joblib", compress=3)
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(config, run_dir / "config.json")
    write_csv(
        metrics_by_year(cache, test_idx, test_true, np.maximum(test_prediction, 0.0)),
        run_dir / "test_metrics_by_year.csv",
    )
    save_predictions(
        run_dir / "test_predictions.npz",
        cache,
        test_idx,
        test_true,
        np.maximum(test_prediction, 0.0),
    )
    print(
        f"[YIELD-ONLY] crop={crop} method={method} "
        f"val_rmse={validation_rmse:.6f} test_rmse={metrics['rmse']:.6f}",
        flush=True,
    )
    return {"crop": crop, "method": method, "seed": seed_label, **metrics}


def run_crop(
    crop: str,
    methods: tuple[str, ...],
    seed: int,
    n_jobs: int,
    force: bool,
) -> list[dict[str, Any]]:
    cache = load_cache(crop)
    stats = compute_normalization(cache)
    train_idx = choose_indices(cache, 0)
    val_idx = choose_indices(cache, 1)
    test_idx = choose_indices(cache, 2)
    history_features, trend_baseline = build_causal_history_features(
        cache, stats.target_mean, stats.target_std
    )
    statistical, statistical_meta = build_statistical_predictions(
        cache,
        history_features,
        trend_baseline,
        stats.target_mean,
        stats.target_std,
    )
    val_true = np.asarray(cache["target"][val_idx], dtype=np.float64)
    test_true = np.asarray(cache["target"][test_idx], dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for method in methods:
        seed_label = "deterministic" if method in STATISTICAL_METHODS else str(seed)
        run_dir = RESULTS_ROOT / crop / method / f"seed_{seed_label}"
        metrics_path = run_dir / "test_metrics.json"
        if (
            metrics_path.exists()
            and (run_dir / "config.json").exists()
            and (run_dir / "test_predictions.npz").exists()
            and not force
        ):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({"crop": crop, "method": method, "seed": seed_label, **metrics})
            print(f"[SKIP] {crop} {method}", flush=True)
            continue

        if method in STATISTICAL_METHODS:
            prediction = statistical[method]
            val_prediction = np.maximum(prediction[val_idx], 0.0)
            val_rmse = float(
                np.sqrt(np.mean(np.square(val_prediction - val_true)))
            )
            rows.append(
                _save_result(
                    crop,
                    method,
                    seed_label,
                    cache,
                    test_idx,
                    test_true,
                    prediction[test_idx],
                    val_rmse,
                    {
                        "crop": crop,
                        "model": method,
                        "modalities": ["historical_yield"],
                        "forecast_protocol": "rolling one-year-ahead; target-year yield excluded",
                        "train_years": [1981, 2011],
                        "validation_years": [2012, 2012],
                        "test_years": [2013, 2016],
                        **statistical_meta,
                    },
                )
            )
            continue

        train_x = np.concatenate(
            (history_features[train_idx], np.asarray(cache["context"][train_idx])),
            axis=1,
        ).astype(np.float32, copy=False)
        val_x = np.concatenate(
            (history_features[val_idx], np.asarray(cache["context"][val_idx])),
            axis=1,
        ).astype(np.float32, copy=False)
        test_x = np.concatenate(
            (history_features[test_idx], np.asarray(cache["context"][test_idx])),
            axis=1,
        ).astype(np.float32, copy=False)
        train_true = np.asarray(cache["target"][train_idx], dtype=np.float64)
        train_base = trend_baseline[train_idx].astype(np.float64)
        val_base = trend_baseline[val_idx].astype(np.float64)
        test_base = trend_baseline[test_idx].astype(np.float64)
        residual = train_true - train_base
        residual_mean = float(residual.mean())
        residual_std = float(residual.std())
        if residual_std < 1.0e-6:
            residual_std = 1.0
        train_y = ((residual - residual_mean) / residual_std).astype(np.float32)
        val_y = (
            (val_true - val_base - residual_mean) / residual_std
        ).astype(np.float32)

        started = time.time()
        if method == "approximate_knn":
            val_residual_prediction, test_residual_prediction, knn_index, knn_mean, knn_std = _fit_predict_knn(
                train_x, train_y, val_x, test_x, n_jobs
            )
            model = None
        else:
            model = build_model(method, seed, n_jobs)
            _fit_model(model, method, train_x, train_y, val_x, val_y)
            val_residual_prediction = model.predict(val_x)
            test_residual_prediction = model.predict(test_x)
        elapsed = time.time() - started
        val_prediction = val_base + val_residual_prediction * residual_std + residual_mean
        test_prediction = test_base + test_residual_prediction * residual_std + residual_mean
        val_rmse = float(
            np.sqrt(np.mean(np.square(np.maximum(val_prediction, 0.0) - val_true)))
        )
        rows.append(
            _save_result(
                crop,
                method,
                seed_label,
                cache,
                test_idx,
                test_true,
                test_prediction,
                val_rmse,
                {
                    "crop": crop,
                    "model": method,
                    "modalities": ["historical_yield"],
                    "forecast_protocol": "rolling one-year-ahead; target-year yield excluded",
                    "history_lag_years": 5,
                    "history_feature_names": list(HISTORY_FEATURE_NAMES),
                    "context_features": [
                        "sin_latitude",
                        "cos_latitude",
                        "sin_longitude",
                        "cos_longitude",
                        "normalized_year",
                    ],
                    "residual_baseline": "causal per-grid historical linear trend",
                    "train_years": [1981, 2011],
                    "validation_years": [2012, 2012],
                    "test_years": [2013, 2016],
                    "seed": seed,
                    "n_features": int(train_x.shape[1]),
                    "train_samples": int(train_idx.size),
                    "validation_samples": int(val_idx.size),
                    "test_samples": int(test_idx.size),
                    "fit_seconds": elapsed,
                    "residual_mean": residual_mean,
                    "residual_std": residual_std,
                    "estimator_parameters": (
                        {
                            "implementation": "FAISS IndexHNSWFlat",
                            "neighbors": 25,
                            "hnsw_m": 32,
                            "ef_construction": 100,
                            "ef_search": 96,
                            "weights": "inverse_squared_l2_distance",
                        }
                        if method == "approximate_knn"
                        else _serializable_parameters(model)
                    ),
                },
                model=model,
            )
        )
        if method == "approximate_knn":
            run_dir = RESULTS_ROOT / crop / method / f"seed_{seed_label}"
            faiss.write_index(knn_index, str(run_dir / "model_best.faiss"))
            np.save(run_dir / "knn_targets.npy", train_y)
            np.savez_compressed(run_dir / "feature_normalization.npz", mean=knn_mean, std=knn_std)
            del knn_index
        del model, train_x, val_x, test_x
        gc.collect()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default=",".join(CROPS))
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    crops = tuple(parse_csv(args.crops))
    methods = tuple(parse_csv(args.methods))
    invalid_crops = sorted(set(crops) - set(CROPS))
    invalid_methods = sorted(set(methods) - set(ALL_METHODS))
    if invalid_crops:
        raise SystemExit(f"Unsupported crops: {invalid_crops}")
    if invalid_methods:
        raise SystemExit(f"Unsupported methods: {invalid_methods}")

    rows: list[dict[str, Any]] = []
    for crop in crops:
        rows.extend(run_crop(crop, methods, args.seed, args.n_jobs, args.force))
        gc.collect()
    summary, completed_runs = rebuild_summary("classical")
    print(
        f"Summary: {summary} ({completed_runs} completed runs)", flush=True
    )


if __name__ == "__main__":
    main()
