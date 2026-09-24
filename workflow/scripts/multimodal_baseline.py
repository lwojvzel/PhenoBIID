#!/usr/bin/env python3
"""Shared data, models, and evaluation utilities for multimodal baselines."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import torch
from numpy.lib.format import open_memmap
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "Data/processed/crop_yield_growing_season"
ERA5_ROOT = PROJECT_ROOT / "Data/era5land/monthly_npy_lon180_0p5deg_by_var"
LAI_ROOT = PROJECT_ROOT / "Data/processed/glass_lai_avhrr_005d/monthly_0p5"
CACHE_ROOT = PROJECT_ROOT / "benchmark/cache/multimodal_main"
RESULTS_ROOT = PROJECT_ROOT / "benchmark/results/multimodal_main"

CROPS = ("maize", "rice", "soybean", "wheat")
MIRCA_YEARS = (2000, 2005, 2010, 2015)
ERA5_VARIABLES = (
    "d2m",
    "t2m",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
    "swvl3",
    "ssrd",
    "pev",
    "u10",
    "v10",
    "sp",
    "tp",
)
MODES = tuple(f"C{idx}" for idx in range(8))
MODE_MODALITIES = {
    "C0": (),
    "C1": ("mirca",),
    "C2": ("weather",),
    "C3": ("lai",),
    "C4": ("weather", "lai"),
    "C5": ("weather", "mirca"),
    "C6": ("lai", "mirca"),
    "C7": ("weather", "lai", "mirca"),
}
CACHE_VERSION = 1
GRID_WIDTH = 720


def nearest_mirca_year(year: int) -> int:
    return min(MIRCA_YEARS, key=lambda value: (abs(year - value), value))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_coordinates() -> tuple[np.ndarray, np.ndarray]:
    candidates = (
        PROCESSED_ROOT,
        PROJECT_ROOT / "Data/GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180",
    )
    for root in candidates:
        lat_path = root / "lat.npy"
        lon_path = root / "lon.npy"
        if lat_path.exists() and lon_path.exists():
            return np.load(lat_path), np.load(lon_path)
    raise FileNotFoundError("Could not locate common lat.npy and lon.npy coordinates.")


def _crop_paths(crop: str, year: int) -> dict[str, Path]:
    mirca_year = nearest_mirca_year(year)
    mirca_root = PROCESSED_ROOT / crop / "mirca" / str(mirca_year)
    return {
        "yield": PROCESSED_ROOT / crop / "yield" / f"yield_{year}.npy",
        "lai": LAI_ROOT / f"lai_monthly_{year}.npy",
        "area_total": mirca_root / "mirca_month_area_total_0p5.npy",
        "area_ir": mirca_root / "mirca_month_area_ir_0p5.npy",
        "area_rf": mirca_root / "mirca_month_area_rf_0p5.npy",
    }


def _valid_sample_mask(crop: str, year: int, area_threshold: float) -> np.ndarray:
    paths = _crop_paths(crop, year)
    target = np.load(paths["yield"], mmap_mode="r")
    lai = np.load(paths["lai"], mmap_mode="r")
    area_total = np.load(paths["area_total"], mmap_mode="r")
    crop_month = area_total > 0.0
    lai_in_crop_month = np.isfinite(lai) & crop_month
    return (
        np.isfinite(target)
        & (target > -1.0e8)
        & (target >= 0.0)
        & (np.max(area_total, axis=0) > area_threshold)
        & (np.sum(lai_in_crop_month, axis=0) >= 3)
    )


def _cache_is_current(cache_dir: Path, area_threshold: float) -> bool:
    metadata_path = cache_dir / "metadata.json"
    required = (
        "weather.npy",
        "lai.npy",
        "lai_valid.npy",
        "mirca_monthly.npy",
        "context.npy",
        "mirca_static.npy",
        "target.npy",
        "year.npy",
        "row.npy",
        "col.npy",
        "split.npy",
        "area_weight.npy",
    )
    if not metadata_path.exists() or any(not (cache_dir / name).exists() for name in required):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        int(metadata.get("cache_version", -1)) == CACHE_VERSION
        and float(metadata.get("area_threshold_ha", -1.0)) == float(area_threshold)
    )


def build_crop_cache(
    crop: str,
    area_threshold: float = 100.0,
    force: bool = False,
) -> Path:
    if crop not in CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    cache_dir = CACHE_ROOT / crop
    if not force and _cache_is_current(cache_dir, area_threshold):
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(1981, 2017))
    counts: dict[int, int] = {}
    for year in years:
        counts[year] = int(_valid_sample_mask(crop, year, area_threshold).sum())
    total_samples = int(sum(counts.values()))
    if total_samples == 0:
        raise RuntimeError(f"No valid samples found for {crop}.")

    shapes = {
        "weather": ((total_samples, 12, len(ERA5_VARIABLES)), np.float32),
        "lai": ((total_samples, 12), np.float32),
        "lai_valid": ((total_samples, 12), np.uint8),
        "mirca_monthly": ((total_samples, 12, 3), np.float32),
        "context": ((total_samples, 5), np.float32),
        "mirca_static": ((total_samples, 3), np.float32),
        "target": ((total_samples,), np.float32),
        "year": ((total_samples,), np.int16),
        "row": ((total_samples,), np.int16),
        "col": ((total_samples,), np.int16),
        "split": ((total_samples,), np.uint8),
        "area_weight": ((total_samples,), np.float32),
    }
    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype) in shapes.items():
        arrays[name] = open_memmap(cache_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    lat, lon = load_coordinates()
    offset = 0
    yearly_rows: list[dict[str, Any]] = []
    for year in years:
        mask = _valid_sample_mask(crop, year, area_threshold)
        rows, cols = np.nonzero(mask)
        count = rows.size
        end = offset + count
        if count == 0:
            yearly_rows.append({"year": year, "n_samples": 0})
            continue

        paths = _crop_paths(crop, year)
        target = np.load(paths["yield"], mmap_mode="r")
        lai = np.load(paths["lai"], mmap_mode="r")
        area_total = np.load(paths["area_total"], mmap_mode="r")
        area_ir = np.load(paths["area_ir"], mmap_mode="r")
        area_rf = np.load(paths["area_rf"], mmap_mode="r")

        for variable_index, variable in enumerate(ERA5_VARIABLES):
            weather_path = ERA5_ROOT / variable / f"{variable}_{year}.npy"
            weather = np.load(weather_path, mmap_mode="r")
            arrays["weather"][offset:end, :, variable_index] = weather[:, rows, cols].T

        lai_values = lai[:, rows, cols].T.astype(np.float32, copy=False)
        arrays["lai"][offset:end] = lai_values
        arrays["lai_valid"][offset:end] = np.isfinite(lai_values).astype(np.uint8)

        area_values = area_total[:, rows, cols].T.astype(np.float32, copy=False)
        ir_values = area_ir[:, rows, cols].T.astype(np.float32, copy=False)
        rf_values = area_rf[:, rows, cols].T.astype(np.float32, copy=False)
        area_sum = area_values.sum(axis=1, keepdims=True, dtype=np.float64)
        area_weight = np.divide(
            area_values,
            area_sum,
            out=np.zeros_like(area_values),
            where=area_sum > 0.0,
        )
        irrigation = np.divide(
            ir_values,
            ir_values + rf_values,
            out=np.zeros_like(ir_values),
            where=(ir_values + rf_values) > 0.0,
        )
        arrays["mirca_monthly"][offset:end, :, 0] = area_values > 0.0
        arrays["mirca_monthly"][offset:end, :, 1] = area_weight
        arrays["mirca_monthly"][offset:end, :, 2] = irrigation

        lat_radians = np.deg2rad(lat[rows])
        lon_radians = np.deg2rad(lon[cols])
        arrays["context"][offset:end, 0] = np.sin(lat_radians)
        arrays["context"][offset:end, 1] = np.cos(lat_radians)
        arrays["context"][offset:end, 2] = np.sin(lon_radians)
        arrays["context"][offset:end, 3] = np.cos(lon_radians)
        arrays["context"][offset:end, 4] = (year - 1981) / (2016 - 1981)

        max_area = area_values.max(axis=1)
        annual_irrigation = np.divide(
            ir_values.sum(axis=1, dtype=np.float64),
            area_values.sum(axis=1, dtype=np.float64),
            out=np.zeros(count, dtype=np.float64),
            where=area_values.sum(axis=1, dtype=np.float64) > 0.0,
        )
        arrays["mirca_static"][offset:end, 0] = np.log1p(max_area)
        arrays["mirca_static"][offset:end, 1] = annual_irrigation
        arrays["mirca_static"][offset:end, 2] = (area_values > 0.0).sum(axis=1) / 12.0

        arrays["target"][offset:end] = target[rows, cols]
        arrays["year"][offset:end] = year
        arrays["row"][offset:end] = rows
        arrays["col"][offset:end] = cols
        arrays["area_weight"][offset:end] = max_area
        if year <= 2011:
            split_value = 0
        elif year == 2012:
            split_value = 1
        else:
            split_value = 2
        arrays["split"][offset:end] = split_value

        yearly_rows.append({"year": year, "n_samples": int(count)})
        offset = end

    for array in arrays.values():
        array.flush()
    del arrays

    split = np.load(cache_dir / "split.npy", mmap_mode="r")
    target = np.load(cache_dir / "target.npy", mmap_mode="r")
    row = np.load(cache_dir / "row.npy", mmap_mode="r")
    col = np.load(cache_dir / "col.npy", mmap_mode="r")
    summary: dict[str, Any] = {}
    for split_name, split_value in (("train", 0), ("validation", 1), ("test", 2)):
        selected = split == split_value
        keys = row[selected].astype(np.int64) * GRID_WIDTH + col[selected].astype(np.int64)
        summary[split_name] = {
            "n_samples": int(selected.sum()),
            "n_unique_grids": int(np.unique(keys).size),
            "yield_mean": float(np.mean(target[selected])),
            "yield_std": float(np.std(target[selected])),
        }

    metadata = {
        "cache_version": CACHE_VERSION,
        "crop": crop,
        "area_threshold_ha": area_threshold,
        "n_samples": total_samples,
        "years": [min(years), max(years)],
        "splits": {"train": [1981, 2011], "validation": [2012, 2012], "test": [2013, 2016]},
        "era5_variables": list(ERA5_VARIABLES),
        "mode_modalities": {key: list(value) for key, value in MODE_MODALITIES.items()},
        "summary": summary,
    }
    save_json(metadata, cache_dir / "metadata.json")
    write_csv(yearly_rows, cache_dir / "samples_by_year.csv")
    return cache_dir


def load_cache(crop: str) -> dict[str, np.ndarray]:
    cache_dir = CACHE_ROOT / crop
    if not _cache_is_current(cache_dir, 100.0):
        raise FileNotFoundError(f"Cache for {crop} is missing or stale: {cache_dir}")
    names = (
        "weather",
        "lai",
        "lai_valid",
        "mirca_monthly",
        "context",
        "mirca_static",
        "target",
        "year",
        "row",
        "col",
        "split",
        "area_weight",
    )
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in names}


@dataclass
class NormalizationStats:
    weather_mean: list[float]
    weather_std: list[float]
    lai_mean: float
    lai_std: float
    mirca_static_mean: list[float]
    mirca_static_std: list[float]
    target_mean: float
    target_std: float


def _stream_mean_std(
    array: np.ndarray,
    indices: np.ndarray,
    reduce_axes: tuple[int, ...],
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    feature_shape = tuple(size for axis, size in enumerate(array.shape[1:]) if axis + 1 not in reduce_axes)
    total_sum = np.zeros(feature_shape, dtype=np.float64)
    total_sq = np.zeros(feature_shape, dtype=np.float64)
    total_count = np.zeros(feature_shape, dtype=np.float64)
    for start in range(0, indices.size, chunk_size):
        values = np.asarray(array[indices[start : start + chunk_size]], dtype=np.float64)
        finite = np.isfinite(values)
        axes = tuple(axis for axis in reduce_axes)
        total_sum += np.where(finite, values, 0.0).sum(axis=axes)
        total_sq += np.where(finite, values * values, 0.0).sum(axis=axes)
        total_count += finite.sum(axis=axes)
    mean = np.divide(total_sum, total_count, out=np.zeros_like(total_sum), where=total_count > 0)
    variance = np.divide(total_sq, total_count, out=np.ones_like(total_sq), where=total_count > 0) - mean * mean
    std = np.sqrt(np.maximum(variance, 1.0e-12))
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean, std


def compute_normalization(cache: dict[str, np.ndarray]) -> NormalizationStats:
    train_indices = np.flatnonzero(cache["split"] == 0)
    weather_mean, weather_std = _stream_mean_std(cache["weather"], train_indices, (0, 1))

    lai_sum = 0.0
    lai_sq = 0.0
    lai_count = 0
    for start in range(0, train_indices.size, 20_000):
        idx = train_indices[start : start + 20_000]
        values = np.asarray(cache["lai"][idx], dtype=np.float64)
        valid = np.asarray(cache["lai_valid"][idx], dtype=bool)
        selected = values[valid & np.isfinite(values)]
        lai_sum += float(selected.sum())
        lai_sq += float(np.square(selected).sum())
        lai_count += int(selected.size)
    lai_mean = lai_sum / max(lai_count, 1)
    lai_var = lai_sq / max(lai_count, 1) - lai_mean * lai_mean
    lai_std = math.sqrt(max(lai_var, 1.0e-12))
    if lai_std < 1.0e-6:
        lai_std = 1.0

    static_mean, static_std = _stream_mean_std(cache["mirca_static"], train_indices, (0,))
    target_values = np.asarray(cache["target"][train_indices], dtype=np.float64)
    target_mean = float(target_values.mean())
    target_std = float(target_values.std())
    if target_std < 1.0e-6:
        target_std = 1.0
    return NormalizationStats(
        weather_mean=weather_mean.astype(float).tolist(),
        weather_std=weather_std.astype(float).tolist(),
        lai_mean=float(lai_mean),
        lai_std=float(lai_std),
        mirca_static_mean=static_mean.astype(float).tolist(),
        mirca_static_std=static_std.astype(float).tolist(),
        target_mean=target_mean,
        target_std=target_std,
    )


def choose_indices(
    cache: dict[str, np.ndarray],
    split_value: int,
    max_samples: int | None = None,
    selection_seed: int = 2026,
) -> np.ndarray:
    indices = np.flatnonzero(cache["split"] == split_value)
    if max_samples is not None and indices.size > max_samples:
        rng = np.random.default_rng(selection_seed + split_value)
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
    return indices


def build_features(
    cache: dict[str, np.ndarray],
    stats: NormalizationStats,
    mode: str,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode not in MODE_MODALITIES:
        raise ValueError(f"Unsupported modality mode: {mode}")
    modalities = MODE_MODALITIES[mode]
    dynamic_parts: list[np.ndarray] = []
    if "weather" in modalities:
        weather = np.asarray(cache["weather"][indices], dtype=np.float32)
        weather_mean = np.asarray(stats.weather_mean, dtype=np.float32)[None, None, :]
        weather_std = np.asarray(stats.weather_std, dtype=np.float32)[None, None, :]
        weather = (weather - weather_mean) / weather_std
        weather = np.where(np.isfinite(weather), weather, 0.0)
        dynamic_parts.append(weather.astype(np.float32, copy=False))
    if "lai" in modalities:
        lai = np.asarray(cache["lai"][indices], dtype=np.float32)
        lai_valid = np.asarray(cache["lai_valid"][indices], dtype=np.float32)
        lai = (lai - stats.lai_mean) / stats.lai_std
        lai = np.where((lai_valid > 0.0) & np.isfinite(lai), lai, 0.0)
        dynamic_parts.extend((lai[:, :, None], lai_valid[:, :, None]))
    if "mirca" in modalities:
        dynamic_parts.append(np.asarray(cache["mirca_monthly"][indices], dtype=np.float32))
    if not dynamic_parts:
        dynamic_parts.append(np.zeros((indices.size, 12, 1), dtype=np.float32))
    dynamic = np.concatenate(dynamic_parts, axis=2)

    static_parts = [np.asarray(cache["context"][indices], dtype=np.float32)]
    if "mirca" in modalities:
        static = np.asarray(cache["mirca_static"][indices], dtype=np.float32)
        mean = np.asarray(stats.mirca_static_mean, dtype=np.float32)[None, :]
        std = np.asarray(stats.mirca_static_std, dtype=np.float32)[None, :]
        static_parts.append((static - mean) / std)
    static_features = np.concatenate(static_parts, axis=1).astype(np.float32, copy=False)
    target = np.asarray(cache["target"][indices], dtype=np.float32)
    target_norm = ((target - stats.target_mean) / stats.target_std).astype(np.float32, copy=False)
    return dynamic, static_features, target_norm


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    area_weight: np.ndarray | None = None,
    latitude_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    centered = y_true - y_true.mean()
    ss_tot = float(np.sum(centered * centered))
    r2 = float(1.0 - np.sum(diff * diff) / ss_tot) if ss_tot > 0.0 else float("nan")
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.size > 1 and true_std > 0.0 and pred_std > 0.0
        else float("nan")
    )
    true_rank = rankdata(y_true)
    pred_rank = rankdata(y_pred)
    spearman = (
        float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if y_true.size > 1 and np.std(true_rank) > 0.0 and np.std(pred_rank) > 0.0
        else float("nan")
    )
    result = {
        "rmse": rmse,
        "mae": mae,
        "nrmse": float(rmse / max(abs(y_true.mean()), 1.0e-12)),
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "n_samples": int(y_true.size),
    }
    for prefix, weight in (("crop_area", area_weight), ("latitude", latitude_weight)):
        if weight is None:
            continue
        weight_values = np.asarray(weight, dtype=np.float64)
        valid = np.isfinite(weight_values) & (weight_values > 0.0)
        if not np.any(valid):
            continue
        normalized = weight_values[valid] / weight_values[valid].sum()
        weighted_diff = diff[valid]
        weighted_true = y_true[valid]
        weighted_pred = y_pred[valid]
        weighted_mean = float(np.sum(normalized * weighted_true))
        weighted_ss_tot = float(np.sum(normalized * np.square(weighted_true - weighted_mean)))
        weighted_ss_res = float(np.sum(normalized * np.square(weighted_diff)))
        result[f"{prefix}_rmse"] = float(np.sqrt(weighted_ss_res))
        result[f"{prefix}_mae"] = float(np.sum(normalized * np.abs(weighted_diff)))
        result[f"{prefix}_r2"] = (
            float(1.0 - weighted_ss_res / weighted_ss_tot) if weighted_ss_tot > 0.0 else float("nan")
        )
        weighted_true_centered = weighted_true - np.sum(normalized * weighted_true)
        weighted_pred_centered = weighted_pred - np.sum(normalized * weighted_pred)
        covariance = float(np.sum(normalized * weighted_true_centered * weighted_pred_centered))
        true_var = float(np.sum(normalized * np.square(weighted_true_centered)))
        pred_var = float(np.sum(normalized * np.square(weighted_pred_centered)))
        result[f"{prefix}_pearson"] = (
            covariance / math.sqrt(true_var * pred_var) if true_var > 0.0 and pred_var > 0.0 else float("nan")
        )
    return result


def evaluation_context(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat, _lon = load_coordinates()
    area_weight = np.asarray(cache["area_weight"][indices], dtype=np.float64)
    latitude_weight = np.cos(np.deg2rad(lat[np.asarray(cache["row"][indices], dtype=np.int64)]))
    return area_weight, latitude_weight


def save_predictions(
    path: Path,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        year=np.asarray(cache["year"][indices]),
        row=np.asarray(cache["row"][indices]),
        col=np.asarray(cache["col"][indices]),
        area_weight=np.asarray(cache["area_weight"][indices]),
        y_true=np.asarray(y_true, dtype=np.float32),
        y_pred=np.asarray(y_pred, dtype=np.float32),
    )


def metrics_by_year(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    years = np.asarray(cache["year"][indices])
    rows: list[dict[str, Any]] = []
    for year in sorted(np.unique(years)):
        selected = years == year
        metrics = regression_metrics(y_true[selected], y_pred[selected])
        rows.append({"year": int(year), **metrics})
    return rows


def run_historical_baselines(crop: str) -> list[dict[str, Any]]:
    cache = load_cache(crop)
    train_idx = choose_indices(cache, 0)
    test_idx = choose_indices(cache, 2)
    train_y = np.asarray(cache["target"][train_idx], dtype=np.float64)
    test_y = np.asarray(cache["target"][test_idx], dtype=np.float64)
    train_year = np.asarray(cache["year"][train_idx], dtype=np.float64)
    test_year = np.asarray(cache["year"][test_idx], dtype=np.float64)
    train_key = (
        np.asarray(cache["row"][train_idx], dtype=np.int64) * GRID_WIDTH
        + np.asarray(cache["col"][train_idx], dtype=np.int64)
    )
    test_key = (
        np.asarray(cache["row"][test_idx], dtype=np.int64) * GRID_WIDTH
        + np.asarray(cache["col"][test_idx], dtype=np.int64)
    )
    n_grid = 360 * GRID_WIDTH
    count = np.bincount(train_key, minlength=n_grid).astype(np.float64)
    sum_y = np.bincount(train_key, weights=train_y, minlength=n_grid)
    grid_mean = np.divide(sum_y, count, out=np.full(n_grid, train_y.mean()), where=count > 0)

    sum_t = np.bincount(train_key, weights=train_year, minlength=n_grid)
    sum_tt = np.bincount(train_key, weights=train_year * train_year, minlength=n_grid)
    sum_ty = np.bincount(train_key, weights=train_year * train_y, minlength=n_grid)
    denominator = count * sum_tt - sum_t * sum_t
    slope = np.divide(
        count * sum_ty - sum_t * sum_y,
        denominator,
        out=np.zeros(n_grid, dtype=np.float64),
        where=np.abs(denominator) > 1.0e-12,
    )
    intercept = np.divide(
        sum_y - slope * sum_t,
        count,
        out=np.full(n_grid, train_y.mean()),
        where=count > 0,
    )

    all_idx = np.concatenate((train_idx, choose_indices(cache, 1), test_idx))
    all_lookup = {
        (int(cache["year"][idx]), int(cache["row"][idx]) * GRID_WIDTH + int(cache["col"][idx])): float(
            cache["target"][idx]
        )
        for idx in all_idx
    }
    persistence = np.asarray(
        [
            all_lookup.get((int(year) - 1, int(key)), grid_mean[int(key)])
            for year, key in zip(test_year, test_key)
        ],
        dtype=np.float64,
    )
    predictions = {
        "A0_global_mean": np.full_like(test_y, train_y.mean()),
        "A1_grid_climatology": grid_mean[test_key],
        "A2_grid_linear_trend": intercept[test_key] + slope[test_key] * test_year,
        "A3_previous_year": persistence,
    }

    area_weight, latitude_weight = evaluation_context(cache, test_idx)
    summary_rows: list[dict[str, Any]] = []
    for experiment_id, prediction in predictions.items():
        run_dir = RESULTS_ROOT / crop / experiment_id / "seed_deterministic"
        metrics = regression_metrics(test_y, prediction, area_weight, latitude_weight)
        save_json(metrics, run_dir / "test_metrics.json")
        write_csv(metrics_by_year(cache, test_idx, test_y, prediction), run_dir / "test_metrics_by_year.csv")
        save_predictions(run_dir / "test_predictions.npz", cache, test_idx, test_y, prediction)
        save_json(
            {
                "crop": crop,
                "experiment_id": experiment_id,
                "train_years": [1981, 2011],
                "test_years": [2013, 2016],
            },
            run_dir / "config.json",
        )
        summary_rows.append(
            {"crop": crop, "model": experiment_id, "mode": "history", "seed": "deterministic", **metrics}
        )
    return summary_rows


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        static_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _output, hidden = self.gru(sequence)
        representation = hidden[-1]
        return self.head(torch.cat((representation, static), dim=1)).squeeze(1)


@torch.no_grad()
def _predict_gru(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for sequence, static, target in loader:
        sequence = sequence.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        pred = model(sequence, static)
        targets.append(target.numpy())
        predictions.append(pred.cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def _make_loader(
    dynamic: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(dynamic),
        torch.from_numpy(static),
        torch.from_numpy(target),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def run_gru(
    crop: str,
    mode: str,
    seed: int,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    cache = load_cache(crop)
    stats = compute_normalization(cache)
    train_idx = choose_indices(cache, 0, max_train_samples)
    val_idx = choose_indices(cache, 1, max_eval_samples)
    test_idx = choose_indices(cache, 2, max_eval_samples)

    train_dynamic, train_static, train_target = build_features(cache, stats, mode, train_idx)
    val_dynamic, val_static, val_target = build_features(cache, stats, mode, val_idx)
    test_dynamic, test_static, test_target = build_features(cache, stats, mode, test_idx)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    train_loader = _make_loader(
        train_dynamic, train_static, train_target, batch_size, True, use_amp
    )
    val_loader = _make_loader(val_dynamic, val_static, val_target, batch_size, False, use_amp)
    test_loader = _make_loader(test_dynamic, test_static, test_target, batch_size, False, use_amp)

    model = GRURegressor(
        input_size=train_dynamic.shape[2],
        static_size=train_static.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_dir = RESULTS_ROOT / crop / f"gru_{mode}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for sequence, static, target in train_loader:
            sequence = sequence.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(sequence, static)
                loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * target.shape[0]
            total_count += target.shape[0]

        val_true_norm, val_pred_norm = _predict_gru(model, val_loader, device)
        val_true = val_true_norm * stats.target_std + stats.target_mean
        val_pred = val_pred_norm * stats.target_std + stats.target_mean
        val_rmse = float(np.sqrt(np.mean(np.square(val_pred - val_true))))
        train_loss = total_loss / max(total_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_rmse": val_rmse})
        print(
            f"[GRU] crop={crop} mode={mode} seed={seed} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_rmse={val_rmse:.6f}",
            flush=True,
        )
        if val_rmse < best_val_rmse - 1.0e-6:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("GRU training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_true_norm, test_pred_norm = _predict_gru(model, test_loader, device)
    test_true = test_true_norm * stats.target_std + stats.target_mean
    test_pred = test_pred_norm * stats.target_std + stats.target_mean
    area_weight, latitude_weight = evaluation_context(cache, test_idx)
    metrics = regression_metrics(test_true, test_pred, area_weight, latitude_weight)
    metrics.update({"best_epoch": best_epoch, "best_validation_rmse": best_val_rmse})

    torch.save(best_state, run_dir / "model_best.pt")
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(asdict(stats), run_dir / "normalization.json")
    write_csv(history, run_dir / "history.csv")
    write_csv(metrics_by_year(cache, test_idx, test_true, test_pred), run_dir / "test_metrics_by_year.csv")
    save_predictions(run_dir / "test_predictions.npz", cache, test_idx, test_true, test_pred)
    save_json(
        {
            "crop": crop,
            "model": "gru",
            "mode": mode,
            "modalities": list(MODE_MODALITIES[mode]),
            "seed": seed,
            "device": str(device),
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "train_samples": int(train_idx.size),
            "validation_samples": int(val_idx.size),
            "test_samples": int(test_idx.size),
            "input_size": int(train_dynamic.shape[2]),
            "static_size": int(train_static.shape[1]),
        },
        run_dir / "config.json",
    )
    return {"crop": crop, "model": "gru", "mode": mode, "seed": seed, **metrics}


def run_hgb(
    crop: str,
    mode: str,
    seed: int = 42,
    max_iter: int = 100,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    cache = load_cache(crop)
    stats = compute_normalization(cache)
    train_idx = choose_indices(cache, 0, max_train_samples)
    val_idx = choose_indices(cache, 1, max_eval_samples)
    test_idx = choose_indices(cache, 2, max_eval_samples)
    train_dynamic, train_static, train_target = build_features(cache, stats, mode, train_idx)
    val_dynamic, val_static, val_target = build_features(cache, stats, mode, val_idx)
    test_dynamic, test_static, test_target = build_features(cache, stats, mode, test_idx)
    train_features = np.concatenate((train_dynamic.reshape(train_idx.size, -1), train_static), axis=1)
    val_features = np.concatenate((val_dynamic.reshape(val_idx.size, -1), val_static), axis=1)
    test_features = np.concatenate((test_dynamic.reshape(test_idx.size, -1), test_static), axis=1)

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
    model.fit(train_features, train_target)
    val_pred_norm = model.predict(val_features)
    test_pred_norm = model.predict(test_features)
    val_true = val_target * stats.target_std + stats.target_mean
    val_pred = val_pred_norm * stats.target_std + stats.target_mean
    test_true = test_target * stats.target_std + stats.target_mean
    test_pred = test_pred_norm * stats.target_std + stats.target_mean
    val_rmse = float(np.sqrt(np.mean(np.square(val_pred - val_true))))
    area_weight, latitude_weight = evaluation_context(cache, test_idx)
    metrics = regression_metrics(test_true, test_pred, area_weight, latitude_weight)
    metrics["validation_rmse"] = val_rmse

    run_dir = RESULTS_ROOT / crop / f"hgb_{mode}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model_best.joblib", compress=3)
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(asdict(stats), run_dir / "normalization.json")
    write_csv(metrics_by_year(cache, test_idx, test_true, test_pred), run_dir / "test_metrics_by_year.csv")
    save_predictions(run_dir / "test_predictions.npz", cache, test_idx, test_true, test_pred)
    save_json(
        {
            "crop": crop,
            "model": "hgb",
            "mode": mode,
            "modalities": list(MODE_MODALITIES[mode]),
            "seed": seed,
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
        f"[HGB] crop={crop} mode={mode} val_rmse={val_rmse:.6f} "
        f"test_rmse={metrics['rmse']:.6f}",
        flush=True,
    )
    return {"crop": crop, "model": "hgb", "mode": mode, "seed": seed, **metrics}


def collect_result_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not RESULTS_ROOT.exists():
        return rows
    for metrics_path in sorted(RESULTS_ROOT.glob("*/*/seed_*/test_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        crop = metrics_path.parents[2].name
        experiment = metrics_path.parents[1].name
        rows.append(
            {
                "crop": crop,
                "experiment": experiment,
                "model": config.get("model", experiment.split("_", 1)[0]),
                "mode": config.get("mode", config.get("experiment_id", "history")),
                "seed": config.get("seed", run_dir.name.removeprefix("seed_")),
                **metrics,
            }
        )
    return rows


def write_summary() -> Path:
    rows = collect_result_rows()
    path = RESULTS_ROOT / "summary" / "p0_results.csv"
    write_csv(rows, path)
    return path
