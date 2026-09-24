from __future__ import annotations
import csv, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.lib.format import open_memmap
PROJECT_ROOT = Path(__import__("os").environ["PHENOBIID_WORKSPACE"]).resolve()

PROCESSED_ROOT = PROJECT_ROOT / "Data/processed/crop_yield_growing_season"

ERA5_ROOT = PROJECT_ROOT / "Data/era5land/monthly_npy_lon180_0p5deg_by_var"

LAI_ROOT = PROJECT_ROOT / "Data/processed/glass_lai_avhrr_005d/monthly_0p5"

CACHE_ROOT = PROJECT_ROOT / "benchmark/cache/multimodal_main"

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
