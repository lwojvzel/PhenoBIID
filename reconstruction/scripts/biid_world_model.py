from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
from multimodal_baseline import (CACHE_ROOT as MULTIMODAL_CACHE_ROOT, CROPS, LAI_ROOT, PROCESSED_ROOT, compute_normalization, load_cache, nearest_mirca_year, save_json)
from run_history_multimodal_baselines import build_causal_history_features
PROJECT_ROOT = Path(__import__("os").environ["PHENOBIID_WORKSPACE"]).resolve()

WORLD_CACHE_ROOT = PROJECT_ROOT / "benchmark/cache/biid_world_model"

LAI_REL_ROOT = PROJECT_ROOT / "Data/processed/glass_lai_avhrr_005d/crops"

WORLD_CACHE_VERSION = 2

@dataclass(frozen=True)
class WorldNormalizationStats:
    weather_mean: list[float]
    weather_std: list[float]
    lai_mean: float
    lai_std: float
    target_mean: float
    target_std: float
    residual_mean: float
    residual_std: float

def world_cache_dir(crop: str) -> Path:
    return WORLD_CACHE_ROOT / crop

def _world_cache_is_current(crop: str) -> bool:
    root = world_cache_dir(crop)
    metadata_path = root / "metadata.json"
    required = (
        "source_indices.npy",
        "weather_rel.npy",
        "target_lai_rel.npy",
        "relative_valid.npy",
        "relative_weight.npy",
        "source_month.npy",
        "previous_lai.npy",
        "previous_lai_valid.npy",
        "history_features.npy",
        "historical_baseline.npy",
    )
    if not metadata_path.exists() or any(not (root / name).exists() for name in required):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("cache_version", -1)) == WORLD_CACHE_VERSION

def build_world_cache(crop: str, force: bool = False) -> Path:
    """Build only the consecutive-year additions to the frozen multimodal cache."""
    if crop not in CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    root = world_cache_dir(crop)
    if not force and _world_cache_is_current(crop):
        return root

    cache = load_cache(crop)
    normalization = compute_normalization(cache)
    years = np.asarray(cache["year"], dtype=np.int64)
    rows = np.asarray(cache["row"], dtype=np.int64)
    cols = np.asarray(cache["col"], dtype=np.int64)
    source_indices = np.flatnonzero(years >= 1982).astype(np.int64)

    n_samples = int(source_indices.size)
    array_specs = {
        "weather_rel": ((n_samples, 12, 13), np.float32),
        "target_lai_rel": ((n_samples, 12), np.float32),
        "relative_valid": ((n_samples, 12), np.uint8),
        "relative_weight": ((n_samples, 12), np.float32),
        "source_month": ((n_samples, 12), np.uint8),
        "previous_lai": ((n_samples, 12), np.float32),
        "previous_lai_valid": ((n_samples, 12), np.uint8),
    }
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in array_specs.items()
    }
    for year in range(1982, 2017):
        local = np.flatnonzero(years[source_indices] == year)
        if local.size == 0:
            continue
        idx = source_indices[local]
        selected_rows = rows[idx]
        selected_cols = cols[idx]
        mirca_year = nearest_mirca_year(year)
        mirca_root = PROCESSED_ROOT / crop / "mirca" / str(mirca_year)

        weather_map = np.load(
            PROCESSED_ROOT / crop / "era5_rel" / f"era5_rel_{year}.npy",
            mmap_mode="r",
        )
        arrays["weather_rel"][local] = np.asarray(
            weather_map[:, :, selected_rows, selected_cols], dtype=np.float32
        ).transpose(2, 1, 0)

        target_lai_map = np.load(
            LAI_REL_ROOT / crop / "lai_rel" / f"lai_rel_{year}.npy",
            mmap_mode="r",
        )
        arrays["target_lai_rel"][local] = np.asarray(
            target_lai_map[:, selected_rows, selected_cols], dtype=np.float32
        ).T

        relative_valid = np.asarray(
            np.load(mirca_root / "valid_rel.npy", mmap_mode="r")[
                :, selected_rows, selected_cols
            ],
            dtype=np.uint8,
        ).T
        relative_weight = np.asarray(
            np.load(mirca_root / "weight_rel.npy", mmap_mode="r")[
                :, selected_rows, selected_cols
            ],
            dtype=np.float32,
        ).T
        source_month = np.asarray(
            np.load(mirca_root / "src_rel.npy", mmap_mode="r")[
                :, selected_rows, selected_cols
            ],
            dtype=np.uint8,
        ).T
        arrays["relative_valid"][local] = relative_valid
        arrays["relative_weight"][local] = relative_weight
        arrays["source_month"][local] = source_month

        previous_natural_map = np.load(
            LAI_ROOT / f"lai_monthly_{year - 1}.npy", mmap_mode="r"
        )
        previous_natural = np.asarray(
            previous_natural_map[:, selected_rows, selected_cols], dtype=np.float32
        ).T
        safe_source = np.clip(source_month.astype(np.int64), 0, 11)
        previous_relative = np.take_along_axis(
            previous_natural, safe_source, axis=1
        )
        previous_valid = (
            (relative_valid > 0)
            & (source_month != 255)
            & np.isfinite(previous_relative)
        )
        arrays["previous_lai"][local] = np.where(
            previous_valid, previous_relative, np.nan
        )
        arrays["previous_lai_valid"][local] = previous_valid.astype(np.uint8)

    for array in arrays.values():
        array.flush()
    del arrays

    history_features, historical_baseline = build_causal_history_features(
        cache,
        target_mean=normalization.target_mean,
        target_std=normalization.target_std,
    )
    np.save(root / "source_indices.npy", source_indices)
    np.save(
        root / "history_features.npy",
        np.asarray(history_features[source_indices], dtype=np.float32),
    )
    np.save(
        root / "historical_baseline.npy",
        np.asarray(historical_baseline[source_indices], dtype=np.float32),
    )
    save_json(
        {
            "cache_version": WORLD_CACHE_VERSION,
            "crop": crop,
            "source_cache": str(MULTIMODAL_CACHE_ROOT / crop),
            "target_years": [1982, 2016],
            "n_samples": int(source_indices.size),
            "weather_shape": [int(source_indices.size), 12, 13],
            "previous_lai_shape": [int(source_indices.size), 12],
            "history_feature_shape": [int(source_indices.size), 15],
            "time_axis": "MIRCA-packed relative phenology slots",
            "time_axis_caveat": (
                "Active natural months are extracted in calendar order and left aligned. "
                "Cross-year planting-to-harvest rotation is not repaired in this cache."
            ),
            "note": (
                "MIRCA supplies relative-slot mapping, validity, and loss weights. Its area "
                "values are not passed as a predictive modality. Previous-year LAI is "
                "reordered with the target year's mapping to keep slots comparable."
            ),
        },
        root / "metadata.json",
    )
    return root

def load_world_cache(crop: str) -> dict[str, np.ndarray]:
    if not _world_cache_is_current(crop):
        raise FileNotFoundError(f"World cache is missing or stale for {crop}")
    root = world_cache_dir(crop)
    names = (
        "source_indices",
        "weather_rel",
        "target_lai_rel",
        "relative_valid",
        "relative_weight",
        "source_month",
        "previous_lai",
        "previous_lai_valid",
        "history_features",
        "historical_baseline",
    )
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}

def compute_world_stats(
    cache: dict[str, np.ndarray], world: dict[str, np.ndarray]
) -> WorldNormalizationStats:
    source = np.asarray(world["source_indices"], dtype=np.int64)
    train_local = np.flatnonzero(np.asarray(cache["split"][source]) == 0)
    train_source = source[train_local]
    base_stats = compute_normalization(cache)

    weather_sum = np.zeros(13, dtype=np.float64)
    weather_square_sum = np.zeros(13, dtype=np.float64)
    weather_count = np.zeros(13, dtype=np.float64)
    lai_sum = 0.0
    lai_square_sum = 0.0
    lai_count = 0
    for start in range(0, train_local.size, 20_000):
        local = train_local[start : start + 20_000]
        valid = np.asarray(world["relative_valid"][local], dtype=bool)
        weather = np.asarray(world["weather_rel"][local], dtype=np.float64)
        weather_valid = valid[:, :, None] & np.isfinite(weather)
        weather_sum += np.where(weather_valid, weather, 0.0).sum(axis=(0, 1))
        weather_square_sum += np.where(
            weather_valid, np.square(weather), 0.0
        ).sum(axis=(0, 1))
        weather_count += weather_valid.sum(axis=(0, 1))

        lai = np.asarray(world["target_lai_rel"][local], dtype=np.float64)
        lai_valid = valid & np.isfinite(lai)
        selected_lai = lai[lai_valid]
        lai_sum += float(selected_lai.sum())
        lai_square_sum += float(np.square(selected_lai).sum())
        lai_count += int(selected_lai.size)
    weather_mean = np.divide(
        weather_sum, weather_count, out=np.zeros_like(weather_sum), where=weather_count > 0
    )
    weather_variance = np.divide(
        weather_square_sum,
        weather_count,
        out=np.ones_like(weather_square_sum),
        where=weather_count > 0,
    ) - np.square(weather_mean)
    weather_std = np.sqrt(np.maximum(weather_variance, 1.0e-12))
    weather_std = np.where(weather_std < 1.0e-6, 1.0, weather_std)
    lai_mean = lai_sum / max(lai_count, 1)
    lai_variance = lai_square_sum / max(lai_count, 1) - lai_mean * lai_mean
    lai_std = math.sqrt(max(lai_variance, 1.0e-12))
    if lai_std < 1.0e-6:
        lai_std = 1.0

    target = np.asarray(cache["target"][train_source], dtype=np.float64)
    baseline = np.asarray(world["historical_baseline"][train_local], dtype=np.float64)
    residual = target - baseline
    residual_std = float(residual.std())
    if residual_std < 1.0e-6:
        residual_std = 1.0
    return WorldNormalizationStats(
        weather_mean=weather_mean.astype(float).tolist(),
        weather_std=weather_std.astype(float).tolist(),
        lai_mean=float(lai_mean),
        lai_std=float(lai_std),
        target_mean=float(base_stats.target_mean),
        target_std=float(base_stats.target_std),
        residual_mean=float(residual.mean()),
        residual_std=residual_std,
    )

def load_or_compute_world_stats(
    crop: str,
    cache: dict[str, np.ndarray],
    world: dict[str, np.ndarray],
    force: bool = False,
) -> WorldNormalizationStats:
    path = world_cache_dir(crop) / "normalization.json"
    if path.exists() and not force:
        return WorldNormalizationStats(**json.loads(path.read_text(encoding="utf-8")))
    stats = compute_world_stats(cache, world)
    save_json(asdict(stats), path)
    return stats

def make_world_arrays(
    cache: dict[str, np.ndarray],
    world: dict[str, np.ndarray],
    stats: WorldNormalizationStats,
    local_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    source_all = np.asarray(world["source_indices"], dtype=np.int64)
    source = source_all[local_indices]
    weather = np.asarray(world["weather_rel"][local_indices], dtype=np.float32)
    weather_mean = np.asarray(stats.weather_mean, dtype=np.float32)[None, None, :]
    weather_std = np.asarray(stats.weather_std, dtype=np.float32)[None, None, :]
    weather = np.where(
        np.isfinite(weather), (weather - weather_mean) / weather_std, 0.0
    ).astype(np.float32, copy=False)

    previous_lai = np.asarray(world["previous_lai"][local_indices], dtype=np.float32)
    previous_valid = np.asarray(
        world["previous_lai_valid"][local_indices], dtype=np.float32
    )
    previous_lai = (previous_lai - stats.lai_mean) / stats.lai_std
    previous_lai = np.where(
        (previous_valid > 0.0) & np.isfinite(previous_lai), previous_lai, 0.0
    ).astype(np.float32, copy=False)

    target_lai_raw = np.asarray(
        world["target_lai_rel"][local_indices], dtype=np.float32
    )
    relative_valid = np.asarray(
        world["relative_valid"][local_indices], dtype=np.float32
    )
    target_lai_valid = (
        relative_valid * np.isfinite(target_lai_raw).astype(np.float32)
    )
    target_lai = (target_lai_raw - stats.lai_mean) / stats.lai_std
    target_lai = np.where(
        (target_lai_valid > 0.0) & np.isfinite(target_lai), target_lai, 0.0
    ).astype(np.float32, copy=False)
    relative_weight = np.asarray(
        world["relative_weight"][local_indices], dtype=np.float32
    )

    target = np.asarray(cache["target"][source], dtype=np.float32)
    baseline = np.asarray(
        world["historical_baseline"][local_indices], dtype=np.float32
    )
    target_residual = (
        (target - baseline - stats.residual_mean) / stats.residual_std
    ).astype(np.float32)
    target_absolute = ((target - stats.target_mean) / stats.target_std).astype(
        np.float32
    )
    return {
        "weather": weather,
        "previous_lai": previous_lai,
        "previous_lai_valid": previous_valid,
        "target_lai": target_lai,
        "target_lai_valid": target_lai_valid,
        "relative_valid": relative_valid,
        "relative_weight": relative_weight,
        "source_month": np.asarray(
            world["source_month"][local_indices], dtype=np.uint8
        ),
        "history": np.asarray(
            world["history_features"][local_indices], dtype=np.float32
        ),
        "context": np.asarray(cache["context"][source], dtype=np.float32),
        "target_residual": target_residual,
        "target_absolute": target_absolute,
        "target": target,
        "baseline": baseline,
        "source_indices": source,
    }
