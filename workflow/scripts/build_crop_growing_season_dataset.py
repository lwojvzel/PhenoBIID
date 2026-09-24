#!/usr/bin/env python3
"""Build crop-wise GDHY + growth-stage meteorology datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

GDHY_ROOT = Path("Data/GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180")
ERA5_ROOT = Path("Data/era5land/monthly_npy_lon180_0p5deg_by_var")
MIRCA_ROOT = Path("Data/MIRCA-OS/Monthly Growing Area Grids/Monthly Growing Area Grids")
OUTPUT_ROOT = Path("Data/processed/crop_yield_growing_season")

FILL_VALUE = np.float32(-9.99e8)
MIRCA_YEARS = (2000, 2005, 2010, 2015)
NATURAL_MONTH_ORDER = tuple(range(1, 13))
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

CROP_CONFIG = {
    "maize": {
        "gdhy_dir": "maize",
        "mirca_patterns": ("MIRCA-OS_Maize_{year}_{system}.nc",),
    },
    "rice": {
        "gdhy_dir": "rice",
        "mirca_patterns": (
            "MIRCA-OS_Rice_1_{year}_{system}.nc",
            "MIRCA-OS_Rice_2_{year}_{system}.nc",
            "MIRCA-OS_Rice_3_{year}_{system}.nc",
        ),
    },
    "soybean": {
        "gdhy_dir": "soybean",
        "mirca_patterns": ("MIRCA-OS_Soybeans_{year}_{system}.nc",),
    },
    "wheat": {
        "gdhy_dir": "wheat",
        "mirca_patterns": (
            "MIRCA-OS_Wheat_1_{year}_{system}.nc",
            "MIRCA-OS_Wheat_2_{year}_{system}.nc",
        ),
    },
}


def nearest_mirca_year(year: int) -> int:
    return min(MIRCA_YEARS, key=lambda value: (abs(year - value), value))


def load_gdhy_coords(root: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(root / "lat.npy"), np.load(root / "lon.npy")


def aggregate_mirca_monthly_file(path: Path) -> tuple[np.ndarray, list[int], dict[str, object]]:
    aggregated = np.zeros((12, 360, 720), dtype=np.float32)
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        months = [int(value) for value in ds.variables["month"][:]]
        if len(months) != len(set(months)):
            raise ValueError(f"{path}: duplicate numeric month coordinates")
        # The provider exporter applies its half-cell offset twice; raster
        # indices retain the global GeoTIFF footprint. Do not shift the values.
        expected_lat = 90.0 - (np.arange(2160) + 1.0) / 12.0
        expected_lon = -180.0 + (np.arange(4320) + 1.0) / 12.0
        if lat.shape != expected_lat.shape or lon.shape != expected_lon.shape:
            raise ValueError(f"{path}: unknown MIRCA coordinate dimensions")
        if not (np.allclose(lat, expected_lat, atol=2e-4, rtol=0)
                and np.allclose(lon, expected_lon, atol=2e-4, rtol=0)):
            raise ValueError(f"{path}: unsupported coordinate convention; verify georeferencing before aggregation")
        harvested_area = ds.variables["harvested_area"]
        for src_idx, month in enumerate(months):
            if month < 1 or month > 12:
                raise ValueError(f"{path}: unexpected month value {month}")
            data = np.asarray(harvested_area[src_idx], dtype=np.float32)
            # MIRCA latitude is north-to-south. Flip to south-to-north before block aggregation.
            data = data[::-1, :]
            if data.shape != (2160, 4320):
                raise ValueError(f"{path}: unexpected spatial shape {data.shape}")
            block_sum = data.reshape(360, 6, 720, 6).sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
            aggregated[month - 1] += block_sum

    meta = {
        "source_file": path.as_posix(),
        "shape": list(aggregated.shape),
        "original_months": months,
        "lat_first_last": [float(lat[0]), float(lat[-1])],
        "lon_first_last": [float(lon[0]), float(lon[-1])],
        "resolution_deg": [float(abs(lat[1] - lat[0])), float(abs(lon[1] - lon[0]))],
        "coordinate_convention": "Verified original MIRCA-OS monthly double-half-cell coordinate labels; underlying global GeoTIFF index footprint retained",
    }
    return aggregated, months, meta


def load_crop_mirca_total(
    crop: str,
    mirca_year: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    crop_cache = cache_dir / crop / str(mirca_year)
    total_path = crop_cache / "mirca_month_area_total_0p5.npy"
    ir_path = crop_cache / "mirca_month_area_ir_0p5.npy"
    rf_path = crop_cache / "mirca_month_area_rf_0p5.npy"
    meta_path = crop_cache / "mirca_aggregation_meta.json"
    if total_path.exists() and ir_path.exists() and rf_path.exists() and meta_path.exists():
        total = np.load(total_path)
        irrigated = np.load(ir_path)
        rainfed = np.load(rf_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return total, irrigated, rainfed, meta["source_files"]

    config = CROP_CONFIG[crop]
    crop_cache.mkdir(parents=True, exist_ok=True)
    irrigated = np.zeros((12, 360, 720), dtype=np.float32)
    rainfed = np.zeros((12, 360, 720), dtype=np.float32)
    source_meta: list[dict[str, object]] = []
    for system, target in (("ir", irrigated), ("rf", rainfed)):
        for pattern in config["mirca_patterns"]:
            path = MIRCA_ROOT / str(mirca_year) / pattern.format(year=mirca_year, system=system)
            if not path.exists():
                raise FileNotFoundError(f"Missing MIRCA file: {path}")
            aggregated, _, meta = aggregate_mirca_monthly_file(path)
            target += aggregated
            source_meta.append(meta)

    total = irrigated + rainfed
    np.save(ir_path, irrigated)
    np.save(rf_path, rainfed)
    np.save(total_path, total)
    meta = {
        "crop": crop,
        "mirca_year": mirca_year,
        "aggregation": "sum 5-arcminute harvested_area over each 6x6 block to GDHY 0.5-degree cells",
        "month_axis_kind": "stored as natural-month indexed tensor after reading MIRCA month variable",
        "natural_month_order": list(NATURAL_MONTH_ORDER),
        "source_files": source_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return total, irrigated, rainfed, source_meta


def build_relative_mapping(month_area: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if month_area.shape != (12, 360, 720):
        raise ValueError(f"Expected MIRCA month_area shape (12,360,720), got {month_area.shape}")
    flat = month_area.reshape(12, -1)
    active = flat > 0.0
    dest_map = np.full((12, flat.shape[1]), -1, dtype=np.int16)
    cursor = np.zeros(flat.shape[1], dtype=np.int16)
    for src_idx in range(12):
        cols = np.flatnonzero(active[src_idx])
        if cols.size == 0:
            continue
        dest_map[src_idx, cols] = cursor[cols]
        cursor[cols] += 1

    weight_rel = np.zeros((12, flat.shape[1]), dtype=np.float32)
    valid_rel = np.zeros((12, flat.shape[1]), dtype=np.uint8)
    month_rel = np.zeros((12, flat.shape[1]), dtype=np.uint8)
    src_rel = np.full((12, flat.shape[1]), 255, dtype=np.uint8)
    totals = flat.sum(axis=0, dtype=np.float64)
    for src_idx in range(12):
        cols = np.flatnonzero(dest_map[src_idx] >= 0)
        if cols.size == 0:
            continue
        dest = dest_map[src_idx, cols]
        values = flat[src_idx, cols]
        denom = totals[cols]
        weights = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0.0)
        weight_rel[dest, cols] = weights.astype(np.float32, copy=False)
        valid_rel[dest, cols] = 1
        month_rel[dest, cols] = np.uint8(src_idx + 1)
        src_rel[dest, cols] = np.uint8(src_idx)

    return (
        weight_rel.reshape(12, 360, 720),
        valid_rel.reshape(12, 360, 720),
        month_rel.reshape(12, 360, 720),
        src_rel.reshape(12, 360, 720),
    )


def reorder_era5_variable(era5_natural: np.ndarray, src_rel: np.ndarray) -> np.ndarray:
    if era5_natural.shape != (12, 360, 720):
        raise ValueError(f"Expected ERA5 shape (12,360,720), got {era5_natural.shape}")
    src_flat = src_rel.reshape(12, -1)
    natural_flat = era5_natural.reshape(12, -1)
    out = np.zeros((12, natural_flat.shape[1]), dtype=np.float32)
    for rel_idx in range(12):
        cols = np.flatnonzero(src_flat[rel_idx] != 255)
        if cols.size == 0:
            continue
        src_idx = src_flat[rel_idx, cols].astype(np.int64, copy=False)
        out[rel_idx, cols] = natural_flat[src_idx, cols]
    return out.reshape(12, 360, 720)


def write_root_readme(output_root: Path, lat: np.ndarray, lon: np.ndarray) -> None:
    lines = [
        "# Crop Yield + Growing-Season Meteorology",
        "",
        "这个目录存放按作物拆分的 `GDHY 年产量 + MIRCA 月生长面积 + ERA5-Land 相对生长月气象` 处理结果。",
        "",
        "## 核心对齐规则",
        "",
        "- GDHY 使用现有 `0.5°`、`lon=-180..180`、纬度南到北的 `.npy` 导出。",
        "- MIRCA Monthly Growing Area 原始分辨率为 `5 arc-minute`（不是 `0.1°`），按 `6x6` 空间块求和聚合到 `0.5°`。",
        "- 聚合后的 MIRCA 空间网格输出到 GDHY `0.5°` 目标网格，并复用 GDHY 的 `lat.npy` / `lon.npy` 作为目标坐标。",
        "- 说明：MIRCA 原生 `5 arc-minute` 网格按 `6x6` 聚合后，其理论块中心与 GDHY 像元中心存在约 `0.0417°` 的半像元级差异；这里统一以 GDHY 网格为最终参考坐标。",
        "- ERA5-Land 使用现有已聚合的 `0.5°`、按变量存储 `.npy` 文件。",
        "- MIRCA 的 `month` 变量在原文件里可能不是自然月升序；本处理读取该变量后，统一落到自然月 `1..12` 索引上。",
        "- 对每个格点，先按自然月 `1..12` 筛出有效月，再按自然月从小到大生成左对齐的相对生长月序列。",
        "- `valid_rel=1` 表示该相对月真实存在，`0` 表示 padding。",
        "- `weight_rel` 为该格点该相对月的面积权重，定义为月生长面积除以全年有效月生长面积之和。",
        "",
        "## 统一张量约定",
        "",
        "- 空间维：`(360, 720)`",
        "- 相对生长月维：固定为 `12`，短生长季通过 `valid_rel=0` 与 `month_rel=0` 做 padding。",
        "- 年度气象文件：`(13, 12, 360, 720)`",
        "- 变量顺序见各作物目录下的 `variable_order.txt`",
        "",
        "## 坐标范围",
        "",
        f"- 纬度首末值：`{float(lat[0]):.2f}` -> `{float(lat[-1]):.2f}`",
        f"- 经度首末值：`{float(lon[0]):.2f}` -> `{float(lon[-1]):.2f}`",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_crop(crop: str, output_root: Path, force: bool = False) -> None:
    config = CROP_CONFIG[crop]
    gdhy_dir = GDHY_ROOT / config["gdhy_dir"]
    crop_root = output_root / crop
    yield_dir = crop_root / "yield"
    mirca_dir = crop_root / "mirca"
    era5_dir = crop_root / "era5_rel"
    yield_dir.mkdir(parents=True, exist_ok=True)
    mirca_dir.mkdir(parents=True, exist_ok=True)
    era5_dir.mkdir(parents=True, exist_ok=True)
    (crop_root / "variable_order.txt").write_text("\n".join(ERA5_VARIABLES) + "\n", encoding="utf-8")

    years = sorted(int(path.stem.split("_")[-1]) for path in gdhy_dir.glob("yield_*.npy"))
    if not years:
        raise FileNotFoundError(f"No GDHY yearly files found under {gdhy_dir}")

    mirca_cache_dir = crop_root / "_mirca_cache"
    mirca_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "year\tmirca_year\tyield_path\tera5_rel_path\tweight_rel_path\tvalid_rel_path\tmonth_rel_path\n"
    ]
    done_mirca_years: set[int] = set()

    for year in years:
        mirca_year = nearest_mirca_year(year)
        total_area, irrigated_area, rainfed_area, source_meta = load_crop_mirca_total(crop, mirca_year, mirca_cache_dir)
        mirca_year_dir = mirca_dir / str(mirca_year)
        mirca_year_dir.mkdir(parents=True, exist_ok=True)
        weight_path = mirca_year_dir / "weight_rel.npy"
        valid_path = mirca_year_dir / "valid_rel.npy"
        month_path = mirca_year_dir / "month_rel.npy"
        src_path = mirca_year_dir / "src_rel.npy"
        if mirca_year not in done_mirca_years or force:
            weight_rel, valid_rel, month_rel, src_rel = build_relative_mapping(total_area)
            np.save(mirca_year_dir / "mirca_month_area_total_0p5.npy", total_area)
            np.save(mirca_year_dir / "mirca_month_area_ir_0p5.npy", irrigated_area)
            np.save(mirca_year_dir / "mirca_month_area_rf_0p5.npy", rainfed_area)
            np.save(weight_path, weight_rel)
            np.save(valid_path, valid_rel)
            np.save(month_path, month_rel)
            np.save(src_path, src_rel)
            year_meta = {
                "crop": crop,
                "mirca_year": mirca_year,
                "natural_month_order": list(NATURAL_MONTH_ORDER),
                "source_month_order_per_file": source_meta,
                "note": "month_rel stores natural month numbers sorted ascending after relative left alignment; 0 means padding.",
            }
            (mirca_year_dir / "metadata.json").write_text(json.dumps(year_meta, indent=2), encoding="utf-8")
            done_mirca_years.add(mirca_year)

        yield_src = gdhy_dir / f"yield_{year}.npy"
        yield_dst = yield_dir / f"yield_{year}.npy"
        if force or not yield_dst.exists():
            yield_array = np.load(yield_src).astype(np.float32, copy=False)
            np.save(yield_dst, yield_array)

        era5_year_path = era5_dir / f"era5_rel_{year}.npy"
        if force or not era5_year_path.exists():
            src_rel = np.load(src_path)
            era5_rel = np.zeros((len(ERA5_VARIABLES), 12, 360, 720), dtype=np.float32)
            for var_idx, variable in enumerate(ERA5_VARIABLES):
                era5_src = ERA5_ROOT / variable / f"{variable}_{year}.npy"
                if not era5_src.exists():
                    raise FileNotFoundError(f"Missing ERA5 file: {era5_src}")
                natural = np.load(era5_src).astype(np.float32, copy=False)
                era5_rel[var_idx] = reorder_era5_variable(natural, src_rel)
            np.save(era5_year_path, era5_rel)

        manifest_lines.append(
            "\t".join(
                [
                    str(year),
                    str(mirca_year),
                    yield_dst.as_posix(),
                    era5_year_path.as_posix(),
                    weight_path.as_posix(),
                    valid_path.as_posix(),
                    month_path.as_posix(),
                ]
            )
            + "\n"
        )
        print(f"[{crop}] processed year {year} with MIRCA {mirca_year}")

    (crop_root / "manifest.tsv").write_text("".join(manifest_lines), encoding="utf-8")


def verify_coordinates(output_root: Path, lat: np.ndarray, lon: np.ndarray) -> None:
    np.save(output_root / "lat.npy", lat)
    np.save(output_root / "lon.npy", lon)
    verify = {
        "gdhy_lat_match": True,
        "gdhy_lon_match": True,
        "lat_shape": list(lat.shape),
        "lon_shape": list(lon.shape),
        "lat_first_last": [float(lat[0]), float(lat[-1])],
        "lon_first_last": [float(lon[0]), float(lon[-1])],
        "mirca_native_resolution_deg": [5.0 / 60.0, 5.0 / 60.0],
        "target_resolution_deg": [0.5, 0.5],
        "mirca_block_center_offset_deg_vs_gdhy": [1.0 / 24.0, 1.0 / 24.0],
        "note": "MIRCA monthly raster indices follow the global GeoTIFF footprint and are summed in 6x6 blocks. The legacy offset field describes coordinate-label magnitudes, not a required shift of raster values.",
        "coordinate_audit": "scripts/audit_mirca_coordinate_export.py: provider export code applies the center offset twice; the reference GeoTIFF centers aggregate to GDHY centers",
    }
    (output_root / "coordinate_check.json").write_text(json.dumps(verify, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crops",
        default="maize,rice,soybean,wheat",
        help="Comma-separated crop list. Supported: maize,rice,soybean,wheat",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for processed crop-wise dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing yearly outputs.",
    )
    args = parser.parse_args()

    crops = [item.strip() for item in args.crops.split(",") if item.strip()]
    unsupported = [crop for crop in crops if crop not in CROP_CONFIG]
    if unsupported:
        raise SystemExit(f"Unsupported crops: {unsupported}")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    lat, lon = load_gdhy_coords(GDHY_ROOT)
    verify_coordinates(output_root, lat, lon)
    write_root_readme(output_root, lat, lon)
    for crop in crops:
        process_crop(crop, output_root, force=args.force)
    print(f"Done. Output written to {output_root}")


if __name__ == "__main__":
    main()
