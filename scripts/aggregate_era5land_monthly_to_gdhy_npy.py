#!/usr/bin/env python3
"""Aggregate ERA5-Land monthly NetCDF files to GDHY 0.5-degree NumPy arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

VARIABLES = [
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
]


def parse_year(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("_", 1)[-1])


def load_gdhy_coords(gdhy_root: Path) -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(gdhy_root / "lat.npy")
    lon = np.load(gdhy_root / "lon.npy")
    return lat, lon


def prepare_lon_order(longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon180 = np.where(longitude >= 180.0, longitude - 360.0, longitude)
    order = np.argsort(lon180)
    return lon180[order], order


def aggregate_variable(
    var_data: np.ndarray,
    lon_order: np.ndarray,
) -> np.ndarray:
    """Convert ERA5-Land (12, 1801, 3600) to GDHY-aligned (12, 360, 720)."""
    data = np.asarray(var_data, dtype=np.float32)
    if data.shape[1:] != (1801, 3600):
        raise ValueError(f"Unexpected ERA5 spatial shape: {data.shape}")

    # North-to-south -> south-to-north.
    data = data[:, ::-1, :]
    # Drop the single 90N row so latitude becomes exactly 1800 = 360 * 5.
    data = data[:, :-1, :]
    # 0..360 -> -180..180 and west-to-east ordering.
    data = data[:, :, lon_order]
    # 0.1-degree -> 0.5-degree block mean, ignoring ocean/masked NaNs.
    data = data.reshape(data.shape[0], 360, 5, 720, 5)
    valid_count = np.sum(np.isfinite(data), axis=(2, 4), dtype=np.int16)
    summed = np.nansum(data, axis=(2, 4), dtype=np.float64)
    out = np.full((data.shape[0], 360, 720), np.nan, dtype=np.float32)
    np.divide(summed, valid_count, out=out, where=valid_count > 0)
    return out


def write_readme(output_root: Path, source_root: Path, gdhy_root: Path) -> None:
    text = "\n".join(
        [
            "# ERA5-Land monthly 0.5 degree npy",
            "",
            f"源目录：`{source_root.as_posix()}`",
            f"目标坐标：`{gdhy_root.as_posix()}`",
            "",
            "这个目录存放将 ERA5-Land 月尺度 NetCDF 文件聚合到 GDHY 0.5° 网格后的 `.npy` 文件。",
            "",
            "## 处理规则",
            "",
            "- 原始经度从 `0..360` 转为 `-180..180` 并重新排序",
            "- 原始纬度从北到南改为南到北",
            "- 对每个月、每个变量执行 `5x5` 空间块平均，将 `0.1°` 聚合到 `0.5°`",
            "- 聚合时忽略块内 `NaN`；只有整块都缺测时才保留 `NaN`",
            "- 输出坐标文件直接复用 GDHY 的 `lat.npy` 与 `lon.npy`",
            "- 当前阶段不做单位换算，保持与原 NetCDF 一致",
            "",
            "## 输出文件",
            "",
            "- `lat.npy`",
            "- `lon.npy`",
            "- `era5land_monthly_0p5deg_<year>.npy`",
            "- `manifest.tsv`",
            "- `variable_order.txt`",
            "",
            "## 张量布局",
            "",
            "每个年度 `.npy` 文件的形状为：",
            "",
            "- `(12, 13, 360, 720)`",
            "",
            "轴顺序为：",
            "",
            "1. 月份（1-12）",
            "2. 变量",
            "3. 纬度（对应 `lat.npy`，南到北）",
            "4. 经度（对应 `lon.npy`，西到东）",
            "",
        ]
    )
    (output_root / "README.md").write_text(text + "\n", encoding="utf-8")
    (output_root / "variable_order.txt").write_text("\n".join(VARIABLES) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("Data/era5land/monthly"),
        help="Directory containing raw ERA5-Land monthly NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/era5land/monthly_npy_lon180_0p5deg"),
        help="Directory for GDHY-aligned aggregated NumPy arrays.",
    )
    parser.add_argument(
        "--gdhy-root",
        type=Path,
        default=Path("Data/GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180"),
        help="GDHY lon180 coordinate directory.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional first year to process.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional last year to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing yearly output files.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    gdhy_root = args.gdhy_root
    output_dir.mkdir(parents=True, exist_ok=True)

    gdhy_lat, gdhy_lon = load_gdhy_coords(gdhy_root)
    np.save(output_dir / "lat.npy", gdhy_lat)
    np.save(output_dir / "lon.npy", gdhy_lon)
    write_readme(output_dir, source_dir, gdhy_root)

    nc_files = sorted(source_dir.glob("era5land_monthly_*.nc"))
    if args.start_year is not None:
        nc_files = [path for path in nc_files if parse_year(path) >= args.start_year]
    if args.end_year is not None:
        nc_files = [path for path in nc_files if parse_year(path) <= args.end_year]
    if not nc_files:
        raise SystemExit("No ERA5-Land NetCDF files selected.")

    manifest_lines = ["year\tsource\ttarget\tshape\tdtype\n"]

    lon_order: np.ndarray | None = None
    for index, path in enumerate(nc_files, start=1):
        year = parse_year(path)
        target = output_dir / f"era5land_monthly_0p5deg_{year}.npy"
        if target.exists() and not args.force:
            manifest_lines.append(
                f"{year}\t{path.as_posix()}\t{target.as_posix()}\t(skip)\t(skip)\n"
            )
            print(f"[{index}/{len(nc_files)}] Skip {year}: output exists")
            continue

        with Dataset(path) as ds:
            if lon_order is None:
                longitude = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
                _, lon_order = prepare_lon_order(longitude)

            year_array = np.empty((12, len(VARIABLES), 360, 720), dtype=np.float32)
            for var_index, var_name in enumerate(VARIABLES):
                aggregated = aggregate_variable(ds.variables[var_name][:], lon_order)
                year_array[:, var_index, :, :] = aggregated

        np.save(target, year_array)
        manifest_lines.append(
            f"{year}\t{path.as_posix()}\t{target.as_posix()}\t{year_array.shape}\t{year_array.dtype}\n"
        )
        print(f"[{index}/{len(nc_files)}] Wrote {target.name}")

    (output_dir / "manifest.tsv").write_text("".join(manifest_lines), encoding="utf-8")
    print(f"Done. Aggregated files written to {output_dir}")


if __name__ == "__main__":
    main()
