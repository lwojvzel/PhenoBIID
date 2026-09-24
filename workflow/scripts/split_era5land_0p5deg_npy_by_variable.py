#!/usr/bin/env python3
"""Split yearly ERA5-Land 0.5-degree tensors into per-variable yearly files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_year(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def load_variable_order(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_readme(output_dir: Path, source_dir: Path, variables: list[str]) -> None:
    lines = [
        "# ERA5-Land monthly 0.5 degree npy by variable",
        "",
        f"源目录：`{source_dir.as_posix()}`",
        "",
        "这个目录把年度 ERA5-Land 0.5° `.npy` 张量拆分为“按变量单独存储”的形式。",
        "",
        "## 目录结构",
        "",
        "- `lat.npy`",
        "- `lon.npy`",
        "- `<variable>/<variable>_<year>.npy`",
        "- `manifest.tsv`",
        "- `variable_order.txt`",
        "",
        "## 单文件张量布局",
        "",
        "每个变量年度文件的形状为：",
        "",
        "- `(12, 360, 720)`",
        "",
        "轴顺序为：",
        "",
        "1. 月份（1-12）",
        "2. 纬度（对应 `lat.npy`，南到北）",
        "3. 经度（对应 `lon.npy`，西到东）",
        "",
        "## 变量列表",
        "",
    ]
    lines.extend(f"- `{name}`" for name in variables)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "variable_order.txt").write_text("\n".join(variables) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("Data/era5land/monthly_npy_lon180_0p5deg"),
        help="Directory containing yearly ERA5-Land 0.5-degree tensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/era5land/monthly_npy_lon180_0p5deg_by_var"),
        help="Directory for per-variable yearly files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = load_variable_order(source_dir / "variable_order.txt")
    lat = np.load(source_dir / "lat.npy")
    lon = np.load(source_dir / "lon.npy")
    np.save(output_dir / "lat.npy", lat)
    np.save(output_dir / "lon.npy", lon)
    write_readme(output_dir, source_dir, variables)

    year_files = sorted(source_dir.glob("era5land_monthly_0p5deg_*.npy"))
    if not year_files:
        raise SystemExit(f"No yearly npy files found under {source_dir}")

    manifest_lines = ["variable\tyear\tsource\ttarget\tshape\tdtype\n"]

    for file_index, year_file in enumerate(year_files, start=1):
        year = parse_year(year_file)
        arr = np.load(year_file)
        expected_shape = (12, len(variables), 360, 720)
        if arr.shape != expected_shape:
            raise ValueError(f"{year_file} has shape {arr.shape}, expected {expected_shape}")

        for var_index, var_name in enumerate(variables):
            var_dir = output_dir / var_name
            var_dir.mkdir(parents=True, exist_ok=True)
            target = var_dir / f"{var_name}_{year}.npy"
            if target.exists() and not args.force:
                manifest_lines.append(
                    f"{var_name}\t{year}\t{year_file.as_posix()}\t{target.as_posix()}\t(skip)\t(skip)\n"
                )
                continue

            var_arr = np.ascontiguousarray(arr[:, var_index, :, :], dtype=np.float32)
            np.save(target, var_arr)
            manifest_lines.append(
                f"{var_name}\t{year}\t{year_file.as_posix()}\t{target.as_posix()}\t{var_arr.shape}\t{var_arr.dtype}\n"
            )

        print(f"[{file_index}/{len(year_files)}] Split {year}")

    (output_dir / "manifest.tsv").write_text("".join(manifest_lines), encoding="utf-8")
    print(f"Done. Split files written to {output_dir}")


if __name__ == "__main__":
    main()
