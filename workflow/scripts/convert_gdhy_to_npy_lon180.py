#!/usr/bin/env python3
"""Convert GDHY NetCDF4/HDF5 yield files to lon -180..180 NumPy arrays.

This script avoids Python HDF5 bindings and uses h5dump to extract the small
datasets as raw little-endian binary, then writes .npy files with NumPy.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


FILL_VALUE = -9.99e8


def h5dump_array(nc_path: Path, dataset: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Read an HDF5 dataset using h5dump's raw binary export."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        subprocess.run(
            [
                "h5dump",
                "-d",
                f"/{dataset}",
                "-b",
                "LE",
                "-o",
                tmp.name,
                str(nc_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        arr = np.fromfile(tmp.name, dtype=dtype)

    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"{nc_path}: dataset /{dataset} has {arr.size} values, expected {expected}"
        )
    return arr.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Data/gdhy_v1.2_v1.3_20190128"),
        help="Source GDHY directory containing crop subdirectories with .nc4 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/gdhy_v1.2_v1.3_20190128_npy_lon180"),
        help="Output directory for converted .npy files.",
    )
    args = parser.parse_args()

    source = args.source
    output = args.output
    nc_files = sorted(source.glob("*/*.nc4"))
    if not nc_files:
        raise SystemExit(f"No .nc4 files found under {source}")

    first_file = nc_files[0]
    lat = h5dump_array(first_file, "lat", np.dtype("<f8"), (360,))
    lon_0360 = h5dump_array(first_file, "lon", np.dtype("<f8"), (720,))
    lon_180 = ((lon_0360 + 180.0) % 360.0) - 180.0
    lon_order = np.argsort(lon_180)
    lon_180 = lon_180[lon_order]

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "lat.npy", lat)
    np.save(output / "lon.npy", lon_180)

    manifest_lines = [
        "source\ttarget\tshape\tdtype\tfill_value\n",
    ]

    for index, nc_file in enumerate(nc_files, start=1):
        rel_path = nc_file.relative_to(source).with_suffix(".npy")
        out_file = output / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        data = h5dump_array(nc_file, "var", np.dtype("<f4"), (360, 720))
        data_lon180 = data[:, lon_order]
        np.save(out_file, data_lon180)

        manifest_lines.append(
            f"{nc_file.as_posix()}\t{out_file.as_posix()}\t{data_lon180.shape}\t"
            f"{data_lon180.dtype}\t{FILL_VALUE:g}\n"
        )

        if index % 25 == 0 or index == len(nc_files):
            print(f"Converted {index}/{len(nc_files)} files")

    (output / "README.md").write_text(
        "\n".join(
            [
                "# GDHY npy lon180",
                "",
                f"Source: `{source.as_posix()}`",
                "",
                "Each crop/year `.npy` file stores the original `/var` data as a "
                "`float32` array with shape `(360, 720)`.",
                "",
                "Dimension order is:",
                "",
                "- axis 0: `lat.npy`, latitude in degrees north, from -89.75 to 89.75",
                "- axis 1: `lon.npy`, longitude in degrees east, converted and sorted "
                "from -179.75 to 179.75",
                "",
                f"Missing values are retained as `{FILL_VALUE:g}`.",
                "",
                "The original file year and crop/season are preserved in the relative "
                "path, for example `maize/yield_1981.npy`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "manifest.tsv").write_text("".join(manifest_lines), encoding="utf-8")

    print(f"Done. Output written to {output}")


if __name__ == "__main__":
    main()
