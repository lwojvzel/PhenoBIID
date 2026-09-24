#!/usr/bin/env python3
"""Process GLASS AVHRR 8-day LAI into monthly and MIRCA-relative tensors."""

from __future__ import annotations

import argparse
import calendar
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pyhdf.SD import SD, SDC


RAW_ROOT = Path("Data/GLASS_LAI_AVHRR_005D")
GROWING_SEASON_ROOT = Path("Data/processed/crop_yield_growing_season")
OUTPUT_ROOT = Path("Data/processed/glass_lai_avhrr_005d")
MIRCA_YEARS = (2000, 2005, 2010, 2015)
DEFAULT_CROPS = ("maize", "rice", "soybean", "wheat")
FILL_VALUE = 2550
VALID_MIN = 0
VALID_MAX = 1000
SCALE_FACTOR = 0.01
FILENAME_RE = re.compile(r"\.A(?P<year>\d{4})(?P<doy>\d{3})\.")


def nearest_mirca_year(year: int) -> int:
    return min(MIRCA_YEARS, key=lambda value: (abs(year - value), value))


def parse_year_doy(path: Path) -> tuple[int, int]:
    match = FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse year/doy from {path.name}")
    return int(match.group("year")), int(match.group("doy"))


def period_month_overlaps(year: int, doy: int) -> list[tuple[int, int]]:
    start = date(year, 1, 1) + timedelta(days=doy - 1)
    next_year = date(year + 1, 1, 1)
    end = min(start + timedelta(days=8), next_year)
    overlaps: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        month_end = date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]) + timedelta(days=1)
        stop = min(month_end, end)
        if cursor.year == year:
            overlaps.append((cursor.month - 1, (stop - cursor).days))
        cursor = stop
    return overlaps


def read_lai_hdf(path: Path) -> np.ndarray:
    hdf = SD(str(path), SDC.READ)
    try:
        sds = hdf.select("LAI")
        raw = sds.get()
    finally:
        hdf.end()
    if raw.shape != (3600, 7200):
        raise ValueError(f"Expected GLASS LAI shape (3600,7200), got {raw.shape} for {path}")
    return raw


def aggregate_005_to_05(raw: np.ndarray) -> np.ndarray:
    valid = (raw != FILL_VALUE) & (raw >= VALID_MIN) & (raw <= VALID_MAX)
    values = np.where(valid, raw, 0).astype(np.float32, copy=False)
    counts = valid.reshape(360, 10, 720, 10).sum(axis=(1, 3), dtype=np.float32)
    sums = values.reshape(360, 10, 720, 10).sum(axis=(1, 3), dtype=np.float32) * np.float32(SCALE_FACTOR)
    out_north_to_south = np.divide(
        sums,
        counts,
        out=np.full((360, 720), np.nan, dtype=np.float32),
        where=counts > 0,
    )
    return np.flipud(out_north_to_south).astype(np.float32, copy=False)


def build_monthly_lai(year: int, raw_root: Path) -> tuple[np.ndarray, dict[str, object]]:
    files = sorted((raw_root / str(year)).glob("*.hdf"))
    if not files:
        raise FileNotFoundError(f"No GLASS LAI HDF files found for {year} under {raw_root / str(year)}")

    accum = np.zeros((12, 360, 720), dtype=np.float32)
    weights = np.zeros((12, 360, 720), dtype=np.float32)
    file_meta = []

    for path in files:
        file_year, doy = parse_year_doy(path)
        if file_year != year:
            raise ValueError(f"{path}: filename year {file_year} does not match requested year {year}")
        lai_05 = aggregate_005_to_05(read_lai_hdf(path))
        finite = np.isfinite(lai_05)
        for month_idx, days in period_month_overlaps(year, doy):
            accum[month_idx, finite] += lai_05[finite] * np.float32(days)
            weights[month_idx, finite] += np.float32(days)
        file_meta.append({"file": path.as_posix(), "doy": doy, "month_overlaps": period_month_overlaps(year, doy)})

    monthly = np.divide(
        accum,
        weights,
        out=np.full_like(accum, np.nan, dtype=np.float32),
        where=weights > 0,
    )
    meta = {
        "year": year,
        "n_hdf_files": len(files),
        "method": "8-day LAI composites aggregated to 0.5-degree by 10x10 valid-pixel mean, then day-overlap weighted to natural months",
        "source_shape": [3600, 7200],
        "output_shape": list(monthly.shape),
        "hdf_lai_fill_value": FILL_VALUE,
        "hdf_lai_valid_range": [VALID_MIN, VALID_MAX],
        "hdf_lai_scale_factor": SCALE_FACTOR,
        "file_meta": file_meta,
    }
    return monthly.astype(np.float32, copy=False), meta


def reorder_relative_months(monthly: np.ndarray, src_rel: np.ndarray) -> np.ndarray:
    if monthly.shape != (12, 360, 720):
        raise ValueError(f"Expected monthly shape (12,360,720), got {monthly.shape}")
    if src_rel.shape != (12, 360, 720):
        raise ValueError(f"Expected src_rel shape (12,360,720), got {src_rel.shape}")

    src_flat = src_rel.reshape(12, -1)
    natural_flat = monthly.reshape(12, -1)
    out = np.full((12, natural_flat.shape[1]), np.nan, dtype=np.float32)
    for rel_idx in range(12):
        cols = np.flatnonzero(src_flat[rel_idx] != 255)
        if cols.size == 0:
            continue
        src_idx = src_flat[rel_idx, cols].astype(np.int64, copy=False)
        out[rel_idx, cols] = natural_flat[src_idx, cols]
    return out.reshape(12, 360, 720)


def write_root_metadata(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lat = np.load(GROWING_SEASON_ROOT / "lat.npy")
    lon = np.load(GROWING_SEASON_ROOT / "lon.npy")
    np.save(output_root / "lat.npy", lat)
    np.save(output_root / "lon.npy", lon)
    readme = [
        "# GLASS LAI AVHRR 0.05D Processed",
        "",
        "This directory stores processed GLASS AVHRR LAI tensors aligned with the GDHY/MIRCA/ERA5 growing-season benchmark.",
        "",
        "Pipeline:",
        "",
        "1. Read GLASS 8-day HDF4 `LAI` grids at 0.05 degrees.",
        "2. Apply `valid_range=0..1000`, `_FillValue=2550`, and `scale_factor=0.01`.",
        "3. Aggregate 0.05-degree grids to 0.5-degree by valid-pixel mean over each 10x10 block.",
        "4. Build natural-month LAI by weighting each 8-day composite by overlap days with each month.",
        "5. Reorder monthly LAI into MIRCA relative growing-month tensors using each crop's `src_rel.npy`.",
        "",
        "Coordinate convention follows `Data/processed/crop_yield_growing_season`:",
        "",
        "- shape `(360, 720)`",
        "- latitude south-to-north",
        "- longitude `-179.75..179.75`",
        "- missing values are stored as `NaN`",
        "",
        "Outputs:",
        "",
        "- `monthly_0p5/lai_monthly_<year>.npy`: shape `(12, 360, 720)`, natural months.",
        "- `crops/<crop>/lai_rel/lai_rel_<year>.npy`: shape `(12, 360, 720)`, relative growing months.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    metadata = {
        "source": "GLASS LAI AVHRR 8-day 0.05-degree HDF4",
        "monthly_output_shape": [12, 360, 720],
        "relative_output_shape": [12, 360, 720],
        "missing_value": "NaN",
        "lat_first_last": [float(lat[0]), float(lat[-1])],
        "lon_first_last": [float(lon[0]), float(lon[-1])],
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def process_year(
    year: int,
    crops: list[str],
    raw_root: Path,
    output_root: Path,
    force: bool,
) -> None:
    monthly_dir = output_root / "monthly_0p5"
    monthly_meta_dir = output_root / "monthly_0p5" / "metadata"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    monthly_meta_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = monthly_dir / f"lai_monthly_{year}.npy"
    monthly_meta_path = monthly_meta_dir / f"lai_monthly_{year}.json"

    if monthly_path.exists() and not force:
        monthly = np.load(monthly_path)
    else:
        monthly, meta = build_monthly_lai(year, raw_root)
        np.save(monthly_path, monthly)
        monthly_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for crop in crops:
        mirca_year = nearest_mirca_year(year)
        src_path = GROWING_SEASON_ROOT / crop / "mirca" / str(mirca_year) / "src_rel.npy"
        valid_path = GROWING_SEASON_ROOT / crop / "mirca" / str(mirca_year) / "valid_rel.npy"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing MIRCA src_rel for crop={crop}, year={mirca_year}: {src_path}")
        crop_dir = output_root / "crops" / crop / "lai_rel"
        crop_dir.mkdir(parents=True, exist_ok=True)
        rel_path = crop_dir / f"lai_rel_{year}.npy"
        if force or not rel_path.exists():
            src_rel = np.load(src_path)
            lai_rel = reorder_relative_months(monthly, src_rel)
            np.save(rel_path, lai_rel.astype(np.float32, copy=False))

        manifest = output_root / "crops" / crop / "manifest.tsv"
        header_needed = not manifest.exists()
        with manifest.open("a", encoding="utf-8") as handle:
            if header_needed:
                handle.write("year\tmirca_year\tmonthly_lai_path\tlai_rel_path\tvalid_rel_path\tsrc_rel_path\n")
            handle.write(
                "\t".join(
                    [
                        str(year),
                        str(mirca_year),
                        monthly_path.as_posix(),
                        rel_path.as_posix(),
                        valid_path.as_posix(),
                        src_path.as_posix(),
                    ]
                )
                + "\n"
            )

    print(f"processed GLASS LAI year {year}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--years", default="1981-2016", help="Year list/range, e.g. 1981-2016 or 1981,1982")
    parser.add_argument("--crops", default=",".join(DEFAULT_CROPS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if "-" in args.years:
        start, end = [int(item) for item in args.years.split("-", 1)]
        years = list(range(start, end + 1))
    else:
        years = [int(item) for item in args.years.split(",") if item.strip()]
    crops = [item.strip() for item in args.crops.split(",") if item.strip()]

    write_root_metadata(args.output_root)
    for crop in crops:
        manifest = args.output_root / "crops" / crop / "manifest.tsv"
        if args.force and manifest.exists():
            manifest.unlink()
    for year in years:
        process_year(year, crops, args.raw_root, args.output_root, force=args.force)
    print(f"Done. Output written to {args.output_root}")


if __name__ == "__main__":
    main()
