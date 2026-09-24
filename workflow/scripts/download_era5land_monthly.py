#!/usr/bin/env python3
"""Download ERA5-Land monthly averaged data from CDS."""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi

DATASET = "reanalysis-era5-land-monthly-means"
VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "surface_solar_radiation_downwards",
    "potential_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
]
MONTHS = [f"{month:02d}" for month in range(1, 13)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-year",
        type=int,
        default=1980,
        help="First year to download.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2017,
        help="Last year to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/era5land/monthly"),
        help="Directory where NetCDF files will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    for year in range(args.start_year, args.end_year + 1):
        target = output_dir / f"era5land_monthly_{year}.nc"
        if target.exists() and not args.force:
            print(f"Skip existing {target}")
            continue

        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": MONTHS,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"Requesting year {year} -> {target}")
        client.retrieve(DATASET, request, str(target))
        print(f"Finished {target}")


if __name__ == "__main__":
    main()
