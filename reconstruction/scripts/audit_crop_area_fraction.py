from __future__ import annotations
from pathlib import Path
import numpy as np
ROOT = Path(__import__("os").environ["PHENOBIID_WORKSPACE"]).resolve()

MIRCA_ROOT = ROOT / "Data/processed/crop_yield_growing_season"

MIRCA_YEARS = (2000, 2005, 2010, 2015)

def nearest_mirca_year(year: int) -> int:
    return min(MIRCA_YEARS, key=lambda value: (abs(year - value), value))

def grid_area_hectares(latitude: np.ndarray) -> np.ndarray:
    radius_m = 6_371_008.8
    half = np.deg2rad(0.25)
    delta_lon = np.deg2rad(0.5)
    center = np.deg2rad(latitude)
    area_m2 = radius_m**2 * delta_lon * (
        np.sin(center + half) - np.sin(center - half)
    )
    return area_m2 / 10_000.0

def maximum_monthly_growing_area(crop: str, year: int) -> np.ndarray:
    path = (
        MIRCA_ROOT
        / crop
        / "mirca"
        / str(year)
        / "mirca_month_area_total_0p5.npy"
    )
    values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    if values.shape != (12, 360, 720):
        raise RuntimeError(f"Unexpected monthly MIRCA shape at {path}: {values.shape}")
    return np.max(values, axis=0)

def sample_fraction(
    crop: str, years: np.ndarray, rows: np.ndarray, cols: np.ndarray, cell_area: np.ndarray
) -> np.ndarray:
    output = np.zeros(years.size, dtype=np.float64)
    for mirca_year in MIRCA_YEARS:
        selected = np.asarray(
            [nearest_mirca_year(int(year)) == mirca_year for year in years], dtype=bool
        )
        if not np.any(selected):
            continue
        area = maximum_monthly_growing_area(crop, mirca_year)
        output[selected] = (
            area[rows[selected], cols[selected]] / cell_area[rows[selected]]
        )
    return output
