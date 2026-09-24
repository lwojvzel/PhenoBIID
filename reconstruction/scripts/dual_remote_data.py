from __future__ import annotations
import numpy as np
from prepare_pku_ndvi import OUT
def extract(a, previous=False):
    n = len(a["year"])
    values = np.full((n, 12), np.nan, np.float32)
    quality = np.zeros_like(values)
    for year in np.unique(a["year"]):
        rows = np.flatnonzero(a["year"] == year)
        source_year = int(year) - int(previous)
        path = OUT / "monthly_0p5" / f"ndvi_monthly_{source_year}.npy"
        if source_year < 1982:
            continue
        grid = np.load(path, mmap_mode="r")
        qgrid = np.load(OUT / "monthly_0p5" / f"quality_fraction_monthly_{source_year}.npy", mmap_mode="r")
        month = np.minimum(a["source_month"][rows], 11)
        r, c = a["row"][rows, None], a["col"][rows, None]
        valid = (a["relative_valid"][rows] > 0) & (a["source_month"][rows] != 255)
        values[rows] = np.where(valid, grid[month, r, c], np.nan)
        quality[rows] = np.where(valid, qgrid[month, r, c], 0)
    return values, quality
