"""Extract SEAS5 ensemble-mean weather on the 0.5-degree crop grid."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from review_revision_data import ROOT
from seas5_reliability_protocol import calendar_plan

PROCESSED = ROOT / "Data/processed/seas5_weather_reliability_v1/native_1deg"


def interpolate_native(grid: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Periodic bilinear interpolation from 1-degree centers to crop cells."""
    grid = np.asarray(grid)
    row, col = np.asarray(row, int), np.asarray(col, int)
    if grid.shape[-3:-1] != (180, 360) or row.shape != col.shape:
        raise ValueError("Unexpected grid or crop index shape")
    lat = -89.75 + .5 * row
    lon = np.mod(-179.75 + .5 * col, 360.)
    yi = np.clip(89.5 - lat, 0., 179.)
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, 179); fy = yi - y0
    xi = np.mod(lon - .5, 360.)
    x0 = np.floor(xi).astype(int); x1 = (x0 + 1) % 360; fx = xi - x0
    a = grid[..., y0, x0, :]
    b = grid[..., y0, x1, :]
    c = grid[..., y1, x0, :]
    d = grid[..., y1, x1, :]
    shape = (1,) * (a.ndim - fx.ndim - 1) + fx.shape + (1,)
    fx, fy = fx.reshape(shape), fy.reshape(shape)
    return (a * (1-fx) * (1-fy) + b * fx * (1-fy) + c * (1-fx) * fy + d * fx * fy)


def extract(year, source_month, active, row, col, ratio):
    """Return [N,12,3] forecasts and a sample-level full-coverage mask."""
    year = np.asarray(year, int); source_month = np.asarray(source_month, int)
    active = np.asarray(active, bool); row = np.asarray(row, int); col = np.asarray(col, int)
    plan = calendar_plan(year, source_month, active, ratio)
    result = np.full((*active.shape, 3), np.nan, np.float32)
    pair = np.column_stack((plan["initialization_year"], plan["initialization_month"]))
    for init_year, init_month in np.unique(pair, axis=0):
        rows = np.flatnonzero((pair[:, 0] == init_year) & (pair[:, 1] == init_month))
        file = PROCESSED / f"seas5_{init_year}_{init_month:02d}.npy"
        if not file.exists():
            raise FileNotFoundError(file)
        native = np.load(file, mmap_mode="r")
        for lead in range(1, 7):
            # Recover slot indices after selecting from the row x slot mask.
            rr, kk = np.nonzero((pair[:, 0, None] == init_year) &
                                (pair[:, 1, None] == init_month) &
                                (plan["leadtime_month"] == lead))
            if not len(rr):
                continue
            vals = interpolate_native(native[lead-1], row[rr], col[rr])
            result[rr, kk] = vals
    hidden = plan["hidden"]
    if np.any(plan["supported"] & hidden.any(1) & ~np.all(np.isfinite(result) | ~hidden[..., None], axis=(1,2))):
        raise ValueError("Supported samples contain missing forecasts")
    return result, plan
