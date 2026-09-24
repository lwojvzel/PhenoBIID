"""Decode downloaded SEAS5 members to audited ensemble-mean native grids.

The output keeps the original 1-degree grid. Spatial interpolation to the
0.5-degree crop grid is performed only for requested crop samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import xarray as xr

from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json

RAW = ROOT / "Data/raw/seas5_weather_reliability_v1"
OUT = ROOT / "Data/processed/seas5_weather_reliability_v1/native_1deg"
VARIABLES = ("t2m", "tp", "ssrd")


def convert(ds: xr.Dataset) -> np.ndarray:
    expected = {"number": 25, "forecast_reference_time": 1, "forecastMonth": 6,
                "latitude": 180, "longitude": 360}
    for key, size in expected.items():
        if ds.sizes.get(key) != size:
            raise ValueError(f"Unexpected {key}: {ds.sizes.get(key)}")
    np.testing.assert_allclose(ds.latitude.values, np.arange(89.5, -90, -1))
    np.testing.assert_allclose(ds.longitude.values, np.arange(.5, 360, 1))
    # Match the physical ERA5-Land interface: K, daily J m-2, daily m.
    t2m = ds.t2m.mean(("number", "forecast_reference_time")).values
    raw_tp = ds.tprate.mean(("number", "forecast_reference_time")).values * 86400.
    if float(raw_tp.min()) < -1e-5:  # 0.01 mm day-1: beyond known GRIB quantization noise.
        raise ValueError(f"Unexpected negative precipitation: {float(raw_tp.min())}")
    tp = np.maximum(raw_tp, 0.)
    ssrd = ds.msdsrf.mean(("number", "forecast_reference_time")).values * 86400.
    result = np.stack((t2m, tp, ssrd), -1).astype(np.float32)
    if result.shape != (6, 180, 360, 3) or not np.isfinite(result).all():
        raise ValueError("Invalid converted SEAS5 array")
    if np.any(result[..., 1] < 0):
        raise ValueError("Negative precipitation after clipping")
    return result


def run(year: int, month: int) -> None:
    source = RAW / f"ecmwf51_members_{year}_{month:02d}" / "download.nc"
    validation = source.parent / "validation.json"
    if not source.exists() or not validation.exists():
        raise FileNotFoundError(source)
    meta = json.loads(validation.read_text())
    if meta.get("status") != "validated" or meta.get("sha256") != sha256(source):
        raise ValueError(f"Unvalidated source: {source}")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"seas5_{year}_{month:02d}.npy"
    marker = OUT / f"seas5_{year}_{month:02d}.json"
    if marker.exists():
        done = json.loads(marker.read_text())
        if sha256(target) != done["output_sha256"] or done["source_sha256"] != meta["sha256"]:
            raise ValueError(f"Changed processed file: {target}")
        return
    started = time.monotonic()
    with xr.open_dataset(source) as ds:
        values = convert(ds)
    np.save(target, values)
    atomic_json(marker, dict(year=year, month=month, variables=VARIABLES,
        shape=list(values.shape), source=str(source), source_sha256=meta["sha256"],
        output_sha256=sha256(target), units=["K", "m day-1", "J m-2 day-1"],
        ensemble_members=25, seconds=time.monotonic()-started))
    print(f"[SEAS5 PROCESS] {year}-{month:02d}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=1981)
    p.add_argument("--end-year", type=int, default=2016)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    jobs = [(y, m) for y in range(a.start_year, a.end_year + 1) for m in range(1, 13)]
    for i, (year, month) in enumerate(jobs):
        if i % a.workers == a.worker:
            run(year, month)
