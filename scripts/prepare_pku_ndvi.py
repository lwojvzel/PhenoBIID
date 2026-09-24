"""Download verified PKU GIMMS NDVI and align it to existing MIRCA slots."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import shutil

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data/external/PKU_GIMMS_NDVI_v1p2"
OUT = ROOT / "Data/processed/pku_gimms_ndvi_v1p2"
RECORD = "https://zenodo.org/api/records/8253971"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def segmented_file(item, path):
    folder = RAW / "segments" / path.name
    folder.mkdir(parents=True, exist_ok=True)
    size = 8 * 2**20
    url = f"https://zenodo.org/records/8253971/files/{item['key']}"
    def fetch(start):
        end = min(start + size, item["size"]) - 1
        destination = folder / f"{start:012d}.bin"
        expected = end - start + 1
        if destination.exists() and destination.stat().st_size == expected:
            return destination
        part = destination.with_suffix(".part")
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            if offset == expected:
                part.replace(destination)
                return destination
            try:
                with requests.get(url, headers={"Range": f"bytes={start+offset}-{end}"},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not content_range.startswith(f"bytes {start+offset}-{end}/"):
                        raise ValueError((response.status_code, content_range))
                    with part.open("ab") as f:
                        for chunk in response.iter_content(262144):
                            f.write(chunk)
                if part.stat().st_size != expected:
                    raise requests.ConnectionError("Truncated byte range")
                part.replace(destination)
                print(f"[SEGMENT] {path.name} {end+1}/{item['size']}", flush=True)
                return destination
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(15 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        segments = list(pool.map(fetch, range(0, item["size"], size)))
    temporary = path.with_suffix(".assembled")
    with temporary.open("wb") as target:
        for segment in segments:
            with segment.open("rb") as source:
                shutil.copyfileobj(source, target, 2**20)
    algorithm, checksum = item["checksum"].split(":")
    if temporary.stat().st_size != item["size"] or digest(temporary, algorithm) != checksum:
        raise ValueError(f"Reassembled archive failed publisher checksum: {temporary}")
    temporary.replace(path)


def download(readme_only=False, file_index=None):
    RAW.mkdir(parents=True, exist_ok=True)
    meta_path = RAW / "zenodo_record.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        response = requests.get(RECORD, timeout=90)
        response.raise_for_status()
        meta = response.json()
        save_json(meta_path, meta)
    files = [f for f in meta["files"] if f["key"].endswith(".pdf") or
             (not readme_only and "AVHRR_MODIS_consolidated" in f["key"])]
    files.sort(key=lambda f: (not f["key"].endswith(".pdf"), f["key"]))
    if file_index is not None:
        files = [files[file_index]]
    verified = []
    for item in files:
        path = RAW / item["key"]
        algorithm, expected = item["checksum"].split(":")
        if path.exists():
            if path.stat().st_size != item["size"] or digest(path, algorithm) != expected:
                raise ValueError(f"Existing file failed integrity check: {path}")
        elif path.suffix == ".zip":
            segmented_file(item, path)
        else:
            part = path.with_suffix(path.suffix + ".part")
            for attempt in range(5):
                offset = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    url = f"https://zenodo.org/records/8253971/files/{item['key']}"
                    with requests.get(url, headers=headers,
                                      stream=True, timeout=(30, 180)) as response:
                        response.raise_for_status()
                        if response.status_code == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned mismatched byte range")
                        mode = "ab" if offset and response.status_code == 206 else "wb"
                        with part.open(mode) as f:
                            for chunk in response.iter_content(2**20):
                                f.write(chunk)
                    if part.stat().st_size != item["size"] or digest(part, algorithm) != expected:
                        raise ValueError(f"Integrity mismatch: {part}")
                    part.replace(path)
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))
        verified.append({"path": str(path), "bytes": path.stat().st_size,
                         "publisher_checksum": item["checksum"], "sha256": digest(path),
                         "url": item["links"]["self"]})
        print(f"[VERIFIED] {path.name} {path.stat().st_size} bytes", flush=True)
    name = "readme_manifest.json" if readme_only else (f"download_part_{file_index}.json" if file_index is not None else "download_manifest.json")
    save_json(RAW / name, verified)
    return verified


def quality_mask(raw, qc, year):
    physical = (raw <= 1000) & (raw != 65535)
    if year <= 2002:
        good = ((qc // 100 >= 1) & (qc // 100 <= 5) &
                ((qc // 10) % 10 == 0) & (qc % 10 == 9))
    else:
        good = (qc == 990)
    return physical & good, physical


def aggregate(raw, mask):
    if raw.shape != (2160, 4320) or mask.shape != raw.shape:
        raise ValueError((raw.shape, mask.shape))
    edges = np.deg2rad(90 - np.arange(2161) / 12)
    w = (np.sin(edges[:-1]) - np.sin(edges[1:]))[:, None]
    denominator = (mask * w).reshape(360, 6, 720, 6).sum((1, 3))
    numerator = (np.where(mask, raw, 0) * w * 0.001).reshape(360, 6, 720, 6).sum((1, 3))
    mean = np.divide(numerator, denominator, out=np.full((360, 720), np.nan), where=denominator > 0)
    full = 6 * w.reshape(360, 6).sum(1)[:, None]
    return np.flipud(mean).astype(np.float32), np.flipud(denominator / full).astype(np.float32)


def process(start=1982, end=2016):
    import rasterio
    from affine import Affine
    from process_glass_lai_avhrr_to_growing_season import nearest_mirca_year, reorder_relative_months
    from review_revision_data import CROPS
    index = {}
    for archive in sorted(RAW.glob("*consolidated*.zip")):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                match = re.search(r"_(\d{4})(\d{2})(0[12])\.tif$", name, re.I)
                if not match:
                    continue
                key = tuple(map(int, match.groups()))
                if key in index:
                    raise ValueError(f"Duplicate composite: {key}")
                index[key] = (archive, name)
    OUT.mkdir(parents=True, exist_ok=True)
    growing = ROOT / "Data/processed/crop_yield_growing_season"
    for coord in ("lat", "lon"):
        np.save(OUT / f"{coord}.npy", np.load(growing / f"{coord}.npy"))
    for year in range(start, end + 1):
        marker = OUT / "metadata" / f"{year}.json"
        if marker.exists():
            print(f"[EXISTS] NDVI {year}", flush=True)
            continue
        monthly = np.full((12, 360, 720), np.nan, dtype=np.float32)
        quality = np.zeros_like(monthly)
        all_fraction = np.zeros_like(monthly)
        sources = []
        for month in range(1, 13):
            accum = np.zeros((360, 720), np.float64)
            mass = np.zeros_like(accum)
            days = calendar.monthrange(year, month)[1]
            for half, duration in ((1, 15), (2, days - 15)):
                archive, name = index[(year, month, half)]
                with rasterio.open(f"/vsizip/{archive}/{name}") as ds:
                    if (ds.height, ds.width, ds.count) != (2160, 4320, 2):
                        raise ValueError((ds.height, ds.width, ds.count))
                    if ds.crs.to_epsg() != 4326 or not ds.transform.almost_equals(Affine(1/12, 0, -180, 0, -1/12, 90), precision=1e-6):
                        raise ValueError((ds.crs, ds.transform))
                    raw, qc = ds.read()
                good, physical = quality_mask(raw, qc, year)
                value, fraction = aggregate(raw, good)
                _, available = aggregate(raw, physical)
                finite = np.isfinite(value)
                accum += np.where(finite, value, 0) * duration
                mass += finite * duration
                quality[month-1] += fraction * (duration / days)
                all_fraction[month-1] += available * (duration / days)
                sources.append({"archive": archive.name, "member": name, "days": duration,
                                "good_native_pixels": int(good.sum()),
                                "physical_native_pixels": int(physical.sum())})
            monthly[month-1] = np.divide(accum, mass, out=np.full_like(accum, np.nan), where=mass > 0)
        paths = []
        for label, data in (("ndvi", monthly), ("quality_fraction", quality), ("available_fraction", all_fraction)):
            folder = OUT / "monthly_0p5"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{label}_monthly_{year}.npy"
            np.save(path, data)
            paths.append(path)
            for crop in CROPS:
                src = np.load(growing / crop / "mirca" / str(nearest_mirca_year(year)) / "src_rel.npy")
                rel = reorder_relative_months(data, src)
                folder = OUT / "crops" / crop / f"{label}_rel"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{label}_rel_{year}.npy"
                np.save(path, rel)
                paths.append(path)
        save_json(marker, {"year": year, "sources": sources,
                           "output_hashes": {str(p.relative_to(OUT)): digest(p) for p in paths}})
        print(f"[PROCESSED] NDVI {year}: 24 composites -> monthly -> 4 crops", flush=True)
    save_json(OUT / "manifest.json", {
        "complete": True, "years": [start, end], "product": "PKU GIMMS NDVI V1.2 consolidated",
        "source_record": RECORD, "source_units": "unitless; scale 0.001; fill 65535",
        "quality": "1982-2002: method 1..5, AVHRR good (0), MODIS NA (9); 2003+: QC990",
        "missing": "NaN, not zero; 1981 absent and never backfilled with later years",
        "aggregation": "6x6 spherical-area valid-pixel mean; half-month day-weighted mean",
        "quality_fraction": "good-pixel spherical area times duration / full grid area times month duration",
        "available_fraction": "same, ignoring QC; diagnostic only, not used to fill good NDVI",
        "orientation": "360x720 south-to-north; longitude -179.75..179.75",
        "phenology": "reuse immutable per-crop src_rel; same-year packed slots, not a new cross-year fix",
        "product_switch": "consolidated AVHRR through 2002; MODIS from 2003",
        "independence": "not independent of optical vegetation signals; not crop-pure; not causal real-time product"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme-only", action="store_true")
    parser.add_argument("--process", action="store_true")
    args = parser.parse_args()
    if args.process:
        process()
    elif not args.readme_only:
        readme = download(True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(lambda i: download(False, i), range(1, 5)))
        save_json(RAW / "download_manifest.json", readme + [item for part in parts for item in part])
    else:
        download(args.readme_only)
