"""Verified annual GPP ingestion; never treat annual files as 8-day states."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urljoin
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import numpy as np
from review_revision_data import ROOT, sha256
from multimodal_baseline import save_json

BASE = "https://glass.hku.hk/archive/GPP/AVHRR/0.05D/GLASS_GPP_0.05D_YEARLY_V40/"
RAW = ROOT / "Data/external/GLASS_GPP_AVHRR_annual_v40"
OUTPUT = ROOT / "Data/processed/glass_gpp_avhrr_annual_0p5"


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.values = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.endswith(".hdf"): self.values.append(href)


def fetch(year):
    destination = RAW / str(year); destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "download.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        path = destination / meta["filename"]
        if path.exists() and sha256(path) == meta["sha256"]: return path, meta
    error = None
    for attempt in range(3):
        try:
            index_url = urljoin(BASE, f"{year}/")
            with urlopen(index_url, timeout=60) as response: index = response.read().decode()
            parser = Links(); parser.feed(index)
            names = [n for n in parser.values if n.startswith("GLASS12B12.")]
            if len(names) != 1: raise ValueError((year, names))
            name = names[0]; url = urljoin(index_url, name)
            with urlopen(url + ".xml", timeout=60) as response: xml = response.read()
            document = ET.fromstring(xml)
            declared_size = int(document.findtext(".//FileSize"))
            declared_checksum = document.findtext(".//Checksum")
            if document.findtext(".//ChecksumType") != "CKSUM": raise ValueError("Unknown checksum")
            path = destination / name; partial = path.with_suffix(".partial")
            sample = Path("/tmp") / name
            if sample.exists() and sample.stat().st_size == declared_size:
                shutil.copy2(sample, partial)
            else:
                with urlopen(url, timeout=180) as response, partial.open("wb") as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
            actual_checksum, actual_size, *_ = subprocess.check_output(["cksum", str(partial)], text=True).split()
            if actual_checksum != declared_checksum or int(actual_size) != declared_size:
                raise ValueError(f"Source checksum mismatch: {year}")
            partial.replace(path)
            path.with_suffix(".hdf.xml").write_bytes(xml)
            meta = {"year": year, "filename": name, "url": url, "bytes": declared_size,
                    "cksum": actual_checksum, "sha256": sha256(path),
                    "archive": "AVHRR annual V4.0", "xml_range_end": document.findtext(".//RangeEndingDate"),
                    "temporal_warning": "XML uses an 8-day template; authoritative SDS long_name and units identify annual accumulation"}
            save_json(meta, marker)
            return path, meta
        except Exception as exc:
            error = exc
            if attempt < 2: time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GPP download failed for {year}: {error}")


def aggregate(raw, attrs):
    if raw.shape != (3600, 7200): raise ValueError(raw.shape)
    if "Annual Accumulation" not in attrs["long_name"] or attrs["units"] != "gC m-2 year-1":
        raise ValueError("Annual GPP semantics not verified")
    lo, hi = attrs["valid_range"]
    valid = np.isfinite(raw) & (raw >= lo) & (raw <= hi)
    physical = raw.astype(np.float64) * float(attrs["scale_factor"]) + float(attrs["add_offset"])
    edges = np.deg2rad(90 - np.arange(3601) * 0.05)
    weights = (np.sin(edges[:-1]) - np.sin(edges[1:]))[:, None]
    numerator = (np.where(valid, physical, 0) * weights).reshape(360, 10, 720, 10).sum((1, 3))
    denominator = (valid * weights).reshape(360, 10, 720, 10).sum((1, 3))
    means = np.divide(numerator, denominator, out=np.full((360, 720), np.nan), where=denominator > 0)
    full = 10 * weights.reshape(360, 10).sum(1)[:, None]
    coverage = denominator / full
    return np.flipud(means).astype(np.float32), np.flipud(coverage).astype(np.float32)


def main():
    from pyhdf.SD import SD, SDC
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1981)
    parser.add_argument("--end", type=int, default=2016)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    items = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, y) for y in range(args.start, args.end + 1)]
        # HDF4 reads are kept in one thread; only HTTP downloads are concurrent.
        for future in as_completed(futures):
            path, item = future.result()
            hdf = SD(str(path), SDC.READ)
            try:
                sds = hdf.select("GPP"); attrs = sds.attributes()
                gpp, coverage = aggregate(sds.get(), attrs)
            finally: hdf.end()
            year = item["year"]
            out = OUTPUT / f"gpp_annual_{year}.npy"
            np.save(out, gpp); np.save(OUTPUT / f"valid_fraction_{year}.npy", coverage)
            item.update(sds_attributes=attrs, output_sha256=sha256(out),
                        finite_cells=int(np.isfinite(gpp).sum()))
            items.append(item)
            save_json(item, OUTPUT / f"metadata_{year}.json")
            print(f"[GPP] {year}: verified, aggregated, {len(items)}/{len(futures)}", flush=True)
    save_json({"complete": True, "years": [args.start, args.end], "items": sorted(items, key=lambda x:x["year"]),
               "grid": "360x720; latitude south-to-north; longitude -180..180",
               "spatial_aggregation": "spherical-area-weighted valid 0.05-degree pixels",
               "units": "gC m-2 year-1", "annual_not_slotwise": True,
               "independent_crop_ground_truth": False,
               "source_dependency": "EC-LUE remotely sensed/environmental model; not crop-masked",
               "metadata_caveat": "Annual SDS units/long_name override stale XML 8-day range; source archive says AVHRR while HDF template retains MODIS fields",
               "release_status": "Local research use only; no redistribution license inferred"}, OUTPUT / "manifest.json")


if __name__ == "__main__": main()
