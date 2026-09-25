"""Package processed numeric arrays without publishing private audit paths."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOTS = {
    "crop_active": "Data/processed/crop_yield_growing_season",
    "era5_land": "Data/era5land/monthly_npy_lon180_0p5deg_by_var",
    "gdhy": "Data/GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180",
    "lai": "Data/processed/glass_lai_avhrr_005d",
    "ndvi": "Data/processed/pku_gimms_ndvi_v1p2",
    "gpp": "Data/processed/reclue_monthly_gpp_v1",
    "seas5": "Data/processed/seas5_weather_reliability_v1",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected(root, product):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "_mirca_cache" in p.parts:
            continue
        if p.suffix != ".npy" and p.name != "variable_order.txt":
            continue
        if product == "era5_land" and p.stem.endswith(("_1980", "_2017")):
            continue
        if product == "gdhy" and p.parent != root and p.relative_to(root).parts[0] not in ("maize", "rice", "soybean", "wheat"):
            continue
        yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, shards = [], []
    for product, relative in ROOTS.items():
        files = list(selected(args.workspace / relative, product))
        if not files:
            raise FileNotFoundError(relative)
        groups, group, size = [], [], 0
        for p in files:
            if group and size + p.stat().st_size > 1024**3:
                groups.append(group)
                group, size = [], 0
            group.append(p)
            size += p.stat().st_size
        if group:
            groups.append(group)
        for index, group in enumerate(groups):
            name = f"shards/{product}-{index:03d}.tar.gz"
            target = args.output / name
            target.parent.mkdir(exist_ok=True)
            part_records = []
            temporary = target.with_suffix(".partial")
            with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
                for p in group:
                    rel = p.relative_to(args.workspace).as_posix()
                    rec = dict(path=rel, bytes=p.stat().st_size, sha256=digest(p), product=product, shard=name)
                    if p.suffix == ".npy":
                        a = np.load(p, mmap_mode="r", allow_pickle=False)
                        rec.update(shape=list(a.shape), dtype=str(a.dtype))
                        del a
                    info = archive.gettarinfo(str(p), arcname=rel)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with p.open("rb") as f:
                        archive.addfile(info, f)
                    part_records.append(rec)
            temporary.replace(target)
            records.extend(part_records)
            shards.append(dict(path=name, product=product, bytes=target.stat().st_size, sha256=digest(target), files=len(group)))
            print(f"{name}: {len(group)} files, {target.stat().st_size / 1024**2:.1f} MiB", flush=True)
    manifest = dict(schema_version=1, products=ROOTS, files=records, shards=shards,
                    unpacked_bytes=sum(r["bytes"] for r in records), packed_bytes=sum(s["bytes"] for s in shards))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("unpacked_bytes", "packed_bytes")}), flush=True)


if __name__ == "__main__":
    main()
