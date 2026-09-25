"""Download and verify CropDynamicsBench shards, then restore Data/ paths."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_verified(archive, workspace, records):
    """Accept only manifest-listed regular files and never overwrite changed data."""
    workspace = Path(workspace).resolve()
    expected = {r["path"]: r for r in records}
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError(f"Unsafe archive member: {member.name}")
            rec = expected.get(member.name)
            if rec is None or member.name in seen or member.size != rec["bytes"]:
                raise ValueError(f"Unexpected archive member: {member.name}")
            seen.add(member.name)
            target = (workspace / member.name).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError(f"Destination escapes workspace: {member.name}")
            if target.exists():
                if digest(target) != rec["sha256"]:
                    raise FileExistsError(f"Existing data differs; choose an empty workspace: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".download-partial")
            with tar.extractfile(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if digest(temporary) != rec["sha256"]:
                raise ValueError(f"Checksum mismatch: {member.name}")
            temporary.replace(target)
    if seen != set(expected):
        raise ValueError("Archive is missing manifest-listed files")


def main():
    from huggingface_hub import HfApi, hf_hub_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--repo-id", default="PHENOBIID/CropDynamicsBench")
    p.add_argument("--revision", help="Override the released commit; main is resolved once before download")
    p.add_argument("--products", nargs="+", default=["crop_active", "era5_land", "gdhy", "lai", "ndvi", "gpp"])
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--list-only", action="store_true")
    args = p.parse_args()
    lock_path = Path(__file__).resolve().parents[1] / "configs/processed_release.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    requested = args.revision or (lock.get("revision") if args.repo_id == lock.get("repo_id") else None) or "main"
    revision = HfApi().repo_info(args.repo_id, repo_type="dataset", revision=requested).sha
    kwargs = dict(repo_id=args.repo_id, repo_type="dataset", revision=revision, cache_dir=args.cache_dir)
    manifest_path = Path(hf_hub_download(filename="manifest.json", **kwargs))
    if args.repo_id == lock.get("repo_id") and revision == lock.get("revision"):
        if digest(manifest_path) != lock["manifest_sha256"]:
            raise ValueError("Release manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    products = set(manifest["products"]) if args.products == ["all"] else set(args.products)
    if products - set(manifest["products"]):
        p.error(f"Unknown products: {sorted(products - set(manifest['products']))}")
    shards = [s for s in manifest["shards"] if s["product"] in products]
    print(f"Dataset revision: {revision}", flush=True)
    print(f"{len(shards)} shards; {sum(s['bytes'] for s in shards) / 1024**3:.2f} GiB download", flush=True)
    if args.list_only:
        for s in shards:
            print(s["path"], s["bytes"])
        return
    for s in shards:
        local = hf_hub_download(filename=s["path"], **kwargs)
        if digest(local) != s["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {s['path']}")
        records = [r for r in manifest["files"] if r["shard"] == s["path"]]
        extract_verified(local, args.workspace, records)
        print(f"Verified and extracted {s['path']}", flush=True)
    receipt = dict(repo_id=args.repo_id, revision=revision, products=sorted(products), verified_files=sum(s["files"] for s in shards))
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "cropdynamicsbench_download.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
