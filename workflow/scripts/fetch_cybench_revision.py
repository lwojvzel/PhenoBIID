"""Read selected official ZIP members through validated HTTP byte ranges."""

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1] / "Data/external/CYBench/full_v1_10"
URL = "https://zenodo.org/api/records/17279151/files/cybench-data.zip/content"
SIZE = 6228749411


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.blocks = OrderedDict()
        self.block_size = 4 * 1024 * 1024
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.pending = {}
        self.member_end_block = -1

    def fetch_block(self, number):
        start = number * self.block_size
        end = min(SIZE-1, start+self.block_size-1)
        directory = ROOT / "range_cache"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{number}.bin"
        if path.exists() and path.stat().st_size == end-start+1:
            return path.read_bytes()
        for attempt in range(4):
            try:
                with requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise IOError("Server did not honor exact byte range")
                    block = response.content
                if len(block) != end-start+1:
                    raise IOError("Incomplete range")
                temporary = path.with_suffix('.part')
                temporary.write_bytes(block)
                temporary.replace(path)
                return block
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def begin_member(self, member):
        self.member_end_block = min(SIZE-1, member.header_offset + member.compress_size + len(member.filename.encode()) + len(member.extra) + 256) // self.block_size

    def prefetch(self, number):
        for n in range(number, min(number+8, self.member_end_block+1)):
            if n not in self.blocks and n not in self.pending:
                self.pending[n] = self.executor.submit(self.fetch_block, n)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        if self.position < 0:
            raise ValueError("Negative position")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = SIZE - self.position
        remaining = min(size, SIZE-self.position)
        result = []
        while remaining > 0:
            number, offset = divmod(self.position, self.block_size)
            self.prefetch(number)
            if number not in self.blocks:
                future = self.pending.pop(number, None)
                self.blocks[number] = future.result() if future else self.fetch_block(number)
                if len(self.blocks) > 3:
                    self.blocks.popitem(last=False)
            self.blocks.move_to_end(number)
            part = self.blocks[number][offset:offset+remaining]
            result.append(part)
            self.position += len(part)
            remaining -= len(part)
        return b"".join(result)

    def close(self):
        self.executor.shutdown(wait=True)
        self.session.close()
        super().close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--countries", default="DE,FR,PL")
    args = p.parse_args()
    countries = args.countries.split(",")
    ROOT.mkdir(parents=True, exist_ok=True)
    with RemoteZip() as remote, zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
        listing = [dict(name=m.filename, size=m.file_size, compressed_size=m.compress_size, crc=m.CRC) for m in members]
        (ROOT / "archive_index.json").write_text(json.dumps(listing, indent=2))
        selected = []
        for crop in ("maize", "wheat"):
            for country in countries:
                for prefix in ("location", "crop_calendar", "yield", "fpar", "meteo"):
                    filename = f"{prefix}_{crop}_{country}.csv"
                    matches = [m for m in members if Path(m.filename).name == filename]
                    if len(matches) != 1:
                        raise RuntimeError(f"Expected exactly one {filename}; found {len(matches)}")
                    selected.append((crop, country, matches[0]))
        print(json.dumps({"members": len(members), "selected": len(selected), "compressed_bytes": sum(m.compress_size for _, _, m in selected), "countries": countries}), flush=True)
        if args.list:
            return
        for crop, country, member in selected:
            output = ROOT / crop / country / Path(member.filename).name
            done = output.with_suffix(".csv.source.json")
            if output.exists() and done.exists() and output.stat().st_size == member.file_size:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".csv.part")
            start = time.monotonic()
            remote.begin_member(member)
            with archive.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(output)
            done.write_text(json.dumps({"record": 17279151, "archive": URL, "member": member.filename, "size": member.file_size, "zip_crc32_verified": member.CRC, "license": "EUPL-1.2 as listed by data record", "seconds": time.monotonic()-start}, indent=2))
            print(f"[FETCH] {crop}/{country}/{output.name} {member.file_size} bytes", flush=True)
    (ROOT / "selected_download_complete.json").write_text(json.dumps({"countries": countries, "crops": ["maize", "wheat"], "record": 17279151, "full_archive_downloaded": False, "members": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
