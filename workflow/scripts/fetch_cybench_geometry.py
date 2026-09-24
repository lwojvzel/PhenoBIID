"""Fetch explicit polygon members from CY-Bench v1.10, retaining provenance."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import fetch_cybench_revision as upstream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Data/external/CYBench/geometry_v1_10'
URL = 'https://zenodo.org/api/records/17279151/files/polygons.zip/content'
SIZE = 104982885
EXPECTED_MD5 = '815d0e94f6746f15febb99b627142a04'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs='*', default=[])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # A distinct cache root prevents interpreting existing 4-MiB chunks as 512-KiB.
    upstream.ROOT = OUT / 'ranges_512k'
    upstream.URL, upstream.SIZE = URL, SIZE
    with upstream.RemoteZip() as remote:
        remote.block_size = 512 * 1024
        with zipfile.ZipFile(remote) as archive:
            members = archive.infolist()
            index = [dict(name=m.filename, size=m.file_size,
                          compressed_size=m.compress_size, crc32=m.CRC)
                     for m in members]
            (OUT / 'archive_index.json').write_text(json.dumps(index, indent=2)+'\n')
            print(json.dumps(dict(members=len(index), requested=args.members,
                                  names=[m['name'] for m in index])), flush=True)
            for name in args.members:
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Unsafe member path: {name}')
                member = archive.getinfo(name)
                if member.is_dir():
                    raise ValueError('Select files, not directories')
                output = OUT / 'selected' / relative
                marker = output.with_name(output.name+'.source.json')
                if output.exists() and marker.exists():
                    saved = json.loads(marker.read_text())
                    digest = hashlib.sha256(output.read_bytes()).hexdigest()
                    if saved['sha256'] == digest and saved['crc32'] == member.CRC:
                        continue
                    raise ValueError(f'Existing member changed: {output}')
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name+'.part')
                remote.begin_member(member)
                with archive.open(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, length=1024*1024)
                if temporary.stat().st_size != member.file_size:
                    raise ValueError('Extracted member has wrong length')
                temporary.replace(output)
                record = dict(record=17279151, archive=URL, archive_bytes=SIZE,
                    archive_md5_expected=EXPECTED_MD5, full_archive_md5_verified=False,
                    member=name, bytes=member.file_size, crc32=member.CRC,
                    crc32_verified=True,
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    license='Check original boundary-provider terms; record license is not blanket permission')
                marker.write_text(json.dumps(record, indent=2)+'\n')
                print(f'[GEOMETRY] {name}: {member.file_size} bytes, CRC verified', flush=True)


if __name__ == '__main__':
    main()
