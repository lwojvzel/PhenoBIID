"""Fetch only the registered sample and algorithm document with source checksums."""
import hashlib
import json
from pathlib import Path
import time

import requests

from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json

RAW = ROOT / 'Data/external/muses_gpp_8day_probe'
URL = 'https://zenodo.org/api/records/3996814'
NAMES = ('Algorithm of global gross and net primary productivity products.pdf', 'GPP1981.tar.gz')


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({'User-Agent': 'AgroClimate-research-data-check/1.0'})
        response = session.get(URL, timeout=(20, 60)); response.raise_for_status()
        record = response.json(); atomic_json(RAW / 'record.json', record)
        files = {f['key']: f for f in record['files']}
        if record['id'] != 3996814 or record['metadata']['license']['id'] != 'cc-by-4.0':
            raise ValueError('Unexpected data identity or license')
        completed = []
        for name in NAMES:
            entry = files[name]; path = RAW / name
            size = entry['size']; algorithm, checksum = entry['checksum'].split(':')
            if algorithm != 'md5' or size > 300*2**20:
                raise ValueError('Unregistered sample size or checksum type')
            if path.exists():
                if path.stat().st_size != size or md5(path) != checksum:
                    raise ValueError('Existing sample differs; not overwriting')
            else:
                partial = path.with_suffix(path.suffix+'.part')
                for attempt in range(4):
                    try:
                        start = partial.stat().st_size if partial.exists() else 0
                        if start > size:
                            raise ValueError('Oversized partial file')
                        if start < size:
                            headers = {'Range': f'bytes={start}-'} if start else {}
                            with session.get(entry['links']['self'], headers=headers,
                                             timeout=(30, 180), stream=True) as response:
                                if response.status_code == 429:
                                    wait = min(max(float(response.headers.get('Retry-After', '60')), 60), 600)
                                    time.sleep(wait); response.raise_for_status()
                                response.raise_for_status()
                                if start and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                                    raise ValueError('Server did not honor resume range')
                                with partial.open('ab' if start else 'xb') as stream:
                                    for block in response.iter_content(2**20):
                                        if start+len(block) > size:
                                            raise ValueError('Response exceeds registered size')
                                        stream.write(block); start += len(block)
                        if partial.stat().st_size != size or md5(partial) != checksum:
                            raise ValueError('Downloaded size/checksum mismatch')
                        partial.replace(path); break
                    except requests.RequestException as exc:
                        print(f'[MUSES RETRY] {name} {attempt+1}: {exc}', flush=True)
                        if attempt == 3:
                            raise
                        time.sleep(10*(attempt+1))
            completed.append(dict(name=name, bytes=size, md5=checksum, sha256=sha256(path), url=entry['links']['self']))
            print(f'[MUSES VERIFIED DOWNLOAD] {name} {size}', flush=True)
        atomic_json(RAW / 'download_audit.json', dict(record=URL, files=completed,
            source_checksums_verified=True, years=[1981], models_fitted=0,
            scope='Data suitability probe only; physical units and temporal contents not yet verified.'))


if __name__ == '__main__':
    main()
