"""Fetch the registered monthly GPP sample without changing existing inputs."""
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fetch_muses_gpp_sample import md5
from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json

RAW = ROOT / 'Data/external/reclue_monthly_gpp_probe'
URL = 'https://zenodo.org/api/records/14350035'
SIZE = 79558668
MD5 = '576903afd3f9b2a742583ef1a57a95b3'


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({'User-Agent': 'AgroClimate-research-data-check/1.0'})
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))))
        response = session.get(URL, timeout=(20, 60)); response.raise_for_status()
        record = response.json()
        if record['id'] != 14350035 or record['metadata']['license']['id'] != 'cc-by-4.0':
            raise ValueError('Unexpected monthly product identity or license')
        entry = next(f for f in record['files'] if f['key'] == '1982.zip')
        if entry['size'] != SIZE or entry['checksum'] != 'md5:'+MD5:
            raise ValueError('Registered sample differs from provider metadata')
        saved = RAW / 'record.json'
        if saved.exists() and json.loads(saved.read_text()) != record:
            raise ValueError('Saved provider metadata changed; preserve for review')
        atomic_json(saved, record)
        path = RAW / '1982.zip'
        if not path.exists():
            partial = RAW / '1982.zip.part'
            start = partial.stat().st_size if partial.exists() else 0
            if start > SIZE:
                raise ValueError('Oversized partial source; not overwriting')
            if start < SIZE:
                headers = {'Range': f'bytes={start}-'} if start else {}
                with session.get(entry['links']['self'], headers=headers,
                                 timeout=(30, 180), stream=True) as response:
                    response.raise_for_status()
                    if start and (response.status_code != 206 or not response.headers.get(
                            'Content-Range', '').startswith(f'bytes {start}-')):
                        raise ValueError('Server did not honor source resume range')
                    with partial.open('ab' if start else 'xb') as stream:
                        for block in response.iter_content(2**20):
                            if start+len(block) > SIZE:
                                raise ValueError('Response exceeds registered size')
                            stream.write(block); start += len(block)
            if partial.stat().st_size != SIZE or md5(partial) != MD5:
                raise ValueError('Downloaded sample size/checksum mismatch')
            partial.replace(path)
        if path.stat().st_size != SIZE or md5(path) != MD5:
            raise ValueError('Existing source checksum mismatch; not overwriting')
        atomic_json(RAW / 'download_audit.json', dict(record=URL, source_checksums_verified=True,
            filename=path.name, bytes=SIZE, md5=MD5, sha256=sha256(path),
            metadata_sha256=sha256(saved), license='CC-BY-4.0', years=[1982], models_fitted=0,
            scope='Monthly source sample only; grid, units and missingness not yet audited.'))
        print(f'[RECLUE VERIFIED DOWNLOAD] {path} {SIZE}', flush=True)


if __name__ == '__main__':
    main()
