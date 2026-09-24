"""Validate all GPP outputs and mappings before opening the feature-build gate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    sys.path.insert(0, str(root / 'scripts'))
    from prepare_reclue_monthly_gpp import OUT, AUDIT, YEARS, mapping_checked
    from review_revision_data import CROPS, sha256
    from run_review_revision_parallel import atomic_json
    files, records = {}, {}
    for year in YEARS:
        path = AUDIT / f'{year}.json'
        record = json.loads(path.read_text())
        if not record['source_format_passed'] or not record['phenology_aligned']:
            raise ValueError(f'Failed GPP processing gate: {year}')
        if sha256(Path(record['archive_path'])) != record['archive_sha256']:
            raise ValueError(f'Changed raw GPP archive: {year}')
        for item in record['outputs']:
            file = Path(item['path'])
            checksum = sha256(file)
            if checksum != item['sha256']:
                raise ValueError(f'Changed GPP output: {file}')
            files[str(file)] = checksum
        for crop in CROPS:
            source = Path(record['crops'][crop]['source_mapping'])
            if sha256(source) != record['crops'][crop]['source_mapping_sha256']:
                raise ValueError('Crop calendar mapping changed')
            mapping = np.load(source)
            for name in ('gpp_monthly_total', 'gpp_daily_rate', 'valid_area_fraction'):
                monthly = np.load(OUT / 'monthly_0p5' / f'{name}_{year}.npy')
                relative = np.load(OUT / 'crops' / crop / name / f'{name}_rel_{year}.npy')
                np.testing.assert_array_equal(relative, mapping_checked(monthly, mapping))
        records[str(path)] = sha256(path)
    if len(files) != 527:
        raise ValueError(f'Expected 527 unique full-period GPP arrays; got {len(files)}')
    atomic_json(OUT / 'manifest.json', dict(complete=True, years=list(YEARS),
        dataset_record='https://zenodo.org/records/14350035',
        audits=records, units='gC m-2 month-1 totals and gC m-2 day-1 rates',
        cohort_support_audited=False, models_fitted=0))
    atomic_json(AUDIT / 'full_period_ready.json', dict(complete=True, years=list(YEARS),
        manifest_sha256=sha256(OUT / 'manifest.json'), files=files,
        all_relative_arrays_rebuilt=True, yield_targets_read=False,
        cohort_support_audited=False, models_fitted=0))
    print('Validated 527 GPP arrays and every crop-month mapping; no yield labels read.')


if __name__ == '__main__':
    main()
