"""Rebuild all GDHY, ERA5 and MIRCA foundation arrays outside the project."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'benchmark/results/inseason_reproducibility_v1/raw_foundation_20260909'
ENV = ROOT / 'Paper/release/inseason_raw_processing_env'
GDHY = Path('Data/GDHY/gdhy_v1.2_v1.3_20190128')
GDHY_OUT = Path('Data/GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180')
ERA5 = Path('Data/era5land/monthly')
ERA5_OUT = Path('Data/era5land/monthly_npy_lon180_0p5deg')
SPLIT_OUT = Path('Data/era5land/monthly_npy_lon180_0p5deg_by_var')
MIRCA = Path('Data/MIRCA-OS/Monthly Growing Area Grids/Monthly Growing Area Grids')
CROP_OUT = Path('Data/processed/crop_yield_growing_season')
SCRIPTS = ('convert_gdhy_to_npy_lon180.py', 'aggregate_era5land_monthly_to_gdhy_npy.py',
           'split_era5land_0p5deg_npy_by_variable.py', 'build_crop_growing_season_dataset.py')
PACKAGES = ['numpy==2.0.2', 'netCDF4==1.7.2', 'cftime==1.6.4.post1', 'certifi==2026.2.25']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def raw_inventory():
    gdhy = sorted((ROOT / GDHY).glob('*/*.nc4'))
    groups = {}
    for p in gdhy:
        groups.setdefault(p.parent.name, set()).add(int(p.stem.rsplit('_', 1)[-1]))
    assert len(groups) == 10 and len(gdhy) == 360
    assert all(years == set(range(1981, 2017)) for years in groups.values())
    era5 = sorted((ROOT / ERA5).glob('era5land_monthly_*.nc'))
    assert len(era5) == 38
    assert {int(p.stem.rsplit('_', 1)[-1]) for p in era5} == set(range(1980, 2018))
    patterns = ('Maize', 'Rice_1', 'Rice_2', 'Rice_3', 'Soybeans', 'Wheat_1', 'Wheat_2')
    mirca = [ROOT / MIRCA / str(y) / f'MIRCA-OS_{crop}_{y}_{system}.nc'
             for y in (2000, 2005, 2010, 2015) for crop in patterns for system in ('ir', 'rf')]
    assert len(mirca) == 56 and all(p.is_file() for p in mirca)
    return gdhy + era5 + mirca


def compare_array(generated, reference):
    a, b = np.load(generated, mmap_mode='r'), np.load(reference, mmap_mode='r')
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f'Array contract differs: {generated}')
    finite = missing = 0
    # Comparison covers every element, using slabs to bound temporary memory.
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    for start in range(0, len(rows_a), 4096):
        x, y = rows_a[start:start+4096], rows_b[start:start+4096]
        if np.issubdtype(a.dtype, np.inexact):
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            missing += int(np.isnan(x).sum())
            finite += int(np.isfinite(x).sum())
        else:
            same = x == y
            finite += x.size
        if not same.all():
            raise ValueError(f'Full array mismatch: {generated}, slab {start}')
    return dict(shape=list(a.shape), dtype=str(a.dtype), elements=int(a.size),
                finite_elements=finite, nan_elements=missing,
                generated_sha256=digest(generated), reference_sha256=digest(reference),
                exact_values_and_missingness=True)


def run_logged(command, name, cwd, env=None):
    log = EVIDENCE / f'{name}.log'
    record = dict(command=list(map(str, command)), cwd=str(cwd), started=time.time())
    write_json(EVIDENCE / f'{name}_command.json', record)
    print(f'[START] {name}', flush=True)
    with log.open('xb') as stream:
        result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, seconds=time.time()-record['started'], log_sha256=digest(log))
    write_json(EVIDENCE / f'{name}_command.json', record)
    if result.returncode:
        raise RuntimeError(f'{name} failed, see {log}')
    print(f'[FINISHED] {name} {record["seconds"]:.1f}s', flush=True)


def main():
    global EVIDENCE, ENV
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit('Use --run for the registered full foundation reconstruction')
    if args.attempt < 1:
        raise ValueError('Attempt identifiers must be positive')
    if args.attempt > 1:
        EVIDENCE = EVIDENCE.with_name(EVIDENCE.name + f'_attempt{args.attempt}')
        ENV = ENV.with_name(ENV.name + f'_attempt{args.attempt}')
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    raw = raw_inventory()
    workspace = Path(tempfile.mkdtemp(prefix='raw-foundation-', dir=ROOT.parent / 'AgroClimate_reproduction_runs'))
    (workspace / 'core').mkdir()
    scripts = {}
    for name in SCRIPTS:
        source = ROOT / 'scripts' / name
        shutil.copy2(source, workspace / 'core' / name)
        scripts[name] = digest(source)
    guard = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_guarded.py'
    shutil.copy2(guard, workspace / 'run_guarded.py')
    for relative in (GDHY, ERA5, MIRCA):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / relative, target_is_directory=True)
    initial = dict(attempt=args.attempt, original_project=str(ROOT), workspace=str(workspace), raw_files=list(map(str, raw)),
                   raw_count=len(raw), scripts=scripts, packages=PACKAGES,
                   guard_sha256=digest(guard), driver_sha256=digest(Path(__file__)),
                   new_scientific_comparisons=0, reference_arrays_used_for_processing=False,
                   remaining_stages=['remote_sensing', 'cohort_features', 'full_new_input_evaluation', 'regional_inputs'])
    write_json(EVIDENCE / 'registration.json', initial)
    write_json(workspace / 'registration.json', initial)
    print(f'[REGISTERED] {workspace}; {len(raw)} raw files', flush=True)
    raw_hashes = {}
    for i, path in enumerate(raw, 1):
        raw_hashes[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
        if i % 40 == 0 or i == len(raw):
            print(f'[SOURCE HASH] {i}/{len(raw)}', flush=True)
    write_json(EVIDENCE / 'raw_sources.json', raw_hashes)
    references = {}
    for relative in (GDHY_OUT, ERA5_OUT, SPLIT_OUT, CROP_OUT):
        paths = sorted((ROOT / relative).rglob('*.npy'))
        references[str(relative)] = {str(p.relative_to(ROOT / relative)): digest(p) for p in paths}
        print(f'[REFERENCE REGISTERED] {relative} {len(paths)} arrays', flush=True)
    assert len(references[str(GDHY_OUT)]) == 362
    assert len(references[str(ERA5_OUT)]) == 40
    assert len(references[str(SPLIT_OUT)]) == 496
    assert len(references[str(CROP_OUT)]) == 450
    write_json(EVIDENCE / 'reference_arrays.json', references)
    if ENV.exists():
        raise FileExistsError('Do not alter an existing processing environment')
    run_logged([sys.executable, '-m', 'venv', ENV], 'environment_create', workspace)
    python = ENV / 'bin/python'
    run_logged([python, '-m', 'pip', 'install', '--disable-pip-version-check', *PACKAGES],
               'environment_install', workspace)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (EVIDENCE / 'pip_freeze.txt').write_text(freeze)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    stages = [
        ('gdhy', GDHY_OUT, [SCRIPTS[0], '--source', str(GDHY), '--output', str(GDHY_OUT)]),
        ('era5', ERA5_OUT, [SCRIPTS[1]]),
        ('era5_split', SPLIT_OUT, [SCRIPTS[2]]),
        ('crop_slots', CROP_OUT, [SCRIPTS[3]]),
    ]
    totals = {}
    for stage, relative, command in stages:
        assert not (workspace / relative).exists(), 'Fresh processing must not reuse cached outputs'
        run_logged([python, workspace / 'run_guarded.py', *command], stage, workspace, env)
        produced = sorted((workspace / relative).rglob('*.npy'))
        names = {str(p.relative_to(workspace / relative)) for p in produced}
        assert names == set(references[str(relative)]), (stage, 'Incomplete output matrix')
        rows = {}
        for index, path in enumerate(produced, 1):
            name = str(path.relative_to(workspace / relative))
            reference = ROOT / relative / name
            if digest(reference) != references[str(relative)][name]:
                raise ValueError(f'Original reference changed: {reference}')
            rows[name] = compare_array(path, reference)
            if index % 50 == 0 or index == len(produced):
                print(f'[ARRAY CHECK] {stage} {index}/{len(produced)}', flush=True)
        total = dict(passed=True, arrays=len(rows), elements=sum(r['elements'] for r in rows.values()),
                     byte_identical_arrays=sum(r['generated_sha256']==r['reference_sha256'] for r in rows.values()),
                     arrays_detail=rows, numerical_and_mask_match=True)
        write_json(EVIDENCE / f'{stage}_verification.json', total)
        totals[stage] = {k:v for k,v in total.items() if k != 'arrays_detail'}
        print(f'[STAGE VERIFIED] {stage}: {total["arrays"]} arrays', flush=True)
    for path, item in raw_hashes.items():
        assert digest(path) == item['sha256'], path
    for name, expected in scripts.items():
        assert digest(ROOT / 'scripts' / name) == digest(workspace / 'core' / name) == expected
    result = dict(passed=True, foundation_reprocessed=True, stages=totals,
                  original_raw_unchanged=True, original_arrays_unchanged=True,
                  workspace=str(workspace), environment=str(ENV), raw_files=len(raw),
                  arrays=sum(r['arrays'] for r in totals.values()),
                  elements=sum(r['elements'] for r in totals.values()),
                  remote_sensing_reprocessed=False, cohort_features_reprocessed=False,
                  full_raw_to_evaluation_reproduced=False, independent_recipe_confirmation=False,
                  new_scientific_comparisons=0, scientific_results_replaced=False,
                  finished=time.time())
    write_json(EVIDENCE / 'complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
