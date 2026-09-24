"""Rebuild every registered LAI, NDVI and GPP raster in a fresh workspace."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import run_raw_foundation_reproduction as foundation
from run_raw_foundation_reproduction import ROOT, compare_array, digest, write_json


EVIDENCE = ROOT / 'benchmark/results/inseason_reproducibility_v1/raw_remote_20260909'
ENV = ROOT / 'Paper/release/inseason_raw_remote_env'
FOUNDATION = ROOT / 'benchmark/results/inseason_reproducibility_v1/raw_foundation_20260909_attempt2'
CROPS = ('maize', 'rice', 'soybean', 'wheat')
GROWING = Path('Data/processed/crop_yield_growing_season')
LAI = Path('Data/GLASS_LAI_AVHRR_005D')
NDVI = Path('Data/external/PKU_GIMMS_NDVI_v1p2')
GPP = Path('Data/external/reclue_monthly_gpp_v1')
PROBE = Path('Data/external/reclue_monthly_gpp_probe')
OUTPUTS = dict(lai=Path('Data/processed/glass_lai_avhrr_005d'),
               ndvi=Path('Data/processed/pku_gimms_ndvi_v1p2'),
               gpp=Path('Data/processed/reclue_monthly_gpp_v1'))
COUNTS = dict(lai=182, ndvi=527, gpp=527)
SCRIPTS = ('process_glass_lai_avhrr_to_growing_season.py',
           'prepare_pku_ndvi.py', 'prepare_reclue_monthly_gpp.py')
EXPORTS = {
    'review_revision_data.py': ('from pathlib import Path\nimport hashlib\n', ('ROOT', 'CROPS', 'sha256')),
    'run_review_revision_parallel.py': ('import json\n', ('atomic_json',)),
    'fetch_muses_gpp_sample.py': ('import hashlib\n', ('md5',)),
    'fetch_reclue_gpp_sample.py': (
        'from review_revision_data import ROOT\nfrom fetch_muses_gpp_sample import md5\n', ('RAW',)),
}
PACKAGES = [
    'numpy==2.0.2', 'pyhdf==0.11.6', 'rasterio==1.4.3', 'affine==3.0.1',
    'requests==2.32.5', 'attrs==25.4.0', 'click==8.1.8', 'click-plugins==1.1.1.2',
    'cligj==0.7.2', 'certifi==2026.2.25', 'urllib3==2.6.3',
    'charset-normalizer==3.4.7', 'idna==3.11', 'pyparsing==3.2.5',
]


def selected_nodes(source, names):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            keys = [node.name]
        elif isinstance(node, ast.Assign):
            keys = [target.id for target in node.targets if isinstance(target, ast.Name)]
        else:
            continue
        for name in keys:
            if name in names:
                if name in found:
                    raise ValueError(f'Duplicate exported symbol: {name}')
                found[name] = node
    if set(found) != set(names):
        raise ValueError(f'Missing exported symbols: {set(names)-set(found)}')
    return found


def export_helpers(source_path, destination, imports, names):
    source = source_path.read_text()
    nodes = selected_nodes(source, names)
    output = imports + '\n\n'.join(ast.get_source_segment(source, nodes[name]) for name in names) + '\n'
    exported = selected_nodes(output, names)
    for name in names:
        assert ast.dump(nodes[name]) == ast.dump(exported[name])
    destination.write_text(output)
    return dict(source_sha256=digest(source_path), exported_sha256=digest(destination),
                symbols=list(names), unchanged_symbol_ast=True)


def raw_inventory():
    lai = sorted((ROOT / LAI).glob('*/*.hdf'))
    assert len(lai) == 1656
    from process_glass_lai_avhrr_to_growing_season import parse_year_doy
    dates = [parse_year_doy(p) for p in lai]
    assert len(set(dates)) == len(dates)
    assert set(dates) == {(year, day) for year in range(1981, 2017) for day in range(1, 366, 8)}
    ndvi = sorted((ROOT / NDVI).glob('*consolidated*.zip'))
    assert len(ndvi) == 4
    ndvi_dates = []
    import re
    for path in ndvi:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                match = re.search(r'_(\d{4})(\d{2})(0[12])\.tif$', name, re.I)
                if match:
                    ndvi_dates.append(tuple(map(int, match.groups())))
    assert len(set(ndvi_dates)) == len(ndvi_dates)
    needed = {(year, month, half) for year in range(1982, 2017)
              for month in range(1, 13) for half in (1, 2)}
    assert needed <= set(ndvi_dates)
    gpp = [ROOT / PROBE / '1982.zip'] + [ROOT / GPP / f'{year}.zip' for year in range(1983, 2017)]
    assert all(path.is_file() for path in gpp)
    for year, path in zip(range(1982, 2017), gpp):
        with zipfile.ZipFile(path) as archive:
            names = {item.filename for item in archive.infolist() if not item.is_dir()}
            assert names == {f'{year}/GLASSGPP_{year}{month:02d}_005D.tif' for month in range(1, 13)}
    return dict(lai=lai, ndvi=ndvi, gpp=gpp)


def verify_providers():
    result = {}
    for product, metadata, paths in [
        ('ndvi', ROOT / NDVI / 'zenodo_record.json', sorted((ROOT / NDVI).glob('*consolidated*.zip'))),
        ('gpp', ROOT / PROBE / 'record.json', [ROOT / PROBE / '1982.zip'] +
         [ROOT / GPP / f'{year}.zip' for year in range(1983, 2017)]),
    ]:
        record = json.loads(metadata.read_text())
        entries = {item['key']: item for item in record['files']}
        rows = {}
        import hashlib
        for path in paths:
            entry = entries[path.name]
            algorithm, expected = entry['checksum'].split(':')
            h = hashlib.new(algorithm)
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*2**20), b''):
                    h.update(block)
            assert path.stat().st_size == entry['size'] and h.hexdigest() == expected, path
            rows[str(path)] = dict(bytes=entry['size'], checksum=entry['checksum'])
        result[product] = dict(metadata=str(metadata), metadata_sha256=digest(metadata), files=rows)
    return result


def main():
    global EVIDENCE, ENV
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not args.run or args.attempt < 1:
        raise SystemExit('Use --run and a positive attempt identifier')
    if args.attempt > 1:
        EVIDENCE = EVIDENCE.with_name(EVIDENCE.name + f'_attempt{args.attempt}')
        ENV = ENV.with_name(ENV.name + f'_attempt{args.attempt}')
    if EVIDENCE.exists() or ENV.exists():
        raise FileExistsError('Preserve all previous executions and environments')
    source_audit = json.loads((FOUNDATION / 'original_source_output_audit.json').read_text())
    assert source_audit['passed'] and source_audit['output_arrays'] == 1348
    for path, expected in source_audit['sources'].items():
        assert digest(path) == expected, path
    source_run = json.loads((FOUNDATION / 'complete.json').read_text())
    new_foundation = Path(source_run['workspace'])
    mapping_names = ['lat.npy', 'lon.npy'] + [
        f'{crop}/mirca/{year}/{name}.npy' for crop in CROPS for year in (2000, 2005, 2010, 2015)
        for name in ('src_rel', 'valid_rel')]
    details = json.loads((FOUNDATION / 'crop_slots_verification.json').read_text())['arrays_detail']
    mappings = {}
    for name in mapping_names:
        path = new_foundation / GROWING / name
        assert digest(path) == details[name]['generated_sha256'], path
        mappings[str(path)] = digest(path)
    products = raw_inventory()
    providers = verify_providers()
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    foundation.EVIDENCE = EVIDENCE
    workspace = Path(tempfile.mkdtemp(prefix='raw-remote-', dir=ROOT.parent / 'AgroClimate_reproduction_runs'))
    (workspace / 'scripts').mkdir()
    (workspace / 'original_scripts').mkdir()
    script_hashes, exports = {}, {}
    for name in (*SCRIPTS, *EXPORTS):
        source = ROOT / 'scripts' / name
        shutil.copy2(source, workspace / 'original_scripts' / name)
        script_hashes[name] = digest(source)
        if name in SCRIPTS:
            shutil.copy2(source, workspace / 'scripts' / name)
        else:
            imports, names = EXPORTS[name]
            exports[name] = export_helpers(source, workspace / 'scripts' / name, imports, names)
    runner = ROOT / 'Paper/iclr2027/reproducibility/raw_processing/run_remote_guarded.py'
    shutil.copy2(runner, workspace / 'run_remote_guarded.py')
    for relative in (LAI, NDVI, GPP, PROBE):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(ROOT / relative, target_is_directory=True)
    target = workspace / GROWING
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(new_foundation / GROWING, target_is_directory=True)
    raw = [path for items in products.values() for path in items]
    initial = dict(attempt=args.attempt, original_project=str(ROOT), workspace=str(workspace),
        new_foundation=str(new_foundation), foundation_audit_sha256=digest(FOUNDATION / 'original_source_output_audit.json'),
        raw_files=list(map(str, raw)), raw_product_counts={name:len(paths) for name, paths in products.items()},
        scripts=script_hashes, helper_exports=exports, runner_sha256=digest(runner),
        driver_sha256=digest(Path(__file__)), packages=PACKAGES, mapping_files=mappings,
        expected_arrays=COUNTS, reference_arrays_used_for_processing=False,
        old_cohort_support_diagnostic_deferred=True, new_scientific_comparisons=0)
    write_json(EVIDENCE / 'registration.json', initial)
    write_json(workspace / 'registration.json', initial)
    write_json(EVIDENCE / 'provider_checks.json', providers)
    print(f'[REGISTERED] {workspace}; {len(raw)} raw files', flush=True)
    raw_hashes = {}
    for index, path in enumerate(raw, 1):
        raw_hashes[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
        if index % 200 == 0 or index == len(raw):
            print(f'[RAW HASH] {index}/{len(raw)}', flush=True)
    write_json(EVIDENCE / 'raw_sources.json', raw_hashes)
    references = {}
    for product, relative in OUTPUTS.items():
        references[product] = {str(path.relative_to(ROOT / relative)): digest(path)
                               for path in sorted((ROOT / relative).rglob('*.npy'))}
        assert len(references[product]) == COUNTS[product], product
    write_json(EVIDENCE / 'reference_arrays.json', references)
    foundation.run_logged([sys.executable, '-m', 'venv', ENV], 'environment_create', workspace)
    python = ENV / 'bin/python'
    foundation.run_logged([python, '-m', 'pip', 'install', '--disable-pip-version-check', *PACKAGES],
                          'environment_install', workspace)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (EVIDENCE / 'pip_freeze.txt').write_text(freeze)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GDAL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    totals = {}
    for product, relative in OUTPUTS.items():
        assert not (workspace / relative).exists(), 'Fresh processing cannot skip cached rasters'
        foundation.run_logged([python, workspace / 'run_remote_guarded.py', product], product, workspace, env)
        generated = sorted((workspace / relative).rglob('*.npy'))
        names = {str(path.relative_to(workspace / relative)) for path in generated}
        assert names == set(references[product]), (product, 'Incomplete output matrix')
        rows = {}
        for index, path in enumerate(generated, 1):
            name = str(path.relative_to(workspace / relative))
            reference = ROOT / relative / name
            assert digest(reference) == references[product][name], reference
            rows[name] = compare_array(path, reference)
            if index % 50 == 0 or index == len(generated):
                print(f'[FULL ARRAY CHECK] {product} {index}/{len(generated)}', flush=True)
        result = dict(passed=True, arrays=len(rows), elements=sum(row['elements'] for row in rows.values()),
                      byte_identical_arrays=sum(row['generated_sha256']==row['reference_sha256'] for row in rows.values()),
                      arrays_detail=rows, all_values_and_missingness=True)
        write_json(EVIDENCE / f'{product}_verification.json', result)
        totals[product] = {key:value for key,value in result.items() if key != 'arrays_detail'}
        print(f'[VERIFIED] {product}: {len(rows)} arrays', flush=True)
    for path, info in raw_hashes.items():
        assert digest(path) == info['sha256'], path
    for path, expected in mappings.items():
        assert digest(path) == expected, path
    for name, expected in script_hashes.items():
        assert digest(ROOT / 'scripts' / name) == digest(workspace / 'original_scripts' / name) == expected
        if name in SCRIPTS:
            assert digest(workspace / 'scripts' / name) == expected
    complete = dict(passed=True, remote_sensing_reprocessed=True, all_registered_years_and_products=True,
        all_quality_and_support_arrays=True, new_foundation_mappings=True, stages=totals,
        arrays=sum(row['arrays'] for row in totals.values()), elements=sum(row['elements'] for row in totals.values()),
        raw_files=len(raw), workspace=str(workspace), environment=str(ENV), original_raw_unchanged=True,
        original_arrays_unchanged=True, scientific_processing_unchanged=True,
        old_cohort_support_diagnostic_deferred=True, cohort_features_reprocessed=False,
        full_raw_to_evaluation_reproduced=False, independent_recipe_confirmation=False,
        new_scientific_comparisons=0, models_fitted=0, scientific_results_replaced=False, finished=time.time())
    write_json(EVIDENCE / 'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
