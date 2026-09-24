"""Run original remote-raster processing without old outputs or cohort caches."""
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


def allowed(path, writing, workspace, project, raw_files, mapping_files, environment, foundation):
    path = Path(path).resolve()
    if writing:
        return path.is_relative_to(workspace) or path.is_relative_to(Path('/tmp')) or path == Path('/dev/null')
    if path.is_relative_to(foundation):
        return path in mapping_files
    if path.is_relative_to(project):
        return path in raw_files or path.is_relative_to(environment)
    return True


def deferred_support(crop, year, rel):
    return dict(executed=False, deferred_to='fresh cohort reconstruction', finite_fraction=None,
                crop=crop, year=year, target_values_accessed=False,
                reason='Original cohort caches are prohibited in raw raster processing')


def main():
    workspace = Path(__file__).resolve().parent
    registration = json.loads((workspace / 'registration.json').read_text())
    project = Path(registration['original_project'])
    raw_files = {Path(path).resolve() for path in registration['raw_files']}
    mappings = {Path(path).resolve() for path in registration['mapping_files']}
    foundation = Path(registration['new_foundation'])
    args = workspace, project, raw_files, mappings, Path(sys.prefix).resolve(), foundation

    def check(path, writing):
        if isinstance(path, (str, bytes, os.PathLike)) and not allowed(os.fsdecode(path), writing, *args):
            raise PermissionError('Old processed data, old cohort caches, or external writes are prohibited')

    def guard(event, values):
        if event == 'open':
            mode, flags = values[1], values[2]
            writing = (isinstance(mode, str) and any(char in mode for char in 'wax+')) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            check(values[0], writing)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            check(values[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(values[0], True)
            check(values[1], True)
        elif event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('Raw reproduction does not download sources')

    sys.addaudithook(guard)
    for target, mode in ((project / 'Data/processed/glass_lai_avhrr_005d/lat.npy', 'rb'),
                         (project / 'benchmark/cache/multimodal_main/maize/year.npy', 'rb'),
                         (next(iter(raw_files)), 'ab'), (next(iter(mappings)), 'ab')):
        try:
            open(target, mode)
        except PermissionError:
            pass
        else:
            raise AssertionError('Guard probe failed')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace / 'scripts'))
    os.chdir(workspace)
    product = sys.argv[1]
    if product == 'lai':
        script = workspace / 'scripts/process_glass_lai_avhrr_to_growing_season.py'
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name='__main__')
    elif product == 'ndvi':
        importlib.import_module('prepare_pku_ndvi').process(1982, 2016)
    elif product == 'gpp':
        module = importlib.import_module('prepare_reclue_monthly_gpp')
        # This diagnostic reads old model identities, not inputs to raster aggregation.
        module.crop_support = deferred_support
        for year in module.YEARS:
            source = module.SAMPLE / '1982.zip' if year == 1982 else module.RAW / f'{year}.zip'
            result = module.process(year, source)
            assert all(not item['original_benchmark_support']['executed'] for item in result['crops'].values())
            if year == 1982:
                assert not result['sample_support_passed']
        module.atomic_json(module.OUT / 'raw_reconstruction_manifest.json', dict(
            years=list(module.YEARS), raster_processing_complete=True,
            original_cohort_support_checked=False, original_crop_support_deferred=True,
            models_fitted=0, uses_fresh_foundation_mappings=True))
    else:
        raise ValueError('Unknown registered product')
    assert 'torch' not in sys.modules, 'Raster processing must not load model dependencies'
    print(f'[RAW PRODUCT FINISHED] {product}', flush=True)


if __name__ == '__main__':
    main()
