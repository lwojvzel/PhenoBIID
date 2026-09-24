"""Rebuild original cohort inputs using registered, newly reconstructed rasters."""
import importlib
import json
import os
from pathlib import Path
import sys


def allowed(path, writing, workspace, project, input_files, environment, runs):
    path = Path(path).resolve()
    if writing:
        return path.is_relative_to(workspace) or path.is_relative_to(Path('/tmp')) or path == Path('/dev/null')
    if path in input_files or path.is_relative_to(workspace) or path.is_relative_to(environment):
        return True
    return not (path.is_relative_to(project) or path.is_relative_to(runs))


def fresh_coverage(crop, seed):
    """Replace only the legacy coverage-loading boundary, never fitted predictions."""
    import numpy as np
    from audit_crop_area_fraction import grid_area_hectares, sample_fraction
    from biid_world_model import load_world_cache
    from multimodal_baseline import load_cache, load_coordinates
    from forward_protocol_revision import index_hash
    cache, world = load_cache(crop), load_world_cache(crop)
    source = np.asarray(world['source_indices'])
    latitude, _ = load_coordinates()
    coverage = np.clip(sample_fraction(crop, cache['year'][source], cache['row'][source],
                                      cache['col'][source], grid_area_hectares(latitude)), 0, 1).astype(np.float32)
    return {'fresh': {'source_indices': source, 'crop_coverage': coverage}}, {
        'sample_index_hash': {'fresh': index_hash(source)}, 'fitted_predictions_loaded': False}


def main():
    workspace = Path(__file__).resolve().parent
    registration = json.loads((workspace / 'registration.json').read_text())
    project = Path(registration['original_project'])
    inputs = {Path(p).resolve() for p in registration['input_files']}
    arguments = workspace, project, inputs, Path(sys.prefix).resolve(), workspace.parent
    accessed = set()

    def check(path, writing):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if not allowed(resolved, writing, *arguments):
            raise PermissionError('Old model inputs, unregistered inputs and external writes are prohibited')
        if resolved in inputs:
            accessed.add(str(resolved))

    def guard(event, values):
        if event == 'open':
            mode, flags = values[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            check(values[0], writing)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            check(values[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(values[0], True)
            check(values[1], True)
        elif event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
            raise PermissionError('Cohort reconstruction needs no network or child processes')

    sys.addaudithook(guard)
    try:
        open(project / 'benchmark/cache/multimodal_main/maize/year.npy', 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('Old-cache read guard failed')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace / 'scripts'))
    os.chdir(workspace)
    import numpy as np
    from multimodal_baseline import CROPS, build_crop_cache, load_cache, save_json
    from biid_world_model import build_world_cache, load_world_cache, load_or_compute_world_stats
    from prepare_reclue_monthly_gpp import OUT as GPP, crop_support
    revision = importlib.import_module('review_revision_data')
    revision.load_shared = fresh_coverage
    from stable_remote_data import ORIGINS, prepare

    completed, support = {}, {}
    for crop in CROPS:
        print(f'[COHORT START] {crop}', flush=True)
        build_crop_cache(crop)
        build_world_cache(crop)
        cache, world = load_cache(crop), load_world_cache(crop)
        load_or_compute_world_stats(crop, cache, world)
        shared, _ = fresh_coverage(crop, 42)
        destination = workspace / 'benchmark/cache/fresh_coverage' / crop
        destination.mkdir(parents=True)
        for name, value in shared['fresh'].items():
            np.save(destination / f'{name}.npy', value)
        for origin in ORIGINS:
            prepare(crop, origin, 0)
        support[crop] = {}
        for year in range(1982, 2017):
            rel = np.load(GPP / 'crops' / crop / 'gpp_daily_rate' / f'gpp_daily_rate_rel_{year}.npy', mmap_mode='r')
            support[crop][str(year)] = crop_support(crop, year, rel)
        completed[crop] = {'base_rows': len(cache['year']), 'world_rows': len(world['source_indices']),
                           'origins': list(ORIGINS)}
        print(f'[COHORT FINISHED] {crop}: {completed[crop]}', flush=True)
    save_json(support, workspace / 'fresh_gpp_support.json')
    save_json({'generated': True, 'crops': completed, 'accessed_input_files': sorted(accessed),
               'old_predictions_loaded': False, 'models_fitted': 0,
               'complete_model_feature_interface': False}, workspace / 'generation.json')
    assert 'torch' not in sys.modules, 'Cohort generation must not load neural models'
    print('[COHORT GENERATION COMPLETE]', flush=True)


if __name__ == '__main__':
    main()
