"""Build physical tracks and model metadata from fresh, registered cohorts."""
import json
import os
from pathlib import Path
import sys

from cohort_guard import allowed


def reject_legacy_predictions(*args, **kwargs):
    raise PermissionError('Feature reconstruction must not load legacy shared predictions')


def main():
    workspace = Path(__file__).resolve().parent
    registration = json.loads((workspace / 'registration.json').read_text())
    inputs = {Path(p).resolve() for p in registration['input_files']}
    project = Path(registration['original_project'])
    arguments = workspace, project, inputs, Path(sys.prefix).resolve(), workspace.parent
    accessed = set()

    def check(path, writing):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(path)).resolve()
        if not allowed(path, writing, *arguments):
            raise PermissionError('Only registered fresh inputs and local outputs are allowed')
        if path in inputs:
            accessed.add(str(path))

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
            raise PermissionError('Feature generation requires no network or subprocesses')

    sys.addaudithook(guard)
    try:
        open(project / 'benchmark/cache/inseason_13year_v1/maize/metadata.npy', 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('Original feature access was not blocked')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace / 'scripts'))
    os.chdir(workspace)
    import numpy as np
    import review_revision_data
    review_revision_data.load_shared = reject_legacy_predictions
    from review_revision_data import CROPS, sha256
    from multimodal_baseline import save_json
    from crop_signal_screen_data import ORIGINS, prepare as prepare_screen
    from forecast_bridge_data import prepare as prepare_bridge, fit_stats
    from inseason_complete_inputs import complete_inputs

    records = {}
    for crop in CROPS:
        print(f'[FEATURE START] {crop}', flush=True)
        for origin in ORIGINS:
            prepare_screen(crop, origin)
            prepare_screen(crop, origin, audit=True)
        prepare_bridge(crop)
        raw, rows, metadata, support, stats, climate, lineage = complete_inputs(crop)
        destination = workspace / 'benchmark/cache/fresh_complete_inputs' / crop
        destination.mkdir(parents=True)
        for name, value in dict(raw, metadata=metadata).items():
            np.save(destination / f'{name}.npy', value)
        np.save(destination / 'support_fit_2009.npy', support)
        statistics = {str(cutoff): fit_stats(raw, np.flatnonzero(raw['year'] <= cutoff))
                      for cutoff in (2001, 2005, 2009)}
        save_json(statistics, destination / 'statistics.json')
        save_json(lineage, destination / 'lineage.json')
        record = {'rows': len(raw['year']), 'fields': sorted(raw), 'metadata_shape': list(metadata.shape),
                  'support_shape': list(support.shape), 'years': np.unique(raw['year']).tolist(),
                  'old_predictions_loaded': False,
                  'files': {p.name: sha256(p) for p in sorted(destination.glob('*.npy'))}}
        save_json(record, destination / 'manifest.json')
        records[crop] = record
        print(f'[FEATURE FINISHED] {crop}: {len(raw["year"])} rows, {metadata.shape[1]} metadata columns', flush=True)
    save_json({'generated': True, 'crops': records, 'accessed_input_files': sorted(accessed),
               'models_fitted': 0, 'old_predictions_loaded': False,
               'complete_model_feature_interface': False}, workspace / 'generation.json')
    assert 'torch' not in sys.modules and 'joblib' not in sys.modules
    print('[FEATURE GENERATION COMPLETE]', flush=True)


if __name__ == '__main__':
    main()
