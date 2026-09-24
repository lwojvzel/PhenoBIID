#!/usr/bin/env python3
"""Build crop cohorts and physical features without loading fitted predictions."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def fresh_coverage(crop, seed):
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


def cohort(crops):
    from multimodal_baseline import build_crop_cache, load_cache
    from biid_world_model import build_world_cache, load_world_cache, load_or_compute_world_stats
    from stable_remote_data import ORIGINS, prepare
    for crop in crops:
        build_crop_cache(crop)
        build_world_cache(crop)
        load_or_compute_world_stats(crop, load_cache(crop), load_world_cache(crop))
        for origin in ORIGINS:
            prepare(crop, origin, 0)
        print(f'Built {crop} cohorts and crop-area support without fitted model predictions', flush=True)


def features(crops, workspace):
    import numpy as np
    from multimodal_baseline import save_json
    from crop_signal_screen_data import ORIGINS, prepare as prepare_screen
    from forecast_bridge_data import prepare as prepare_bridge, fit_stats
    from inseason_complete_inputs import complete_inputs
    for crop in crops:
        for origin in ORIGINS:
            prepare_screen(crop, origin)
            prepare_screen(crop, origin, audit=True)
        prepare_bridge(crop)
        raw, rows, metadata, support, stats, climate, lineage = complete_inputs(crop)
        destination = workspace / 'benchmark/cache/fresh_complete_inputs' / crop
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite generated feature interface: {destination}')
        destination.mkdir(parents=True)
        for name, value in dict(raw, metadata=metadata).items():
            np.save(destination / f'{name}.npy', value)
        np.save(destination / 'support_fit_2009.npy', support)
        statistics = {str(c): fit_stats(raw, np.flatnonzero(raw['year'] <= c)) for c in (2001, 2005, 2009)}
        save_json(statistics, destination / 'statistics.json')
        save_json(lineage, destination / 'lineage.json')
        print(f'{crop}: {len(raw["year"])} rows, metadata {metadata.shape}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--stage', choices=('cohort', 'features', 'all', 'check-imports'), default='all')
    parser.add_argument('--crop', choices=('all', 'maize', 'rice', 'soybean', 'wheat'), default='all')
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        parser.error('Initialize the raw reconstruction workspace first')
    os.environ['PHENOBIID_WORKSPACE'] = str(workspace)
    os.chdir(workspace)
    sys.path.insert(0, str(REPO / 'reconstruction/scripts'))
    import review_revision_data
    review_revision_data.load_shared = fresh_coverage
    # Both reconstruction stages import only NumPy-based definitions.
    import inseason_complete_inputs
    import stable_remote_data
    if 'torch' in sys.modules or 'joblib' in sys.modules:
        raise RuntimeError('Preprocessing unexpectedly loaded a model runtime')
    if args.stage == 'check-imports':
        print('Standalone cohort and feature imports passed; no model runtime loaded.')
        return
    crops = review_revision_data.CROPS if args.crop == 'all' else (args.crop,)
    if args.stage in ('cohort', 'all'):
        cohort(crops)
    if args.stage in ('features', 'all'):
        features(crops, workspace)


if __name__ == '__main__':
    main()
