"""Contracts for pairing seasonal states, historical anchors, and readouts."""
import numpy as np


RATIOS = (.1, .3, .5)
MODES = ('biid', 'climatology')


def component_identity(config, crop, cutoff, seed, product=None):
    expected = dict(crop=crop, cutoff=cutoff, seed=seed, smoke=False)
    if product is not None:
        expected.update(product=product, architecture='biid')
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Component crop, window, seed, or product differs')


def state_training(config, marker, trace, normalization, original_normalization):
    if (marker['smoke'] or marker['evaluation_arrays_loaded'] or
            marker['full_fit_cutoff'] != config['cutoff'] or
            config['selection_years'] != [config['cutoff']-1, config['cutoff']]):
        raise ValueError('State training crossed the registered cutoff')
    inner = [row for row in trace if row['stage'] == 'inner_selection']
    full = [row for row in trace if row['stage'] == 'full_refit']
    if not inner or len(inner)+len(full) != len(trace):
        raise ValueError('Missing or unknown state training stages')
    if trace != inner+full or any(set(r['rmse']) != {str(y) for y in config['selection_years']} for r in inner):
        raise ValueError('State selection used unexpected stages or years')
    if ([r['epoch'] for r in inner] != list(range(1, len(inner)+1)) or
            not 1 <= len(inner) <= config['epochs']):
        raise ValueError('Invalid inner training epochs')
    scores = [float(np.mean(list(row['rmse'].values()))) for row in inner]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite inner state scores')
    selected = int(np.argmin(scores))+1
    if marker['selected_epochs'] != selected or [r['epoch'] for r in full] != list(range(1, selected+1)):
        raise ValueError('Selected state was not completely refitted')
    if normalization != original_normalization:
        raise ValueError('State normalization changed across paired seeds')
    return selected


def preserve_prefix(observed, mixed, active, tail):
    if (observed.shape != mixed.shape or active.shape != tail.shape or
            active.shape != observed.shape or np.any(tail & ~active)):
        raise ValueError('Invalid seasonal completion shapes or mask')
    np.testing.assert_array_equal(mixed[~tail], observed[~tail])
    if not np.isfinite(mixed[tail]).all():
        raise ValueError('Nonfinite predicted suffix')


def reference_map(rows):
    result = {row['crop']: row for row in rows}
    if len(rows) != 4 or set(result) != {'maize', 'rice', 'soybean', 'wheat'}:
        raise ValueError('Expected four fixed named main-table references')
    if any(not row['baseline'].startswith('extended:') for row in rows):
        raise ValueError('Registered main-table historical reference changed')
    return result
