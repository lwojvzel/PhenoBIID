"""Pair fixed terminal features with the same seed's historical anchor."""
import numpy as np

from inseason_nested_common import LABELS


def paired_prediction(saved, expected):
    ids = np.asarray(saved['source_indices'])
    wanted = np.asarray(expected['source_indices'])
    prediction = np.asarray(saved['prediction'])
    if ids.ndim != 1 or wanted.ndim != 1 or prediction.shape != ids.shape:
        raise ValueError('History identity/prediction shape differs')
    if len(np.unique(ids)) != len(ids) or len(np.unique(wanted)) != len(wanted):
        raise ValueError('Duplicate history identities')
    order = np.argsort(ids)
    offsets = np.searchsorted(ids[order], wanted)
    if np.any(offsets >= len(ids)):
        raise ValueError('Missing historical prediction')
    take = order[offsets]
    for name in LABELS:
        np.testing.assert_array_equal(saved[name][take], expected[name])
    result = prediction[take].astype(float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite historical prediction')
    return result


def fixed_parameters(parameters, trees, seed, smoke=False):
    if parameters.get('random_state') != 42 or parameters.get('n_jobs') != 4:
        raise ValueError('Unexpected original terminal recipe')
    if trees < 1 or seed not in (42, 45, 48):
        raise ValueError('Unregistered seed or unfitted original tree')
    return dict(parameters, n_estimators=min(trees, 8) if smoke else trees,
                random_state=seed)


def residual_labels(target, base, normalization):
    target, base = np.asarray(target, float), np.asarray(base, float)
    center = normalization.get('center', normalization.get('residual_mean'))
    scale = normalization.get('scale', normalization.get('residual_std'))
    if target.ndim != 1 or target.shape != base.shape:
        raise ValueError('Residual inputs must be paired vectors')
    if center is None or scale is None or not np.isfinite([center, scale]).all() or scale <= 0:
        raise ValueError('Invalid original residual normalization')
    result = (target - base - center) / scale
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite residual target')
    return result


def combine(crop, parts):
    if crop not in ('maize', 'rice', 'soybean', 'wheat'):
        raise ValueError('Unknown crop')
    if len(parts) != (2 if crop == 'maize' else 1):
        raise ValueError('Original crop branch count changed')
    arrays = [np.asarray(p) for p in parts]
    if any(p.ndim != 1 or p.shape != arrays[0].shape or not np.isfinite(p).all() for p in arrays):
        raise ValueError('Invalid terminal branch outputs')
    return .5 * arrays[0] + .5 * arrays[1] if crop == 'maize' else arrays[0]


def validate_feature_schema(features, names):
    expected = [f'history_{i}' for i in range(15)] + [f'context_{i}' for i in range(5)]
    if names['history'] != expected or features.ndim != 2 or features.shape[1] not in (501, 537):
        raise ValueError('Terminal history must be the original raw history20 interface')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite original features')
