"""Fixed-capacity readout ablations on the retained world-model interface."""
import numpy as np

CONDITIONS = ('full', 'no_weather', 'no_quality', 'no_weather_quality')
RATIOS = (.1, .3, .5)
MODES = ('biid', 'climatology')


def removed_columns(width, condition):
    if width not in (501, 537) or condition not in CONDITIONS:
        raise ValueError('Unregistered terminal layout or condition')
    metadata = np.arange(21, 309).reshape(12, 24)
    columns = []
    if condition in ('no_weather', 'no_weather_quality'):
        columns.extend(metadata[:, 11:24].ravel())
        columns.extend(range(309, 465))
    if condition in ('no_quality', 'no_weather_quality'):
        columns.extend(metadata[:, 5:11].ravel())
    return np.asarray(sorted(columns), dtype=np.int64)


def mask_features(features, condition):
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite two-dimensional terminal features')
    columns = removed_columns(features.shape[1], condition)
    result = features.copy()
    result[:, columns] = 0
    return result


def check_splits(model, condition):
    importance = model.booster_.feature_importance(importance_type='split')
    columns = removed_columns(len(importance), condition)
    if np.any(importance[columns] != 0):
        raise ValueError('A removed terminal feature was used in a tree')
    return columns.tolist()
