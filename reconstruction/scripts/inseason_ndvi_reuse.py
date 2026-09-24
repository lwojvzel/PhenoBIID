from __future__ import annotations
import numpy as np
from observed_remote_anomaly import seasonal_anomalies
from observed_remote_benchmark import trajectory_features
class TrajectoryEncoder:
    def __init__(self, raw, fit, take, scale):
        self.active = raw['relative_valid'][take] > 0
        self.support = np.isfinite(raw['observed_ndvi'][take]) & self.active
        self.scale = scale
        self.fields = {k: raw[k][take] for k in ('row', 'col', 'source_month', 'relative_valid')}
        active_fit = raw['relative_valid'][fit] > 0
        valid_fit = active_fit & np.isfinite(raw['observed_ndvi'][fit])
        self.train = dict(observed_ndvi=np.where(valid_fit,
            (raw['observed_ndvi'][fit]-scale['mean'])/scale['std'], 0).astype(np.float32),
            observed_ndvi_valid=valid_fit.astype(np.float32),
            **{k:raw[k][fit] for k in self.fields})

    def encode(self, physical, tail=None, available=False):
        valid = self.support if not available else self.active & np.isfinite(physical)
        if available and tail is None:
            raise ValueError('Cutoff mask required')
        values = np.where(valid, (physical-self.scale['mean'])/self.scale['std'], 0).astype(np.float32)
        evaluation = dict(**self.fields, observed_ndvi=values, observed_ndvi_valid=valid.astype(np.float32))
        anomalies = seasonal_anomalies({'train': self.train, 'validation': evaluation}, 'ndvi')['validation']
        return np.concatenate((trajectory_features(values, valid), trajectory_features(anomalies, valid)), 1)

def historical_support(raw, fit, take, metadata_train):
    """Grid/month support averages fitted only on pre-evaluation years."""
    if metadata_train.shape != (len(fit), 289):
        raise ValueError('Expected frozen 289-column support interface')
    values = metadata_train[:, 1:].reshape(-1, 12, 24)[..., 5:11]
    active = raw['relative_valid'][fit] > 0
    month = np.minimum(raw['source_month'][fit], 11).astype(np.int64)
    cell = raw['row'][fit].astype(np.int64)*720+raw['col'][fit]
    keys = (cell[:, None]*12+month)[active]
    keys_unique, inverse = np.unique(keys, return_inverse=True)
    count = np.bincount(inverse)
    averages = np.stack([np.bincount(inverse, weights=values[..., j][active])/count for j in range(6)], -1)
    global_month = np.stack([values[active & (month == m)].mean(0) if np.any(active & (month == m))
                             else values[active].mean(0) for m in range(12)])
    target_month = np.minimum(raw['source_month'][take], 11).astype(np.int64)
    target_cell = raw['row'][take].astype(np.int64)*720+raw['col'][take]
    target_key = target_cell[:, None]*12+target_month
    index = np.searchsorted(keys_unique, target_key)
    clipped = np.minimum(index, len(keys_unique)-1)
    found = (index < len(keys_unique)) & (keys_unique[clipped] == target_key)
    result = np.where(found[..., None], averages[clipped], global_month[target_month])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite support imputation')
    return result.astype(np.float32)
