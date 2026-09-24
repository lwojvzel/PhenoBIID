"""Frozen-head temporal replacement with observable-prefix feature boundaries."""
import numpy as np

from observed_remote_anomaly import seasonal_anomalies
from observed_remote_benchmark import trajectory_features
from ndvi_tail_replacement import RATIOS, MODES, tail_mask, mix_trajectory, prefix_values, prefix_rollout

SUPPORTS = ('legacy_support', 'available_prefix')


def lead_times(years, months, active, tail):
    """Time to the final supplied active-month end, not a verified harvest date."""
    active, tail = np.asarray(active, bool), np.asarray(tail, bool)
    if active.shape != months.shape or tail.shape != active.shape or np.any(tail & ~active):
        raise ValueError('Mismatched temporal masks')
    if np.any(active.sum(1) == 0):
        raise ValueError('Undefined cutoff for empty activity calendar')
    prefix = active & ~tail
    first = np.where(active, months, 12).min(1)
    last = np.where(active, months, -1).max(1)
    known_last = np.where(prefix, months, -1).max(1)
    boundary = np.where(prefix.any(1), known_last+1, first)
    jan = np.asarray([f'{int(y):04d}-01' for y in years], dtype='datetime64[M]')
    issue_exclusive = (jan+boundary.astype('timedelta64[M]')).astype('datetime64[D]')
    end_exclusive = (jan+(last+1).astype('timedelta64[M]')).astype('datetime64[D]')
    return dict(lead_days=(end_exclusive-issue_exclusive).astype(np.int16),
                lead_calendar_months=(last+1-boundary).astype(np.int8),
                issue_date=(issue_exclusive-np.timedelta64(1, 'D')).astype('int64'),
                activity_end_date=(end_exclusive-np.timedelta64(1, 'D')).astype('int64'),
                remaining_active_slots=tail.sum(1).astype(np.int8))


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


def available_features(fixed, encoded, tail, support):
    # Existing features are history20 + metadata289 + weather156 + NDVI36.
    if fixed.shape[1] != 501 or support.shape != (*tail.shape, 6):
        raise ValueError('Unexpected frozen feature layout')
    out = np.array(fixed, copy=True)
    metadata = out[:, 21:309].reshape(-1, 12, 24)
    metadata[..., 5:11] = np.where(tail[..., None], support, metadata[..., 5:11])
    out[:, -36:] = encoded
    return out
