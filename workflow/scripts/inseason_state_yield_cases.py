"""Outcome-blind case selection and pointwise state-to-yield diagnostics."""
import hashlib

import numpy as np

YEAR = 2010
CUTOFF = 2009
SEED = 42
PERCENTS = (10, 30, 50)
PRIMARY = 30
MIN_ACTIVE = 4
PRODUCTS = {'maize': ('gpp',), 'rice': ('ndvi',),
            'soybean': ('ndvi',), 'wheat': ('ndvi', 'gpp')}
FIELDS = ('year', 'row', 'col', 'source_indices', 'relative_valid')


def select_case(crop, inputs):
    if crop not in PRODUCTS or set(inputs) != set(FIELDS):
        raise ValueError('Selection accepts only registered identity and activity fields')
    n = len(inputs['year'])
    for key in FIELDS[:-1]:
        a = np.asarray(inputs[key])
        if a.shape != (n,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('Invalid integer identity vector')
    valid = np.asarray(inputs['relative_valid'])
    if valid.shape != (n, 12) or not np.isfinite(valid).all():
        raise ValueError('Invalid activity mask')
    if not np.isin(valid, (0, 1)).all():
        raise ValueError('Activity must be binary')
    active = valid > 0
    if np.any(np.diff(active.astype(int), axis=1) > 0):
        raise ValueError('Activity must be a packed prefix')
    if (np.any((inputs['row'] < 0) | (inputs['row'] >= 360)) or
            np.any((inputs['col'] < 0) | (inputs['col'] >= 720))):
        raise ValueError('Coordinates outside the registered grid')
    ids = np.stack([inputs[k] for k in ('year', 'row', 'col')], axis=1)
    if len(np.unique(ids, axis=0)) != n or len(np.unique(inputs['source_indices'])) != n:
        raise ValueError('Duplicate sample identity')
    candidates = np.flatnonzero((inputs['year'] == YEAR) & (active.sum(1) >= MIN_ACTIVE))
    if not len(candidates):
        raise ValueError('No eligible case; do not select a different year silently')
    def digest(i):
        key = f'state-yield-case-v1|{crop}|{YEAR}|{int(inputs["row"][i])}|{int(inputs["col"][i])}'
        return hashlib.sha256(key.encode('ascii')).hexdigest()
    i = min(candidates, key=digest)
    return dict(crop=crop, year=YEAR, row=int(inputs['row'][i]), col=int(inputs['col'][i]),
        source_indices=int(inputs['source_indices'][i]), active_slots=int(active[i].sum()),
        candidate_count=len(candidates), selection_digest=digest(i))


def locate(inputs, case):
    match = np.ones(len(inputs['year']), bool)
    for key in ('year', 'row', 'col', 'source_indices'):
        match &= inputs[key] == case[key]
    ix = np.flatnonzero(match)
    if len(ix) != 1:
        raise ValueError('Selected identity not uniquely present')
    return int(ix[0])


def suffix(active, percent):
    active = np.asarray(active, bool)
    if active.ndim != 1 or active.shape != (12,) or not active.any():
        raise ValueError('Expected a nonempty twelve-slot activity vector')
    if percent not in PERCENTS:
        raise ValueError('Unregistered case cutoff')
    count = int(active.sum())
    hidden = (count * percent + 99) // 100
    return active & (np.cumsum(active) > count - hidden)


def state_scores(truth, forecast, active, tail):
    truth, forecast = np.asarray(truth), np.asarray(forecast)
    if truth.shape != (12,) or forecast.shape != truth.shape or np.any(tail & ~active):
        raise ValueError('Invalid case trajectory')
    np.testing.assert_array_equal(truth[~tail], forecast[~tail])
    if not np.isfinite(forecast[tail]).all():
        raise ValueError('Nonfinite completion')
    valid = active & tail & np.isfinite(truth)
    error = forecast[valid].astype(float) - truth[valid].astype(float)
    return dict(valid_hidden_slots=int(valid.sum()),
        state_rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        state_mae=float(np.mean(np.abs(error))) if len(error) else None)
