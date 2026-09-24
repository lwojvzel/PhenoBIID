"""Repair derived timing without changing any frozen prediction artifacts."""
import numpy as np
from inseason_ndvi_reuse import lead_times


def corrected_lead_times(years, months, active, tail):
    months = np.asarray(months, dtype=np.int64)
    active, tail = np.asarray(active, bool), np.asarray(tail, bool)
    if np.any(active & ((months < 0) | (months > 11))):
        raise ValueError('Invalid active calendar month')
    result = lead_times(years, months, active, tail)
    if np.any((result['lead_days'] < 0) | (result['lead_days'] > 366)):
        raise ValueError('Invalid same-year lead interval')
    if np.any((result['lead_days'] == 0) != (tail.sum(1) == 0)):
        raise ValueError('Lead interval disagrees with prediction suffix')
    return result
