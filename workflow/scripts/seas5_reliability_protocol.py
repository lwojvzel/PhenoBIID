"""Calendar and information-boundary checks for the SEAS5 pilot."""
import numpy as np

WEATHER_INDICES = (1, 12, 7)  # t2m, tp, ssrd in the existing physical cache.


def calendar_plan(year, source_month, active, ratio):
    """Issue at first hidden month's start; use the previous month's run.

    source_month is zero-based. CDS monthly lead 1 denotes the initialization
    calendar month. The previous run avoids treating a run issued during the
    hidden month as if it were available at that month's beginning.
    """
    year = np.asarray(year, dtype=np.int64)
    month = np.asarray(source_month, dtype=np.int64)
    active = np.asarray(active, dtype=bool)
    if month.shape != active.shape or month.shape != (len(year), 12):
        raise ValueError('Expected N by 12 packed crop-active slots')
    if not 0 < ratio <= 1 or np.any(active.sum(1) == 0):
        raise ValueError('Invalid suffix ratio or empty season')
    if np.any(np.diff(active.astype(int), axis=1) > 0):
        raise ValueError('Active slots must precede padding')
    if np.any(active & ((month < 0) | (month > 11))):
        raise ValueError('Invalid natural month')
    if np.any(active[:, 1:] & (np.diff(month, axis=1) <= 0)):
        raise ValueError('Expected ascending natural months, not repaired cross-year seasons')
    count = active.sum(1)
    hidden_count = np.ceil(ratio * count - 1e-9).astype(int)
    first = count - hidden_count
    hidden = active & (np.arange(12)[None, :] >= first[:, None])
    issue_month = year * 12 + month[np.arange(len(year)), first]
    init_month = issue_month - 1
    valid_month = year[:, None] * 12 + month
    leads = valid_month - init_month[:, None] + 1
    supported = np.all(~hidden | ((leads >= 1) & (leads <= 6)), axis=1)
    return dict(hidden=hidden, initialization_year=init_month // 12,
                initialization_month=init_month % 12 + 1,
                issue_month=issue_month % 12 + 1,
                leadtime_month=np.where(hidden, leads, 0), supported=supported)


def replace_future_weather(actual, predicted, hidden):
    """Restrict both routes to three shared physical variables; never fill truth."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    hidden = np.asarray(hidden, dtype=bool)
    if actual.shape[-1] != 13 or predicted.shape != actual.shape[:-1] + (3,):
        raise ValueError('Expected original 13 channels and three forecast channels')
    if hidden.shape != actual.shape[:-1]:
        raise ValueError('Weather and suffix shapes differ')
    if not np.isfinite(predicted[hidden]).all():
        raise ValueError('Missing forecast: do not backfill with future reanalysis')
    return np.where(hidden[..., None], predicted, actual[..., WEATHER_INDICES])
