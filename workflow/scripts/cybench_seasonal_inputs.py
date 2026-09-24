"""Monthly CY-Bench adaptation with explicit calendar support and issue masks."""
import numpy as np
import pandas as pd

SLOTS = 12
PRODUCTS = ('ndvi', 'gpp')
KNOWN_FIELDS = frozenset((
    'source_month_id', 'previous_source_month_id', 'month_start_day',
    'month_end_day', 'previous_month_end_day', 'calendar_overlap_days',
    'active_mask', 'calendar_sos_day', 'calendar_eos_day', 'cross_year',
    'region_index', 'year', 'context', 'weather',
    'weather_finite_grid_area_fraction', 'previous_vegetation',
    'previous_vegetation_support', 'previous_valid'))


def day_number(date):
    return np.datetime64(date, 'D').astype(np.int32)


def season_calendar(year, sos, eos):
    if not (np.isfinite(sos) and np.isfinite(eos) and 0 <= sos <= 366 and 0 <= eos <= 366):
        raise ValueError('Invalid provider SOS/EOS')
    sos, eos = max(1, int(sos)), max(1, int(eos))
    start = pd.to_datetime(year*1000+sos, format='%Y%j')
    end = pd.to_datetime(year*1000+eos, format='%Y%j')
    if sos > eos:
        start = start-pd.DateOffset(years=1)
    if not 0 <= (end-start).days <= 366:
        raise ValueError('Invalid season duration')
    months = pd.period_range(start, end, freq='M')
    if not 1 <= len(months) <= SLOTS:
        raise ValueError('Calendar touches more than twelve months; explicit protocol needed')
    result = {key:np.zeros(SLOTS, dtype=np.int32) for key in (
        'source_month_id', 'previous_source_month_id', 'month_start_day',
        'month_end_day', 'previous_month_end_day', 'calendar_overlap_days')}
    result['active_mask'] = np.arange(SLOTS) < len(months)
    result['calendar_sos_day'] = day_number(start)
    result['calendar_eos_day'] = day_number(end)
    result['cross_year'] = np.bool_(sos > eos)
    for k, month in enumerate(months):
        prior = month-12
        result['source_month_id'][k] = month.year*100+month.month
        result['previous_source_month_id'][k] = prior.year*100+prior.month
        result['month_start_day'][k] = day_number(month.start_time)
        result['month_end_day'][k] = day_number(month.end_time)
        result['previous_month_end_day'][k] = day_number(prior.end_time)
        result['calendar_overlap_days'][k] = (
            min(end, month.end_time.normalize())-max(start, month.start_time)).days+1
    return result


def issue_masks(active, percent):
    active = np.asarray(active)
    if (active.dtype != np.bool_ or active.ndim != 2 or active.shape[1] != SLOTS
            or isinstance(percent, bool) or not isinstance(percent, (int, np.integer))
            or not 0 <= percent <= 100):
        raise ValueError('Expected packed bool slots and an integer percentage')
    counts = active.sum(1)
    positions = np.arange(SLOTS)[None, :]
    if np.any(counts == 0) or not np.array_equal(active, positions < counts[:, None]):
        raise ValueError('Active slots must be a nonempty packed prefix')
    hidden_counts = (counts*percent+99)//100
    visible = active & (positions < (counts-hidden_counts)[:, None])
    return visible, active & ~visible


def issue_view(known, state_targets, percent):
    """Only this view supplies current remote values to prediction-time models."""
    if set(known) != KNOWN_FIELDS or set(state_targets) != {'vegetation', 'vegetation_support'}:
        raise ValueError('Unexpected fields in model inputs or state supervision')
    visible, hidden = issue_masks(known['active_mask'], percent)
    vegetation = np.asarray(state_targets['vegetation'])
    support = np.asarray(state_targets['vegetation_support'])
    expected = visible.shape+(len(PRODUCTS),)
    if vegetation.shape != expected or support.shape != expected:
        raise ValueError('Vegetation target schema mismatch')
    if np.any(~np.isfinite(support)) or np.any((support < 0) | (support > 1+1e-6)):
        raise ValueError('Invalid vegetation support')
    n_visible = visible.sum(1)
    ends = known['month_end_day'][np.arange(len(visible)), np.maximum(n_visible-1, 0)]
    issue = np.where(n_visible > 0, ends, known['month_start_day'][:, 0]-1)
    if percent > 0 and np.any(issue >= known['calendar_eos_day']):
        raise ValueError('In-season issue must precede supplied EOS')
    if np.any(known['active_mask'] & (known['previous_month_end_day'] > issue[:, None])):
        raise ValueError('Historical vegetation overlaps the issue time')
    observed_valid = visible[..., None] & np.isfinite(vegetation) & (support > 0)
    result = dict(known)
    result.update(observed_vegetation=np.where(observed_valid, vegetation, np.nan),
        observed_vegetation_support=np.where(visible[..., None], support, 0),
        observed_valid=observed_valid, visible_mask=visible, hidden_mask=hidden,
        issue_day=issue.astype(np.int32),
        days_to_supplied_eos=(known['calendar_eos_day']-issue).astype(np.int32))
    return result
