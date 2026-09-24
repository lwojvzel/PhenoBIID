"""Original terminal feature layout with explicit issue-time masking."""
import numpy as np

from forecast_bridge_data import fit_stats, history_features, remote_features
from inseason_complete_inputs import mask_metadata
from inseason_ndvi_reuse import historical_support
from ndvi_tail_replacement import tail_mask


def local_climate(raw, fit, take, product, fallback_mean):
    grid = raw['row'].astype(np.int64)*720+raw['col']
    months = np.minimum(raw['source_month'], 11).astype(np.int64)
    values = raw[f'observed_{product}'][fit]
    valid = np.isfinite(values) & (raw['relative_valid'][fit] > 0)
    keys = grid[fit, None]*12+months[fit]
    totals = np.bincount(keys[valid], weights=values[valid], minlength=360*720*12)
    counts = np.bincount(keys[valid], minlength=len(totals))
    mt = np.bincount(months[fit][valid], weights=values[valid], minlength=12)
    mc = np.bincount(months[fit][valid], minlength=12)
    query = grid[take, None]*12+months[take]
    numerator, denominator = totals[query], counts[query]
    fallback_sum, fallback_count = mt[months[take]], mc[months[take]]
    # Exclude the queried fitting record, matching the original anomaly encoder.
    own = raw[f'observed_{product}'][take]
    exclude = np.isin(take, fit)[:, None] & np.isfinite(own) & (raw['relative_valid'][take] > 0)
    numerator = numerator-np.where(exclude, own, 0)
    denominator = denominator-exclude
    fallback_sum = fallback_sum-np.where(exclude, own, 0)
    fallback_count = fallback_count-exclude
    fallback = np.divide(fallback_sum, fallback_count, out=np.full(own.shape, fallback_mean), where=fallback_count > 0)
    return np.divide(numerator, denominator, out=fallback.copy(), where=denominator > 0).astype(np.float32)


def context(raw, fit, take, recipe):
    stats = fit_stats(raw, fit)
    history, _ = history_features(raw, take, stats)
    active = raw['relative_valid'][take] > 0
    weather = np.nan_to_num((raw['weather'][take]-np.asarray(stats['weather_mean'])) /
        np.asarray(stats['weather_std']), nan=0., posinf=0., neginf=0.).astype(np.float32)
    support = historical_support(raw, fit, take, raw['metadata'][fit])
    products = tuple(recipe.split('_'))
    return dict(history=history, active=active, weather=weather,
        metadata=np.array(raw['metadata'][take], copy=True), support=support, stats=stats,
        products=products, observed={p: raw[f'observed_{p}'][take] for p in products},
        climate={p: local_climate(raw, fit, take, p, stats[p]['mean']) for p in products})


def features(ctx, indices, ratio, mode, forecast=None, forward_climate=None):
    if mode not in ('observed', 'prefix', 'climatology', 'biid'):
        raise ValueError('Unregistered completion mode')
    active = ctx['active'][indices]
    tail = tail_mask(active, 0. if mode == 'observed' else ratio)
    metadata = ctx['metadata'][indices]
    slots = mask_metadata(metadata, tail, ctx['support'][indices])
    parts = [ctx['history'][indices], metadata[:, :1], slots.reshape(len(indices), -1),
             ctx['weather'][indices].reshape(len(indices), -1)]
    for product in ctx['products']:
        observed = ctx['observed'][product][indices]
        climo = ctx['climate'][product][indices]
        if mode == 'observed':
            value = observed
        elif mode == 'prefix':
            value = np.where(tail, np.nan, observed)
        else:
            if mode == 'biid':
                if forecast is None:
                    raise ValueError('BIID requires predicted, not true, suffix values')
                replacement = forecast[product][indices]
            else:
                replacement = climo if forward_climate is None else forward_climate[product][indices]
            value = np.where(tail, replacement, observed)
        parts.append(remote_features(value, active, climo, ctx['stats'][product]))
    result = np.concatenate(parts, 1).astype(np.float32)
    expected = 465+36*len(ctx['products'])
    if result.shape != (len(indices), expected) or not np.isfinite(result).all():
        raise ValueError('Invalid nested terminal input')
    return result
