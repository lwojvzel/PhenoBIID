"""Matched vegetation-value interfaces without product-specific prior quality."""
import numpy as np

from run_inseason_13year_direct import make_design

CONDITIONS = ('metadata', 'lai', 'ndvi', 'gpp', 'ndvi_gpp')


def mask_values(inputs, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unregistered signal condition')
    result = dict(inputs, sequence=inputs['sequence'].copy(), static=inputs['static'].copy())
    products = 2 if condition == 'ndvi_gpp' else 1
    if result['sequence'].shape[1:] != (12, 38+7*products):
        raise ValueError('Unexpected signal sequence shape')
    if result['static'].shape[1:] != (21+24*products,):
        raise ValueError('Unexpected signal static shape')
    # LAI has no comparable native prior-quality field in this physical cache.
    result['sequence'][..., 41::7] = 0
    if condition == 'metadata':
        result['sequence'][..., 38:] = 0
        result['static'][:, 21:] = 0
    return result


def design(raw, fit, other, condition):
    inputs = dict(raw)
    inputs['previous_lai_quality'] = np.zeros_like(raw['previous_lai'])
    recipe = 'lai' if condition == 'metadata' else condition
    x, residual, normalization = make_design(inputs, fit, other, recipe, .1)
    return {key: mask_values(value, condition) for key, value in x.items()}, residual, normalization
