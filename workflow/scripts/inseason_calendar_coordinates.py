"""Move already-masked active tokens to calendar positions without new inputs."""
import numpy as np


def to_calendar(inputs, source_month):
    active = inputs['valid'] > 0
    if active.shape != source_month.shape or active.shape[1:] != (12,):
        raise ValueError('Expected twelve source-month slots')
    if not np.issubdtype(source_month.dtype, np.integer):
        raise ValueError('Source months must be integer indices')
    if np.any(active.sum(1) == 0) or np.any(np.diff(active.astype(np.int8), axis=1) > 0):
        raise ValueError('Expected nonempty packed active slots')
    if np.any((source_month[active] < 0) | (source_month[active] > 11)):
        raise ValueError('Invalid active source month')
    paired = active[:, 1:] & active[:, :-1]
    if np.any(np.diff(source_month.astype(np.int64), axis=1)[paired] <= 0):
        raise ValueError('Active source months must be unique and increasing')
    if np.any(inputs['tail'] & ~active) or np.any(inputs['sequence'][~active] != 0):
        raise ValueError('Inactive tokens must be masked before coordinate conversion')
    row, slot = np.where(active)
    month = source_month[row, slot]
    result = {k: v.copy() for k, v in inputs.items()}
    for key in ('sequence', 'valid', 'tail'):
        result[key].fill(0)
        result[key][row, month] = inputs[key][row, slot]
        np.testing.assert_array_equal(result[key][row, month], inputs[key][row, slot])
    if not np.array_equal(result['valid'].sum(1), inputs['valid'].sum(1)):
        raise ValueError('Coordinate transformation changed active support')
    return result
