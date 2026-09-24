from __future__ import annotations
import numpy as np
def tail_mask(active, ratio):
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2 or not 0 <= ratio <= 1:
        raise ValueError('Expected a batch of active slots and a fraction in [0,1]')
    count = active.sum(1)
    replaced = np.ceil(count*ratio-1e-9).astype(int)
    replaced = np.clip(replaced, 0, count)
    ordinal = np.cumsum(active, axis=1)
    return active & (ordinal > (count-replaced)[:, None])
