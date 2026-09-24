"""Chronological tail masks and frozen, observed-prefix state updates."""
import numpy as np
import torch


RATIOS = tuple(i/10 for i in range(11))
MODES = ('annual_forecast', 'prefix_forecast', 'climatology')


def tail_mask(active, ratio):
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2 or not 0 <= ratio <= 1:
        raise ValueError('Expected a batch of active slots and a fraction in [0,1]')
    count = active.sum(1)
    replaced = np.ceil(count*ratio-1e-9).astype(int)
    replaced = np.clip(replaced, 0, count)
    ordinal = np.cumsum(active, axis=1)
    return active & (ordinal > (count-replaced)[:, None])


def mix_trajectory(observed, forecast, tail):
    if observed.shape != forecast.shape or tail.shape != observed.shape:
        raise ValueError('Mismatched trajectory shapes')
    return np.where(tail, forecast, observed)


def prefix_values(observed, active, tail, mean, std):
    known = np.asarray(active, bool) & ~tail & np.isfinite(observed)
    values = np.where(known, (observed-mean)/std, 0).astype(np.float32)
    return values, known


@torch.no_grad()
def prefix_rollout(model, batch, values, known):
    """Use only available prefix observations for the existing scalar feedback."""
    if model.architecture != 'biid':
        raise ValueError('This registered diagnostic uses the existing BIID only')
    if values.shape != batch['previous'].shape or known.shape != values.shape:
        raise ValueError('Invalid prefix shape')
    if torch.any(values[~known] != 0):
        raise ValueError('Future or missing observed values entered prefix inputs')
    zeros = torch.zeros_like(batch['previous'])
    inputs = dict(weather=batch['weather'], previous_lai=zeros, previous_lai_valid=zeros,
        previous_ndvi=batch['previous'], previous_ndvi_valid=batch['previous_valid'],
        previous_ndvi_quality=batch['previous_quality'], relative_valid=batch['relative_valid'],
        context=batch['context'])
    dynamics = model.dynamics
    states = dynamics.initial_states(inputs)
    weather = dynamics.weather(inputs['weather'], inputs['context'])
    outputs = []
    for k in range(12):
        active = inputs['relative_valid'][:, k, None, None].bool()
        states = {p: torch.where(active, dynamics.transitions[p](s, weather[:, k]), s)
                  for p, s in states.items()}
        previous = torch.where(inputs['previous_ndvi_valid'][:, k].bool(),
                               inputs['previous_ndvi'][:, k], zeros[:, k])
        predicted = previous+dynamics.heads['ndvi'](states['ndvi'].mean(1)).squeeze(-1)
        outputs.append(predicted)
        feedback = torch.where(known[:, k], values[:, k], predicted)
        message = dynamics.feedback['ndvi'](feedback[:, None])[:, None]
        states['ndvi'] = torch.where(active, dynamics.norms['ndvi'](states['ndvi']+message), states['ndvi'])
    return torch.stack(outputs, 1)
