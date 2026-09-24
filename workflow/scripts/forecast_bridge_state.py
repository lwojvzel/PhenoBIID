"""Forecast-only state interface on annual, supplied active-slot calendars."""
import numpy as np
import torch
from torch import nn

from token_retention_state import RetainedDynamics

INPUTS = ('weather', 'previous', 'previous_valid', 'previous_quality', 'relative_valid', 'context')


def batch_arrays(a, take, stats, product):
    scale = stats[product]
    physical = a[f'previous_{product}'][take]
    valid = np.isfinite(physical) & (a['relative_valid'][take] > 0)
    old = np.where(valid, (physical-scale['mean'])/scale['std'], 0).astype(np.float32)
    quality = a[f'previous_{product}_quality'][take] if product in ('ndvi', 'gpp') else valid
    weather = (a['weather'][take]-np.array(stats['weather_mean'], np.float32))/np.array(stats['weather_std'], np.float32)
    return dict(weather=np.nan_to_num(weather, nan=0., posinf=0., neginf=0.).astype(np.float32),
                previous=old, previous_valid=valid.astype(np.float32),
                previous_quality=np.where(valid, quality, 0).astype(np.float32),
                relative_valid=a['relative_valid'][take].astype(np.float32),
                context=a['context'][take].astype(np.float32))


class ForecastState(nn.Module):
    def __init__(self, architecture='biid', dim=128):
        super().__init__()
        self.architecture = architecture
        if architecture == 'biid':
            # The generic scalar product uses the existing NDVI branch layout;
            # each physical product is normalized and fitted independently.
            self.dynamics = RetainedDynamics('ndvi', 'retain', dim=dim)
        elif architecture == 'gru':
            self.initial = nn.Linear(36+5, dim)
            self.cell = nn.GRUCell(13+3+1+12, dim)
            self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            raise ValueError(architecture)

    def forward(self, b, feedback=True):
        if set(b) != set(INPUTS):
            raise ValueError('Only registered forecast inputs are accepted')
        if self.architecture == 'biid':
            zeros = torch.zeros_like(b['previous'])
            inputs = dict(weather=b['weather'], previous_lai=zeros, previous_lai_valid=zeros,
                previous_ndvi=b['previous'], previous_ndvi_valid=b['previous_valid'],
                previous_ndvi_quality=b['previous_quality'], relative_valid=b['relative_valid'], context=b['context'])
            return self.dynamics(inputs, feedback=feedback)['ndvi']
        initial = torch.cat((b['previous'], b['previous_valid'], b['previous_quality'], b['context']), 1)
        state = self.initial(initial)
        last = torch.zeros_like(b['previous'][:, :1])
        positions = torch.eye(12, device=state.device, dtype=state.dtype)
        outputs = []
        for k in range(12):
            phase = positions[k].expand(len(state), -1)
            x = torch.cat((b['weather'][:, k], b['previous'][:, k:k+1],
                b['previous_valid'][:, k:k+1], b['previous_quality'][:, k:k+1], last, phase), 1)
            active = b['relative_valid'][:, k:k+1].bool()
            state = torch.where(active, self.cell(x, state), state)
            value = b['previous'][:, k]+self.readout(state).squeeze(-1)
            outputs.append(value)
            if feedback:
                last = torch.where(active, value[:, None], last)
        return torch.stack(outputs, 1)


def masked_loss(prediction, target, active):
    valid = torch.isfinite(target) & active.bool()
    truth = torch.where(valid, target, torch.zeros_like(target))
    return ((prediction-truth).square()*valid).sum()/valid.sum().clamp_min(1)
