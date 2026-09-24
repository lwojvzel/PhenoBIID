"""Matched recurrent state models with previous-season weather information."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gru_task_priority import RecurrentWorld
from task_aligned_world import INPUTS, calendar
from diagnose_weather_innovation import paired_weather
from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json

MODES = ('continued', 'appended', 'innovation')
PAIRED_INPUTS = (*INPUTS, 'weather_difference')
RESULT = ROOT / 'benchmark/results/paired_weather_world_v1'
CACHE = ROOT / 'benchmark/cache/paired_weather_world_v1'


def run_root(crop, origin, mode, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}' / 'seed_42' / mode


def weather_cache(crop, origin, arrays, meta):
    import fcntl
    root = CACHE / crop / f'origin_{origin}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sources, errors = {}, {}
        recipe = dict(source_files=meta['files'], normalization=meta['normalization'],
            code_sha256=sha256(Path(__file__)),
            gather_sha256=sha256(ROOT / 'scripts/diagnose_weather_innovation.py'))
        path = root / 'manifest.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['recipe'] != recipe:
                raise ValueError('Paired weather cache recipe changed')
            for split, expected in saved['files'].items():
                if sha256(root / f'{split}.npy') != expected:
                    raise ValueError('Weather difference cache changed')
        else:
            for split, a in arrays.items():
                previous, error = paired_weather(a, meta['normalization'], sources)
                difference = np.where(a['relative_valid'][..., None] > 0, a['weather']-previous, 0).astype(np.float32)
                np.save(root / f'{split}.npy', difference)
                errors[split] = error
            saved = dict(recipe=recipe, weather_sources=sources, current_weather_errors=errors,
                files={split: sha256(root / f'{split}.npy') for split in arrays})
            atomic_json(path, saved)
        for split, a in arrays.items():
            a['weather_difference'] = np.load(root / f'{split}.npy')
            if a['weather_difference'].shape != a['weather'].shape:
                raise ValueError('Difference shape mismatch')
    return saved


class PairedWeatherWorld(RecurrentWorld):
    def __init__(self, mode, dim=128, dropout=.1):
        if mode not in MODES:
            raise ValueError('Unknown forcing mode')
        super().__init__(dim, dropout)
        self.mode = mode
        self.direct = nn.GRU(63, dim, num_layers=2, batch_first=True, dropout=dropout)

    def initialize(self, state):
        state = {k: v.detach().clone() for k, v in state.items()}
        expanded = torch.zeros_like(self.direct.weight_ih_l0)
        expanded[:, :50] = state['direct.weight_ih_l0']
        state['direct.weight_ih_l0'] = expanded
        self.load_state_dict(state)

    def forward(self, b, observation_feedback=True):
        if set(b) != set(PAIRED_INPUTS):
            raise ValueError('Only declared inference inputs are allowed')
        mask = b['relative_valid'] > 0
        difference = torch.where(mask[..., None], b['weather_difference'], 0)
        anomaly, cumulative = b['weather_anomaly'], b['weather_cumulative']
        if self.mode == 'innovation':
            anomaly = difference
            cumulative = difference.cumsum(1)/mask.cumsum(1).clamp_min(1).sqrt()[..., None]
        extra = difference if self.mode == 'appended' else torch.zeros_like(difference)
        sequence = torch.cat((b['weather_climo'], anomaly, cumulative,
            b['previous_lai'][..., None], b['previous_lai_valid'][..., None], b['lai_climo'][..., None],
            calendar(b['source_month']), b['context'][:, None].expand(-1, 12, -1)), -1)
        initial = torch.cat((torch.where(mask, b['previous_lai'], 0), b['previous_lai_valid']*mask), -1)
        hidden = self.initial(initial).reshape(len(initial), 2, -1).transpose(0, 1).contiguous()
        feedback = torch.zeros_like(b['previous_lai'][:, 0])
        states, predictions = [], []
        for phase in range(12):
            tokens = torch.cat((sequence[:, phase], feedback[:, None], extra[:, phase]), -1)[:, None]
            _, candidate = self.direct(tokens, hidden)
            hidden = torch.where(mask[:, phase][None, :, None], candidate, hidden)
            state = hidden[-1]
            prediction = b['previous_lai'][:, phase]+self.observation(state).squeeze(-1)
            prediction = torch.where(mask[:, phase], prediction, b['previous_lai'][:, phase])
            if observation_feedback:
                feedback = torch.where(mask[:, phase], prediction-b['lai_climo'][:, phase], feedback)
            states.append(state*mask[:, phase, None]); predictions.append(prediction)
        prediction = torch.stack(predictions, 1)
        encoded, _ = self.memory_readout(torch.stack(states, 1))
        weight = mask/mask.sum(1, keepdim=True).clamp_min(1)
        parts = (b['history'], b['context'], torch.where(mask, prediction-b['lai_climo'], 0),
                 (encoded*weight[..., None]).sum(1))
        return self.yield_head(torch.cat(parts, -1)).squeeze(-1), prediction


@torch.no_grad()
def predict_state(model, data, size=256):
    model.eval(); output = []
    for start in range(0, len(data['previous_lai']), size):
        b = {key: data[key][start:start+size].cuda() for key in PAIRED_INPUTS}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, state = model(b)
        output.append(state.float().cpu().numpy())
    return np.concatenate(output)
