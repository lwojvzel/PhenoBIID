"""Compact recurrent corrections to a frozen LAI prior, with outcome-aware fidelity."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from paired_weather_world import PAIRED_INPUTS
from task_aligned_world import calendar
from linear_state_yield import inputs as load_sources, RESULT as READOUTS, PENALTIES
from review_revision_data import ROOT, sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/yield_sensitive_state_v1'
CACHE = ROOT / 'benchmark/cache/yield_sensitive_state_v1'
STATE_INPUTS = (*[k for k in PAIRED_INPUTS if k != 'history'], 'prior_lai')
MODES = dict(state_only=0., projection_tenth=.1, projection_one=1.)


def run_root(crop, origin, mode, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / f'origin_{origin}' / 'seed_42' / mode


def teacher_sensitivity(bundle, column):
    return (bundle['model'].coef_[column, -12:]/bundle['scaler'].scale_[-12:]).astype(np.float32)


def prepare(crop, origin):
    import fcntl
    root = CACHE / crop / f'origin_{origin}'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arrays, meta, difference, _, forecasts, sources = load_sources(crop, origin)
        teacher = READOUTS / crop / f'origin_{origin}/observed_state.joblib'
        complete = json.loads((teacher.parent / 'complete.json').read_text())
        if sha256(teacher) != complete['weights_sha256'][str(teacher)]:
            raise ValueError('Teacher weight changed')
        column = int(np.flatnonzero(PENALTIES == .01)[0])
        beta = teacher_sensitivity(joblib.load(teacher), column)
        for split, a in arrays.items():
            a['weather_difference'] = difference[split]
            a['prior_lai'] = forecasts[split]['state_innovation'].astype(np.float32)
        a = arrays['train']; mask = (a['target_lai_valid'] > 0) & (a['relative_valid'] > 0)
        error = (a['prior_lai'].astype(float)-a['target_lai'].astype(float))*mask
        projection = (error*beta.astype(float)).sum(1)
        scales = dict(lai=max(float(np.mean(error[mask]**2)), 1e-6),
                      projection=max(float(np.mean(projection**2)), 1e-6))
        spec = dict(input_manifest=meta, sources=sources, teacher_weight=str(teacher), teacher_sha256=sha256(teacher),
            teacher_penalty=.01, teacher_output_column=column, beta=beta.tolist(), scales=scales,
            prior_source='W9 full-window innovation, validation-selected state penalty',
            code_hashes={name: sha256(ROOT / 'scripts' / name) for name in
                ('yield_sensitive_state.py', 'linear_state_yield.py', 'diagnose_weather_innovation.py')})
        path = root / 'manifest.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['spec'] != spec:
                raise ValueError('Sensitive-state input recipe changed')
            for split, digest in saved['files'].items():
                if sha256(root / f'{split}.npz') != digest:
                    raise ValueError('Sensitive-state arrays changed')
        else:
            for split, a in arrays.items():
                np.savez(root / f'{split}.npz', **{k: a[k] for k in (*STATE_INPUTS, 'target_lai', 'target_lai_valid',
                    'source_indices', 'row', 'col', 'year')})
            atomic_json(path, dict(spec=spec, files={split: sha256(root / f'{split}.npz') for split in arrays}))
    return root


def load(crop, origin):
    root = CACHE / crop / f'origin_{origin}'
    meta = json.loads((root / 'manifest.json').read_text())
    arrays = {}
    for split in ('train', 'validation', 'test'):
        with np.load(root / f'{split}.npz') as f:
            arrays[split] = {key: f[key] for key in f.files}
    return arrays, meta


class YieldSensitiveState(nn.Module):
    def __init__(self, dim=32, limit=.25):
        super().__init__()
        self.limit = limit
        self.initial = nn.Linear(24, 2*dim)
        self.transition = nn.GRU(63, dim, num_layers=2, batch_first=True, dropout=.1)
        self.observation = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, 1))
        nn.init.zeros_(self.initial.weight); nn.init.zeros_(self.initial.bias)
        nn.init.zeros_(self.observation[-1].weight); nn.init.zeros_(self.observation[-1].bias)

    def forward(self, b, observation_feedback=True, return_latent=False):
        if set(b) != set(STATE_INPUTS):
            raise ValueError('Only declared state inputs are allowed')
        mask = b['relative_valid'] > 0
        initial = torch.cat((torch.where(mask, b['previous_lai'], 0), b['previous_lai_valid']*mask), -1)
        hidden = self.initial(initial).reshape(len(initial), 2, -1).transpose(0, 1).contiguous()
        sequence = torch.cat((b['weather_climo'], b['weather_anomaly'], b['weather_cumulative'],
            b['previous_lai'][..., None], b['previous_lai_valid'][..., None], b['lai_climo'][..., None],
            calendar(b['source_month']), b['context'][:, None].expand(-1, 12, -1)), -1)
        feedback = torch.zeros_like(b['previous_lai'][:, 0]); predictions, latent = [], []
        for phase in range(12):
            forcing = torch.cat((sequence[:, phase], feedback[:, None], b['weather_difference'][:, phase]), -1)
            _, candidate = self.transition(forcing[:, None], hidden)
            hidden = torch.where(mask[:, phase][None, :, None], candidate, hidden)
            correction = self.limit*torch.tanh(self.observation(hidden[-1]).squeeze(-1))
            prediction = b['prior_lai'][:, phase]+torch.where(mask[:, phase], correction, 0)
            if observation_feedback:
                feedback = torch.where(mask[:, phase], prediction-b['lai_climo'][:, phase], feedback)
            predictions.append(prediction); latent.append(hidden[-1]*mask[:, phase, None])
        state = torch.stack(predictions, 1)
        return (state, torch.stack(latent, 1)) if return_latent else state


def projection_rmse(a, prediction, beta):
    mask = (a['relative_valid'] > 0) & (a['target_lai_valid'] > 0)
    error = ((prediction.astype(float)-a['target_lai'].astype(float))*mask*beta.astype(float)).sum(1)
    return float(np.sqrt(np.mean(error**2)))


@torch.no_grad()
def predict(model, data, size=256):
    model.eval(); output = []
    for start in range(0, len(data['previous_lai']), size):
        b = {key: data[key][start:start+size].cuda() for key in STATE_INPUTS}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = model(b)
        output.append(state.float().cpu().numpy())
    return np.concatenate(output)
