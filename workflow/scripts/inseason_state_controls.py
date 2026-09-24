"""Observable-prefix recurrence for the existing GRU state family."""
import json

import torch

from inseason_13year_data import ROOT, RECIPES, BLOCKS
from ndvi_tail_replacement import prefix_rollout
from review_revision_data import sha256
from run_forecast_bridge_state import run_root as state_root
from run_ndvi_signal_permutation import check_files

OUT = ROOT / 'benchmark/results/inseason_state_controls_v1'
METHODS = ('biid', 'gru', 'no_feedback', 'climatology', 'previous', 'persistence')


def checked_state(crop, product, architecture, cutoff):
    root = state_root(crop, product, architecture, cutoff, 42)
    marker = json.loads((root / 'complete.json').read_text())
    check_files(root, marker['files'])
    config = json.loads((root / 'config.json').read_text())
    for name, digest in config['code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError('Changed original state implementation')
    if marker['smoke'] or marker['selected_epochs'] < 1 or marker['full_fit_cutoff'] != cutoff:
        raise ValueError('Invalid state checkpoint')
    norm = json.loads((root / 'normalization.json').read_text())
    if max(norm['fit_years']) != cutoff or config['selection_years'] != [cutoff-1, cutoff]:
        raise ValueError('Incorrect state normalization or selection years')
    return root, norm, config


@torch.no_grad()
def gru_prefix_rollout(model, batch, values, known):
    if model.architecture != 'gru':
        raise ValueError('Expected the registered GRU state model')
    if values.shape != batch['previous'].shape or known.shape != values.shape:
        raise ValueError('Invalid prefix shape')
    if torch.any(values[~known] != 0):
        raise ValueError('Hidden observed values entered GRU feedback')
    state = model.initial(torch.cat((batch['previous'], batch['previous_valid'],
        batch['previous_quality'], batch['context']), 1))
    last = torch.zeros_like(batch['previous'][:, :1])
    positions = torch.eye(12, device=state.device, dtype=state.dtype)
    outputs = []
    for k in range(12):
        phase = positions[k].expand(len(state), -1)
        x = torch.cat((batch['weather'][:, k], batch['previous'][:, k:k+1],
            batch['previous_valid'][:, k:k+1], batch['previous_quality'][:, k:k+1], last, phase), 1)
        active = batch['relative_valid'][:, k:k+1].bool()
        state = torch.where(active, model.cell(x, state), state)
        value = batch['previous'][:, k]+model.readout(state).squeeze(-1)
        outputs.append(value)
        observed_or_predicted = torch.where(known[:,k], values[:,k], value)
        last = torch.where(active, observed_or_predicted[:,None], last)
    return torch.stack(outputs, 1)
