"""Full-slot recurrent world models with observable or latent terminal readouts."""
import math

import torch
from torch import nn

from biid_world_model import CropWorldDynamics

VARIANTS = ('history_mlp', 'direct_gru', 'biid_joint', 'slot_observable', 'slot_joint', 'forcing_joint')
INPUTS = ('weather', 'weather_climo', 'weather_anomaly', 'weather_cumulative', 'previous_lai',
          'previous_lai_valid', 'relative_valid', 'source_month', 'context', 'history', 'lai_climo')


def calendar(month):
    angle = month.clamp(0, 11).float() * (2 * math.pi / 12)
    return torch.stack((angle.sin(), angle.cos()), -1)


class GatedCrossLayer(nn.Module):
    def __init__(self, dim=128, dropout=.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.forcing_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, state, forcing):
        key = self.forcing_norm(forcing)
        message, _ = self.cross(self.query_norm(state), key, key, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat((state, message), -1)))
        state = state + gate * message
        return self.output_norm(state + self.mlp(self.mlp_norm(state)))


class FullSlotTransition(nn.Module):
    def __init__(self, decomposed=False, dim=128, dropout=.1):
        super().__init__()
        self.decomposed = decomposed
        self.initial = nn.Linear(5, dim)
        self.context = nn.Linear(5, dim)
        self.phase = nn.Parameter(torch.randn(1, 12, dim) * .02)
        encoder = nn.TransformerEncoderLayer(dim, 4, 2 * dim, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.initial_encoder = nn.TransformerEncoder(encoder, 1)
        self.forcing = nn.Linear(3 if decomposed else 1, dim)
        self.variable = nn.Parameter(torch.randn(1, 13, dim) * .02)
        self.calendar_projection = nn.Linear(2, dim)
        self.layers = nn.ModuleList([GatedCrossLayer(dim, dropout) for _ in range(2)])
        self.observation = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.feedback = nn.Linear(1, dim)
        self.feedback_norm = nn.LayerNorm(dim)

    def forward(self, b, observation_feedback=True):
        mask = b['relative_valid'] > 0
        cal = calendar(b['source_month'])
        initial = torch.cat((b['previous_lai'][..., None], b['previous_lai_valid'][..., None], b['lai_climo'][..., None], cal), -1)
        state = self.initial(initial) + self.phase + self.context(b['context'])[:, None]
        padding = ~mask
        # One neutral key avoids all-masked attention for an empty season.
        safe_padding = padding.clone(); safe_padding[padding.all(1), 0] = False
        state = self.initial_encoder(state, src_key_padding_mask=safe_padding)
        state = state * mask[..., None]
        forcing = torch.stack((b['weather_climo'], b['weather_anomaly'], b['weather_cumulative']), -1) if self.decomposed else b['weather'][..., None]
        forcing = self.forcing(forcing) + self.variable[:, None]
        forcing = forcing + self.calendar_projection(cal)[:, :, None] + self.context(b['context'])[:, None, None]
        predictions, trajectory = [], []
        for phase in range(12):
            before = state
            candidate = state
            for layer in self.layers:
                candidate = layer(candidate, forcing[:, phase])
            valid = mask[:, phase, None, None]
            state = torch.where(valid, candidate, before)
            # Read the corresponding retained slot, not a mean of compressed queries.
            observation = b['previous_lai'][:, phase] + self.observation(state[:, phase]).squeeze(-1)
            observation = torch.where(mask[:, phase], observation, b['previous_lai'][:, phase])
            predictions.append(observation)
            if observation_feedback:
                feedback = self.feedback((observation - b['lai_climo'][:, phase])[:, None])[:, None]
                state = torch.where(valid, self.feedback_norm(state + feedback), state)
            trajectory.append(state[:, phase] * mask[:, phase, None])
        return torch.stack(predictions, 1), torch.stack(trajectory, 1)


class TaskAlignedWorld(nn.Module):
    def __init__(self, variant, dim=128, dropout=.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        if variant == 'biid_joint':
            self.transition = CropWorldDynamics('biid_climate', dim=dim)
        elif variant in ('slot_observable', 'slot_joint', 'forcing_joint'):
            self.transition = FullSlotTransition(variant == 'forcing_joint', dim, dropout)
        elif variant == 'direct_gru':
            self.direct = nn.GRU(49, dim, num_layers=2, batch_first=True, dropout=dropout)
        self.use_memory = variant in ('biid_joint', 'slot_joint', 'forcing_joint', 'direct_gru')
        if self.use_memory:
            self.memory_readout = nn.GRU(dim, 64, batch_first=True)
        width = 20 + (0 if variant == 'history_mlp' else 12) + (64 if self.use_memory else 0)
        self.yield_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        nn.init.zeros_(self.yield_head[-1].weight); nn.init.zeros_(self.yield_head[-1].bias)

    def forward(self, b, observation_feedback=True):
        if set(b) != set(INPUTS):
            raise ValueError('Only declared inference inputs are allowed')
        mask = b['relative_valid'] > 0
        if self.variant == 'history_mlp':
            prediction = b['previous_lai']; memory = None
        elif self.variant == 'direct_gru':
            sequence = torch.cat((b['weather_climo'], b['weather_anomaly'], b['weather_cumulative'],
                b['previous_lai'][..., None], b['previous_lai_valid'][..., None], b['lai_climo'][..., None],
                calendar(b['source_month']), b['context'][:, None].expand(-1, 12, -1)), -1)
            memory, _ = self.direct(sequence)
            prediction = b['previous_lai']
        elif self.variant == 'biid_joint':
            # Legacy transition excludes yield history; only the terminal map sees it.
            prediction, memory = self.transition(b['weather'], b['previous_lai'], b['previous_lai_valid'],
                b['relative_valid'], torch.zeros_like(b['history']), b['context'], return_trajectory=True,
                observation_feedback=observation_feedback)
        else:
            prediction, memory = self.transition(b, observation_feedback)
        parts = [b['history'], b['context']]
        if self.variant != 'history_mlp':
            parts.append(torch.where(mask, prediction - b['lai_climo'], 0))
        if self.use_memory:
            encoded, _ = self.memory_readout(memory * mask[..., None])
            weight = mask / mask.sum(1, keepdim=True).clamp_min(1)
            parts.append((encoded * weight[..., None]).sum(1))
        return self.yield_head(torch.cat(parts, -1)).squeeze(-1), prediction
