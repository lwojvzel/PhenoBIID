"""A supervised recurrent world extension of the matched direct climate GRU."""
import torch
from torch import nn

from task_aligned_world import INPUTS, calendar

MODES = ('direct_continued', 'joint_one', 'joint_tenth', 'joint_hundredth', 'yield_priority')
STATE_WEIGHTS = dict(direct_continued=0., joint_one=1., joint_tenth=.1, joint_hundredth=.01, yield_priority=.1)


class RecurrentWorld(nn.Module):
    def __init__(self, dim=128, dropout=.1):
        super().__init__()
        self.initial = nn.Linear(24, 2 * dim)
        self.direct = nn.GRU(50, dim, num_layers=2, batch_first=True, dropout=dropout)
        self.observation = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.memory_readout = nn.GRU(dim, 64, batch_first=True)
        self.yield_head = nn.Sequential(nn.Linear(96, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        nn.init.zeros_(self.initial.weight); nn.init.zeros_(self.initial.bias)
        nn.init.zeros_(self.observation[-1].weight); nn.init.zeros_(self.observation[-1].bias)

    def initialize_from_direct(self, source):
        state = source.direct.state_dict()
        expanded = torch.zeros_like(self.direct.weight_ih_l0)
        expanded[:, :49] = state['weight_ih_l0']
        state['weight_ih_l0'] = expanded
        self.direct.load_state_dict(state)
        self.memory_readout.load_state_dict(source.memory_readout.state_dict())
        self.yield_head.load_state_dict(source.yield_head.state_dict())

    def transition_parameters(self):
        return [p for module in (self.initial, self.direct, self.observation) for p in module.parameters()]

    def forward(self, b, observation_feedback=True):
        if set(b) != set(INPUTS):
            raise ValueError('Only declared inference inputs are allowed')
        mask = b['relative_valid'] > 0
        cal = calendar(b['source_month'])
        sequence = torch.cat((b['weather_climo'], b['weather_anomaly'], b['weather_cumulative'],
            b['previous_lai'][..., None], b['previous_lai_valid'][..., None], b['lai_climo'][..., None],
            cal, b['context'][:, None].expand(-1, 12, -1)), -1)
        initial = torch.cat((torch.where(mask, b['previous_lai'], 0), b['previous_lai_valid'] * mask), -1)
        hidden = self.initial(initial).reshape(len(initial), 2, -1).transpose(0, 1).contiguous()
        feedback = torch.zeros_like(b['previous_lai'][:, 0])
        states, predictions = [], []
        for phase in range(12):
            tokens = torch.cat((sequence[:, phase], feedback[:, None]), -1)[:, None]
            state, candidate = self.direct(tokens, hidden)
            hidden = torch.where(mask[:, phase][None, :, None], candidate, hidden)
            state = hidden[-1]
            prediction = b['previous_lai'][:, phase] + self.observation(state).squeeze(-1)
            prediction = torch.where(mask[:, phase], prediction, b['previous_lai'][:, phase])
            if observation_feedback:
                feedback = torch.where(mask[:, phase], prediction - b['lai_climo'][:, phase], feedback)
            states.append(state * mask[:, phase, None]); predictions.append(prediction)
        prediction = torch.stack(predictions, 1)
        trajectory = torch.stack(states, 1)
        encoded, _ = self.memory_readout(trajectory)
        weight = mask / mask.sum(1, keepdim=True).clamp_min(1)
        parts = (b['history'], b['context'], torch.where(mask, prediction - b['lai_climo'], 0),
                 (encoded * weight[..., None]).sum(1))
        return self.yield_head(torch.cat(parts, -1)).squeeze(-1), prediction


def primary_preserving_gradients(primary, auxiliary, weight):
    """One-sided conflicting-gradient projection on shared transition parameters."""
    dot = sum((a * b).sum() for a, b in zip(primary, auxiliary))
    norm_y = sum(a.square().sum() for a in primary)
    norm_l = sum(a.square().sum() for a in auxiliary)
    coefficient = dot.clamp(max=0) / norm_y.clamp_min(1e-20)
    projected = [b - coefficient * a for a, b in zip(primary, auxiliary)]
    merged = [a + weight * b for a, b in zip(primary, projected)]
    metrics = dict(cosine=float(dot / (norm_y * norm_l).sqrt().clamp_min(1e-20)),
                   auxiliary_primary_norm_ratio=float((norm_l / norm_y.clamp_min(1e-20)).sqrt()),
                   conflicting=bool(dot < 0))
    return merged, metrics
