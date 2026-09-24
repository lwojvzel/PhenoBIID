"""Causal BIID state transitions for two separately observed vegetation products."""
from __future__ import annotations

import torch
from torch import nn

from biid_world_model import BIIDStack, HistoryEncoder, WeatherTokenizer

MODES = ("lai", "ndvi", "joint", "dual")
INPUTS = ("weather", "previous_lai", "previous_lai_valid", "previous_ndvi",
          "previous_ndvi_valid", "previous_ndvi_quality", "relative_valid", "context")


class ObservationEncoder(nn.Module):
    def __init__(self, dim=128, tokens=8):
        super().__init__()
        self.project = nn.Linear(3, dim)
        self.position = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        self.null = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, 0.1, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
        self.context = nn.Linear(5, dim)
        self.pool = nn.MultiheadAttention(dim, 4, 0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, value, valid, quality, context):
        value = torch.where(valid.bool(), value, torch.zeros_like(value))
        tokens = self.project(torch.stack((value, valid, quality * valid), -1)) + self.position
        tokens = torch.cat((tokens, self.null.expand(len(tokens), -1, -1)), 1)
        padding = torch.cat((~valid.bool(), torch.zeros((len(valid), 1), dtype=torch.bool, device=valid.device)), 1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        query = self.query + self.context(context)[:, None]
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=padding, need_weights=False)
        return self.norm(query + pooled)


class ClimateTransition(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.biid = BIIDStack(dim, 2, 0.1)
        self.attention = nn.MultiheadAttention(dim, 4, 0.1, batch_first=True)
        self.gate = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, state, climate):
        modified, weather = self.biid(state, climate)
        message, _ = self.attention(modified, weather, weather, need_weights=False)
        candidate = modified + message
        gate = torch.sigmoid(self.gate(torch.cat((state, candidate), -1)))
        return self.norm(gate * candidate + (1 - gate) * state)


class DualRemoteDynamics(nn.Module):
    def __init__(self, mode, dim=128):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.products = (mode,) if mode in ("lai", "ndvi") else ("lai", "ndvi")
        self.encoders = nn.ModuleDict({p: ObservationEncoder(dim) for p in self.products})
        self.weather = WeatherTokenizer(dim, 13)
        names = ("joint",) if mode == "joint" else self.products
        self.transitions = nn.ModuleDict({p: ClimateTransition(dim) for p in names})
        self.identity = nn.ParameterDict({p: nn.Parameter(torch.randn(1, 1, dim) * 0.02) for p in self.products})
        self.heads = nn.ModuleDict({p: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, 1)) for p in self.products})
        self.feedback = nn.ModuleDict({p: nn.Linear(1, dim) for p in self.products})
        self.norms = nn.ModuleDict({p: nn.LayerNorm(dim) for p in self.products})
        if mode == "joint":
            self.shared_query = nn.Parameter(torch.randn(1, 8, dim) * .02)
            self.shared_pool = nn.MultiheadAttention(dim, 4, .1, batch_first=True)
            self.shared_norm = nn.LayerNorm(dim)

    def forward(self, inputs, feedback=True):
        if set(inputs) != set(INPUTS):
            raise ValueError("Dynamics requires only declared past observations, masks, context and weather")
        states = {}
        for p in self.products:
            valid = inputs[f"previous_{p}_valid"]
            quality = inputs["previous_ndvi_quality"] if p == "ndvi" else valid
            states[p] = self.encoders[p](inputs[f"previous_{p}"], valid, quality, inputs["context"]) + self.identity[p]
        weather = self.weather(inputs["weather"], inputs["context"])
        outputs = {p: [] for p in self.products}
        if self.mode == "joint":
            initial = torch.cat([states[p] for p in self.products], 1)
            query = self.shared_query.expand(len(initial), -1, -1)
            shared, _ = self.shared_pool(query, initial, initial, need_weights=False)
            shared = self.shared_norm(query + shared)
            for k in range(12):
                active = inputs["relative_valid"][:, k, None, None].bool()
                candidate = self.transitions["joint"](shared, weather[:, k])
                shared = torch.where(active, candidate, shared)
                messages = []
                for p in self.products:
                    previous = inputs[f"previous_{p}"][:, k]
                    valid = inputs[f"previous_{p}_valid"][:, k].bool()
                    value = torch.where(valid, previous, torch.zeros_like(previous)) + self.heads[p](shared.mean(1)).squeeze(-1)
                    outputs[p].append(value)
                    messages.append(self.feedback[p](value[:, None])[:, None])
                if feedback:
                    updated = self.shared_norm(shared + torch.stack(messages).mean(0))
                    shared = torch.where(active, updated, shared)
            return {p: torch.stack(values, 1) for p, values in outputs.items()}
        for k in range(12):
            before = states
            states = {p: self.transitions[p](before[p], weather[:, k]) for p in self.products}
            active = inputs["relative_valid"][:, k, None, None].bool()
            for p in self.products:
                states[p] = torch.where(active, states[p], before[p])
                previous = inputs[f"previous_{p}"][:, k]
                valid = inputs[f"previous_{p}_valid"][:, k].bool()
                value = torch.where(valid, previous, torch.zeros_like(previous)) + self.heads[p](states[p].mean(1)).squeeze(-1)
                outputs[p].append(value)
                if feedback:
                    updated = self.norms[p](states[p] + self.feedback[p](value[:, None])[:, None])
                    states[p] = torch.where(active, updated, states[p])
        return {p: torch.stack(values, 1) for p, values in outputs.items()}


YIELD_INPUTS = (*INPUTS, "state_lai", "state_ndvi", "history", "history_base", "crop_coverage")


class DualRemoteYield(nn.Module):
    def __init__(self, mode, climate=False, dim=128):
        super().__init__()
        self.products = (mode,) if mode in ("lai", "ndvi") else ("lai", "ndvi")
        self.climate = climate
        self.project = nn.ModuleDict({p: nn.Linear(4, dim) for p in self.products})
        self.identity = nn.ParameterDict({p: nn.Parameter(torch.randn(1, 1, dim) * .02) for p in self.products})
        self.phase = nn.Parameter(torch.randn(1, 12, dim) * .02)
        self.history = HistoryEncoder(dim)
        self.context = nn.Linear(6, dim)
        self.biid = BIIDStack(dim, 2, .1)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.pool = nn.MultiheadAttention(dim, 4, .1, batch_first=True)
        self.weather_project = nn.Linear(13, dim) if climate else None
        self.output = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(.1), nn.Linear(dim, 2))
        nn.init.constant_(self.output[-1].bias[1], -2.2)

    def forward(self, inputs):
        if set(inputs) != set(YIELD_INPUTS):
            raise ValueError("Yield forward cannot receive target observations or targets")
        tokens = []
        for p in self.products:
            valid = inputs[f"previous_{p}_valid"]
            previous = torch.where(valid.bool(), inputs[f"previous_{p}"], torch.zeros_like(valid))
            quality = inputs["previous_ndvi_quality"] if p == "ndvi" else valid
            features = torch.stack((inputs[f"state_{p}"], previous, valid, quality * valid), -1)
            tokens.append(self.project[p](features) + self.identity[p] + self.phase)
        token_mask = inputs["relative_valid"].bool().repeat(1, len(self.products))
        tokens = torch.cat(tokens, 1)
        context = self.context(torch.cat((inputs["context"], inputs["crop_coverage"][:, None]), -1))
        history = self.history(inputs["history"]) + context[:, None]
        tokens, history = self.biid(tokens, history, token_mask, None)
        if self.climate:
            tokens = torch.cat((tokens, self.weather_project(inputs["weather"]) + self.phase), 1)
            token_mask = torch.cat((token_mask, inputs["relative_valid"].bool()), 1)
        # An always-valid history token handles samples with no usable season slots.
        tokens = torch.cat((tokens, history.mean(1, keepdim=True)), 1)
        token_mask = torch.cat((token_mask, torch.ones((len(tokens), 1), dtype=torch.bool, device=tokens.device)), 1)
        pooled, _ = self.pool(self.query + context[:, None], tokens, tokens, key_padding_mask=~token_mask, need_weights=False)
        delta, logits = self.output(torch.cat((pooled[:, 0], history.mean(1)), -1)).unbind(-1)
        gate = torch.sigmoid(logits)
        base = inputs["history_base"]
        return {"prediction": base + gate * delta, "candidate": base + delta, "gate": gate}


def state_loss(predictions, labels, relative_weight):
    losses = []
    for p, prediction in predictions.items():
        weight = labels[f"target_{p}_valid"] * relative_weight
        if weight.sum() > 0:
            losses.append(((prediction.float() - labels[f"target_{p}"])**2 * weight).sum() / weight.sum())
    if not losses:
        return next(iter(predictions.values())).sum() * 0
    return torch.stack(losses).mean()
