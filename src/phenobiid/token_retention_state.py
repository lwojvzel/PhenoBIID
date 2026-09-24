"""Preserve short observation sequences while testing latent-state capacity.

The query8 control is the unchanged two-product implementation. Additional
memory tokens are inferred features, not additional remote observations.
"""
from __future__ import annotations

import torch
from torch import nn

from .dual_remote_state import DualRemoteDynamics, INPUTS

STRATEGIES = ("query8", "query24", "retain", "expand")


class RetainedObservationEncoder(nn.Module):
    def __init__(self, original, strategy):
        super().__init__()
        if strategy not in ("retain", "expand"):
            raise ValueError(strategy)
        self.strategy = strategy
        for name in ("project", "position", "null", "encoder", "context", "norm"):
            setattr(self, name, getattr(original, name))
        if strategy == "expand":
            self.memory = nn.Parameter(torch.randn(1, 12, self.position.shape[-1]) * .02)
            self.pool = original.pool

    def forward(self, value, valid, quality, context):
        valid = valid.bool()
        value = torch.where(valid, value, torch.zeros_like(value))
        quality = torch.where(valid, quality, torch.zeros_like(quality))
        tokens = self.project(torch.stack((value, valid.to(value.dtype), quality), -1)) + self.position
        tokens = torch.cat((tokens, self.null.expand(len(tokens), -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros((len(valid), 1), dtype=torch.bool, device=valid.device)), 1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        context = self.context(context)[:, None]
        # Missing slots remain labeled latent slots, not claimed observations.
        retained = self.norm(encoded[:, :12] + context)
        if self.strategy == "retain":
            return retained
        query = self.memory + context
        extra, _ = self.pool(query, encoded, encoded, key_padding_mask=padding, need_weights=False)
        return torch.cat((retained, self.norm(query + extra)), 1)


class LengthMatchedModulation(nn.Module):
    """Optional control matching the original eight-target modulation scale."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, source, target, source_mask=None, target_mask=None):
        value = self.original(source, target, source_mask, target_mask)
        count = target.shape[1] if target_mask is None else target_mask.sum(1)[:, None, None]
        return value * (count / 8.)


class RetainedDynamics(DualRemoteDynamics):
    def __init__(self, mode, strategy, dim=128, biid_scale="original"):
        super().__init__(mode, dim)
        if strategy not in STRATEGIES or biid_scale not in ("original", "length_matched"):
            raise ValueError((strategy, biid_scale))
        self.strategy, self.biid_scale = strategy, biid_scale
        if strategy == "query24":
            for encoder in self.encoders.values():
                encoder.query = nn.Parameter(torch.randn(1, 24, dim) * .02)
            if mode == "joint":
                self.shared_query = nn.Parameter(torch.randn(1, 48, dim) * .02)
        elif strategy in ("retain", "expand"):
            self.encoders = nn.ModuleDict({p: RetainedObservationEncoder(e, strategy)
                                          for p, e in self.encoders.items()})
            if mode == "joint":
                del self.shared_query, self.shared_pool
        if biid_scale == "length_matched":
            for transition in self.transitions.values():
                for layer in transition.biid.layers:
                    # BIID source is state; only weather -> state changes length.
                    layer.target_to_source = LengthMatchedModulation(layer.target_to_source)

    def initial_states(self, inputs):
        states = {}
        for p in self.products:
            valid = inputs[f"previous_{p}_valid"]
            quality = inputs["previous_ndvi_quality"] if p == "ndvi" else valid
            states[p] = self.encoders[p](inputs[f"previous_{p}"], valid, quality, inputs["context"]) + self.identity[p]
        if self.mode != "joint":
            return states
        initial = torch.cat([states[p] for p in self.products], 1)
        if self.strategy in ("retain", "expand"):
            return {"joint": self.shared_norm(initial)}
        query = self.shared_query.expand(len(initial), -1, -1)
        pooled, _ = self.shared_pool(query, initial, initial, need_weights=False)
        return {"joint": self.shared_norm(query + pooled)}

    def forward(self, inputs, feedback=True):
        if self.strategy == "query8" and self.biid_scale == "original":
            return super().forward(inputs, feedback)
        if set(inputs) != set(INPUTS):
            raise ValueError("State forward accepts past observations, masks, context and weather only")
        states = self.initial_states(inputs)
        weather = self.weather(inputs["weather"], inputs["context"])
        outputs = {p: [] for p in self.products}
        for k in range(12):
            active = inputs["relative_valid"][:, k, None, None].bool()
            states = {p: torch.where(active, self.transitions[p](state, weather[:, k]), state)
                      for p, state in states.items()}
            messages = []
            for p in self.products:
                key = "joint" if self.mode == "joint" else p
                previous = inputs[f"previous_{p}"][:, k]
                valid = inputs[f"previous_{p}_valid"][:, k].bool()
                value = torch.where(valid, previous, torch.zeros_like(previous)) + self.heads[p](states[key].mean(1)).squeeze(-1)
                outputs[p].append(value)
                if feedback:
                    message = self.feedback[p](value[:, None])[:, None]
                    if self.mode == "joint":
                        messages.append(message)
                    else:
                        states[p] = torch.where(active, self.norms[p](states[p] + message), states[p])
            if feedback and self.mode == "joint":
                states["joint"] = torch.where(active, self.shared_norm(states["joint"] + torch.stack(messages).mean(0)), states["joint"])
        return {p: torch.stack(values, 1) for p, values in outputs.items()}


def state_lengths(mode, strategy):
    length = {"query8": 8, "query24": 24, "retain": 12, "expand": 24}[strategy]
    if mode == "joint":
        return {"joint": 8 if strategy == "query8" else 2 * length}
    return {p: length for p in ((mode,) if mode in ("lai", "ndvi") else ("lai", "ndvi"))}
