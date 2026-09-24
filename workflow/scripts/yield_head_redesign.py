"""Frozen-state yield readouts with token-preserving, gated value exchange.

These are head-only alternatives. Standard source-value attention is not the
original target-relevance BIID operator, and memory tokens are not new data.
"""
from __future__ import annotations

import torch
from torch import nn

from biid_world_model import FeedForward
from dual_remote_state import DualRemoteYield, YIELD_INPUTS

HEADS = ("original_query", "original_flat", "cross_flat", "gated_cross_query", "gated_cross_flat", "gated_cross_memory")


class GatedValueLayer(nn.Module):
    def __init__(self, dim=128, learned_gate=True):
        super().__init__()
        self.learned_gate = learned_gate
        self.s_norm, self.h_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.s_from_h = nn.MultiheadAttention(dim, 4, dropout=.1, batch_first=True)
        self.h_from_s = nn.MultiheadAttention(dim, 4, dropout=.1, batch_first=True)
        self.s_ff_norm, self.h_ff_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.s_ff, self.h_ff = FeedForward(dim, .1), FeedForward(dim, .1)
        self.s_gate, self.h_gate = nn.Linear(2 * dim, dim), nn.Linear(2 * dim, dim)
        for gate in (self.s_gate, self.h_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.197224577)
            gate.requires_grad_(learned_gate)

    def forward(self, source, history, source_mask=None, history_mask=None):
        if source_mask is None:
            source_mask = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
        if history_mask is None:
            history_mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        source_mask, history_mask = source_mask.bool(), history_mask.bool()
        s, h = self.s_norm(source), self.h_norm(history)
        # A zero sentinel prevents all-masked softmax; its output is then
        # explicitly discarded when the source branch has no valid tokens.
        def message(attention, query, values, valid):
            nonempty = valid.any(1)
            safe = valid.clone()
            safe[:, 0] |= ~nonempty
            values = torch.where(valid[..., None], values, torch.zeros_like(values))
            out, _ = attention(query, values, values, key_padding_mask=~safe, need_weights=False)
            return torch.where(nonempty[:, None, None], out, torch.zeros_like(out))
        ms = message(self.s_from_h, s, h, history_mask)
        mh = message(self.h_from_s, h, s, source_mask)
        gs = torch.sigmoid(self.s_gate(torch.cat((s, ms), -1))) if self.learned_gate else .1
        gh = torch.sigmoid(self.h_gate(torch.cat((h, mh), -1))) if self.learned_gate else .1
        # Both updates read the same pre-update state; no sequential leakage.
        sb, hb = source + gs * ms, history + gh * mh
        sn, hn = sb + self.s_ff(self.s_ff_norm(sb)), hb + self.h_ff(self.h_ff_norm(hb))
        return (torch.where(source_mask[..., None], sn, source),
                torch.where(history_mask[..., None], hn, history))


class GatedValueStack(nn.Module):
    def __init__(self, dim=128, learned_gate=True):
        super().__init__()
        self.layers = nn.ModuleList([GatedValueLayer(dim, learned_gate) for _ in range(2)])

    def forward(self, source, history, source_mask=None, history_mask=None):
        for layer in self.layers:
            source, history = layer(source, history, source_mask, history_mask)
        return source, history


class RedesignedYieldHead(DualRemoteYield):
    def __init__(self, mode, head, climate=False, dim=128):
        super().__init__(mode, climate, dim)
        if head not in HEADS:
            raise ValueError(head)
        self.head = head
        if head.startswith("gated_cross") or head == "cross_flat":
            self.biid = GatedValueStack(dim, learned_gate=head != "cross_flat")
        self.memory_count = 12 * len(self.products) if head.endswith("memory") else 0
        if self.memory_count:
            self.memory = nn.Parameter(torch.randn(1, self.memory_count, dim) * .02)
            self.memory_attention = nn.MultiheadAttention(dim, 4, dropout=.1, batch_first=True)
            self.memory_norm = nn.LayerNorm(dim)
        if head.endswith(("flat", "memory")):
            n = 12 * len(self.products) + 12 * int(climate) + self.memory_count
            del self.query, self.pool
            self.output = nn.Sequential(nn.LayerNorm((n + 1) * dim), nn.Linear((n + 1) * dim, dim),
                                        nn.GELU(), nn.Dropout(.1), nn.Linear(dim, 2))
            nn.init.constant_(self.output[-1].bias[1], -2.2)

    def encode_tokens(self, inputs):
        active = inputs["relative_valid"].bool()
        tokens = []
        for p in self.products:
            valid = inputs[f"previous_{p}_valid"].bool() & active
            previous = torch.where(valid, inputs[f"previous_{p}"], torch.zeros_like(inputs[f"previous_{p}"]))
            state = torch.where(active, inputs[f"state_{p}"], torch.zeros_like(previous))
            quality = inputs["previous_ndvi_quality"] if p == "ndvi" else valid.to(previous.dtype)
            quality = torch.where(valid, quality, torch.zeros_like(previous))
            features = torch.stack((state, previous, valid.to(previous.dtype), quality), -1)
            tokens.append(self.project[p](features) + self.identity[p] + self.phase)
        tokens = torch.cat(tokens, 1)
        mask = active.repeat(1, len(self.products))
        context = self.context(torch.cat((inputs["context"], inputs["crop_coverage"][:, None]), -1))
        history = self.history(inputs["history"]) + context[:, None]
        return tokens, mask, history, context

    def forward(self, inputs):
        if self.head == "original_query":
            return super().forward(inputs)
        if set(inputs) != set(YIELD_INPUTS):
            raise ValueError("Yield head accepts frozen states and declared past inputs, never labels")
        tokens, mask, history, context = self.encode_tokens(inputs)
        tokens, history = self.biid(tokens, history, mask, None)
        if self.climate:
            active = inputs["relative_valid"].bool()
            weather = torch.where(active[..., None], inputs["weather"], torch.zeros_like(inputs["weather"]))
            tokens = torch.cat((tokens, self.weather_project(weather) + self.phase), 1)
            mask = torch.cat((mask, active), 1)
        history_pool = history.mean(1)
        if self.memory_count:
            # Preserve all observation tokens. Memory reads them but never
            # replaces them; all coordinates reach the final regression layer.
            values = torch.cat((tokens, history_pool[:, None]), 1)
            valid = torch.cat((mask, torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device)), 1)
            query = self.memory + context[:, None]
            extra, _ = self.memory_attention(query, values, values, key_padding_mask=~valid, need_weights=False)
            tokens = torch.cat((tokens, self.memory_norm(query + extra)), 1)
            mask = torch.cat((mask, torch.ones((len(mask), self.memory_count), dtype=torch.bool, device=mask.device)), 1)
        if self.head.endswith(("flat", "memory")):
            tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
            features = torch.cat((tokens.flatten(1), history_pool), -1)
        else:
            values = torch.cat((tokens, history_pool[:, None]), 1)
            valid = torch.cat((mask, torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device)), 1)
            pooled, _ = self.pool(self.query + context[:, None], values, values, key_padding_mask=~valid, need_weights=False)
            features = torch.cat((pooled[:, 0], history_pool), -1)
        delta, logits = self.output(features).unbind(-1)
        gate = torch.sigmoid(logits)
        base = inputs["history_base"]
        return dict(prediction=base + gate * delta, candidate=base + delta, gate=gate)
