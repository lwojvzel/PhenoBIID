"""Explicit alternatives to mean-query BIID; old checkpoint APIs stay unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from biid_world_model import BIIDStack, CropWorldDynamics, FeedForward

KINDS = ("original", "mean_normalized", "pairwise_relevance", "cross_value")


def masked_softmax(logits, mask):
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)
    weights = weights * mask
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)


class NormalizedMessage(nn.Module):
    def __init__(self, dim, kind, heads=4):
        super().__init__()
        assert dim % heads == 0 and kind in KINDS[1:]
        self.kind, self.heads, self.head_dim = kind, heads, dim // heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.log_temperature = nn.Parameter(torch.full((heads, 1, 1), math.log(math.sqrt(self.head_dim))))

    def heads_of(self, values):
        return values.reshape(values.shape[0], values.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, source, target, source_mask=None, target_mask=None):
        if source_mask is None:
            source_mask = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
        if target_mask is None:
            target_mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        source_mask, target_mask = source_mask.bool(), target_mask.bool()
        cross = self.kind == "cross_value"
        q = self.heads_of(self.query(target if cross else source))
        k = self.heads_of(self.key(source if cross else target))
        v = self.heads_of(self.value(source if cross else target))
        with torch.autocast(device_type=source.device.type, enabled=False):
            q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
            temperature = self.log_temperature.float().clamp(math.log(0.1), math.log(100)).exp()
            if cross:
                message = F.scaled_dot_product_attention(
                    q * temperature, k, v.float(),
                    attn_mask=source_mask[:, None, None, :], scale=1.0, dropout_p=0.0)
            else:
                affinity = (q @ k.transpose(-1, -2)) * temperature
                source_weight = source_mask[:, None, :, None].float()
                if self.kind == "mean_normalized":
                    logits = (affinity * source_weight).sum(-2) / source_weight.sum(-2).clamp_min(1)
                    relevance = masked_softmax(logits, target_mask[:, None, :])
                else:
                    rows = masked_softmax(affinity, target_mask[:, None, None, :])
                    relevance = (rows * source_weight).sum(-2) / source_weight.sum(-2).clamp_min(1)
                relevance = relevance * target_mask.sum(-1)[:, None, None]
                message = relevance[..., None] * v.float()
            message = message * source_mask.any(-1)[:, None, None, None]
            message = message * target_mask[:, None, :, None]
        merged = message.transpose(1, 2).reshape(target.shape).to(target.dtype)
        return self.output(merged) * target_mask[..., None]


class NormalizedInteractionLayer(nn.Module):
    def __init__(self, dim, kind, dropout):
        super().__init__()
        self.source_norm, self.target_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.source_to_target = NormalizedMessage(dim, kind)
        self.target_to_source = NormalizedMessage(dim, kind)
        self.source_mlp_norm, self.target_mlp_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.source_mlp, self.target_mlp = FeedForward(dim, dropout), FeedForward(dim, dropout)
        self.source_scale = nn.Parameter(torch.full((dim,), math.atanh(0.1)))
        self.target_scale = nn.Parameter(torch.full((dim,), math.atanh(0.1)))

    def forward(self, source, target, source_mask=None, target_mask=None):
        s, t = self.source_norm(source), self.target_norm(target)
        sb = source + self.source_scale.tanh() * self.target_to_source(t, s, target_mask, source_mask)
        tb = target + self.target_scale.tanh() * self.source_to_target(s, t, source_mask, target_mask)
        sn = sb + self.source_mlp(self.source_mlp_norm(sb))
        tn = tb + self.target_mlp(self.target_mlp_norm(tb))
        if source_mask is not None: sn = torch.where(source_mask[..., None].bool(), sn, source)
        if target_mask is not None: tn = torch.where(target_mask[..., None].bool(), tn, target)
        return sn, tn


class InteractionStack(nn.Module):
    def __init__(self, dim, layers, dropout, kind):
        super().__init__()
        self.layers = nn.ModuleList(NormalizedInteractionLayer(dim, kind, dropout) for _ in range(layers))

    def forward(self, source, target, source_mask=None, target_mask=None):
        for layer in self.layers:
            source, target = layer(source, target, source_mask, target_mask)
        return source, target


def replace_interactions(module, kind):
    if kind == "original": return
    for name, child in list(module.named_children()):
        if isinstance(child, BIIDStack):
            dim = child.layers[0].source_norm.normalized_shape[0]
            setattr(module, name, InteractionStack(dim, len(child.layers), 0.1, kind))
        else:
            replace_interactions(child, kind)


class InteractionDynamics(CropWorldDynamics):
    def __init__(self, kind):
        super().__init__("biid_climate")
        self.interaction_kind = kind
        if kind != "original":
            self.climate_biid = InteractionStack(self.dim, 2, 0.1, kind)
