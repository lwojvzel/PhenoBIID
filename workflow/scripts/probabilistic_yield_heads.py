"""Conditional residual distributions on immutable phenological state forecasts."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DDIMScheduler

from biid_world_model import HistoryEncoder
from dual_remote_state import YIELD_INPUTS

HEADS = ("deterministic", "gaussian", "mdn", "diffusion")
CONDITIONS = ("no_remote", "previous", "predicted")


class FullSlotCondition(nn.Module):
    def __init__(self, condition, mode="lai", climate=True, dim=128, hidden=256):
        super().__init__()
        if condition not in CONDITIONS or mode not in ("lai", "ndvi", "joint", "dual"):
            raise ValueError((condition, mode))
        self.condition, self.climate = condition, climate
        self.products = () if condition == "no_remote" else ((mode,) if mode in ("lai", "ndvi") else ("lai", "ndvi"))
        self.project = nn.ModuleDict({p: nn.Linear(6, dim) for p in self.products})
        self.identity = nn.ParameterDict({p: nn.Parameter(torch.randn(1, 1, dim) * .02) for p in self.products})
        self.phase = nn.Parameter(torch.randn(1, 12, dim) * .02)
        self.history = HistoryEncoder(dim)
        self.weather = nn.Linear(13, dim) if climate else None
        count = 6 + 12 * (len(self.products) + int(climate))
        self.output = nn.Sequential(nn.LayerNorm(count * dim + 7), nn.Linear(count * dim + 7, hidden),
                                    nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, hidden), nn.GELU())

    def forward(self, b):
        if set(b) != set(YIELD_INPUTS):
            raise ValueError("Only declared past inputs and frozen forecasts may enter the condition encoder")
        active = b["relative_valid"].bool()
        tokens = []
        for p in self.products:
            valid = b[f"previous_{p}_valid"].bool() & active
            previous = torch.where(valid, b[f"previous_{p}"], 0.)
            raw = previous if self.condition == "previous" else b[f"state_{p}"]
            state = torch.where(active, raw, 0.)
            change = torch.where(valid, state - previous, 0.)
            quality = b["previous_ndvi_quality"] if p == "ndvi" else valid.float()
            quality = torch.where(valid, quality, 0.)
            x = torch.stack((state, previous, change, valid.float(), quality, active.float()), -1)
            z = self.project[p](x) + self.identity[p] + self.phase
            tokens.append(torch.where(active[..., None], z, 0.).flatten(1))
        tokens.append(self.history(b["history"]).flatten(1))
        if self.climate:
            weather = torch.where(active[..., None], b["weather"], 0.)
            z = self.weather(weather) + self.phase
            tokens.append(torch.where(active[..., None], z, 0.).flatten(1))
        tokens.extend((b["context"], b["crop_coverage"][:, None], b["history_base"][:, None]))
        return self.output(torch.cat(tokens, -1))


class ConditionalDenoiser(nn.Module):
    def __init__(self, hidden=256, layers=4):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(32) / 31))
        self.input = nn.Linear(65, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.films = nn.ModuleList([nn.Linear(hidden, 2 * hidden) for _ in range(layers)])
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value, time, condition):
        angles = time.float()[:, None] * self.frequencies[None]
        h = self.input(torch.cat((value[:, None], angles.sin(), angles.cos()), -1))
        for norm, film, block in zip(self.norms, self.films, self.blocks):
            scale, shift = film(condition).chunk(2, -1)
            h = h + block(norm(h) * (1 + .1 * scale) + shift)
        return self.output(h).squeeze(-1)


def scheduler(steps=1000):
    return DDIMScheduler(num_train_timesteps=steps, beta_schedule="squaredcos_cap_v2",
                         prediction_type="v_prediction", clip_sample=False, timestep_spacing="trailing")


class ProbabilisticYieldHead(nn.Module):
    def __init__(self, head, condition, mode="lai", climate=True, dim=128, hidden=256, steps=1000):
        super().__init__()
        if head not in HEADS:
            raise ValueError(head)
        self.head, self.steps = head, steps
        self.encoder = FullSlotCondition(condition, mode, climate, dim, hidden)
        if head == "diffusion":
            self.denoiser = ConditionalDenoiser(hidden)
            self.register_buffer("alphas_cumprod", scheduler(steps).alphas_cumprod)
        else:
            self.components = 3 if head == "mdn" else 1
            self.output = nn.Linear(hidden, 1 if head == "deterministic" else 3 * self.components)

    def distribution(self, encoded):
        if self.head not in ("gaussian", "mdn"):
            raise ValueError("No explicit density for this head")
        logits, loc, raw_scale = self.output(encoded).chunk(3, -1)
        scale = F.softplus(raw_scale.float()) + 1e-3
        return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits.float()),
                                                     torch.distributions.Normal(loc.float(), scale))

    def loss(self, b, residual):
        encoded = self.encoder(b)
        if self.head == "deterministic":
            return F.mse_loss(self.output(encoded).squeeze(-1), residual)
        if self.head in ("gaussian", "mdn"):
            return -self.distribution(encoded).log_prob(residual).mean()
        t = torch.randint(self.steps, residual.shape, device=residual.device)
        noise = torch.randn_like(residual)
        a = self.alphas_cumprod[t]
        noisy = a.sqrt() * residual + (1 - a).sqrt() * noise
        # v-parameterization avoids dividing an untrained noise prediction by
        # near-zero signal at the end of the cosine schedule.
        target_v = a.sqrt() * noise - (1 - a).sqrt() * residual
        return F.mse_loss(self.denoiser(noisy, t, encoded), target_v)

    def mean(self, encoded):
        if self.head == "deterministic":
            return self.output(encoded).squeeze(-1)
        return self.distribution(encoded).mean

    @torch.no_grad()
    def sample(self, encoded, count, generator, inference_steps=50):
        if self.head == "deterministic":
            raise ValueError("A point predictor does not define a predictive distribution")
        if self.head in ("gaussian", "mdn"):
            dist = self.distribution(encoded)
            index = torch.multinomial(dist.mixture_distribution.probs, count, replacement=True, generator=generator)
            loc = dist.component_distribution.loc.gather(1, index)
            scale = dist.component_distribution.scale.gather(1, index)
            return loc + scale * torch.randn(loc.shape, device=loc.device, generator=generator)
        process = scheduler(self.steps)
        process.set_timesteps(inference_steps, device=encoded.device)
        expanded = encoded.repeat_interleave(count, dim=0)
        value = torch.randn(len(expanded), device=encoded.device, generator=generator)
        for t in process.timesteps:
            v = self.denoiser(value, t.expand(len(value)), expanded)
            value = process.step(v, int(t), value, eta=0.).prev_sample
        return value.reshape(len(encoded), count)


def fair_crps(samples, target):
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("CRPS requires at least two independent samples per condition")
    n = samples.shape[1]
    ordered = samples.sort(dim=1).values
    weights = 2 * torch.arange(n, device=samples.device, dtype=samples.dtype) - n + 1
    pairs = (ordered * weights).sum(1)
    return (samples - target[:, None]).abs().mean(1) - pairs / (n * (n - 1))
