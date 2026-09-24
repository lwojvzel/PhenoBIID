#!/usr/bin/env python3
"""Yield fusion heads for frozen climate-driven BIID LAI trajectories."""

from __future__ import annotations

import torch
from torch import nn

from biid_world_model import BIIDStack, HistoryEncoder


FUSION_V1_VARIANTS = (
    "concat_residual_gate",
    "biid_residual_nogate",
    "biid_residual_gate",
    "biid_moe_gate",
    "biid_phase_residual_gate",
)
FUSION_V2_VARIANTS = (
    "biid_gru_moe_gate",
    "biid_yield_query_moe_gate",
    "biid_anomaly_query_moe_gate",
    "biid_query_reliability_gate",
    "biid_query_reliability_climate_gate",
    "biid_query_reliability_residual_climate_gate",
)
FUSION_ABLATION_VARIANTS = (
    "biid_query_reliability_no_candidate",
    "biid_query_reliability_no_bce",
    "biid_reliability_no_query",
    "biid_query_reliability_observed_lai",
)
FUSION_COVERAGE_VARIANTS = (
    "biid_query_reliability_coverage_moddrop_gate",
    "biid_query_reliability_climate_coverage_input_gate",
    "biid_query_reliability_climate_coverage_scale_gate",
    "biid_query_reliability_climate_static_coverage_gate",
    "biid_query_reliability_climate_coverage_logit_prior_gate",
    "biid_query_reliability_climate_coverage_blend_gate",
    "biid_query_reliability_climate_coverage_film_gate",
    "biid_query_reliability_climate_coverage_query_gate",
    "biid_query_reliability_climate_coverage_moddrop_gate",
    "biid_query_reliability_climate_coverage_weighted_loss_gate",
)
STATE_ONLY_COVERAGE_VARIANTS = {
    "biid_query_reliability_coverage_moddrop_gate",
}
FUSION_VARIANTS = (
    FUSION_V1_VARIANTS
    + FUSION_V2_VARIANTS
    + FUSION_ABLATION_VARIANTS
    + FUSION_COVERAGE_VARIANTS
)
LATENT_STATE_FUSION_VARIANT = "latent_state_reliability_gate"

YIELD_QUERY_VARIANTS = {
    "biid_yield_query_moe_gate",
    "biid_anomaly_query_moe_gate",
    "biid_query_reliability_gate",
    "biid_query_reliability_climate_gate",
    "biid_query_reliability_residual_climate_gate",
    "biid_query_reliability_no_candidate",
    "biid_query_reliability_no_bce",
    "biid_query_reliability_observed_lai",
    *FUSION_COVERAGE_VARIANTS,
}
MOE_VARIANTS = {
    "biid_moe_gate",
    "biid_gru_moe_gate",
    "biid_reliability_no_query",
    *YIELD_QUERY_VARIANTS,
}
RELIABILITY_VARIANTS = {
    "biid_query_reliability_gate",
    "biid_query_reliability_climate_gate",
    "biid_query_reliability_residual_climate_gate",
    "biid_query_reliability_no_candidate",
    "biid_query_reliability_no_bce",
    "biid_reliability_no_query",
    "biid_query_reliability_observed_lai",
    *FUSION_COVERAGE_VARIANTS,
}
SCALED_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_coverage_scale_gate",
}
STATIC_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_static_coverage_gate",
}
COVERAGE_GATE_INPUT_VARIANTS = {
    "biid_query_reliability_coverage_moddrop_gate",
    "biid_query_reliability_climate_coverage_input_gate",
    "biid_query_reliability_climate_coverage_moddrop_gate",
    "biid_query_reliability_climate_coverage_weighted_loss_gate",
}
LOGIT_PRIOR_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_coverage_logit_prior_gate",
}
BLENDED_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_coverage_blend_gate",
}
FILM_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_coverage_film_gate",
}
QUERY_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_coverage_query_gate",
}
MODDROP_COVERAGE_VARIANTS = {
    "biid_query_reliability_coverage_moddrop_gate",
    "biid_query_reliability_climate_coverage_moddrop_gate",
}
WEIGHTED_LOSS_COVERAGE_VARIANTS = {
    "biid_query_reliability_climate_coverage_weighted_loss_gate",
}


class LAITrajectoryEncoder(nn.Module):
    def __init__(
        self, dim: int, heads: int, dropout: float, input_channels: int = 2
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.projection = nn.Linear(input_channels, dim)
        self.phase_embedding = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(
        self,
        lai: torch.Tensor,
        valid: torch.Tensor,
        previous_lai: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_channels == 3:
            if previous_lai is None:
                raise RuntimeError("LAI anomaly encoding requires previous-year LAI.")
            features = torch.stack((lai, lai - previous_lai, valid), dim=-1)
        else:
            features = torch.stack((lai, valid), dim=-1)
        tokens = self.projection(features)
        return self.encoder(
            tokens + self.phase_embedding,
            src_key_padding_mask=valid <= 0.0,
        )


class ClimateTrajectoryEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, weather_variables: int = 13) -> None:
        super().__init__()
        self.projection = nn.Linear(weather_variables, dim)
        self.phase_embedding = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(
        self, climate: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(
            self.projection(climate) + self.phase_embedding,
            src_key_padding_mask=valid <= 0.0,
        )


class YieldFusionModel(nn.Module):
    """Protect a frozen historical-yield prediction and learn LAI corrections."""

    def __init__(
        self,
        variant: str,
        dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        gate_bias: float = -2.0,
        coverage_tau: float = 0.05,
        coverage_floor: float = 0.10,
        weather_variables: int = 13,
    ) -> None:
        super().__init__()
        if variant not in FUSION_VARIANTS:
            raise ValueError(f"Unsupported yield fusion variant: {variant}")
        self.variant = variant
        if coverage_tau <= 0.0:
            raise ValueError("coverage_tau must be positive")
        if not 0.0 <= coverage_floor < 1.0:
            raise ValueError("coverage_floor must be in [0, 1)")
        self.coverage_tau = coverage_tau
        self.coverage_floor = coverage_floor
        self.use_coverage_gate = variant in FUSION_COVERAGE_VARIANTS
        self.history_encoder = HistoryEncoder(dim)
        lai_channels = 3 if variant == "biid_anomaly_query_moe_gate" else 2
        self.lai_encoder = LAITrajectoryEncoder(
            dim, heads, dropout, input_channels=lai_channels
        )
        self.context_encoder = nn.Sequential(nn.Linear(5, dim), nn.GELU())
        self.biid = BIIDStack(dim, layers=1, dropout=dropout)
        self.use_climate_readout = variant in {
            "biid_query_reliability_climate_gate",
            "biid_query_reliability_residual_climate_gate",
            *FUSION_COVERAGE_VARIANTS,
        } and variant not in STATE_ONLY_COVERAGE_VARIANTS
        self.climate_encoder = (
            ClimateTrajectoryEncoder(dim, heads, dropout, weather_variables)
            if self.use_climate_readout
            else None
        )
        self.climate_biid = (
            BIIDStack(dim, layers=1, dropout=dropout)
            if self.use_climate_readout
            else None
        )
        self.lai_gru = (
            nn.GRU(dim, dim, batch_first=True)
            if variant == "biid_gru_moe_gate"
            else None
        )
        if variant in YIELD_QUERY_VARIANTS:
            self.yield_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.yield_attention = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            self.yield_query_norm = nn.LayerNorm(dim)
        else:
            self.register_parameter("yield_query", None)
            self.yield_attention = None
            self.yield_query_norm = None

        feature_dim = dim * (4 if self.use_climate_readout else 3)
        self.delta_head = self._head(feature_dim, dim, dropout)
        self.candidate_head = self._head(feature_dim, dim, dropout)
        gate_feature_dim = (
            feature_dim + 2
            if variant in COVERAGE_GATE_INPUT_VARIANTS
            else feature_dim
        )
        self.gate_head = self._head(gate_feature_dim, dim, dropout)
        self.coverage_film = (
            nn.Sequential(
                nn.Linear(2, dim),
                nn.GELU(),
                nn.Linear(dim, dim * 2),
            )
            if variant in FILM_COVERAGE_VARIANTS
            else None
        )
        self.coverage_query = (
            nn.Sequential(
                nn.Linear(2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            if variant in QUERY_COVERAGE_VARIANTS
            else None
        )
        self.coverage_logit_strength = (
            nn.Parameter(torch.tensor(-0.43275213))
            if variant in LOGIT_PRIOR_COVERAGE_VARIANTS
            else None
        )
        self.coverage_logit_center = (
            nn.Parameter(torch.tensor(-0.40546511))
            if variant in LOGIT_PRIOR_COVERAGE_VARIANTS
            else None
        )
        self.coverage_blend_logit = (
            nn.Parameter(torch.tensor(-1.38629436))
            if variant in BLENDED_COVERAGE_VARIANTS
            else None
        )
        self.phase_score = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, gate_bias)
        if self.coverage_film is not None:
            nn.init.zeros_(self.coverage_film[-1].weight)
            nn.init.zeros_(self.coverage_film[-1].bias)
        if self.coverage_query is not None:
            nn.init.zeros_(self.coverage_query[-1].weight)
            nn.init.zeros_(self.coverage_query[-1].bias)

    @staticmethod
    def _head(
        input_dim: int, hidden_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _masked_mean(
        tokens: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        weight = valid.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def coverage_features(
        self,
        crop_coverage: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if crop_coverage is None:
            raise RuntimeError("Coverage-aware fusion requires crop coverage.")
        coverage = crop_coverage.reshape(-1).to(reference.dtype).clamp(0.0, 1.0)
        reliability = self.coverage_floor + (
            1.0 - self.coverage_floor
        ) * coverage / (coverage + self.coverage_tau)
        features = torch.stack((coverage, reliability), dim=-1)
        return coverage, reliability, features

    def _phase_pool(
        self,
        lai_tokens: torch.Tensor,
        history_pool: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens = history_pool.unsqueeze(1).expand(-1, lai_tokens.shape[1], -1)
        logits = self.phase_score(torch.cat((lai_tokens, history_tokens), dim=-1))
        logits = logits.squeeze(-1).masked_fill(
            valid <= 0.0, torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return (lai_tokens * weights.unsqueeze(-1)).sum(dim=1), weights

    def forward(
        self,
        lai: torch.Tensor,
        valid: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor,
        history_base: torch.Tensor,
        previous_lai: torch.Tensor | None = None,
        climate: torch.Tensor | None = None,
        crop_coverage: torch.Tensor | None = None,
        extra_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        lai_tokens = self.lai_encoder(lai, valid, previous_lai)
        coverage_reliability = torch.ones_like(history_base)
        coverage_features: torch.Tensor | None = None
        if self.use_coverage_gate:
            _coverage, coverage_reliability, coverage_features = (
                self.coverage_features(crop_coverage, history_base)
            )
        if self.coverage_film is not None:
            if coverage_features is None:
                raise RuntimeError("Coverage FiLM requires coverage features.")
            scale, shift = self.coverage_film(coverage_features).chunk(2, dim=-1)
            lai_tokens = lai_tokens * (
                1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)
            ) + 0.1 * torch.tanh(shift).unsqueeze(1)
        climate_tokens: torch.Tensor | None = None
        if self.use_climate_readout:
            if climate is None or self.climate_encoder is None or self.climate_biid is None:
                raise RuntimeError("Climate readout requires target-year climate tokens.")
            climate_tokens = self.climate_encoder(climate, valid)
            lai_tokens, climate_tokens = self.climate_biid(
                lai_tokens,
                climate_tokens,
                source_mask=valid > 0.0,
                target_mask=valid > 0.0,
            )
        history_tokens = self.history_encoder(history)
        context_pool = self.context_encoder(context)
        if extra_context is not None:
            if extra_context.shape != context_pool.shape:
                raise ValueError("Extra readout context must match the context embedding")
            context_pool = context_pool + extra_context

        if self.variant == "concat_residual_gate":
            lai_pool = self._masked_mean(lai_tokens, valid)
            history_pool = history_tokens.mean(dim=1)
            phase_weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            history_mask = torch.ones(
                history_tokens.shape[:2],
                dtype=torch.bool,
                device=history_tokens.device,
            )
            lai_tokens, history_tokens = self.biid(
                lai_tokens,
                history_tokens,
                source_mask=valid > 0.0,
                target_mask=history_mask,
            )
            history_pool = history_tokens.mean(dim=1)
            if self.variant == "biid_phase_residual_gate":
                lai_pool, phase_weights = self._phase_pool(
                    lai_tokens, history_pool, valid
                )
            elif self.variant == "biid_gru_moe_gate":
                if self.lai_gru is None:
                    raise RuntimeError("GRU trajectory decoder is not initialized.")
                lai_sequence, _hidden = self.lai_gru(lai_tokens)
                lai_pool = self._masked_mean(lai_sequence, valid)
                phase_weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            elif self.variant in YIELD_QUERY_VARIANTS:
                if self.yield_attention is None or self.yield_query_norm is None:
                    raise RuntimeError("Yield-query attention is not initialized.")
                query = self.yield_query.expand(lai_tokens.shape[0], -1, -1)
                if self.coverage_query is not None:
                    if coverage_features is None:
                        raise RuntimeError(
                            "Coverage-conditioned query requires coverage features."
                        )
                    query = query + 0.1 * torch.tanh(
                        self.coverage_query(coverage_features)
                    ).unsqueeze(1)
                attended, attention = self.yield_attention(
                    query,
                    lai_tokens,
                    lai_tokens,
                    key_padding_mask=valid <= 0.0,
                    need_weights=True,
                )
                lai_pool = self.yield_query_norm(query + attended).squeeze(1)
                phase_weights = attention.squeeze(1) * valid.to(attention.dtype)
                phase_weights = phase_weights / phase_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0e-6)
            else:
                lai_pool = self._masked_mean(lai_tokens, valid)
                phase_weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        feature_parts = [lai_pool, history_pool, context_pool]
        if climate_tokens is not None:
            feature_parts.append(self._masked_mean(climate_tokens, valid))
        features = torch.cat(feature_parts, dim=-1)
        delta = self.delta_head(features).squeeze(-1)
        gate_features = features
        if self.variant in COVERAGE_GATE_INPUT_VARIANTS:
            if coverage_features is None:
                raise RuntimeError("Coverage-conditioned gate requires coverage features.")
            gate_features = torch.cat((features, coverage_features), dim=-1)
        learned_gate = torch.sigmoid(self.gate_head(gate_features).squeeze(-1))
        if self.variant in STATIC_COVERAGE_VARIANTS:
            gate = coverage_reliability
        elif self.variant in SCALED_COVERAGE_VARIANTS:
            gate = coverage_reliability * learned_gate
        elif self.variant in LOGIT_PRIOR_COVERAGE_VARIANTS:
            if (
                self.coverage_logit_strength is None
                or self.coverage_logit_center is None
            ):
                raise RuntimeError("Coverage logit prior is not initialized.")
            strength = nn.functional.softplus(self.coverage_logit_strength)
            center = torch.sigmoid(self.coverage_logit_center)
            base_logit = torch.logit(learned_gate.clamp(1.0e-5, 1.0 - 1.0e-5))
            gate = torch.sigmoid(
                base_logit + strength * (coverage_reliability - center)
            )
        elif self.variant in BLENDED_COVERAGE_VARIANTS:
            if self.coverage_blend_logit is None:
                raise RuntimeError("Coverage blend weight is not initialized.")
            blend = 0.5 * torch.sigmoid(self.coverage_blend_logit)
            gate = (1.0 - blend) * learned_gate + blend * coverage_reliability
        else:
            gate = learned_gate

        candidate = history_base + delta
        if self.variant == "biid_residual_nogate":
            prediction = history_base + delta
            gate = torch.ones_like(history_base)
        elif (
            self.variant in MOE_VARIANTS
            and self.variant != "biid_query_reliability_residual_climate_gate"
        ):
            candidate = self.candidate_head(features).squeeze(-1)
            prediction = history_base + gate * (candidate - history_base)
        else:
            prediction = history_base + gate * delta

        return {
            "prediction": prediction,
            "history_base": history_base,
            "gate": gate,
            "learned_gate": learned_gate,
            "coverage_reliability": coverage_reliability,
            "delta": delta,
            "candidate": candidate,
            "phase_weights": phase_weights,
        }


class LatentStateYieldFusionModel(nn.Module):
    """Decode yield from predicted LAI and the frozen climate-driven latent state."""

    variant = LATENT_STATE_FUSION_VARIANT

    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.lai_encoder = LAITrajectoryEncoder(dim, heads, dropout)
        self.state_norm = nn.LayerNorm(dim)
        self.state_lai_biid = BIIDStack(dim, layers=1, dropout=dropout)
        self.history_encoder = HistoryEncoder(dim)
        self.history_state_biid = BIIDStack(dim, layers=1, dropout=dropout)
        self.context_encoder = nn.Sequential(nn.Linear(5, dim), nn.GELU())
        self.yield_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.yield_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.yield_query_norm = nn.LayerNorm(dim)
        feature_dim = dim * 3
        self.delta_head = YieldFusionModel._head(feature_dim, dim, dropout)
        self.gate_head = YieldFusionModel._head(feature_dim, dim, dropout)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, gate_bias)

    def forward(
        self,
        lai: torch.Tensor,
        valid: torch.Tensor,
        latent_state: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor,
        history_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lai_tokens = self.lai_encoder(lai, valid)
        state_tokens = self.state_norm(latent_state)
        state_mask = torch.ones(
            state_tokens.shape[:2], dtype=torch.bool, device=state_tokens.device
        )
        state_tokens, lai_tokens = self.state_lai_biid(
            state_tokens,
            lai_tokens,
            source_mask=state_mask,
            target_mask=valid > 0.0,
        )
        world_tokens = torch.cat((state_tokens, lai_tokens), dim=1)
        world_mask = torch.cat((state_mask, valid > 0.0), dim=1)
        history_tokens = self.history_encoder(history)
        history_mask = torch.ones(
            history_tokens.shape[:2], dtype=torch.bool, device=history_tokens.device
        )
        world_tokens, history_tokens = self.history_state_biid(
            world_tokens,
            history_tokens,
            source_mask=world_mask,
            target_mask=history_mask,
        )
        query = self.yield_query.expand(lai.shape[0], -1, -1)
        attended, attention = self.yield_attention(
            query,
            world_tokens,
            world_tokens,
            key_padding_mask=~world_mask,
            need_weights=True,
        )
        world_pool = self.yield_query_norm(query + attended).squeeze(1)
        history_pool = history_tokens.mean(dim=1)
        context_pool = self.context_encoder(context)
        features = torch.cat((world_pool, history_pool, context_pool), dim=-1)
        delta = self.delta_head(features).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(features).squeeze(-1))
        candidate = history_base + delta
        prediction = history_base + gate * delta
        phase_attention = attention.squeeze(1)[:, state_tokens.shape[1] :]
        phase_attention = phase_attention * valid.to(phase_attention.dtype)
        phase_attention = phase_attention / phase_attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        return {
            "prediction": prediction,
            "history_base": history_base,
            "gate": gate,
            "delta": delta,
            "candidate": candidate,
            "phase_weights": phase_attention,
        }
