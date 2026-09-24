"""Unmodified definitions extracted from the registered source snapshot."""
from __future__ import annotations
import math
import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)

class RelevanceModulation(nn.Module):
    """Target-side relevance modulation from the provided BIID formulation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.scale = dim**-0.5

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        affinity = torch.matmul(
            self.query(source), self.key(target).transpose(-1, -2)
        ) * self.scale
        if source_mask is None:
            pooled_affinity = affinity.mean(dim=1)
        else:
            source_weight = source_mask.to(affinity.dtype).unsqueeze(-1)
            pooled_affinity = (affinity * source_weight).sum(dim=1)
            pooled_affinity = pooled_affinity / source_weight.sum(dim=1).clamp_min(1.0)
        if target_mask is not None:
            pooled_affinity = pooled_affinity.masked_fill(
                ~target_mask.to(torch.bool), torch.finfo(affinity.dtype).min
            )
        relevance = torch.softmax(pooled_affinity, dim=-1).unsqueeze(-1)
        modulation = self.output(relevance * self.value(target))
        if target_mask is not None:
            modulation = modulation * target_mask.to(modulation.dtype).unsqueeze(-1)
        return modulation

class BIIDLayer(nn.Module):
    """Synchronous bidirectional target-side modulation."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.source_to_target = RelevanceModulation(dim)
        self.target_to_source = RelevanceModulation(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.target_norm = nn.LayerNorm(dim)
        self.source_mlp = FeedForward(dim, dropout)
        self.target_mlp = FeedForward(dim, dropout)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_bar = target + self.source_to_target(
            source, target, source_mask, target_mask
        )
        source_bar = source + self.target_to_source(
            target, source, target_mask, source_mask
        )
        target_next = target_bar + self.target_mlp(self.target_norm(target_bar))
        source_next = source_bar + self.source_mlp(self.source_norm(source_bar))
        return source_next, target_next

class BIIDStack(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BIIDLayer(dim, dropout) for _ in range(layers)
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            source, target = layer(
                source, target, source_mask=source_mask, target_mask=target_mask
            )
        return source, target

class HistoryEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lag_projection = nn.Linear(2, dim)
        self.summary_projection = nn.Linear(5, dim)
        self.lag_embedding = nn.Parameter(torch.randn(1, 5, dim) * 0.02)
        self.summary_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        lag_values = history[:, :5]
        lag_masks = history[:, 5:10]
        lag_tokens = self.lag_projection(
            torch.stack((lag_values, lag_masks), dim=-1)
        ) + self.lag_embedding
        summary = self.summary_projection(history[:, 10:]).unsqueeze(1)
        summary = summary + self.summary_embedding
        return self.norm(torch.cat((lag_tokens, summary), dim=1))

class PreviousLAIStateEncoder(nn.Module):
    def __init__(
        self, dim: int, state_tokens: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, dim)
        self.phase_embedding = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.state_queries = nn.Parameter(torch.randn(1, state_tokens, dim) * 0.02)
        self.context_projection = nn.Linear(5, dim)
        self.pool = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        previous_lai: torch.Tensor,
        previous_valid: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.input_projection(
            torch.stack((previous_lai, previous_valid), dim=-1)
        ) + self.phase_embedding
        padding_mask = previous_valid <= 0.0
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        queries = self.state_queries.expand(previous_lai.shape[0], -1, -1)
        queries = queries + self.context_projection(context).unsqueeze(1)
        pooled, _weights = self.pool(
            queries,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.norm(queries + pooled)

class WeatherTokenizer(nn.Module):
    def __init__(self, dim: int, weather_variables: int = 13) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, dim)
        self.variable_embedding = nn.Parameter(torch.randn(1, weather_variables, dim) * 0.02)
        self.phase_embedding = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        self.context_projection = nn.Linear(5, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, weather: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(weather.unsqueeze(-1))
        tokens = tokens + self.variable_embedding.unsqueeze(1)
        tokens = tokens + self.phase_embedding.unsqueeze(2)
        tokens = tokens + self.context_projection(context)[:, None, None, :]
        return self.norm(tokens)

class CropWorldDynamics(nn.Module):
    """Shared transition unrolled over 12 MIRCA-packed phenology slots."""

    def __init__(
        self,
        variant: str,
        dim: int = 128,
        state_tokens: int = 8,
        heads: int = 4,
        biid_layers: int = 2,
        dropout: float = 0.1,
        weather_variables: int = 13,
    ) -> None:
        super().__init__()
        if variant not in {
            "gru_transition",
            "transformer_transition",
            "gated_concat",
            "cross_attention",
            "biid_climate",
            "biid_prior",
        }:
            raise ValueError(f"Unsupported trainable dynamics variant: {variant}")
        self.variant = variant
        self.dim = dim
        self.state_tokens = state_tokens
        self.sequence_input_projection: nn.Module | None = None
        self.sequence_transition: nn.Module | None = None
        self.sequence_lai_head: nn.Module | None = None
        if variant in {"gru_transition", "transformer_transition"}:
            self.sequence_input_projection = nn.Sequential(
                nn.Linear(weather_variables + 7, dim), nn.LayerNorm(dim)
            )
            if variant == "gru_transition":
                self.sequence_transition = nn.GRU(
                    dim,
                    dim,
                    num_layers=2,
                    dropout=dropout,
                    batch_first=True,
                )
            else:
                sequence_layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.sequence_transition = nn.TransformerEncoder(
                    sequence_layer, num_layers=2
                )
            self.sequence_lai_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )
        self.state_encoder = PreviousLAIStateEncoder(
            dim, state_tokens, heads, dropout
        )
        self.weather_tokenizer = WeatherTokenizer(dim, weather_variables)
        self.history_encoder = HistoryEncoder(dim)
        self.climate_biid = BIIDStack(dim, biid_layers, dropout)
        self.prior_biid = BIIDStack(dim, biid_layers, dropout)
        self.prior_history_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.concat_update = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.gate = nn.Linear(dim * 2, dim)
        self.state_norm = nn.LayerNorm(dim)
        self.lai_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        self.lai_feedback = nn.Linear(1, dim)

    def _gated_update(
        self, previous: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((previous, candidate), dim=-1)))
        return self.state_norm(gate * candidate + (1.0 - gate) * previous)

    def forward(
        self,
        weather: torch.Tensor,
        previous_lai: torch.Tensor,
        previous_valid: torch.Tensor,
        relative_valid: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor,
        return_trajectory: bool = False,
        observation_feedback: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant in {"gru_transition", "transformer_transition"}:
            if (
                self.sequence_input_projection is None
                or self.sequence_transition is None
                or self.sequence_lai_head is None
            ):
                raise RuntimeError("Sequence transition modules are not initialized.")
            repeated_context = context.unsqueeze(1).expand(-1, 12, -1)
            sequence = torch.cat(
                (
                    weather,
                    previous_lai.unsqueeze(-1),
                    previous_valid.unsqueeze(-1),
                    repeated_context,
                ),
                dim=-1,
            )
            tokens = self.sequence_input_projection(sequence)
            if self.variant == "gru_transition":
                transitioned, _hidden = self.sequence_transition(tokens)
            else:
                transitioned = self.sequence_transition(
                    tokens, src_key_padding_mask=relative_valid <= 0.0
                )
            prediction = previous_lai + self.sequence_lai_head(
                transitioned
            ).squeeze(-1)
            weights = relative_valid / relative_valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            pooled = (transitioned * weights.unsqueeze(-1)).sum(dim=1)
            state = pooled.unsqueeze(1).expand(-1, self.state_tokens, -1)
            if return_trajectory:
                return prediction, transitioned
            return prediction, state

        state = self.state_encoder(previous_lai, previous_valid, context)
        history_tokens = (
            self.history_encoder(history) if self.variant == "biid_prior" else None
        )
        all_weather_tokens = self.weather_tokenizer(weather, context)
        predictions: list[torch.Tensor] = []
        trajectory: list[torch.Tensor] = []

        for phase in range(12):
            weather_tokens = all_weather_tokens[:, phase]
            previous_state = state
            if self.variant == "gated_concat":
                weather_mean = weather_tokens.mean(dim=1, keepdim=True).expand(
                    -1, self.state_tokens, -1
                )
                candidate = state + self.concat_update(
                    torch.cat((state, weather_mean), dim=-1)
                )
                state = self._gated_update(previous_state, candidate)
            elif self.variant == "cross_attention":
                message, _weights = self.cross_attention(
                    state, weather_tokens, weather_tokens, need_weights=False
                )
                state = self._gated_update(previous_state, state + message)
            elif self.variant == "biid_climate":
                state_mod, weather_mod = self.climate_biid(state, weather_tokens)
                message, _weights = self.cross_attention(
                    state_mod, weather_mod, weather_mod, need_weights=False
                )
                state = self._gated_update(previous_state, state_mod + message)
            else:
                if history_tokens is None:
                    raise RuntimeError("Prior-conditioned BIID requires history tokens.")
                fused = torch.cat((state, weather_tokens), dim=1)
                fused, history_tokens = self.prior_biid(fused, history_tokens)
                history_tokens = self.prior_history_norm(history_tokens)
                state_mod = fused[:, : self.state_tokens]
                weather_mod = fused[:, self.state_tokens :]
                message, _weights = self.cross_attention(
                    state_mod, weather_mod, weather_mod, need_weights=False
                )
                state = self._gated_update(previous_state, state_mod + message)

            phase_mask = relative_valid[:, phase].view(-1, 1, 1)
            state = phase_mask * state + (1.0 - phase_mask) * previous_state

            delta = self.lai_head(state.mean(dim=1)).squeeze(-1)
            prediction = previous_lai[:, phase] + delta
            predictions.append(prediction)
            feedback = self.lai_feedback(prediction.unsqueeze(-1)).unsqueeze(1) if observation_feedback else torch.zeros_like(state)
            feedback_state = self.state_norm(state + feedback)
            state = phase_mask * feedback_state + (1.0 - phase_mask) * state
            if return_trajectory:
                trajectory.append(state.mean(dim=1))

        memory = torch.stack(trajectory, dim=1) if return_trajectory else state
        return torch.stack(predictions, dim=1), memory
