#!/usr/bin/env python3
"""Data and model components for the climate-driven BIID crop world model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import nn

from multimodal_baseline import (
    CACHE_ROOT as MULTIMODAL_CACHE_ROOT,
    CROPS,
    LAI_ROOT,
    PROCESSED_ROOT,
    compute_normalization,
    load_cache,
    nearest_mirca_year,
    save_json,
)
from run_history_multimodal_baselines import build_causal_history_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLD_CACHE_ROOT = PROJECT_ROOT / "benchmark/cache/biid_world_model"
WORLD_RESULTS_ROOT = PROJECT_ROOT / "benchmark/results/biid_world_model"
LAI_REL_ROOT = PROJECT_ROOT / "Data/processed/glass_lai_avhrr_005d/crops"
WORLD_CACHE_VERSION = 2
WORLD_VARIANTS = (
    "persistence",
    "gru_transition",
    "transformer_transition",
    "gated_concat",
    "cross_attention",
    "biid_climate",
    "biid_prior",
    "observed_upper",
)


@dataclass(frozen=True)
class WorldNormalizationStats:
    weather_mean: list[float]
    weather_std: list[float]
    lai_mean: float
    lai_std: float
    target_mean: float
    target_std: float
    residual_mean: float
    residual_std: float


def world_cache_dir(crop: str) -> Path:
    return WORLD_CACHE_ROOT / crop


def _world_cache_is_current(crop: str) -> bool:
    root = world_cache_dir(crop)
    metadata_path = root / "metadata.json"
    required = (
        "source_indices.npy",
        "weather_rel.npy",
        "target_lai_rel.npy",
        "relative_valid.npy",
        "relative_weight.npy",
        "source_month.npy",
        "previous_lai.npy",
        "previous_lai_valid.npy",
        "history_features.npy",
        "historical_baseline.npy",
    )
    if not metadata_path.exists() or any(not (root / name).exists() for name in required):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("cache_version", -1)) == WORLD_CACHE_VERSION


def build_world_cache(crop: str, force: bool = False) -> Path:
    """Build only the consecutive-year additions to the frozen multimodal cache."""
    if crop not in CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    root = world_cache_dir(crop)
    if not force and _world_cache_is_current(crop):
        return root

    cache = load_cache(crop)
    normalization = compute_normalization(cache)
    years = np.asarray(cache["year"], dtype=np.int64)
    rows = np.asarray(cache["row"], dtype=np.int64)
    cols = np.asarray(cache["col"], dtype=np.int64)
    source_indices = np.flatnonzero(years >= 1982).astype(np.int64)

    n_samples = int(source_indices.size)
    array_specs = {
        "weather_rel": ((n_samples, 12, 13), np.float32),
        "target_lai_rel": ((n_samples, 12), np.float32),
        "relative_valid": ((n_samples, 12), np.uint8),
        "relative_weight": ((n_samples, 12), np.float32),
        "source_month": ((n_samples, 12), np.uint8),
        "previous_lai": ((n_samples, 12), np.float32),
        "previous_lai_valid": ((n_samples, 12), np.uint8),
    }
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in array_specs.items()
    }
    for year in range(1982, 2017):
        local = np.flatnonzero(years[source_indices] == year)
        if local.size == 0:
            continue
        idx = source_indices[local]
        selected_rows = rows[idx]
        selected_cols = cols[idx]
        mirca_year = nearest_mirca_year(year)
        mirca_root = PROCESSED_ROOT / crop / "mirca" / str(mirca_year)

        weather_map = np.load(
            PROCESSED_ROOT / crop / "era5_rel" / f"era5_rel_{year}.npy",
            mmap_mode="r",
        )
        arrays["weather_rel"][local] = np.asarray(
            weather_map[:, :, selected_rows, selected_cols], dtype=np.float32
        ).transpose(2, 1, 0)

        target_lai_map = np.load(
            LAI_REL_ROOT / crop / "lai_rel" / f"lai_rel_{year}.npy",
            mmap_mode="r",
        )
        arrays["target_lai_rel"][local] = np.asarray(
            target_lai_map[:, selected_rows, selected_cols], dtype=np.float32
        ).T

        relative_valid = np.asarray(
            np.load(mirca_root / "valid_rel.npy", mmap_mode="r")[
                :, selected_rows, selected_cols
            ],
            dtype=np.uint8,
        ).T
        relative_weight = np.asarray(
            np.load(mirca_root / "weight_rel.npy", mmap_mode="r")[
                :, selected_rows, selected_cols
            ],
            dtype=np.float32,
        ).T
        source_month = np.asarray(
            np.load(mirca_root / "src_rel.npy", mmap_mode="r")[
                :, selected_rows, selected_cols
            ],
            dtype=np.uint8,
        ).T
        arrays["relative_valid"][local] = relative_valid
        arrays["relative_weight"][local] = relative_weight
        arrays["source_month"][local] = source_month

        previous_natural_map = np.load(
            LAI_ROOT / f"lai_monthly_{year - 1}.npy", mmap_mode="r"
        )
        previous_natural = np.asarray(
            previous_natural_map[:, selected_rows, selected_cols], dtype=np.float32
        ).T
        safe_source = np.clip(source_month.astype(np.int64), 0, 11)
        previous_relative = np.take_along_axis(
            previous_natural, safe_source, axis=1
        )
        previous_valid = (
            (relative_valid > 0)
            & (source_month != 255)
            & np.isfinite(previous_relative)
        )
        arrays["previous_lai"][local] = np.where(
            previous_valid, previous_relative, np.nan
        )
        arrays["previous_lai_valid"][local] = previous_valid.astype(np.uint8)

    for array in arrays.values():
        array.flush()
    del arrays

    history_features, historical_baseline = build_causal_history_features(
        cache,
        target_mean=normalization.target_mean,
        target_std=normalization.target_std,
    )
    np.save(root / "source_indices.npy", source_indices)
    np.save(
        root / "history_features.npy",
        np.asarray(history_features[source_indices], dtype=np.float32),
    )
    np.save(
        root / "historical_baseline.npy",
        np.asarray(historical_baseline[source_indices], dtype=np.float32),
    )
    save_json(
        {
            "cache_version": WORLD_CACHE_VERSION,
            "crop": crop,
            "source_cache": str(MULTIMODAL_CACHE_ROOT / crop),
            "target_years": [1982, 2016],
            "n_samples": int(source_indices.size),
            "weather_shape": [int(source_indices.size), 12, 13],
            "previous_lai_shape": [int(source_indices.size), 12],
            "history_feature_shape": [int(source_indices.size), 15],
            "time_axis": "MIRCA-packed relative phenology slots",
            "time_axis_caveat": (
                "Active natural months are extracted in calendar order and left aligned. "
                "Cross-year planting-to-harvest rotation is not repaired in this cache."
            ),
            "note": (
                "MIRCA supplies relative-slot mapping, validity, and loss weights. Its area "
                "values are not passed as a predictive modality. Previous-year LAI is "
                "reordered with the target year's mapping to keep slots comparable."
            ),
        },
        root / "metadata.json",
    )
    return root


def load_world_cache(crop: str) -> dict[str, np.ndarray]:
    if not _world_cache_is_current(crop):
        raise FileNotFoundError(f"World cache is missing or stale for {crop}")
    root = world_cache_dir(crop)
    names = (
        "source_indices",
        "weather_rel",
        "target_lai_rel",
        "relative_valid",
        "relative_weight",
        "source_month",
        "previous_lai",
        "previous_lai_valid",
        "history_features",
        "historical_baseline",
    )
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}


def compute_world_stats(
    cache: dict[str, np.ndarray], world: dict[str, np.ndarray]
) -> WorldNormalizationStats:
    source = np.asarray(world["source_indices"], dtype=np.int64)
    train_local = np.flatnonzero(np.asarray(cache["split"][source]) == 0)
    train_source = source[train_local]
    base_stats = compute_normalization(cache)

    weather_sum = np.zeros(13, dtype=np.float64)
    weather_square_sum = np.zeros(13, dtype=np.float64)
    weather_count = np.zeros(13, dtype=np.float64)
    lai_sum = 0.0
    lai_square_sum = 0.0
    lai_count = 0
    for start in range(0, train_local.size, 20_000):
        local = train_local[start : start + 20_000]
        valid = np.asarray(world["relative_valid"][local], dtype=bool)
        weather = np.asarray(world["weather_rel"][local], dtype=np.float64)
        weather_valid = valid[:, :, None] & np.isfinite(weather)
        weather_sum += np.where(weather_valid, weather, 0.0).sum(axis=(0, 1))
        weather_square_sum += np.where(
            weather_valid, np.square(weather), 0.0
        ).sum(axis=(0, 1))
        weather_count += weather_valid.sum(axis=(0, 1))

        lai = np.asarray(world["target_lai_rel"][local], dtype=np.float64)
        lai_valid = valid & np.isfinite(lai)
        selected_lai = lai[lai_valid]
        lai_sum += float(selected_lai.sum())
        lai_square_sum += float(np.square(selected_lai).sum())
        lai_count += int(selected_lai.size)
    weather_mean = np.divide(
        weather_sum, weather_count, out=np.zeros_like(weather_sum), where=weather_count > 0
    )
    weather_variance = np.divide(
        weather_square_sum,
        weather_count,
        out=np.ones_like(weather_square_sum),
        where=weather_count > 0,
    ) - np.square(weather_mean)
    weather_std = np.sqrt(np.maximum(weather_variance, 1.0e-12))
    weather_std = np.where(weather_std < 1.0e-6, 1.0, weather_std)
    lai_mean = lai_sum / max(lai_count, 1)
    lai_variance = lai_square_sum / max(lai_count, 1) - lai_mean * lai_mean
    lai_std = math.sqrt(max(lai_variance, 1.0e-12))
    if lai_std < 1.0e-6:
        lai_std = 1.0

    target = np.asarray(cache["target"][train_source], dtype=np.float64)
    baseline = np.asarray(world["historical_baseline"][train_local], dtype=np.float64)
    residual = target - baseline
    residual_std = float(residual.std())
    if residual_std < 1.0e-6:
        residual_std = 1.0
    return WorldNormalizationStats(
        weather_mean=weather_mean.astype(float).tolist(),
        weather_std=weather_std.astype(float).tolist(),
        lai_mean=float(lai_mean),
        lai_std=float(lai_std),
        target_mean=float(base_stats.target_mean),
        target_std=float(base_stats.target_std),
        residual_mean=float(residual.mean()),
        residual_std=residual_std,
    )


def load_or_compute_world_stats(
    crop: str,
    cache: dict[str, np.ndarray],
    world: dict[str, np.ndarray],
    force: bool = False,
) -> WorldNormalizationStats:
    path = world_cache_dir(crop) / "normalization.json"
    if path.exists() and not force:
        return WorldNormalizationStats(**json.loads(path.read_text(encoding="utf-8")))
    stats = compute_world_stats(cache, world)
    save_json(asdict(stats), path)
    return stats


def local_indices_for_split(
    cache: dict[str, np.ndarray], world: dict[str, np.ndarray], split_value: int
) -> np.ndarray:
    source = np.asarray(world["source_indices"], dtype=np.int64)
    return np.flatnonzero(np.asarray(cache["split"][source]) == split_value)


def make_world_arrays(
    cache: dict[str, np.ndarray],
    world: dict[str, np.ndarray],
    stats: WorldNormalizationStats,
    local_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    source_all = np.asarray(world["source_indices"], dtype=np.int64)
    source = source_all[local_indices]
    weather = np.asarray(world["weather_rel"][local_indices], dtype=np.float32)
    weather_mean = np.asarray(stats.weather_mean, dtype=np.float32)[None, None, :]
    weather_std = np.asarray(stats.weather_std, dtype=np.float32)[None, None, :]
    weather = np.where(
        np.isfinite(weather), (weather - weather_mean) / weather_std, 0.0
    ).astype(np.float32, copy=False)

    previous_lai = np.asarray(world["previous_lai"][local_indices], dtype=np.float32)
    previous_valid = np.asarray(
        world["previous_lai_valid"][local_indices], dtype=np.float32
    )
    previous_lai = (previous_lai - stats.lai_mean) / stats.lai_std
    previous_lai = np.where(
        (previous_valid > 0.0) & np.isfinite(previous_lai), previous_lai, 0.0
    ).astype(np.float32, copy=False)

    target_lai_raw = np.asarray(
        world["target_lai_rel"][local_indices], dtype=np.float32
    )
    relative_valid = np.asarray(
        world["relative_valid"][local_indices], dtype=np.float32
    )
    target_lai_valid = (
        relative_valid * np.isfinite(target_lai_raw).astype(np.float32)
    )
    target_lai = (target_lai_raw - stats.lai_mean) / stats.lai_std
    target_lai = np.where(
        (target_lai_valid > 0.0) & np.isfinite(target_lai), target_lai, 0.0
    ).astype(np.float32, copy=False)
    relative_weight = np.asarray(
        world["relative_weight"][local_indices], dtype=np.float32
    )

    target = np.asarray(cache["target"][source], dtype=np.float32)
    baseline = np.asarray(
        world["historical_baseline"][local_indices], dtype=np.float32
    )
    target_residual = (
        (target - baseline - stats.residual_mean) / stats.residual_std
    ).astype(np.float32)
    target_absolute = ((target - stats.target_mean) / stats.target_std).astype(
        np.float32
    )
    return {
        "weather": weather,
        "previous_lai": previous_lai,
        "previous_lai_valid": previous_valid,
        "target_lai": target_lai,
        "target_lai_valid": target_lai_valid,
        "relative_valid": relative_valid,
        "relative_weight": relative_weight,
        "source_month": np.asarray(
            world["source_month"][local_indices], dtype=np.uint8
        ),
        "history": np.asarray(
            world["history_features"][local_indices], dtype=np.float32
        ),
        "context": np.asarray(cache["context"][source], dtype=np.float32),
        "target_residual": target_residual,
        "target_absolute": target_absolute,
        "target": target,
        "baseline": baseline,
        "source_indices": source,
    }


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


class YieldHeads(nn.Module):
    """Three parameter-independent yield branches trained in one pass."""

    def __init__(
        self, dim: int = 128, heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.history_only_encoder = HistoryEncoder(dim)
        self.history_only_context = nn.Sequential(nn.Linear(5, dim), nn.GELU())
        self.history_head = self._head(dim, dim, dropout)

        self.lai_only_projection = nn.Linear(2, dim)
        self.lai_only_phase_embedding = nn.Parameter(
            torch.randn(1, 12, dim) * 0.02
        )
        self.lai_only_encoder = self._lai_encoder(dim, heads, dropout)
        self.lai_only_context = nn.Sequential(nn.Linear(5, dim), nn.GELU())
        self.lai_head = self._head(dim, dim, dropout)

        self.fusion_history_encoder = HistoryEncoder(dim)
        self.fusion_lai_projection = nn.Linear(2, dim)
        self.fusion_lai_phase_embedding = nn.Parameter(
            torch.randn(1, 12, dim) * 0.02
        )
        self.fusion_lai_encoder = self._lai_encoder(dim, heads, dropout)
        self.fusion = BIIDStack(dim, layers=1, dropout=dropout)
        self.fusion_context = nn.Sequential(nn.Linear(5, dim), nn.GELU())
        self.fusion_head = self._head(dim * 2, dim, dropout)

    @staticmethod
    def _lai_encoder(dim: int, heads: int, dropout: float) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=1)

    @staticmethod
    def _head(input_dim: int, context_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim + context_dim, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, 1),
        )

    def branch_parameter_groups(self) -> tuple[list[nn.Parameter], ...]:
        history_modules = (
            self.history_only_encoder,
            self.history_only_context,
            self.history_head,
        )
        lai_modules = (
            self.lai_only_projection,
            self.lai_only_encoder,
            self.lai_only_context,
            self.lai_head,
        )
        fusion_modules = (
            self.fusion_history_encoder,
            self.fusion_lai_projection,
            self.fusion_lai_encoder,
            self.fusion,
            self.fusion_context,
            self.fusion_head,
        )
        history_parameters = [
            parameter for module in history_modules for parameter in module.parameters()
        ]
        lai_parameters = [self.lai_only_phase_embedding] + [
            parameter for module in lai_modules for parameter in module.parameters()
        ]
        fusion_parameters = [self.fusion_lai_phase_embedding] + [
            parameter for module in fusion_modules for parameter in module.parameters()
        ]
        return (
            history_parameters,
            lai_parameters,
            fusion_parameters,
        )

    @staticmethod
    def _encode_lai(
        lai_trajectory: torch.Tensor,
        relative_valid: torch.Tensor,
        projection: nn.Linear,
        phase_embedding: torch.Tensor,
        encoder: nn.TransformerEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = projection(torch.stack((lai_trajectory, relative_valid), dim=-1))
        padding = relative_valid <= 0.0
        return encoder(tokens + phase_embedding, src_key_padding_mask=padding), padding

    def forward(
        self,
        lai_trajectory: torch.Tensor,
        relative_valid: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_only_tokens = self.history_only_encoder(history)
        history_only_context = self.history_only_context(context)
        history_only_pool = history_only_tokens.mean(dim=1)
        history_prediction = self.history_head(
            torch.cat((history_only_pool, history_only_context), dim=-1)
        ).squeeze(-1)

        lai_only_tokens, _lai_only_padding = self._encode_lai(
            lai_trajectory,
            relative_valid,
            self.lai_only_projection,
            self.lai_only_phase_embedding,
            self.lai_only_encoder,
        )
        lai_weight = relative_valid.unsqueeze(-1)
        lai_only_pool = (lai_only_tokens * lai_weight).sum(dim=1) / lai_weight.sum(
            dim=1
        ).clamp_min(1.0)
        lai_only_context = self.lai_only_context(context)
        lai_prediction = self.lai_head(
            torch.cat((lai_only_pool, lai_only_context), dim=-1)
        ).squeeze(-1)

        fusion_lai_tokens, _fusion_lai_padding = self._encode_lai(
            lai_trajectory,
            relative_valid,
            self.fusion_lai_projection,
            self.fusion_lai_phase_embedding,
            self.fusion_lai_encoder,
        )
        fusion_history_tokens = self.fusion_history_encoder(history)
        history_mask = torch.ones(
            fusion_history_tokens.shape[:2],
            dtype=torch.bool,
            device=fusion_history_tokens.device,
        )
        lai_fused, history_fused = self.fusion(
            fusion_lai_tokens,
            fusion_history_tokens,
            source_mask=relative_valid > 0.0,
            target_mask=history_mask,
        )
        lai_fused_pool = (lai_fused * lai_weight).sum(dim=1) / lai_weight.sum(
            dim=1
        ).clamp_min(1.0)
        fusion_pool = torch.cat(
            (lai_fused_pool, history_fused.mean(dim=1)), dim=-1
        )
        fusion_context = self.fusion_context(context)
        fusion_prediction = self.fusion_head(
            torch.cat((fusion_pool, fusion_context), dim=-1)
        ).squeeze(-1)
        return {
            "history": history_prediction,
            "lai": lai_prediction,
            "fusion": fusion_prediction,
        }


def masked_lai_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    relative_weight: torch.Tensor,
    relative_weight_factor: float = 1.0,
) -> torch.Tensor:
    valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    phase_emphasis = relative_weight * valid_count
    weight = valid_mask * (1.0 + relative_weight_factor * phase_emphasis)
    squared_error = torch.square(prediction - target) * weight
    return squared_error.sum() / weight.sum().clamp_min(1.0)


def lai_metrics(
    target_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    valid_mask: np.ndarray,
    relative_weight: np.ndarray,
    stats: WorldNormalizationStats,
) -> dict[str, Any]:
    target = target_normalized.astype(np.float64) * stats.lai_std + stats.lai_mean
    prediction = (
        prediction_normalized.astype(np.float64) * stats.lai_std + stats.lai_mean
    )

    def summarize(mask: np.ndarray) -> dict[str, float]:
        selected = np.asarray(mask, dtype=bool)
        difference = prediction[selected] - target[selected]
        if difference.size == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "n_values": 0}
        return {
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "mae": float(np.mean(np.abs(difference))),
            "n_values": int(selected.sum()),
        }

    selected = np.asarray(valid_mask, dtype=bool)
    normalized_weight = np.asarray(relative_weight, dtype=np.float64)
    normalized_weight = np.where(selected, normalized_weight, 0.0)
    weight_sum = float(normalized_weight.sum())
    difference = prediction - target
    weighted_rmse = (
        float(np.sqrt(np.sum(normalized_weight * np.square(difference)) / weight_sum))
        if weight_sum > 0.0
        else float("nan")
    )
    by_phase: list[dict[str, float]] = []
    for phase in range(12):
        phase_selected = np.asarray(valid_mask[:, phase], dtype=bool)
        phase_difference = (
            prediction[phase_selected, phase] - target[phase_selected, phase]
        )
        phase_summary = (
            {
                "rmse": float(np.sqrt(np.mean(np.square(phase_difference)))),
                "mae": float(np.mean(np.abs(phase_difference))),
            }
            if phase_difference.size
            else {"rmse": float("nan"), "mae": float("nan")}
        )
        by_phase.append(
            {
                "relative_phase": phase + 1,
                **phase_summary,
                "n_values": int(phase_selected.sum()),
            }
        )
    return {
        "valid_relative_phases": summarize(valid_mask),
        "mirca_weighted_rmse": weighted_rmse,
        "by_relative_phase": by_phase,
    }


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def stats_to_dict(stats: WorldNormalizationStats) -> dict[str, Any]:
    return asdict(stats)
