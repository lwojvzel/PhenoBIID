#!/usr/bin/env python3
"""Train gated yield fusion heads on frozen climate-BIID LAI predictions."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn

from biid_world_model import (
    WORLD_RESULTS_ROOT,
    CropWorldDynamics,
    YieldHeads,
    build_world_cache,
    load_or_compute_world_stats,
    load_world_cache,
    local_indices_for_split,
    make_world_arrays,
    parameter_count,
)
from biid_yield_fusion import (
    FUSION_VARIANTS,
    LATENT_STATE_FUSION_VARIANT,
    MODDROP_COVERAGE_VARIANTS,
    RELIABILITY_VARIANTS,
    STATIC_COVERAGE_VARIANTS,
    WEIGHTED_LOSS_COVERAGE_VARIANTS,
    LatentStateYieldFusionModel,
    YieldFusionModel,
)
from multimodal_baseline import (
    CROPS,
    compute_normalization,
    evaluation_context,
    load_cache,
    load_coordinates,
    regression_metrics,
    save_json,
    set_seed,
    write_csv,
)
from audit_crop_area_fraction import grid_area_hectares, sample_fraction
from run_biid_world_model import (
    TensorBatchLoader,
    dynamics_amp_dtype,
    make_loader,
    predict_dynamics,
    select_local_indices,
    unpack_batch,
)
from run_history_multimodal_baselines import (
    build_causal_history_features,
    build_model_features,
)
from run_yield_only_neural_baselines import HistorySequenceRegressor


FUSION_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "benchmark/results/biid_yield_fusion"


def load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def predict_history_base(
    model: YieldHeads,
    loader: TensorBatchLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for raw_batch in loader:
        batch = unpack_batch(raw_batch)
        history = batch["history"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            tokens = model.history_only_encoder(history)
            context_token = model.history_only_context(context)
            prediction = model.history_head(
                torch.cat((tokens.mean(dim=1), context_token), dim=-1)
            ).squeeze(-1)
        predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


@torch.no_grad()
def predict_dynamics_with_state(
    model: CropWorldDynamics,
    loader: TensorBatchLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both the decoded LAI trajectory and its recurrent world state."""
    model.eval()
    lai_predictions: list[np.ndarray] = []
    latent_states: list[np.ndarray] = []
    for raw_batch in loader:
        batch = unpack_batch(raw_batch)
        with torch.autocast(
            device_type=device.type,
            dtype=dynamics_amp_dtype(model.variant),
            enabled=device.type == "cuda",
        ):
            prediction, state = model(
                batch["weather"].to(device, non_blocking=True),
                batch["previous_lai"].to(device, non_blocking=True),
                batch["previous_lai_valid"].to(device, non_blocking=True),
                batch["relative_valid"].to(device, non_blocking=True),
                batch["history"].to(device, non_blocking=True),
                batch["context"].to(device, non_blocking=True),
            )
        lai_predictions.append(prediction.float().cpu().numpy())
        latent_states.append(state.float().cpu().numpy())
    return (
        np.concatenate(lai_predictions).astype(np.float32, copy=False),
        np.concatenate(latent_states).astype(np.float32, copy=False),
    )


def predict_hgb_history_base(
    crop: str,
    hgb_seed: int,
    cache: dict[str, np.ndarray],
    arrays: dict[str, dict[str, np.ndarray]],
    world_stats: Any,
) -> tuple[dict[str, np.ndarray], Path]:
    """Express a frozen HGB yield forecast in the world-model residual space."""
    run_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmark/results/multimodal_main"
        / crop
        / "hgb_history_yield"
        / f"seed_{hgb_seed}"
    )
    model_path = run_dir / "model_best.joblib"
    normalization_path = run_dir / "normalization.json"
    if not model_path.exists() or not normalization_path.exists():
        raise FileNotFoundError(
            f"Missing frozen HGB history expert in {run_dir}"
        )

    model = joblib.load(model_path)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    baseline_stats = compute_normalization(cache)
    history_features, historical_baseline = build_causal_history_features(
        cache,
        target_mean=baseline_stats.target_mean,
        target_std=baseline_stats.target_std,
    )

    predictions: dict[str, np.ndarray] = {}
    for name, split_arrays in arrays.items():
        source_indices = np.asarray(split_arrays["source_indices"], dtype=np.int64)
        features = build_model_features(
            cache,
            baseline_stats,
            history_features,
            source_indices,
            "history_yield",
        )
        physical_prediction = (
            historical_baseline[source_indices].astype(np.float64)
            + model.predict(features) * float(normalization["residual_std"])
            + float(normalization["residual_mean"])
        )
        world_baseline = np.asarray(split_arrays["baseline"], dtype=np.float64)
        predictions[name] = (
            (
                physical_prediction
                - world_baseline
                - float(world_stats.residual_mean)
            )
            / float(world_stats.residual_std)
        ).astype(np.float32)
    return predictions, model_path


@torch.no_grad()
def predict_mlp_history_base(
    crop: str,
    seed: int,
    cache: dict[str, np.ndarray],
    arrays: dict[str, dict[str, np.ndarray]],
    world_stats: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], Path]:
    """Express the frozen causal MLP forecast in world-model residual space."""
    run_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmark/results/yield_only_baselines"
        / crop
        / "mlp"
        / f"seed_{seed}"
    )
    model_path = run_dir / "model_best.pt"
    config_path = run_dir / "config.json"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing frozen MLP history expert in {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    residual_mean = float(config["residual_mean"])
    residual_std = float(config["residual_std"])
    baseline_stats = compute_normalization(cache)
    history_features, historical_baseline = build_causal_history_features(
        cache,
        target_mean=baseline_stats.target_mean,
        target_std=baseline_stats.target_std,
    )
    model = HistorySequenceRegressor(
        architecture="mlp",
        static_size=10,
    ).to(device)
    model.load_state_dict(load_state(model_path, device))
    model.eval()

    predictions: dict[str, np.ndarray] = {}
    for name, split_arrays in arrays.items():
        source_indices = np.asarray(split_arrays["source_indices"], dtype=np.int64)
        history_values = history_features[source_indices, :5][:, ::-1]
        history_masks = history_features[source_indices, 5:10][:, ::-1]
        history_sequence = np.stack(
            (history_values, history_masks), axis=2
        ).astype(np.float32, copy=False)
        static = np.concatenate(
            (
                history_features[source_indices, 10:].astype(np.float32, copy=False),
                np.asarray(cache["context"][source_indices], dtype=np.float32),
            ),
            axis=1,
        )
        residual_prediction: list[np.ndarray] = []
        for start in range(0, source_indices.size, batch_size):
            stop = min(source_indices.size, start + batch_size)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    torch.from_numpy(history_sequence[start:stop]).to(
                        device, non_blocking=True
                    ),
                    torch.from_numpy(static[start:stop]).to(
                        device, non_blocking=True
                    ),
                )
            residual_prediction.append(output.float().cpu().numpy())
        normalized_residual = np.concatenate(residual_prediction)
        physical_prediction = np.maximum(
            historical_baseline[source_indices].astype(np.float64)
            + normalized_residual * residual_std
            + residual_mean,
            0.0,
        )
        world_baseline = np.asarray(split_arrays["baseline"], dtype=np.float64)
        predictions[name] = (
            (
                physical_prediction
                - world_baseline
                - float(world_stats.residual_mean)
            )
            / float(world_stats.residual_std)
        ).astype(np.float32)
    del model
    return predictions, model_path


def make_fusion_loader(
    input_lai: np.ndarray,
    lai_valid: np.ndarray,
    arrays: dict[str, np.ndarray],
    history_base: np.ndarray,
    batch_size: int,
    shuffle: bool,
    crop_coverage: np.ndarray | None = None,
) -> TensorBatchLoader:
    if crop_coverage is None:
        crop_coverage = np.zeros_like(history_base, dtype=np.float32)
    tensors = (
        torch.from_numpy(input_lai),
        torch.from_numpy(arrays["previous_lai"]),
        torch.from_numpy(lai_valid),
        torch.from_numpy(arrays["weather"]),
        torch.from_numpy(arrays["history"]),
        torch.from_numpy(arrays["context"]),
        torch.from_numpy(history_base),
        torch.from_numpy(crop_coverage),
        torch.from_numpy(arrays["target_residual"]),
        torch.from_numpy(arrays["target"]),
        torch.from_numpy(arrays["baseline"]),
    )
    return TensorBatchLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def make_latent_fusion_loader(
    input_lai: np.ndarray,
    lai_valid: np.ndarray,
    latent_state: np.ndarray,
    arrays: dict[str, np.ndarray],
    history_base: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> TensorBatchLoader:
    tensors = (
        torch.from_numpy(input_lai),
        torch.from_numpy(lai_valid),
        torch.from_numpy(latent_state),
        torch.from_numpy(arrays["history"]),
        torch.from_numpy(arrays["context"]),
        torch.from_numpy(history_base),
        torch.from_numpy(arrays["target_residual"]),
        torch.from_numpy(arrays["target"]),
        torch.from_numpy(arrays["baseline"]),
    )
    return TensorBatchLoader(tensors, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_latent_fusion(
    model: LatentStateYieldFusionModel,
    loader: TensorBatchLoader,
    residual_mean: float,
    residual_std: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "target": [],
        "history": [],
        "fusion": [],
        "gate": [],
        "delta": [],
        "phase_weights": [],
    }
    for batch in loader:
        (
            lai,
            valid,
            latent_state,
            history,
            context,
            history_base,
            _target_residual,
            target,
            baseline,
        ) = batch
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                lai.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
                latent_state.to(device, non_blocking=True),
                history.to(device, non_blocking=True),
                context.to(device, non_blocking=True),
                history_base.to(device, non_blocking=True),
            )
        history_physical = baseline.numpy() + (
            history_base.numpy() * residual_std + residual_mean
        )
        fusion_physical = baseline.numpy() + (
            output["prediction"].float().cpu().numpy() * residual_std
            + residual_mean
        )
        collected["target"].append(target.numpy())
        collected["history"].append(history_physical)
        collected["fusion"].append(fusion_physical)
        collected["gate"].append(output["gate"].float().cpu().numpy())
        collected["delta"].append(output["delta"].float().cpu().numpy())
        collected["phase_weights"].append(
            output["phase_weights"].float().cpu().numpy()
        )
    return {
        name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for name, values in collected.items()
    }


@torch.no_grad()
def predict_fusion(
    model: YieldFusionModel,
    loader: TensorBatchLoader,
    residual_mean: float,
    residual_std: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "target": [],
        "history": [],
        "fusion": [],
        "gate": [],
        "delta": [],
        "phase_weights": [],
        "learned_gate": [],
        "coverage_reliability": [],
        "crop_coverage": [],
    }
    for batch in loader:
        (
            lai,
            previous_lai,
            valid,
            climate,
            history,
            context,
            history_base,
            crop_coverage,
            _residual,
            target,
            baseline,
        ) = batch
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                lai.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
                history.to(device, non_blocking=True),
                context.to(device, non_blocking=True),
                history_base.to(device, non_blocking=True),
                previous_lai.to(device, non_blocking=True),
                climate.to(device, non_blocking=True),
                crop_coverage.to(device, non_blocking=True),
            )
        history_physical = baseline.numpy() + (
            history_base.numpy() * residual_std + residual_mean
        )
        fusion_physical = baseline.numpy() + (
            output["prediction"].float().cpu().numpy() * residual_std + residual_mean
        )
        collected["target"].append(target.numpy())
        collected["history"].append(history_physical)
        collected["fusion"].append(fusion_physical)
        collected["gate"].append(output["gate"].float().cpu().numpy())
        collected["learned_gate"].append(
            output["learned_gate"].float().cpu().numpy()
        )
        collected["coverage_reliability"].append(
            output["coverage_reliability"].float().cpu().numpy()
        )
        collected["crop_coverage"].append(crop_coverage.numpy())
        collected["delta"].append(output["delta"].float().cpu().numpy())
        collected["phase_weights"].append(
            output["phase_weights"].float().cpu().numpy()
        )
    return {
        name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for name, values in collected.items()
    }


def validation_rmse(
    model: YieldFusionModel,
    loader: TensorBatchLoader,
    residual_mean: float,
    residual_std: float,
    device: torch.device,
) -> float:
    predictions = predict_fusion(
        model, loader, residual_mean, residual_std, device
    )
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    predictions["fusion"].astype(np.float64)
                    - predictions["target"].astype(np.float64)
                )
            )
        )
    )


def train_fusion(
    model: YieldFusionModel,
    train_loader: TensorBatchLoader,
    validation_loader: TensorBatchLoader,
    residual_mean: float,
    residual_std: float,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gate_penalty: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_rmse = float("inf")
    stale_epochs = 0
    rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_gate = 0.0
        total_count = 0
        for batch in train_loader:
            (
                lai,
                previous_lai,
                valid,
                climate,
                history,
                context,
                history_base,
                crop_coverage,
                residual,
                _target,
                _baseline,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            lai_device = lai.to(device, non_blocking=True)
            previous_lai_device = previous_lai.to(device, non_blocking=True)
            history_base_device = history_base.to(device, non_blocking=True)
            coverage_device = crop_coverage.to(device, non_blocking=True)
            if model.variant in MODDROP_COVERAGE_VARIANTS:
                with torch.no_grad():
                    _coverage, reliability, _coverage_features = (
                        model.coverage_features(
                            coverage_device, history_base_device
                        )
                    )
                    drop_probability = 0.5 * (1.0 - reliability)
                    drop_state = torch.rand_like(drop_probability) < drop_probability
                    lai_device = torch.where(
                        drop_state.unsqueeze(-1), previous_lai_device, lai_device
                    )
                    coverage_device = torch.where(
                        drop_state, torch.zeros_like(coverage_device), coverage_device
                    )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    lai_device,
                    valid.to(device, non_blocking=True),
                    history.to(device, non_blocking=True),
                    context.to(device, non_blocking=True),
                    history_base_device,
                    previous_lai_device,
                    climate.to(device, non_blocking=True),
                    coverage_device,
                )
                residual_device = residual.to(device, non_blocking=True)
                loss = criterion(output["prediction"], residual_device)
                if model.variant in RELIABILITY_VARIANTS:
                    candidate_error = torch.square(
                        output["candidate"] - residual_device
                    )
                    oracle_use_fusion = (
                        torch.abs(output["candidate"].detach() - residual_device)
                        < torch.abs(output["history_base"] - residual_device)
                    ).to(output["gate"].dtype)
                    with torch.autocast(
                        device_type=device.type, enabled=False
                    ):
                        reliability_error = nn.functional.binary_cross_entropy(
                            output["gate"].float().clamp(1.0e-5, 1.0 - 1.0e-5),
                            oracle_use_fusion.float(),
                            reduction="none",
                        )
                    if model.variant in WEIGHTED_LOSS_COVERAGE_VARIANTS:
                        quality_weight = output["coverage_reliability"].detach()
                        quality_weight = quality_weight / quality_weight.mean().clamp_min(
                            1.0e-6
                        )
                        candidate_loss = (candidate_error * quality_weight).mean()
                        reliability_loss = (
                            reliability_error * quality_weight.float()
                        ).mean()
                    else:
                        candidate_loss = candidate_error.mean()
                        reliability_loss = reliability_error.mean()
                    candidate_weight = (
                        0.0
                        if model.variant == "biid_query_reliability_no_candidate"
                        else 0.25
                    )
                    reliability_weight = (
                        0.0
                        if model.variant
                        in {
                            "biid_query_reliability_no_bce",
                            *STATIC_COVERAGE_VARIANTS,
                        }
                        else 0.1
                    )
                    loss = (
                        loss
                        + candidate_weight * candidate_loss
                        + reliability_weight * reliability_loss
                    )
                if gate_penalty > 0.0 and model.variant != "biid_residual_nogate":
                    loss = loss + gate_penalty * output["gate"].mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(lai.shape[0])
            total_loss += float(loss.detach().cpu()) * count
            total_gate += float(output["gate"].detach().mean().cpu()) * count
            total_count += count

        val_rmse = validation_rmse(
            model, validation_loader, residual_mean, residual_std, device
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_gate_mean": total_gate / max(total_count, 1),
            "validation_rmse": val_rmse,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"[FUSION] variant={model.variant} epoch={epoch} "
            f"loss={row['train_loss']:.6f} gate={row['train_gate_mean']:.4f} "
            f"val_rmse={val_rmse:.6f} seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        if val_rmse < best_rmse - 1.0e-6:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("Fusion training did not produce a checkpoint.")
    return best_state, rows, best_epoch, best_rmse


def train_latent_fusion(
    model: LatentStateYieldFusionModel,
    train_loader: TensorBatchLoader,
    validation_loader: TensorBatchLoader,
    residual_mean: float,
    residual_std: float,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_rmse = float("inf")
    stale_epochs = 0
    rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_gate = 0.0
        total_count = 0
        for batch in train_loader:
            (
                lai,
                valid,
                latent_state,
                history,
                context,
                history_base,
                target_residual,
                _target,
                _baseline,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    lai.to(device, non_blocking=True),
                    valid.to(device, non_blocking=True),
                    latent_state.to(device, non_blocking=True),
                    history.to(device, non_blocking=True),
                    context.to(device, non_blocking=True),
                    history_base.to(device, non_blocking=True),
                )
                target_device = target_residual.to(device, non_blocking=True)
                prediction_loss = criterion(output["prediction"], target_device)
                candidate_loss = criterion(output["candidate"], target_device)
                oracle_use_state = (
                    torch.abs(output["candidate"].detach() - target_device)
                    < torch.abs(output["history_base"] - target_device)
                ).to(output["gate"].dtype)
                with torch.autocast(device_type=device.type, enabled=False):
                    reliability_loss = nn.functional.binary_cross_entropy(
                        output["gate"].float().clamp(1.0e-5, 1.0 - 1.0e-5),
                        oracle_use_state.float(),
                    )
                loss = (
                    prediction_loss
                    + 0.25 * candidate_loss
                    + 0.1 * reliability_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(lai.shape[0])
            total_loss += float(loss.detach().cpu()) * count
            total_gate += float(output["gate"].detach().mean().cpu()) * count
            total_count += count

        validation = predict_latent_fusion(
            model,
            validation_loader,
            residual_mean,
            residual_std,
            device,
        )
        validation_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        validation["target"].astype(np.float64)
                        - validation["fusion"].astype(np.float64)
                    )
                )
            )
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_gate_mean": total_gate / max(total_count, 1),
            "validation_rmse": validation_rmse,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"[LATENT FUSION] epoch={epoch} loss={row['train_loss']:.6f} "
            f"gate={row['train_gate_mean']:.4f} "
            f"val_rmse={validation_rmse:.6f} "
            f"seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        if validation_rmse < best_rmse - 1.0e-6:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("Latent-state fusion training produced no checkpoint.")
    return best_state, rows, best_epoch, best_rmse


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    build_world_cache(args.crop)
    cache = load_cache(args.crop)
    world = load_world_cache(args.crop)
    stats = load_or_compute_world_stats(args.crop, cache, world)
    local_indices = {
        "train": select_local_indices(
            local_indices_for_split(cache, world, 0),
            args.max_train_samples,
            args.seed + 101,
        ),
        "validation": select_local_indices(
            local_indices_for_split(cache, world, 1),
            args.max_eval_samples,
            args.seed + 102,
        ),
        "test": select_local_indices(
            local_indices_for_split(cache, world, 2),
            args.max_eval_samples,
            args.seed + 103,
        ),
    }
    arrays = {
        name: make_world_arrays(cache, world, stats, indices)
        for name, indices in local_indices.items()
    }
    latitude, _longitude = load_coordinates()
    cell_area = grid_area_hectares(latitude)
    crop_coverage = {}
    for name, values in arrays.items():
        source_indices = np.asarray(values["source_indices"], dtype=np.int64)
        crop_coverage[name] = sample_fraction(
            args.crop,
            np.asarray(cache["year"][source_indices], dtype=np.int64),
            np.asarray(cache["row"][source_indices], dtype=np.int64),
            np.asarray(cache["col"][source_indices], dtype=np.int64),
            cell_area,
        ).astype(np.float32)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaders = {
        name: make_loader(
            value,
            args.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        for name, value in arrays.items()
    }

    dynamics_reference_dir = (
        WORLD_RESULTS_ROOT
        / args.crop
        / args.dynamics_variant
        / f"seed_{args.seed}"
    )
    dynamics_path = dynamics_reference_dir / "dynamics_best.pt"
    if not dynamics_path.exists():
        raise FileNotFoundError(
            f"Missing frozen dynamics checkpoint in {dynamics_reference_dir}"
        )

    dynamics = CropWorldDynamics(
        variant=args.dynamics_variant,
        dim=args.dim,
        state_tokens=args.state_tokens,
        heads=args.heads,
        biid_layers=args.dynamics_biid_layers,
        dropout=args.dropout,
    ).to(device)
    missing, unexpected = dynamics.load_state_dict(
        load_state(dynamics_path, device), strict=False
    )
    allowed_missing = {
        "prior_history_norm.weight",
        "prior_history_norm.bias",
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible dynamics checkpoint: missing={missing}, unexpected={unexpected}"
        )
    latent_states: dict[str, np.ndarray] | None = None
    if args.fusion_variant == LATENT_STATE_FUSION_VARIANT:
        predicted_lai = {}
        latent_states = {}
        for name in ("train", "validation", "test"):
            prediction, state = predict_dynamics_with_state(
                dynamics, loaders[name], device
            )
            predicted_lai[name] = prediction
            latent_states[name] = state
    else:
        predicted_lai = {
            name: predict_dynamics(
                dynamics, loaders[name], args.dynamics_variant, device
            )
            for name in ("train", "validation", "test")
        }
    del dynamics
    if device.type == "cuda":
        torch.cuda.empty_cache()

    use_observed_lai = (
        args.fusion_variant == "biid_query_reliability_observed_lai"
    )
    fusion_input_lai = {
        name: (
            arrays[name]["target_lai"]
            if use_observed_lai
            else predicted_lai[name]
        )
        for name in ("train", "validation", "test")
    }
    fusion_lai_valid = {
        name: (
            arrays[name]["target_lai_valid"]
            if use_observed_lai
            else arrays[name]["relative_valid"]
        )
        for name in ("train", "validation", "test")
    }

    if args.history_base_source == "hgb":
        history_base, history_path = predict_hgb_history_base(
            args.crop,
            args.hgb_history_seed,
            cache,
            arrays,
            stats,
        )
    elif args.history_base_source == "mlp":
        history_base, history_path = predict_mlp_history_base(
            args.crop,
            args.seed,
            cache,
            arrays,
            stats,
            device,
            args.batch_size,
        )
    elif args.history_base_source == "hgb_mlp_ensemble":
        hgb_base, hgb_path = predict_hgb_history_base(
            args.crop,
            args.hgb_history_seed,
            cache,
            arrays,
            stats,
        )
        mlp_base, mlp_path = predict_mlp_history_base(
            args.crop,
            args.seed,
            cache,
            arrays,
            stats,
            device,
            args.batch_size,
        )
        history_base = {
            name: 0.5 * (hgb_base[name] + mlp_base[name])
            for name in ("train", "validation", "test")
        }
        history_path = f"{hgb_path}|{mlp_path}"
    else:
        history_reference_dir = (
            WORLD_RESULTS_ROOT
            / args.crop
            / args.history_reference_variant
            / f"seed_{args.seed}"
        )
        history_path = history_reference_dir / "yield_history_only_best.pt"
        if not history_path.exists():
            raise FileNotFoundError(
                f"Missing frozen history checkpoint in {history_reference_dir}"
            )
        history_model = YieldHeads(
            dim=args.dim, heads=args.heads, dropout=args.dropout
        ).to(device)
        history_model.load_state_dict(load_state(history_path, device))
        history_base = {
            name: predict_history_base(history_model, loaders[name], device)
            for name in ("train", "validation", "test")
        }
        del history_model
    del loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.fusion_variant == LATENT_STATE_FUSION_VARIANT:
        if latent_states is None:
            raise RuntimeError("Latent-state fusion requires dynamics states.")
        fusion_loaders = {
            name: make_latent_fusion_loader(
                fusion_input_lai[name],
                fusion_lai_valid[name],
                latent_states[name],
                arrays[name],
                history_base[name],
                args.batch_size,
                shuffle=name == "train",
            )
            for name in ("train", "validation", "test")
        }
        model = LatentStateYieldFusionModel(
            dim=args.dim,
            heads=args.heads,
            dropout=args.dropout,
            gate_bias=args.gate_bias,
        ).to(device)
        best_state, history_rows, best_epoch, best_validation_rmse = (
            train_latent_fusion(
                model,
                fusion_loaders["train"],
                fusion_loaders["validation"],
                stats.residual_mean,
                stats.residual_std,
                device,
                args.epochs,
                args.patience,
                args.learning_rate,
                args.weight_decay,
            )
        )
        model.load_state_dict(best_state)
        validation_predictions = predict_latent_fusion(
            model,
            fusion_loaders["validation"],
            stats.residual_mean,
            stats.residual_std,
            device,
        )
        test_predictions = predict_latent_fusion(
            model,
            fusion_loaders["test"],
            stats.residual_mean,
            stats.residual_std,
            device,
        )
    else:
        fusion_loaders = {
            name: make_fusion_loader(
                fusion_input_lai[name],
                fusion_lai_valid[name],
                arrays[name],
                history_base[name],
                args.batch_size,
                shuffle=name == "train",
                crop_coverage=crop_coverage[name],
            )
            for name in ("train", "validation", "test")
        }
        model = YieldFusionModel(
            variant=args.fusion_variant,
            dim=args.dim,
            heads=args.heads,
            dropout=args.dropout,
            gate_bias=args.gate_bias,
            coverage_tau=args.coverage_tau,
            coverage_floor=args.coverage_floor,
        ).to(device)
        best_state, history_rows, best_epoch, best_validation_rmse = train_fusion(
            model,
            fusion_loaders["train"],
            fusion_loaders["validation"],
            stats.residual_mean,
            stats.residual_std,
            device,
            args.epochs,
            args.patience,
            args.learning_rate,
            args.weight_decay,
            args.gate_penalty,
        )
        model.load_state_dict(best_state)
        validation_predictions = predict_fusion(
            model,
            fusion_loaders["validation"],
            stats.residual_mean,
            stats.residual_std,
            device,
        )
        test_predictions = predict_fusion(
            model,
            fusion_loaders["test"],
            stats.residual_mean,
            stats.residual_std,
            device,
        )

    if args.fusion_variant == LATENT_STATE_FUSION_VARIANT:
        for name, predictions in (
            ("validation", validation_predictions),
            ("test", test_predictions),
        ):
            predictions["learned_gate"] = predictions["gate"]
            predictions["crop_coverage"] = crop_coverage[name]
            predictions["coverage_reliability"] = np.ones_like(
                predictions["gate"], dtype=np.float32
            )

    test_source = arrays["test"]["source_indices"]
    area_weight, latitude_weight = evaluation_context(cache, test_source)
    history_metrics = regression_metrics(
        test_predictions["target"],
        test_predictions["history"],
        area_weight,
        latitude_weight,
    )
    fusion_metrics = regression_metrics(
        test_predictions["target"],
        test_predictions["fusion"],
        area_weight,
        latitude_weight,
    )
    gain_percent = 100.0 * (
        history_metrics["rmse"] - fusion_metrics["rmse"]
    ) / history_metrics["rmse"]
    metrics = {
        "crop": args.crop,
        "fusion_variant": args.fusion_variant,
        "dynamics_variant": args.dynamics_variant,
        "history_reference_variant": args.history_reference_variant,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_validation_rmse,
        "history_metrics": history_metrics,
        "fusion_metrics": fusion_metrics,
        "fusion_gain_percent": gain_percent,
        "gate_mean": float(test_predictions["gate"].mean()),
        "gate_std": float(test_predictions["gate"].std()),
        "gate_q10": float(np.quantile(test_predictions["gate"], 0.1)),
        "gate_q50": float(np.quantile(test_predictions["gate"], 0.5)),
        "gate_q90": float(np.quantile(test_predictions["gate"], 0.9)),
        "crop_coverage_mean": float(test_predictions["crop_coverage"].mean()),
        "coverage_reliability_mean": float(
            test_predictions["coverage_reliability"].mean()
        ),
    }

    experiment_name = args.fusion_variant
    if (
        args.dynamics_variant != "biid_climate"
        or args.history_reference_variant != "biid_climate"
    ):
        experiment_name = (
            f"{args.fusion_variant}__dynamics_{args.dynamics_variant}"
        )
    if args.history_base_source != "neural":
        experiment_name = (
            f"{experiment_name}__history_{args.history_base_source}"
        )
    run_dir = (
        FUSION_RESULTS_ROOT
        / args.crop
        / experiment_name
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "fusion_best.pt")
    write_csv(history_rows, run_dir / "training_history.csv")
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "run_tag": args.run_tag,
            "crop": args.crop,
            "fusion_variant": args.fusion_variant,
            "experiment_name": experiment_name,
            "dynamics_variant": args.dynamics_variant,
            "history_reference_variant": args.history_reference_variant,
            "seed": args.seed,
            "protocol": "frozen_dynamics_and_frozen_history_base",
            "history_base_source": args.history_base_source,
            "hgb_history_seed": (
                args.hgb_history_seed
                if args.history_base_source == "hgb"
                else None
            ),
            "lai_source": "observed" if use_observed_lai else "predicted",
            "dynamics_checkpoint": str(dynamics_path),
            "history_checkpoint": str(history_path),
            "time_axis": "MIRCA-packed relative phenology slots",
            "train_years": [1982, 2011],
            "validation_year": 2012,
            "test_years": [2013, 2016],
            "dim": args.dim,
            "state_tokens": args.state_tokens,
            "heads": args.heads,
            "dynamics_biid_layers": args.dynamics_biid_layers,
            "fusion_biid_layers": (
                2
                if args.fusion_variant == LATENT_STATE_FUSION_VARIANT
                else 1
            ),
            "parameters": parameter_count(model),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gate_bias": args.gate_bias,
            "gate_penalty": args.gate_penalty,
            "coverage_tau": args.coverage_tau,
            "coverage_floor": args.coverage_floor,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "actual_device": str(device),
            "peak_allocated_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
            "split_sizes": {
                name: int(value["target"].shape[0])
                for name, value in arrays.items()
            },
        },
        run_dir / "config.json",
    )
    np.savez_compressed(
        run_dir / "validation_predictions.npz",
        target=validation_predictions["target"],
        history_prediction=validation_predictions["history"],
        fusion_prediction=validation_predictions["fusion"],
        gate=validation_predictions["gate"],
        learned_gate=validation_predictions["learned_gate"],
        crop_coverage=validation_predictions["crop_coverage"],
        coverage_reliability=validation_predictions["coverage_reliability"],
        delta=validation_predictions["delta"],
        phase_weights=validation_predictions["phase_weights"],
        predicted_lai=predicted_lai["validation"],
        fusion_input_lai=fusion_input_lai["validation"],
        relative_valid=fusion_lai_valid["validation"],
    )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        year=np.asarray(cache["year"][test_source]),
        row=np.asarray(cache["row"][test_source]),
        col=np.asarray(cache["col"][test_source]),
        target=test_predictions["target"],
        history_prediction=test_predictions["history"],
        fusion_prediction=test_predictions["fusion"],
        gate=test_predictions["gate"],
        learned_gate=test_predictions["learned_gate"],
        crop_coverage=test_predictions["crop_coverage"],
        coverage_reliability=test_predictions["coverage_reliability"],
        delta=test_predictions["delta"],
        phase_weights=test_predictions["phase_weights"],
        predicted_lai=predicted_lai["test"],
        fusion_input_lai=fusion_input_lai["test"],
        relative_valid=fusion_lai_valid["test"],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    del arrays, predicted_lai, history_base, fusion_loaders, model
    if latent_states is not None:
        del latent_states
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=CROPS, required=True)
    parser.add_argument(
        "--fusion-variant",
        choices=(*FUSION_VARIANTS, LATENT_STATE_FUSION_VARIANT),
        required=True,
    )
    parser.add_argument(
        "--dynamics-variant",
        choices=("biid_climate", "cross_attention", "transformer_transition"),
        default="biid_climate",
    )
    parser.add_argument(
        "--history-reference-variant",
        choices=("biid_climate",),
        default="biid_climate",
    )
    parser.add_argument(
        "--history-base-source",
        choices=("neural", "hgb", "mlp", "hgb_mlp_ensemble"),
        default="neural",
    )
    parser.add_argument("--hgb-history-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="fusion_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dynamics-biid-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--gate-penalty", type=float, default=0.0)
    parser.add_argument("--coverage-tau", type=float, default=0.05)
    parser.add_argument("--coverage-floor", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)
    run_one(args)


if __name__ == "__main__":
    main()
