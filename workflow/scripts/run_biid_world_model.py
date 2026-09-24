#!/usr/bin/env python3
"""Train and evaluate relative-phenology BIID crop world-model experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from biid_world_model import (
    WORLD_RESULTS_ROOT,
    WORLD_VARIANTS,
    CropWorldDynamics,
    WorldNormalizationStats,
    YieldHeads,
    build_world_cache,
    lai_metrics,
    load_world_cache,
    load_or_compute_world_stats,
    local_indices_for_split,
    make_world_arrays,
    masked_lai_loss,
    parameter_count,
    stats_to_dict,
)
from multimodal_baseline import (
    CROPS,
    evaluation_context,
    load_cache,
    regression_metrics,
    save_json,
    set_seed,
    write_csv,
)


DYNAMICS_VARIANTS = {
    "gru_transition",
    "transformer_transition",
    "gated_concat",
    "cross_attention",
    "biid_climate",
    "biid_prior",
}
ARRAY_ORDER = (
    "weather",
    "previous_lai",
    "previous_lai_valid",
    "target_lai",
    "target_lai_valid",
    "relative_valid",
    "relative_weight",
    "history",
    "context",
    "target_residual",
    "target_absolute",
    "target",
    "baseline",
)


class TensorBatchLoader:
    """Vectorized in-memory batching without per-sample Python collation."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.tensors = tuple(tensors)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = int(self.tensors[0].shape[0])
        if any(int(tensor.shape[0]) != self.size for tensor in self.tensors):
            raise ValueError("All tensors must have the same leading dimension.")

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.size)
            for start in range(0, self.size, self.batch_size):
                indices = order[start : start + self.batch_size]
                yield tuple(tensor[indices] for tensor in self.tensors)
            return
        for start in range(0, self.size, self.batch_size):
            stop = min(start + self.batch_size, self.size)
            yield tuple(tensor[start:stop] for tensor in self.tensors)


def make_loader(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> TensorBatchLoader:
    del pin_memory
    tensors = [torch.from_numpy(np.asarray(arrays[name])) for name in ARRAY_ORDER]
    return TensorBatchLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def unpack_batch(batch: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return dict(zip(ARRAY_ORDER, batch))


def select_local_indices(
    indices: np.ndarray, maximum: int | None, seed: int
) -> np.ndarray:
    if maximum is None or indices.size <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def physical_lai_rmse(
    target_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    valid: np.ndarray,
    stats: WorldNormalizationStats,
) -> float:
    selected = np.asarray(valid, dtype=bool)
    difference = (
        prediction_normalized[selected].astype(np.float64)
        - target_normalized[selected].astype(np.float64)
    ) * stats.lai_std
    return float(np.sqrt(np.mean(np.square(difference))))


def dynamics_amp_dtype(variant: str) -> torch.dtype:
    return torch.bfloat16 if variant == "biid_prior" else torch.float16


@torch.no_grad()
def predict_dynamics(
    model: CropWorldDynamics | None,
    loader: TensorBatchLoader,
    variant: str,
    device: torch.device,
) -> np.ndarray:
    if model is not None:
        model.eval()
    predictions: list[np.ndarray] = []
    for raw_batch in loader:
        batch = unpack_batch(raw_batch)
        if variant == "persistence":
            prediction = batch["previous_lai"]
        elif variant == "observed_upper":
            prediction = batch["target_lai"]
        else:
            if model is None:
                raise RuntimeError("Trainable dynamics variant has no model.")
            with torch.autocast(
                device_type=device.type,
                dtype=dynamics_amp_dtype(variant),
                enabled=device.type == "cuda",
            ):
                prediction, _state = model(
                    batch["weather"].to(device, non_blocking=True),
                    batch["previous_lai"].to(device, non_blocking=True),
                    batch["previous_lai_valid"].to(device, non_blocking=True),
                    batch["relative_valid"].to(device, non_blocking=True),
                    batch["history"].to(device, non_blocking=True),
                    batch["context"].to(device, non_blocking=True),
                )
            prediction = prediction.cpu()
        predictions.append(prediction.numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32, copy=False)


def train_dynamics(
    model: CropWorldDynamics,
    train_loader: TensorBatchLoader,
    val_loader: TensorBatchLoader,
    val_arrays: dict[str, np.ndarray],
    stats: WorldNormalizationStats,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    micro_batch_size: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    use_amp = device.type == "cuda"
    amp_dtype = dynamics_amp_dtype(model.variant)
    use_scaler = use_amp and amp_dtype == torch.float16
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=use_amp,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_rmse = float("inf")
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        total_count = 0
        for raw_batch in train_loader:
            full = unpack_batch(raw_batch)
            n = raw_batch[0].shape[0]
            chunk_size = n if micro_batch_size is None else micro_batch_size
            valid = full["target_lai_valid"]
            full_mass = (valid * (1 + full["relative_weight"] * valid.sum(1, keepdim=True).clamp_min(1))).sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for start in range(0, n, chunk_size):
                batch = {k: v[start:start + chunk_size].to(device, non_blocking=True) for k, v in full.items()}
                valid = batch["target_lai_valid"]
                mass = (valid * (1 + batch["relative_weight"] * valid.sum(1, keepdim=True).clamp_min(1))).sum().clamp_min(1)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    prediction, _state = model(batch["weather"], batch["previous_lai"], batch["previous_lai_valid"],
                                               batch["relative_valid"], batch["history"], batch["context"])
                    loss = masked_lai_loss(prediction, batch["target_lai"], valid, batch["relative_weight"])
                    loss = loss * (mass / full_mass.to(device))
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite dynamics objective")
                scaler.scale(loss).backward()
                batch_loss += loss.detach()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += batch_loss * n
            total_count += n

        val_prediction = predict_dynamics(model, val_loader, model.variant, device)
        val_rmse = physical_lai_rmse(
            val_arrays["target_lai"],
            val_prediction,
            val_arrays["target_lai_valid"],
            stats,
        )
        train_loss = float((total_loss / max(total_count, 1)).cpu())
        elapsed = time.perf_counter() - epoch_started
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_lai_rmse": val_rmse,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[DYNAMICS] variant={model.variant} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_lai_rmse={val_rmse:.6f} "
            f"seconds={elapsed:.1f}",
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
        raise RuntimeError("Dynamics training did not produce a checkpoint.")
    return best_state, history_rows, best_epoch, best_rmse


def make_yield_loader(
    predicted_lai: np.ndarray,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> TensorBatchLoader:
    tensors = (
        torch.from_numpy(predicted_lai),
        torch.from_numpy(arrays["relative_valid"]),
        torch.from_numpy(arrays["history"]),
        torch.from_numpy(arrays["context"]),
        torch.from_numpy(arrays["target_residual"]),
        torch.from_numpy(arrays["target_absolute"]),
        torch.from_numpy(arrays["target"]),
        torch.from_numpy(arrays["baseline"]),
    )
    del pin_memory
    return TensorBatchLoader(tensors, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_yield_heads(
    model: YieldHeads,
    loader: TensorBatchLoader,
    stats: WorldNormalizationStats,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected = {"history": [], "lai": [], "fusion": [], "target": [], "baseline": []}
    for batch in loader:
        lai, valid, history, context, _residual, _absolute, target, baseline = batch
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(
                lai.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
                history.to(device, non_blocking=True),
                context.to(device, non_blocking=True),
            )
        collected["history"].append(outputs["history"].cpu().numpy())
        collected["lai"].append(outputs["lai"].cpu().numpy())
        collected["fusion"].append(outputs["fusion"].cpu().numpy())
        collected["target"].append(target.numpy())
        collected["baseline"].append(baseline.numpy())
    normalized = {key: np.concatenate(values) for key, values in collected.items()}
    baseline = normalized["baseline"].astype(np.float64)
    normalized["history"] = (
        baseline
        + normalized["history"] * stats.residual_std
        + stats.residual_mean
    )
    normalized["fusion"] = (
        baseline
        + normalized["fusion"] * stats.residual_std
        + stats.residual_mean
    )
    normalized["lai"] = (
        normalized["lai"] * stats.target_std + stats.target_mean
    )
    return normalized


def train_yield_heads(
    model: YieldHeads,
    train_loader: TensorBatchLoader,
    val_loader: TensorBatchLoader,
    stats: WorldNormalizationStats,
    device: torch.device,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, float],
]:
    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=use_amp,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.MSELoss()
    branch_parameter_groups = model.branch_parameter_groups()
    head_names = ("history", "lai", "fusion")
    best_states: dict[str, dict[str, torch.Tensor] | None] = {
        name: None for name in head_names
    }
    best_epochs = {name: -1 for name in head_names}
    best_rmses = {name: float("inf") for name in head_names}
    stale_epochs = {name: 0 for name in head_names}
    rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        total_count = 0
        for batch in train_loader:
            lai, valid, history, context, residual, absolute, _target, _baseline = batch
            lai = lai.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            history = history.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)
            absolute = absolute.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(lai, valid, history, context)
                fusion_loss = criterion(outputs["fusion"], residual)
                history_loss = criterion(outputs["history"], residual)
                lai_loss = criterion(outputs["lai"], absolute)
                loss = fusion_loss + history_loss + lai_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for parameters in branch_parameter_groups:
                nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * lai.shape[0]
            total_count += int(lai.shape[0])

        val_predictions = predict_yield_heads(model, val_loader, stats, device)
        validation_rmses = {
            name: float(
                regression_metrics(val_predictions["target"], val_predictions[name])[
                    "rmse"
                ]
            )
            for name in head_names
        }
        train_loss = float((total_loss / max(total_count, 1)).cpu())
        elapsed = time.perf_counter() - epoch_started
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_history_rmse": validation_rmses["history"],
                "validation_lai_rmse": validation_rmses["lai"],
                "validation_fusion_rmse": validation_rmses["fusion"],
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[YIELD] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_history_rmse={validation_rmses['history']:.6f} "
            f"val_lai_rmse={validation_rmses['lai']:.6f} "
            f"val_fusion_rmse={validation_rmses['fusion']:.6f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        for name in head_names:
            if validation_rmses[name] < best_rmses[name] - 1.0e-6:
                best_rmses[name] = validation_rmses[name]
                best_epochs[name] = epoch
                best_states[name] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs[name] = 0
            else:
                stale_epochs[name] += 1
        if all(value >= patience for value in stale_epochs.values()):
            break
    if any(state is None for state in best_states.values()):
        raise RuntimeError("Yield-branch training did not produce every checkpoint.")
    complete_states = {
        name: state for name, state in best_states.items() if state is not None
    }
    return complete_states, rows, best_epochs, best_rmses


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    build_world_cache(args.crop)
    cache = load_cache(args.crop)
    world = load_world_cache(args.crop)
    stats = load_or_compute_world_stats(args.crop, cache, world)
    local_indices = {
        "train": select_local_indices(
            local_indices_for_split(cache, world, 0), args.max_train_samples, args.seed + 1
        ),
        "validation": select_local_indices(
            local_indices_for_split(cache, world, 1), args.max_eval_samples, args.seed + 2
        ),
        "test": select_local_indices(
            local_indices_for_split(cache, world, 2), args.max_eval_samples, args.seed + 3
        ),
    }
    arrays = {
        name: make_world_arrays(cache, world, stats, indices)
        for name, indices in local_indices.items()
    }
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(arrays["train"], args.batch_size, True, pin_memory),
        "validation": make_loader(
            arrays["validation"], args.batch_size, False, pin_memory
        ),
        "test": make_loader(arrays["test"], args.batch_size, False, pin_memory),
    }
    run_dir = WORLD_RESULTS_ROOT / args.crop / args.variant / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dynamics: CropWorldDynamics | None = None
    dynamics_history: list[dict[str, Any]] = []
    dynamics_best_epoch = 0
    dynamics_best_rmse = float("nan")
    dynamics_parameters = 0
    if args.variant in DYNAMICS_VARIANTS:
        dynamics = CropWorldDynamics(
            variant=args.variant,
            dim=args.dim,
            state_tokens=args.state_tokens,
            heads=args.heads,
            biid_layers=args.biid_layers,
            dropout=args.dropout,
        ).to(device)
        dynamics_parameters = parameter_count(dynamics)
        best_state, dynamics_history, dynamics_best_epoch, dynamics_best_rmse = train_dynamics(
            dynamics,
            loaders["train"],
            loaders["validation"],
            arrays["validation"],
            stats,
            device,
            args.dynamics_epochs,
            args.patience,
            args.learning_rate,
            args.weight_decay,
        )
        dynamics.load_state_dict(best_state)
        dynamics.to(device)
        torch.save(best_state, run_dir / "dynamics_best.pt")
        write_csv(dynamics_history, run_dir / "dynamics_history.csv")

    prediction_splits = ("test",) if args.dynamics_only else ("train", "validation", "test")
    predicted_lai = {
        name: predict_dynamics(dynamics, loaders[name], args.variant, device)
        for name in prediction_splits
    }
    test_lai_metrics = lai_metrics(
        arrays["test"]["target_lai"],
        predicted_lai["test"],
        arrays["test"]["target_lai_valid"],
        arrays["test"]["relative_weight"],
        stats,
    )

    if args.dynamics_only:
        metrics = {
            "crop": args.crop,
            "variant": args.variant,
            "seed": args.seed,
            "dynamics_best_epoch": dynamics_best_epoch,
            "dynamics_best_validation_lai_rmse": dynamics_best_rmse,
            "test_lai": test_lai_metrics,
            "test_yield": {},
        }
        save_json(metrics, run_dir / "test_metrics.json")
        save_json(stats_to_dict(stats), run_dir / "normalization.json")
        save_json(
            {
                "crop": args.crop,
                "variant": args.variant,
                "seed": args.seed,
                "run_tag": args.run_tag,
                "dynamics_only": True,
                "time_axis": "MIRCA-packed relative phenology slots",
                "train_years": [1982, 2011],
                "validation_year": 2012,
                "test_years": [2013, 2016],
                "oracle_future_climate": True,
                "history_in_lai_dynamics": False,
                "mirca_as_predictive_modality": False,
                "dim": args.dim,
                "heads": args.heads,
                "dynamics_parameters": dynamics_parameters,
                "batch_size": args.batch_size,
                "dynamics_epochs": args.dynamics_epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "actual_device": str(device),
                "split_sizes": {
                    name: int(value["target"].shape[0])
                    for name, value in arrays.items()
                },
            },
            run_dir / "config.json",
        )
        test_source = arrays["test"]["source_indices"]
        np.savez_compressed(
            run_dir / "test_predictions.npz",
            year=np.asarray(cache["year"][test_source]),
            row=np.asarray(cache["row"][test_source]),
            col=np.asarray(cache["col"][test_source]),
            target_lai=arrays["test"]["target_lai"].astype(np.float32),
            predicted_lai=predicted_lai["test"].astype(np.float32),
            relative_valid=arrays["test"]["relative_valid"].astype(np.uint8),
            relative_weight=arrays["test"]["relative_weight"].astype(np.float32),
            source_month=arrays["test"]["source_month"].astype(np.uint8),
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        del arrays, loaders, predicted_lai
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics

    yield_loaders = {
        "train": make_yield_loader(
            predicted_lai["train"], arrays["train"], args.batch_size, True, pin_memory
        ),
        "validation": make_yield_loader(
            predicted_lai["validation"],
            arrays["validation"],
            args.batch_size,
            False,
            pin_memory,
        ),
        "test": make_yield_loader(
            predicted_lai["test"], arrays["test"], args.batch_size, False, pin_memory
        ),
    }
    yield_model = YieldHeads(
        dim=args.dim, heads=args.heads, dropout=args.dropout
    ).to(device)
    yield_parameters = parameter_count(yield_model)
    best_yield_states, yield_history, yield_best_epochs, yield_best_rmses = train_yield_heads(
        yield_model,
        yield_loaders["train"],
        yield_loaders["validation"],
        stats,
        device,
        args.yield_epochs,
        args.patience,
        args.learning_rate,
        args.weight_decay,
    )
    checkpoint_names = {
        "history": "yield_history_only_best.pt",
        "lai": "yield_predicted_lai_only_best.pt",
        "fusion": "yield_fusion_best.pt",
    }
    for name, state in best_yield_states.items():
        torch.save(state, run_dir / checkpoint_names[name])
    torch.save(best_yield_states["fusion"], run_dir / "yield_heads_best.pt")
    write_csv(yield_history, run_dir / "yield_history.csv")

    test_predictions: dict[str, np.ndarray] = {}
    for name in ("history", "lai", "fusion"):
        yield_model.load_state_dict(best_yield_states[name])
        yield_model.to(device)
        branch_predictions = predict_yield_heads(
            yield_model, yield_loaders["test"], stats, device
        )
        test_predictions[name] = branch_predictions[name]
        if "target" not in test_predictions:
            test_predictions["target"] = branch_predictions["target"]
            test_predictions["baseline"] = branch_predictions["baseline"]
    test_source = arrays["test"]["source_indices"]
    area_weight, latitude_weight = evaluation_context(cache, test_source)
    yield_metrics = {
        name: regression_metrics(
            test_predictions["target"], prediction, area_weight, latitude_weight
        )
        for name, prediction in (
            ("history_only", test_predictions["history"]),
            ("predicted_lai_only", test_predictions["lai"]),
            ("history_predicted_lai_fusion", test_predictions["fusion"]),
        )
    }
    metrics = {
        "crop": args.crop,
        "variant": args.variant,
        "seed": args.seed,
        "dynamics_best_epoch": dynamics_best_epoch,
        "dynamics_best_validation_lai_rmse": dynamics_best_rmse,
        "yield_best_epoch": yield_best_epochs["fusion"],
        "yield_best_validation_rmse": yield_best_rmses["fusion"],
        "yield_best_epochs": yield_best_epochs,
        "yield_best_validation_rmse_by_head": yield_best_rmses,
        "test_lai": test_lai_metrics,
        "test_yield": yield_metrics,
    }
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(stats_to_dict(stats), run_dir / "normalization.json")
    save_json(
        {
            "crop": args.crop,
            "variant": args.variant,
            "seed": args.seed,
            "run_tag": args.run_tag,
            "experiment_version": 2,
            "yield_head_protocol": "independent_encoders_and_validation_checkpoints",
            "batch_loader": "vectorized_full_sample_permutation",
            "time_axis": "MIRCA-packed relative phenology slots",
            "train_years": [1982, 2011],
            "validation_year": 2012,
            "test_years": [2013, 2016],
            "oracle_future_climate": True,
            "history_in_lai_dynamics": args.variant == "biid_prior",
            "dynamics_numeric_precision": (
                "bfloat16" if args.variant == "biid_prior" else "float16"
            ),
            "biid_prior_history_normalization": args.variant == "biid_prior",
            "mirca_as_predictive_modality": False,
            "dim": args.dim,
            "state_tokens": args.state_tokens,
            "heads": args.heads,
            "biid_layers_per_shared_transition": args.biid_layers,
            "dynamics_parameters": dynamics_parameters,
            "yield_head_parameters": yield_parameters,
            "batch_size": args.batch_size,
            "cpu_threads": args.cpu_threads,
            "cpu_interop_threads": 1,
            "dynamics_epochs": args.dynamics_epochs,
            "yield_epochs": args.yield_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "requested_device": args.device,
            "actual_device": str(device),
            "peak_allocated_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_memory_mib": (
                torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
            "split_sizes": {
                name: int(value["target"].shape[0]) for name, value in arrays.items()
            },
        },
        run_dir / "config.json",
    )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        year=np.asarray(cache["year"][test_source]),
        row=np.asarray(cache["row"][test_source]),
        col=np.asarray(cache["col"][test_source]),
        target_yield=test_predictions["target"].astype(np.float32),
        history_prediction=test_predictions["history"].astype(np.float32),
        lai_prediction=test_predictions["lai"].astype(np.float32),
        fusion_prediction=test_predictions["fusion"].astype(np.float32),
        target_lai=arrays["test"]["target_lai"].astype(np.float32),
        predicted_lai=predicted_lai["test"].astype(np.float32),
        relative_valid=arrays["test"]["relative_valid"].astype(np.uint8),
        relative_weight=arrays["test"]["relative_weight"].astype(np.float32),
        source_month=arrays["test"]["source_month"].astype(np.uint8),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    del arrays, loaders, predicted_lai, yield_loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def build_summary(run_tag: str) -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted(WORLD_RESULTS_ROOT.glob("*/*/seed_*/test_metrics.json")):
        config_path = path.with_name("config.json")
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("run_tag") != run_tag:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        base = {
            "crop": data["crop"],
            "variant": data["variant"],
            "seed": data["seed"],
            "lai_rmse": data["test_lai"]["valid_relative_phases"]["rmse"],
            "lai_weighted_rmse": data["test_lai"]["mirca_weighted_rmse"],
        }
        for head, metrics in data["test_yield"].items():
            rows.append({**base, "yield_head": head, **metrics})
    output = WORLD_RESULTS_ROOT / "summary" / "results.csv"
    write_csv(rows, output)
    return output


def build_stats(crops: list[str], force: bool = False) -> None:
    for crop in crops:
        build_world_cache(crop)
        cache = load_cache(crop)
        world = load_world_cache(crop)
        stats = load_or_compute_world_stats(crop, cache, world, force=force)
        print(f"{crop}: {json.dumps(stats_to_dict(stats), ensure_ascii=False)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("cache", "stats", "train", "summary"), default="train"
    )
    parser.add_argument("--crops", default=",".join(CROPS))
    parser.add_argument("--crop", choices=CROPS, default="maize")
    parser.add_argument("--variant", choices=WORLD_VARIANTS, default="biid_climate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="adhoc")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--state-tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--biid-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dynamics-epochs", type=int, default=40)
    parser.add_argument("--yield-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--force-stats", action="store_true")
    parser.add_argument("--dynamics-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)
    if args.stage == "cache":
        crops = [part.strip() for part in args.crops.split(",") if part.strip()]
        for crop in crops:
            print(build_world_cache(crop), flush=True)
        return
    if args.stage == "stats":
        crops = [part.strip() for part in args.crops.split(",") if part.strip()]
        build_stats(crops, force=args.force_stats)
        return
    if args.stage == "summary":
        print(build_summary(args.run_tag), flush=True)
        return
    run_one(args)


if __name__ == "__main__":
    main()
