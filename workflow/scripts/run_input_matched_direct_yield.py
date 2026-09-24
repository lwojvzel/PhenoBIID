#!/usr/bin/env python3
"""Direct yield baselines with exactly the raw inputs available to PhenoBIID.

The models consume causal yield history, previous-year LAI, target-season
weather, validity masks, and spatial/year context. Target-season LAI is never
loaded into a model input and is not an auxiliary training target.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from biid_world_model import (
    load_or_compute_world_stats,
    load_world_cache,
    local_indices_for_split,
    make_world_arrays,
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
from run_biid_world_model import TensorBatchLoader


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOTS = {
    "causal_trend": ROOT / "benchmark/results/input_matched_direct_yield_v1",
    "strong_history": ROOT / "benchmark/results/input_matched_direct_yield_v2_strong_history",
}
STRONG_HISTORY_ROOT = ROOT / "benchmark/cache/input_matched_direct_yield/strong_history"
MODEL_NAMES = ("lightgbm", "mlp", "gru", "transformer")


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def prepare_inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the common direct-prediction input without target-season LAI."""
    valid = np.asarray(arrays["relative_valid"], dtype=np.float32)
    weather = np.asarray(arrays["weather"], dtype=np.float32)
    weather = weather * valid[:, :, None]
    previous_lai = np.asarray(arrays["previous_lai"], dtype=np.float32)
    previous_valid = np.asarray(arrays["previous_lai_valid"], dtype=np.float32)
    sequence = np.concatenate(
        (
            weather,
            previous_lai[:, :, None],
            previous_valid[:, :, None],
            valid[:, :, None],
        ),
        axis=2,
    ).astype(np.float32, copy=False)
    static = np.concatenate(
        (
            np.asarray(arrays["history"], dtype=np.float32),
            np.asarray(arrays["context"], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return {
        "sequence": sequence,
        "static": static,
        "valid": valid,
        "target_residual": np.asarray(arrays["target_residual"], dtype=np.float32),
        "target": np.asarray(arrays["target"], dtype=np.float32),
        "baseline": np.asarray(arrays["baseline"], dtype=np.float32),
        "source_indices": np.asarray(arrays["source_indices"], dtype=np.int64),
    }


def apply_strong_history_anchor(
    crop: str, inputs: dict[str, dict[str, np.ndarray]]
) -> tuple[float, float]:
    for split, values in inputs.items():
        path = STRONG_HISTORY_ROOT / crop / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing strong-history cache {path}; run prepare_strong_history_expert.py first."
            )
        with np.load(path) as cached:
            source = np.asarray(cached["source_indices"], dtype=np.int64)
            if not np.array_equal(source, values["source_indices"]):
                raise RuntimeError(f"Strong-history alignment mismatch for {crop}/{split}")
            values["baseline"] = np.asarray(
                cached["history_prediction"], dtype=np.float32
            ).copy()
    train_residual = (
        inputs["train"]["target"].astype(np.float64)
        - inputs["train"]["baseline"].astype(np.float64)
    )
    residual_mean = float(train_residual.mean())
    residual_std = float(train_residual.std())
    if residual_std < 1.0e-6:
        residual_std = 1.0
    for values in inputs.values():
        values["target_residual"] = (
            (
                values["target"].astype(np.float64)
                - values["baseline"].astype(np.float64)
                - residual_mean
            )
            / residual_std
        ).astype(np.float32)
    return residual_mean, residual_std


class DirectMLP(nn.Module):
    def __init__(self, sequence_features: int, static_features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12 * sequence_features + static_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        del valid
        return self.net(torch.cat((sequence.flatten(1), static), dim=1)).squeeze(1)


class DirectGRU(nn.Module):
    def __init__(self, sequence_features: int, static_features: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            sequence_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(128 + static_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        lengths = valid.sum(dim=1).long().clamp_min(1).cpu()
        packed = pack_padded_sequence(
            sequence, lengths, batch_first=True, enforce_sorted=False
        )
        _output, hidden = self.gru(packed)
        return self.head(torch.cat((hidden[-1], static), dim=1)).squeeze(1)


class DirectTransformer(nn.Module):
    def __init__(self, sequence_features: int, static_features: int, dropout: float) -> None:
        super().__init__()
        dim = 128
        self.projection = nn.Linear(sequence_features, dim)
        self.position = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(dim + static_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.encoder(
            self.projection(sequence) + self.position,
            src_key_padding_mask=valid <= 0.0,
        )
        weights = valid.unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat((pooled, static), dim=1)).squeeze(1)


def make_neural_model(
    model_name: str, sequence_features: int, static_features: int, dropout: float
) -> nn.Module:
    if model_name == "mlp":
        return DirectMLP(sequence_features, static_features, dropout)
    if model_name == "gru":
        return DirectGRU(sequence_features, static_features, dropout)
    if model_name == "transformer":
        return DirectTransformer(sequence_features, static_features, dropout)
    raise ValueError(f"Unsupported neural model: {model_name}")


def make_loader(
    inputs: dict[str, np.ndarray], batch_size: int, shuffle: bool
) -> TensorBatchLoader:
    tensors = (
        torch.from_numpy(inputs["sequence"]),
        torch.from_numpy(inputs["static"]),
        torch.from_numpy(inputs["valid"]),
        torch.from_numpy(inputs["target_residual"]),
    )
    return TensorBatchLoader(tensors, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_neural(
    model: nn.Module, loader: TensorBatchLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for sequence, static, valid, _target in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                sequence.to(device, non_blocking=True),
                static.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
            )
        predictions.append(output.float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def physical_prediction(
    residual_prediction: np.ndarray,
    inputs: dict[str, np.ndarray],
    residual_mean: float,
    residual_std: float,
) -> np.ndarray:
    return (
        inputs["baseline"].astype(np.float64)
        + residual_prediction.astype(np.float64) * residual_std
        + residual_mean
    )


def save_split_predictions(
    path: Path,
    cache: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> None:
    source = inputs["source_indices"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        year=np.asarray(cache["year"][source]),
        row=np.asarray(cache["row"][source]),
        col=np.asarray(cache["col"][source]),
        area_weight=np.asarray(cache["area_weight"][source]),
        target=inputs["target"],
        prediction=np.asarray(prediction, dtype=np.float32),
        historical_baseline=inputs["baseline"],
        source_indices=source,
    )


def evaluate(
    cache: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> dict[str, float]:
    area, latitude = evaluation_context(cache, inputs["source_indices"])
    return regression_metrics(inputs["target"], prediction, area, latitude)


def run_lightgbm(
    crop: str,
    seed: int,
    inputs: dict[str, dict[str, np.ndarray]],
    cache: dict[str, np.ndarray],
    stats: Any,
    result_root: Path,
    residual_anchor: str,
    residual_mean: float,
    residual_std: float,
    max_estimators: int,
    cpu_threads: int,
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = result_root / crop / "lightgbm" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_x = np.concatenate(
        (inputs["train"]["sequence"].reshape(len(inputs["train"]["target"]), -1),
         inputs["train"]["static"]), axis=1
    )
    validation_x = np.concatenate(
        (inputs["validation"]["sequence"].reshape(len(inputs["validation"]["target"]), -1),
         inputs["validation"]["static"]), axis=1
    )
    test_x = np.concatenate(
        (inputs["test"]["sequence"].reshape(len(inputs["test"]["target"]), -1),
         inputs["test"]["static"]), axis=1
    )
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=max_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=cpu_threads,
        verbosity=-1,
    )
    started = time.time()
    model.fit(
        train_x,
        inputs["train"]["target_residual"],
        eval_set=[(validation_x, inputs["validation"]["target_residual"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    validation_prediction = physical_prediction(
        model.predict(validation_x), inputs["validation"], residual_mean, residual_std
    )
    test_prediction = physical_prediction(
        model.predict(test_x), inputs["test"], residual_mean, residual_std
    )
    metrics = evaluate(cache, inputs["test"], test_prediction)
    validation_rmse = float(np.sqrt(np.mean(np.square(
        validation_prediction - inputs["validation"]["target"]
    ))))
    metrics.update(
        validation_rmse=validation_rmse,
        best_iteration=int(model.best_iteration_),
        elapsed_seconds=float(time.time() - started),
    )
    joblib.dump(model, run_dir / "model_best.joblib", compress=3)
    save_split_predictions(
        run_dir / "validation_predictions.npz", cache, inputs["validation"], validation_prediction
    )
    save_split_predictions(
        run_dir / "test_predictions.npz", cache, inputs["test"], test_prediction
    )
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "crop": crop,
            "model": "lightgbm",
            "seed": seed,
            "input_boundary": "H_<t + previous-year LAI + target-season ERA5 + context",
            "target_season_lai_used": False,
            "auxiliary_lai_supervision": False,
            "prediction_target": "annual GDHY yield",
            "residual_anchor": residual_anchor,
            "anchor_residual_mean": residual_mean,
            "anchor_residual_std": residual_std,
            "train_years": [1982, 2011],
            "validation_years": [2012, 2012],
            "test_years": [2013, 2016],
            "sequence_shape": list(inputs["train"]["sequence"].shape[1:]),
            "static_features": int(inputs["train"]["static"].shape[1]),
            "normalization": asdict(stats),
            "max_estimators": max_estimators,
            "cpu_threads": cpu_threads,
        },
        run_dir / "config.json",
    )
    del train_x, validation_x, test_x, model
    return {"crop": crop, "model": "lightgbm", "seed": seed, **metrics}


def run_neural(
    crop: str,
    model_name: str,
    seed: int,
    inputs: dict[str, dict[str, np.ndarray]],
    cache: dict[str, np.ndarray],
    stats: Any,
    result_root: Path,
    residual_anchor: str,
    residual_mean: float,
    residual_std: float,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
) -> dict[str, Any]:
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    model = make_neural_model(
        model_name,
        inputs["train"]["sequence"].shape[2],
        inputs["train"]["static"].shape[1],
        dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loaders = {
        name: make_loader(values, batch_size, name == "train")
        for name, values in inputs.items()
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation_rmse = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for sequence, static, valid, target in loaders["train"]:
            sequence = sequence.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(sequence, static, valid)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * target.shape[0]
            count += int(target.shape[0])

        validation_residual = predict_neural(model, loaders["validation"], device)
        validation_prediction = physical_prediction(
            validation_residual, inputs["validation"], residual_mean, residual_std
        )
        validation_rmse = float(np.sqrt(np.mean(np.square(
            validation_prediction - inputs["validation"]["target"]
        ))))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "validation_rmse": validation_rmse,
            }
        )
        print(
            f"[DIRECT] crop={crop} model={model_name} seed={seed} epoch={epoch} "
            f"val_rmse={validation_rmse:.6f}",
            flush=True,
        )
        if validation_rmse < best_validation_rmse - 1.0e-6:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No neural direct-yield checkpoint was selected.")
    model.load_state_dict(best_state)
    model.to(device)
    validation_residual = predict_neural(model, loaders["validation"], device)
    test_residual = predict_neural(model, loaders["test"], device)
    validation_prediction = physical_prediction(
        validation_residual, inputs["validation"], residual_mean, residual_std
    )
    test_prediction = physical_prediction(
        test_residual, inputs["test"], residual_mean, residual_std
    )
    metrics = evaluate(cache, inputs["test"], test_prediction)
    metrics.update(
        best_validation_rmse=best_validation_rmse,
        best_epoch=best_epoch,
        elapsed_seconds=float(time.time() - started),
        peak_allocated_memory_mib=float(torch.cuda.max_memory_allocated(device) / 2**20),
    )
    run_dir = result_root / crop / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "model_best.pt")
    write_csv(history, run_dir / "history.csv")
    save_split_predictions(
        run_dir / "validation_predictions.npz", cache, inputs["validation"], validation_prediction
    )
    save_split_predictions(
        run_dir / "test_predictions.npz", cache, inputs["test"], test_prediction
    )
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "crop": crop,
            "model": model_name,
            "seed": seed,
            "input_boundary": "H_<t + previous-year LAI + target-season ERA5 + context",
            "target_season_lai_used": False,
            "auxiliary_lai_supervision": False,
            "prediction_target": "annual GDHY yield",
            "residual_anchor": residual_anchor,
            "anchor_residual_mean": residual_mean,
            "anchor_residual_std": residual_std,
            "train_years": [1982, 2011],
            "validation_years": [2012, 2012],
            "test_years": [2013, 2016],
            "sequence_shape": list(inputs["train"]["sequence"].shape[1:]),
            "sequence_channels": [
                "13 normalized ERA5 variables",
                "previous-year LAI",
                "previous-year LAI validity",
                "relative-slot validity",
            ],
            "static_features": int(inputs["train"]["static"].shape[1]),
            "normalization": asdict(stats),
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "device": str(device),
        },
        run_dir / "config.json",
    )
    del model, loaders, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return {"crop": crop, "model": model_name, "seed": seed, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=CROPS, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--seeds", default="42,45,48")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-estimators", type=int, default=800)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--anchor", choices=tuple(RESULTS_ROOTS), default="causal_trend"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in parse_csv(args.seeds)]
    cache = load_cache(args.crop)
    world = load_world_cache(args.crop)
    stats = load_or_compute_world_stats(args.crop, cache, world)
    local = {
        "train": local_indices_for_split(cache, world, 0),
        "validation": local_indices_for_split(cache, world, 1),
        "test": local_indices_for_split(cache, world, 2),
    }
    arrays = {
        name: make_world_arrays(cache, world, stats, indices)
        for name, indices in local.items()
    }
    inputs = {name: prepare_inputs(values) for name, values in arrays.items()}
    del arrays, world
    result_root = RESULTS_ROOTS[args.anchor]
    if args.anchor == "strong_history":
        residual_mean, residual_std = apply_strong_history_anchor(args.crop, inputs)
        residual_anchor = "frozen crop-specific strong history expert"
    else:
        residual_mean = float(stats.residual_mean)
        residual_std = float(stats.residual_std)
        residual_anchor = "causal per-grid historical trend"

    device = torch.device(args.device)
    if args.model != "lightgbm" and (
        device.type != "cuda" or not torch.cuda.is_available()
    ):
        raise SystemExit("Neural input-matched baselines require a visible CUDA GPU.")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = result_root / args.crop / args.model / f"seed_{seed}"
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.exists() and not args.force:
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")) | {
                "crop": args.crop, "model": args.model, "seed": seed
            })
            print(f"[SKIP] Existing result: {metrics_path}", flush=True)
            continue
        if args.model == "lightgbm":
            row = run_lightgbm(
                args.crop, seed, inputs, cache, stats, result_root,
                residual_anchor, residual_mean, residual_std,
                args.max_estimators, args.cpu_threads,
            )
        else:
            row = run_neural(
                args.crop, args.model, seed, inputs, cache, stats, result_root,
                residual_anchor, residual_mean, residual_std, device,
                args.max_epochs, args.patience, args.batch_size,
                args.learning_rate, args.weight_decay, args.dropout,
            )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary_path = result_root / args.crop / args.model / "summary.csv"
    write_csv(rows, summary_path)
    print(f"[DONE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
