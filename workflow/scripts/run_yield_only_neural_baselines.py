#!/usr/bin/env python3
"""Run causal neural sequence baselines using historical yield only."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from multimodal_baseline import (
    CROPS,
    choose_indices,
    compute_normalization,
    evaluation_context,
    load_cache,
    metrics_by_year,
    regression_metrics,
    save_json,
    save_predictions,
    set_seed,
    write_csv,
)
from run_history_multimodal_baselines import (
    HISTORY_FEATURE_NAMES,
    build_causal_history_features,
    parse_csv,
)
from rebuild_yield_only_summaries import rebuild_summary


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmark/results/yield_only_baselines"
MODELS = ("mlp", "rnn", "lstm", "gru", "tcn", "transformer")


class HistorySequenceRegressor(nn.Module):
    def __init__(
        self,
        architecture: str,
        static_size: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.hidden_size = hidden_size
        if architecture == "mlp":
            representation_size = 10
            self.sequence_encoder: nn.Module | None = None
        elif architecture in {"rnn", "lstm", "gru"}:
            recurrent_class = {
                "rnn": nn.RNN,
                "lstm": nn.LSTM,
                "gru": nn.GRU,
            }[architecture]
            self.sequence_encoder = recurrent_class(
                input_size=2,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            representation_size = hidden_size
        elif architecture == "tcn":
            self.sequence_encoder = nn.Sequential(
                nn.Conv1d(2, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                ),
                nn.GELU(),
            )
            representation_size = hidden_size
        elif architecture == "transformer":
            self.input_projection = nn.Linear(2, hidden_size)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, 5, hidden_size)
            )
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=2)
            representation_size = hidden_size
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.head = nn.Sequential(
            nn.Linear(representation_size + static_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode_sequence(self, history: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            return history.flatten(1)
        if self.architecture in {"rnn", "gru"}:
            assert self.sequence_encoder is not None
            _output, hidden = self.sequence_encoder(history)
            return hidden[-1]
        if self.architecture == "lstm":
            assert self.sequence_encoder is not None
            _output, (hidden, _cell) = self.sequence_encoder(history)
            return hidden[-1]
        if self.architecture == "tcn":
            assert self.sequence_encoder is not None
            encoded = self.sequence_encoder(history.transpose(1, 2))
            mask = history[:, :, 1].unsqueeze(1)
            denominator = mask.sum(dim=2).clamp_min(1.0)
            return (encoded * mask).sum(dim=2) / denominator
        if self.architecture == "transformer":
            assert self.sequence_encoder is not None
            encoded = self.input_projection(history) + self.position_embedding
            missing = history[:, :, 1] < 0.5
            all_missing = missing.all(dim=1)
            if all_missing.any():
                missing = missing.clone()
                missing[all_missing, -1] = False
            encoded = self.sequence_encoder(encoded, src_key_padding_mask=missing)
            valid = (~missing).unsqueeze(2).to(encoded.dtype)
            return (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        raise RuntimeError("Unreachable architecture branch")

    def forward(self, history: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_sequence(history)
        return self.head(torch.cat((sequence, static), dim=1)).squeeze(1)


def make_arrays(
    cache: dict[str, np.ndarray],
    history_features: np.ndarray,
    trend_baseline: np.ndarray,
    indices: np.ndarray,
    residual_mean: float,
    residual_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = history_features[indices, :5][:, ::-1]
    masks = history_features[indices, 5:10][:, ::-1]
    sequence = np.stack((values, masks), axis=2).astype(np.float32, copy=False)
    static = np.concatenate(
        (
            history_features[indices, 10:].astype(np.float32, copy=False),
            np.asarray(cache["context"][indices], dtype=np.float32),
        ),
        axis=1,
    )
    target = np.asarray(cache["target"][indices], dtype=np.float32)
    baseline = trend_baseline[indices].astype(np.float32)
    residual = ((target - baseline - residual_mean) / residual_std).astype(
        np.float32, copy=False
    )
    return sequence, static, residual


def make_loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(*(torch.from_numpy(array) for array in arrays)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        persistent_workers=True,
        pin_memory=pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for history, static, _target in loader:
        output = model(
            history.to(device, non_blocking=True),
            static.to(device, non_blocking=True),
        )
        predictions.append(output.cpu().numpy())
    return np.concatenate(predictions)


def run_one(
    crop: str,
    architecture: str,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    run_dir = RESULTS_ROOT / crop / architecture / f"seed_{seed}"
    metrics_path = run_dir / "test_metrics.json"
    if (
        metrics_path.exists()
        and (run_dir / "config.json").exists()
        and (run_dir / "test_predictions.npz").exists()
        and not force
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[SKIP] {crop} {architecture} seed={seed}", flush=True)
        return {"crop": crop, "method": architecture, "seed": seed, **metrics}

    set_seed(seed)
    if device.type == "cuda" and architecture == "tcn":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    cache = load_cache(crop)
    stats = compute_normalization(cache)
    train_idx = choose_indices(cache, 0)
    val_idx = choose_indices(cache, 1)
    test_idx = choose_indices(cache, 2)
    history_features, trend_baseline = build_causal_history_features(
        cache, stats.target_mean, stats.target_std
    )
    train_true = np.asarray(cache["target"][train_idx], dtype=np.float64)
    train_base = trend_baseline[train_idx].astype(np.float64)
    train_residual = train_true - train_base
    residual_mean = float(train_residual.mean())
    residual_std = float(train_residual.std())
    if residual_std < 1.0e-6:
        residual_std = 1.0

    train_arrays = make_arrays(
        cache, history_features, trend_baseline, train_idx, residual_mean, residual_std
    )
    val_arrays = make_arrays(
        cache, history_features, trend_baseline, val_idx, residual_mean, residual_std
    )
    test_arrays = make_arrays(
        cache, history_features, trend_baseline, test_idx, residual_mean, residual_std
    )
    use_amp = device.type == "cuda"
    train_loader = make_loader(train_arrays, batch_size, True, use_amp)
    val_loader = make_loader(val_arrays, batch_size, False, use_amp)
    test_loader = make_loader(test_arrays, batch_size, False, use_amp)

    model = HistorySequenceRegressor(
        architecture=architecture,
        static_size=int(train_arrays[1].shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    val_true = np.asarray(cache["target"][val_idx], dtype=np.float64)
    val_base = trend_baseline[val_idx].astype(np.float64)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_rmse = float("inf")
    no_improvement = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for history, static, target in train_loader:
            history = history.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(history, static)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * target.shape[0]
            total_count += target.shape[0]

        val_residual_prediction = predict(model, val_loader, device)
        val_prediction = np.maximum(
            val_base + val_residual_prediction * residual_std + residual_mean,
            0.0,
        )
        val_rmse = float(np.sqrt(np.mean(np.square(val_prediction - val_true))))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
                "validation_rmse": val_rmse,
            }
        )
        print(
            f"[YIELD-NEURAL] crop={crop} model={architecture} seed={seed} "
            f"epoch={epoch} val_rmse={val_rmse:.6f}",
            flush=True,
        )
        if val_rmse < best_val_rmse - 1.0e-6:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Neural baseline did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_residual_prediction = predict(model, test_loader, device)
    test_true = np.asarray(cache["target"][test_idx], dtype=np.float64)
    test_base = trend_baseline[test_idx].astype(np.float64)
    test_prediction = np.maximum(
        test_base + test_residual_prediction * residual_std + residual_mean,
        0.0,
    )
    area_weight, latitude_weight = evaluation_context(cache, test_idx)
    metrics = regression_metrics(
        test_true, test_prediction, area_weight, latitude_weight
    )
    metrics.update(
        {"best_epoch": best_epoch, "best_validation_rmse": best_val_rmse}
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "model_best.pt")
    save_json(metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "crop": crop,
            "model": architecture,
            "modalities": ["historical_yield"],
            "forecast_protocol": "rolling one-year-ahead; target-year yield excluded",
            "history_lag_years": 5,
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
            "context_features": [
                "sin_latitude",
                "cos_latitude",
                "sin_longitude",
                "cos_longitude",
                "normalized_year",
            ],
            "residual_baseline": "causal per-grid historical linear trend",
            "train_years": [1981, 2011],
            "validation_years": [2012, 2012],
            "test_years": [2013, 2016],
            "seed": seed,
            "device": str(device),
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_samples": int(train_idx.size),
            "validation_samples": int(val_idx.size),
            "test_samples": int(test_idx.size),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
        },
        run_dir / "config.json",
    )
    write_csv(history_rows, run_dir / "history.csv")
    write_csv(
        metrics_by_year(cache, test_idx, test_true, test_prediction),
        run_dir / "test_metrics_by_year.csv",
    )
    save_predictions(
        run_dir / "test_predictions.npz",
        cache,
        test_idx,
        test_true,
        test_prediction,
    )
    print(
        f"[YIELD-NEURAL-DONE] crop={crop} model={architecture} seed={seed} "
        f"test_rmse={metrics['rmse']:.6f}",
        flush=True,
    )
    return {"crop": crop, "method": architecture, "seed": seed, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default=",".join(CROPS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    crops = tuple(parse_csv(args.crops))
    models = tuple(parse_csv(args.models))
    seeds = tuple(int(seed) for seed in parse_csv(args.seeds))
    invalid_crops = sorted(set(crops) - set(CROPS))
    invalid_models = sorted(set(models) - set(MODELS))
    if invalid_crops:
        raise SystemExit(f"Unsupported crops: {invalid_crops}")
    if invalid_models:
        raise SystemExit(f"Unsupported models: {invalid_models}")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    rows: list[dict[str, Any]] = []
    for crop in crops:
        for architecture in models:
            for seed in seeds:
                rows.append(
                    run_one(
                        crop,
                        architecture,
                        seed,
                        args.max_epochs,
                        args.patience,
                        args.batch_size,
                        args.learning_rate,
                        args.weight_decay,
                        device,
                        args.force,
                    )
                )
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    summary, completed_runs = rebuild_summary("neural")
    print(
        f"Summary: {summary} ({completed_runs} completed runs)", flush=True
    )


if __name__ == "__main__":
    main()
