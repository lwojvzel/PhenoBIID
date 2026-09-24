"""Matched readout controls; target LAI is never a yield-forward input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from biid_yield_fusion import YieldFusionModel
from multimodal_baseline import regression_metrics, save_json, set_seed, write_csv
from review_revision_data import CROPS, RESULT_ROOT, SPLITS, load_shared, reliability, sha256
from run_biid_world_model import TensorBatchLoader
from run_input_matched_direct_yield import DirectMLP, DirectGRU, DirectTransformer
from run_review_revision_parallel import experiment_id

MODELS = ("lightgbm", "mlp", "gru", "transformer", "fusion", "multitask_gru", "trajectory_mlp", "trajectory_gru")
INPUT_KEYS = ("weather", "previous_lai", "previous_lai_valid", "relative_valid", "history", "context", "history_base", "crop_coverage", "state", "climatology_lai")
LABEL_KEYS = ("target_residual", "target_lai", "target_lai_valid", "target", "baseline")


def state_input(a, source):
    key = {"predicted": "predicted_lai", "previous": "previous_lai", "climatology": "climatology_lai"}.get(source)
    return np.zeros_like(a["previous_lai"]) if source == "constant" else a[key].copy()


def direct_features(b):
    valid = b["relative_valid"]
    sequence = torch.cat((b["weather"] * valid[..., None], b["previous_lai"][..., None], b["previous_lai_valid"][..., None], valid[..., None]), dim=-1)
    q = b["crop_coverage"]
    r = 0.1 + 0.9 * q / (q + 0.05)
    static = torch.cat((b["history"], b["context"], q[:, None], r[:, None], b["history_base"][:, None]), dim=-1)
    return sequence, static, valid


def apply_state_dropout(b, kind, uniform_probability):
    result = dict(b)
    if kind == "none":
        return result
    q = b["crop_coverage"]
    p = torch.full_like(q, uniform_probability) if kind == "uniform" else 0.5 * (1 - (0.1 + 0.9 * q / (q + 0.05)))
    mask = torch.rand_like(p) < p
    result["state"] = torch.where(mask[:, None], b["previous_lai"], b["state"])
    result["crop_coverage"] = torch.where(mask, torch.zeros_like(q), q)
    return result


class RevisionModel(nn.Module):
    def __init__(self, model, history_size, context_size, climate=False, gate="bce", weather_variables=13):
        super().__init__()
        self.kind, self.gate = model, gate
        static_dim = history_size + context_size + 3
        if model == "fusion":
            variant = "biid_query_reliability_climate_coverage_moddrop_gate" if climate else "biid_query_reliability_coverage_moddrop_gate"
            self.network = YieldFusionModel(variant, dim=128, heads=4, dropout=0.1, weather_variables=weather_variables)
        elif model in ("mlp", "gru", "transformer"):
            self.network = {"mlp": DirectMLP, "gru": DirectGRU, "transformer": DirectTransformer}[model](weather_variables + 3, static_dim, 0.1)
        elif model == "multitask_gru":
            self.encoder = nn.GRU(weather_variables + 3, 128, num_layers=2, dropout=0.1, batch_first=True)
            self.lai_head = nn.Linear(128, 1)
            self.head = nn.Sequential(nn.Linear(128 + static_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        elif model == "trajectory_mlp":
            self.head = nn.Sequential(nn.Linear(12 * 6 + static_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        elif model == "trajectory_gru":
            self.encoder = nn.GRU(6, 128, batch_first=True)
            self.head = nn.Sequential(nn.Linear(128 + static_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        else:
            raise ValueError(model)

    def forward(self, b):
        if any(k in b for k in LABEL_KEYS):
            raise ValueError("Labels must not enter model forward")
        if self.kind == "fusion":
            out = self.network(b["state"], b["relative_valid"], b["history"], b["context"], b["history_base"], b["previous_lai"], b["weather"], b["crop_coverage"])
            if self.gate.startswith("fixed_"):
                alpha = 0.1 if self.gate == "fixed_01" else 0.5
                out["gate"] = torch.full_like(out["gate"], alpha)
                out["prediction"] = b["history_base"] + alpha * (out["candidate"] - b["history_base"])
            return out
        sequence, static, valid = direct_features(b)
        auxiliary = None
        if self.kind in ("mlp", "gru", "transformer"):
            correction = self.network(sequence, static, valid)
        else:
            if self.kind.startswith("trajectory"):
                state = b["state"]
                prev, climo = b["previous_lai"], b["climatology_lai"]
                sequence = torch.stack((state, prev, climo, state - prev, state - climo, valid), dim=-1) * valid[..., None]
            if self.kind == "trajectory_mlp":
                correction = self.head(torch.cat((sequence.flatten(1), static), dim=-1)).squeeze(-1)
            else:
                tokens, _ = self.encoder(sequence)
                lengths = valid.sum(1).long().clamp_min(1)
                pooled = tokens[torch.arange(tokens.shape[0], device=tokens.device), lengths - 1]
                correction = self.head(torch.cat((pooled, static), dim=-1)).squeeze(-1)
                if self.kind == "multitask_gru":
                    auxiliary = self.lai_head(tokens).squeeze(-1)
        prediction = b["history_base"] + correction
        return {"prediction": prediction, "candidate": prediction, "gate": torch.ones_like(prediction), "auxiliary_lai": auxiliary}


def make_batches(a, batch_size, shuffle):
    keys = (*INPUT_KEYS, *LABEL_KEYS)
    tensors = [torch.from_numpy(np.asarray(a[key], dtype=np.float32)) for key in keys]
    return TensorBatchLoader(tensors, batch_size, shuffle)


def split_batch(raw, device):
    batch = {key: value.to(device) for key, value in zip((*INPUT_KEYS, *LABEL_KEYS), raw)}
    return {k: batch[k] for k in INPUT_KEYS}, {k: batch[k] for k in LABEL_KEYS}


@torch.no_grad()
def predict(model, loader, stats, device):
    model.eval()
    collected = {k: [] for k in ("prediction", "candidate", "gate")}
    for raw in loader:
        inputs, labels = split_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inputs)
        for key in collected:
            value = out[key].float()
            if key != "gate":
                value = labels["baseline"] + value * stats["residual_std"] + stats["residual_mean"]
            collected[key].append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def rmse(target, pred):
    return float(np.sqrt(np.mean((target.astype(np.float64) - pred.astype(np.float64)) ** 2)))


def train_neural(args, arrays, meta, destination, model_override=None):
    device = torch.device("cuda")
    model = model_override if model_override is not None else RevisionModel(args.model, arrays["train"]["history"].shape[1], arrays["train"]["context"].shape[1], args.climate, args.gate, arrays["train"]["weather"].shape[-1])
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()
    loaders = {s: make_batches(a, getattr(args, "batch_size", 4096), s == "train") for s, a in arrays.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    stats = meta["normalization"]
    best, best_epoch, stale, rows = float("inf"), -1, 0, []
    state = None
    accumulation = getattr(args, "gradient_accumulation", 1)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        total_loss, count = 0.0, 0
        for step, raw in enumerate(loaders["train"]):
            inputs, labels = split_batch(raw, device)
            if args.model == "fusion":
                inputs = apply_state_dropout(inputs, args.moddrop, meta["uniform_drop_probability"])
            if step % accumulation == 0: optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inputs)
                target = labels["target_residual"]
                loss = nn.functional.mse_loss(out["prediction"], target)
                if args.model == "fusion":
                    loss = loss + 0.25 * nn.functional.mse_loss(out["candidate"], target)
                    if args.gate == "bce":
                        better = (abs(out["candidate"].detach() - target) < abs(inputs["history_base"] - target)).float()
                        with torch.autocast("cuda", enabled=False):
                            loss = loss + 0.1 * nn.functional.binary_cross_entropy(out["gate"].float().clamp(1e-5, 1-1e-5), better)
                if args.model == "multitask_gru":
                    mask = labels["target_lai_valid"]
                    loss = loss + ((out["auxiliary_lai"] - labels["target_lai"]) ** 2 * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            loader = loaders["train"]
            window_start = (step // accumulation) * accumulation * loader.batch_size
            window_size = min(accumulation * loader.batch_size, loader.size - window_start)
            scaler.scale(loss * (len(target) / window_size)).backward()
            if (step + 1) % accumulation == 0 or (step + 1) * loader.batch_size >= loader.size:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach()) * len(target)
            count += len(target)
        val = predict(model, loaders["validation"], stats, device)
        score = rmse(arrays["validation"]["target"], val["prediction"])
        rows.append(dict(epoch=epoch, train_loss=total_loss/count, validation_rmse=score, seconds=time.monotonic()-start))
        print(f"[TRAIN] {args.crop}/{args.model}/{args.state} seed={args.seed} epoch={epoch} val={score:.6f} sec={rows[-1]['seconds']:.1f}", flush=True)
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError("No finite selected checkpoint")
    model.load_state_dict(state)
    torch.save(state, destination / "model_best.pt")
    write_csv(rows, destination / "training_history.csv")
    predictions = {s: predict(model, loaders[s], stats, device) for s in ("validation", "test")}
    return predictions, dict(best_epoch=best_epoch, validation_rmse=best, parameters=sum(p.numel() for p in model.parameters()), peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)


def tree_features(a):
    inputs = {key: torch.from_numpy(a[key]) for key in INPUT_KEYS}
    sequence, static, _ = direct_features(inputs)
    return np.concatenate((sequence.numpy().reshape(len(sequence), -1), static.numpy()), axis=1)


def train_tree(args, arrays, meta, destination):
    matrices = {s: tree_features(a) for s, a in arrays.items()}
    model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40, colsample_bytree=0.9, reg_lambda=1, random_state=42, n_jobs=2, verbosity=-1)
    target = {s: a["target_residual"] - a["history_base"] for s, a in arrays.items()}
    model.fit(matrices["train"], target["train"], eval_set=[(matrices["validation"], target["validation"])], callbacks=[lgb.early_stopping(40, verbose=False)])
    predictions = {}
    stats = meta["normalization"]
    for s in ("validation", "test"):
        prediction = arrays[s]["baseline"] + (arrays[s]["history_base"] + model.predict(matrices[s])) * stats["residual_std"] + stats["residual_mean"]
        predictions[s] = dict(prediction=prediction.astype(np.float32), candidate=prediction.astype(np.float32), gate=np.ones(len(prediction), dtype=np.float32))
    joblib.dump(model, destination / "model_best.joblib")
    return predictions, dict(best_iteration=int(model.best_iteration_), validation_rmse=rmse(arrays["validation"]["target"], predictions["validation"]["prediction"]), parameters=None, peak_memory_mib=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--state", choices=("predicted", "previous", "climatology", "constant"), default="predicted")
    p.add_argument("--climate", action="store_true")
    p.add_argument("--moddrop", choices=("none", "coverage", "uniform", "shuffled"), default="coverage")
    p.add_argument("--gate", choices=("bce", "no_bce", "fixed_01", "fixed_05"), default="bce")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    key = experiment_id(vars(args))
    root = RESULT_ROOT / "smoke" if args.smoke else RESULT_ROOT
    destination = root / args.crop / key / f"seed_{args.seed}"
    if (destination / "test_metrics.json").exists():
        existing = json.loads((destination / "config.json").read_text())
        if existing["epochs"] != args.epochs or existing["patience"] != args.patience:
            raise RuntimeError("Completed run has a different training budget")
        print(f"[EXISTS] {destination}")
        return
    arrays, meta = load_shared(args.crop, args.seed)
    for s, a in arrays.items():
        a["state"] = state_input(a, args.state)
        if args.model == "fusion" and args.moddrop == "shuffled":
            a["crop_coverage"] = a["shuffled_coverage"].copy()
        if args.smoke:
            arrays[s] = {k: v[:128].copy() for k, v in a.items()}
    destination.mkdir(parents=True, exist_ok=True)
    save_json({**vars(args), "input_manifest": meta, "implementation_sha256": sha256(Path(__file__)), "same_anchor_for_all_methods": True, "yield_forward_target_lai": False, "auxiliary_lai_weight": 1.0 if args.model == "multitask_gru" else 0.0, "learning_rate": 3e-4, "batch_size": 4096, "dropout": 0.1, "selection": "2012 validation only"}, destination / "config.json")
    start = time.monotonic()
    predictions, training = train_tree(args, arrays, meta, destination) if args.model == "lightgbm" else train_neural(args, arrays, meta, destination)
    stats = meta["normalization"]
    summary = {}
    for s, values in predictions.items():
        a = arrays[s]
        history = a["baseline"] + a["history_base"] * stats["residual_std"] + stats["residual_mean"]
        saved = {**values, **{k: a[k] for k in ("target", "source_indices", "row", "col", "year", "crop_coverage")}, "history_prediction": history}
        np.savez_compressed(destination / f"{s}_predictions.npz", **saved)
        metrics = regression_metrics(a["target"], values["prediction"])
        h = rmse(a["target"], history)
        better = abs(values["candidate"] - a["target"]) < abs(history - a["target"])
        metrics.update(history_rmse=h, gain_over_history_percent=100*(h-metrics["rmse"])/h, gate_mean=float(values["gate"].mean()), gate_auc=float(roc_auc_score(better, values["gate"])) if len(np.unique(better)) == 2 else None, gate_brier=float(np.mean((values["gate"]-better)**2)))
        summary[s] = metrics
    result = {"crop": args.crop, "seed": args.seed, "experiment": key, **training, "elapsed_seconds": time.monotonic()-start, "metrics": summary}
    temporary = destination / "test_metrics.tmp.json"
    save_json(result, temporary)
    temporary.replace(destination / "test_metrics.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
