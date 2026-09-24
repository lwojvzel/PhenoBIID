"""Train only probabilistic yield readouts; never refit upstream forecasts."""
from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from dual_remote_state import YIELD_INPUTS
from forward_protocol_revision import index_hash
from multimodal_baseline import regression_metrics, set_seed, write_csv
from probabilistic_yield_heads import HEADS, CONDITIONS, ProbabilisticYieldHead, fair_crps
from review_revision_data import ROOT, CROPS, sha256
from run_dual_remote_state import loader, assign_states
from run_review_revision_parallel import atomic_json
from run_yield_head_redesign import load_forward, CODE_FILES as SOURCE_CODE

RESULT = ROOT / "benchmark/results/probabilistic_yield_head_v1"
CODE_FILES = tuple(dict.fromkeys(("probabilistic_yield_heads.py", "run_probabilistic_yield_heads.py", *SOURCE_CODE)))


def memory_guard(limit=16000):
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise RuntimeError("Exactly one physical GPU must be declared")
    used = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True).strip())
    if used > limit:
        raise RuntimeError(f"GPU {gpu} exceeds {limit} MiB; stop our job without touching other processes")


def atomic_weights(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, a, meta, residual_scale, args, count, save=None):
    model.eval()
    generator = torch.Generator(device="cuda").manual_seed(args.sample_seed)
    n = len(a["target"])
    stats = meta["normalization"]
    physical_scale = float(stats["residual_std"] * residual_scale)
    history = a["baseline"].astype(np.float64) + a["history_base"] * stats["residual_std"] + stats["residual_mean"]
    keys = (*YIELD_INPUTS, "target_residual")
    output, draws_saved, density = {}, [], []
    started = time.monotonic()
    offset = 0
    for raw in loader(a, keys, args.eval_batch):
        b = {k: t.cuda() for k, t in zip(keys, raw)}
        encoded = model.encoder({k: b[k] for k in YIELD_INPUTS})
        size = len(encoded)
        anchor = torch.as_tensor(history[offset:offset+size], device="cuda", dtype=torch.float32)
        truth = torch.as_tensor(a["target"][offset:offset+size], device="cuda", dtype=torch.float32)
        values = {}
        if model.head == "deterministic":
            values["prediction"] = anchor + physical_scale * model.mean(encoded)
        else:
            samples = model.sample(encoded, count, generator, args.inference_steps)
            physical = anchor[:, None] + physical_scale * samples
            mean = samples.mean(1) if model.head == "diffusion" else model.mean(encoded)
            values["prediction"] = anchor + physical_scale * mean
            values["crps"] = fair_crps(physical, truth)
            values["predictive_std"] = physical.std(1)
            values["mean_mc_se"] = physical.std(1) / np.sqrt(count)
            values["sample_mean_64"] = physical[:, :min(64, count)].mean(1)
            values["sample_mean"] = physical.mean(1)
            values["pit_empirical"] = (physical <= truth[:, None]).float().mean(1)
            values["negative_probability"] = (physical < 0).float().mean(1)
            for level in (.5, .8, .9, .95):
                alpha = 1 - level
                low, high = torch.quantile(physical, torch.tensor([alpha/2, 1-alpha/2], device="cuda"), dim=1)
                name = str(round(100*level))
                values[f"lower_{name}"], values[f"upper_{name}"] = low, high
                values[f"coverage_{name}"] = ((truth >= low) & (truth <= high)).float()
                values[f"width_{name}"] = high - low
                values[f"interval_score_{name}"] = high-low + 2/alpha*((low-truth).clamp_min(0)+(truth-high).clamp_min(0))
            if model.head in ("gaussian", "mdn"):
                residual = (b["target_residual"] - b["history_base"]) / residual_scale
                values["nll"] = -model.distribution(encoded).log_prob(residual) + np.log(physical_scale)
            if save is not None:
                draws_saved.append(physical.cpu().numpy())
        for k, v in values.items():
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"Nonfinite evaluation {k}")
            output.setdefault(k, []).append(v.cpu().numpy())
        offset += size
    assert offset == n
    output = {k: np.concatenate(v) for k, v in output.items()}
    metrics = regression_metrics(a["target"], output["prediction"])
    metrics["history_rmse"] = regression_metrics(a["target"], history)["rmse"]
    metrics.update({k: float(v.astype(np.float64).mean()) for k, v in output.items()
                    if k in ("crps", "nll", "negative_probability", "mean_mc_se") or k.startswith(("coverage_", "width_", "interval_score_"))})
    metrics.update(samples=0 if model.head == "deterministic" else count,
                   seconds=time.monotonic()-started, mean_estimator="Monte Carlo" if model.head == "diffusion" else "analytic",
                   nll_kind="exact" if model.head in ("gaussian", "mdn") else "not available")
    if "sample_mean" in output:
        metrics["sample_mean_64_vs_full_rms"] = float(np.sqrt(np.mean((output["sample_mean_64"]-output["sample_mean"])**2)))
    if save is not None:
        saved = {**output, **{k: a[k] for k in ("target", "source_indices", "row", "col", "year", "crop_coverage")},
                 "history_prediction": history}
        if draws_saved:
            saved["samples"] = np.concatenate(draws_saved)
        np.savez_compressed(save, **saved)
    return metrics


def fit(model, arrays, meta, residual_scale, out, args):
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
    keys = (*YIELD_INPUTS, "target_residual")
    best, best_crps, stale, rows = float("inf"), float("inf"), 0, []
    best_epoch, best_crps_epoch = None, None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        memory_guard()
        model.train()
        total, count, epoch_start = 0., 0, time.monotonic()
        for raw in loader(arrays["train"], keys, args.batch, True):
            full, n = dict(zip(keys, raw)), len(raw[0])
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, n, args.micro_batch):
                b = {k: v[start:start+args.micro_batch].cuda() for k, v in full.items()}
                residual = (b["target_residual"] - b["history_base"]) / residual_scale
                loss = model.loss({k: b[k] for k in YIELD_INPUTS}, residual) * len(residual)/n
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                total += float(loss.detach()) * n
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count += n
        row = dict(epoch=epoch, training_loss=total/count, training_seconds=time.monotonic()-epoch_start)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            scores = evaluate(model, arrays["validation"], meta, residual_scale, args, args.validation_samples)
            row.update(validation_rmse=scores["rmse"], validation_crps=scores.get("crps"), validation_seconds=scores["seconds"])
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if scores["rmse"] < best - 1e-6:
                best, best_epoch, stale = scores["rmse"], epoch, 0
                atomic_weights(state, out / "model_best.pt")
            else:
                stale += 1
            if scores.get("crps", float("inf")) < best_crps:
                best_crps, best_crps_epoch = scores["crps"], epoch
                atomic_weights(state, out / "model_best_crps.pt")
        rows.append(row)
        write_csv(rows, out / "training_history.csv")
        atomic_json(out / "progress.json", {**row, "best_epoch": best_epoch, "updated_unix": time.time()})
        print(f"[PROB] {out} {json.dumps(row)}", flush=True)
        if stale >= args.patience:
            break
    if best_epoch is None:
        raise RuntimeError("No validation checkpoint")
    model.load_state_dict(torch.load(out / "model_best.pt", map_location="cpu", weights_only=True))
    return model, dict(best_epoch=best_epoch, best_validation_rmse_64=best, best_crps_epoch=best_crps_epoch,
                       elapsed_seconds=time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20, epochs_completed=epoch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--head", choices=HEADS, required=True)
    p.add_argument("--conditions", default="predicted,previous,no_remote")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--origin", type=int, default=2012)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--micro-batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--validation-samples", type=int, default=64)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--sample-seed", type=int, default=20260906)
    p.add_argument("--allocator-mib", type=int, default=1024)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    conditions = args.conditions.split(",")
    if len(set(conditions)) != len(conditions) or not all(c in CONDITIONS for c in conditions):
        p.error("Unknown or duplicate condition")
    if min(args.epochs, args.patience, args.eval_every, args.batch, args.micro_batch, args.eval_batch) < 1 or min(args.samples, args.validation_samples) < 2:
        p.error("Positive training budgets and at least two probability samples required")
    memory_guard()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.allocator_mib*2**20/torch.cuda.get_device_properties(0).total_memory)
    hashes = {name: sha256(ROOT / "scripts" / name) for name in CODE_FILES}
    root = RESULT / ("smoke" if args.smoke else "pipelines") / "forward_lai" / args.crop / f"origin_{args.origin}" / args.head / f"seed_{args.seed}"
    spec = {**vars(args), "code_hashes": hashes, "state_fits": 0, "mode": "lai", "path": "full", "precision": "float32",
            "token_dim": 128, "condition_dim": 256, "diffusion_blocks": 4, "diffusion_training_steps": 1000,
            "diffusion_prediction_type": "v_prediction", "diffusion_schedule": "squaredcos_cap_v2",
            "scheduler_spacing": "trailing", "sampler": "DDIM eta=0; random initial noise", "clip_sample": False,
            "selection": "validation RMSE; secondary CRPS checkpoint is not substituted using test results",
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "diffusers", "numpy")}}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = root / "config.json"
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError(f"Configuration or source changed: {root}")
        atomic_json(config, spec)
        if (root / "complete.json").exists():
            return
        arrays, meta = load_forward(args.crop, args.seed, args.origin)
        if args.smoke:
            arrays = {s: {k: v[:128 if s == "train" else 64].copy() for k, v in a.items()} for s, a in arrays.items()}
        for a in arrays.values():
            assign_states(a, "predicted")
            for key in (*YIELD_INPUTS, "target", "target_residual"):
                if not np.isfinite(a[key]).all():
                    raise ValueError(f"Nonfinite frozen input: {key}")
        residual_scale = max(float(np.std(arrays["train"]["target_residual"].astype(np.float64)-arrays["train"]["history_base"])), 1e-6)
        meta = {**meta, "residual_scale_relative_to_target_normalization": residual_scale,
                "sample_hashes": {s: index_hash(a["source_indices"]) for s, a in arrays.items()},
                "sample_sizes": {s: len(a["target"]) for s, a in arrays.items()}}
        atomic_json(root / "input_manifest.json", meta)
        for condition in conditions:
            out = root / condition
            out.mkdir(parents=True, exist_ok=True)
            if (out / "metrics.json").exists():
                continue
            set_seed(args.seed)
            model = ProbabilisticYieldHead(args.head, condition)
            model, training = fit(model, arrays, meta, residual_scale, out, args)
            scores = {s: evaluate(model, arrays[s], meta, residual_scale, args, args.samples, out / f"{s}_predictions.npz")
                      for s in ("validation", "test")}
            secondary = None
            if (out / "model_best_crps.pt").exists():
                model.load_state_dict(torch.load(out / "model_best_crps.pt", map_location="cpu", weights_only=True))
                secondary = evaluate(model, arrays["validation"], meta, residual_scale, args, args.samples)
            training["peak_memory_mib"] = torch.cuda.max_memory_allocated()/2**20
            atomic_json(out / "metrics.json", {"scores": scores, "training": training, "crps_checkpoint_validation": secondary,
                                               "checkpoint_sha256": sha256(out / "model_best.pt"), "condition": condition})
            print(f"[PROB COMPLETE] {out} test_rmse={scores['test']['rmse']:.6f}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for path, expected in meta["frozen_weight_hashes"].items():
            if sha256(Path(path)) != expected:
                raise RuntimeError("Frozen state weights changed")
        atomic_json(root / "complete.json", {"state_fits": 0, "conditions": conditions, "source_weights_unchanged": True})


if __name__ == "__main__":
    main()
