"""Forward-only state/history caches and paired readouts for review revision v4.

This diagnostic deliberately uses a common HGB anchor for all crops. It is not
a numerical rerun of the older maize-MLP-anchor main table.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from biid_world_model import CropWorldDynamics, WorldNormalizationStats, load_world_cache, make_world_arrays
from multimodal_baseline import load_cache, regression_metrics, save_json, set_seed, write_csv
from review_revision_data import CROPS, ROOT, load_shared, reliability, sha256, training_climatology
from run_biid_world_model import make_loader, predict_dynamics, train_dynamics
from run_crossfit_readout_revision import expert
from run_history_multimodal_baselines import build_causal_history_features
from run_review_revision import train_neural

OUTPUT = ROOT / "benchmark/results/review_revision_v4"
CONDITIONS = (
    ("strict_in_sample", "fusion", "in_sample", False),
    ("strict_forward", "fusion", "forward", False),
    ("strict_previous", "fusion", "previous", False),
    ("strict_climatology", "fusion", "climatology", False),
    ("full_in_sample", "fusion", "in_sample", True),
    ("full_forward", "fusion", "forward", True),
    ("full_previous", "fusion", "previous", True),
    ("direct_gru", "gru", "previous", True),
)


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def forward_windows(origin):
    return [(start - 2, start - 1, start, start + 3)
            for start in range(origin - 12, origin, 4)]


def split_indices(years, origin):
    return {"train": np.flatnonzero(years < origin),
            "validation": np.flatnonzero(years == origin),
            "test": np.flatnonzero((years > origin) & (years <= origin + 4))}


def safe_std(values):
    return max(float(np.std(values, dtype=np.float64)), 1e-6)


class RawInputs:
    def __init__(self, crop):
        self.crop = crop
        self.cache, self.world = load_cache(crop), load_world_cache(crop)
        self.source = np.asarray(self.world["source_indices"])
        self.years = np.asarray(self.cache["year"])[self.source]
        # Coverage is deterministic source metadata; do not reuse fitted states,
        # history predictions or normalization from the old revision cache.
        shared, metadata = load_shared(crop, 42)
        ids = np.concatenate([a["source_indices"] for a in shared.values()])
        q = np.concatenate([a["crop_coverage"] for a in shared.values()])
        order = np.argsort(ids)
        if len(np.unique(ids)) != len(ids) or not np.array_equal(ids[order], self.source):
            raise ValueError("Coverage/sample universe differs from raw world cache")
        self.coverage = q[order]
        self.source_manifest = {"coverage_cache": metadata["sample_index_hash"],
                                "world_source_indices": index_hash(self.source)}

    def normalization(self, indices):
        total = np.zeros(13, dtype=np.float64)
        square = total.copy()
        count = total.copy()
        lai_total = lai_square = 0.0
        lai_count = 0
        for start in range(0, len(indices), 10000):
            ix = indices[start:start + 10000]
            valid = np.asarray(self.world["relative_valid"][ix], dtype=bool)
            weather = np.asarray(self.world["weather_rel"][ix], dtype=np.float64)
            mask = valid[..., None] & np.isfinite(weather)
            total += np.where(mask, weather, 0).sum((0, 1))
            square += np.where(mask, weather * weather, 0).sum((0, 1))
            count += mask.sum((0, 1))
            lai = np.asarray(self.world["target_lai_rel"][ix], dtype=np.float64)
            values = lai[valid & np.isfinite(lai)]
            lai_total += values.sum()
            lai_square += np.square(values).sum()
            lai_count += len(values)
        if not lai_count or np.any(count == 0):
            raise ValueError("Missing training observations in normalization")
        wm = total / count
        ws = np.sqrt(np.maximum(square / count - wm * wm, 1e-12))
        lm = float(lai_total / lai_count)
        ls = float(np.sqrt(max(lai_square / lai_count - lm * lm, 1e-12)))
        target = np.asarray(self.cache["target"])[self.source[indices]].astype(np.float64)
        mean, std = float(target.mean()), safe_std(target)
        history, baseline = build_causal_history_features(self.cache, mean, std)
        residual = target - baseline[self.source[indices]]
        stats = WorldNormalizationStats(wm.tolist(), ws.tolist(), lm, ls, mean, std,
                                        float(residual.mean()), safe_std(residual))
        return stats, history, baseline

    def arrays(self, indices, fitted):
        stats, history, baseline = fitted
        a = make_world_arrays(self.cache, self.world, stats, indices)
        source = a["source_indices"]
        a["history"] = history[source].copy()
        a["baseline"] = baseline[source].copy()
        a["target_residual"] = ((a["target"] - a["baseline"] - stats.residual_mean)
                                / stats.residual_std).astype(np.float32)
        for key in ("year", "row", "col"):
            a[key] = np.asarray(self.cache[key])[source].copy()
        a["crop_coverage"] = self.coverage[indices].copy()
        return a


def model_contract(raw, fit, validation, seed, epochs, patience, batch_size):
    return {"schema": 1, "crop": raw.crop, "seed": seed,
            "fit_hash": index_hash(raw.source[fit]),
            "validation_hash": index_hash(raw.source[validation]),
            "fit_years": [int(raw.years[fit].min()), int(raw.years[fit].max())],
            "validation_years": np.unique(raw.years[validation]).tolist(),
            "epochs": epochs, "patience": patience, "batch_size": batch_size,
            "code_hashes": {name: sha256(ROOT / "scripts" / name) for name in
                            ("forward_protocol_revision.py", "biid_world_model.py", "run_biid_world_model.py")}}


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def fit_upstream(raw, fit_end, val_year, seed, args, root):
    fit = np.flatnonzero(raw.years <= fit_end)
    validation = np.flatnonzero(raw.years == val_year)
    if not len(fit) or not len(validation) or fit_end >= val_year:
        raise ValueError("Invalid forward fit/validation boundary")
    if args.smoke:
        fit, validation = fit[:128], validation[:64]
    fitted = raw.normalization(fit)
    destination = root / "upstream" / raw.crop / f"seed_{seed}" / f"fit_{fit_end}_val_{val_year}"
    destination.mkdir(parents=True, exist_ok=True)
    contract = model_contract(raw, fit, validation, seed, args.state_epochs, args.state_patience, args.state_batch)
    with (destination / "fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = destination / "complete.json"
        if marker.exists():
            if json.loads(marker.read_text())["contract"] != contract:
                raise RuntimeError(f"Upstream provenance mismatch: {destination}")
        else:
            started = time.monotonic()
            set_seed(seed)
            train, val = raw.arrays(fit, fitted), raw.arrays(validation, fitted)
            model = CropWorldDynamics("biid_climate").to("cuda")
            state, trace, epoch, score = train_dynamics(
                model, make_loader(train, args.state_batch, True, False),
                make_loader(val, args.state_batch, False, False), val, fitted[0],
                torch.device("cuda"), args.state_epochs, args.state_patience, 3e-4, 1e-4)
            torch.save(state, destination / "dynamics_best.pt")
            write_csv(trace, destination / "training_history.csv")
            hgb = expert()
            hgb.fit(np.concatenate((train["history"], train["context"]), axis=1), train["target_residual"])
            joblib.dump(hgb, destination / "history.joblib")
            save_json({"contract": contract, "normalization": asdict(fitted[0]),
                       "selected_epoch": epoch, "validation_lai_rmse": score,
                       "state_sha256": sha256(destination / "dynamics_best.pt"),
                       "history_sha256": sha256(destination / "history.joblib"),
                       "elapsed_seconds": time.monotonic() - started}, marker)
            del model, state, train, val, hgb
            gc.collect()
            torch.cuda.empty_cache()
    return destination, fitted, fit


def upstream_predict(raw, indices, destination, fitted, batch_size):
    a = raw.arrays(indices, fitted)
    model = CropWorldDynamics("biid_climate").to("cuda")
    model.load_state_dict(load_torch(destination / "dynamics_best.pt"))
    pred = predict_dynamics(model, make_loader(a, batch_size, False, False),
                            "biid_climate", torch.device("cuda"))
    stats = fitted[0]
    pred = pred * stats.lai_std + stats.lai_mean
    hgb = joblib.load(destination / "history.joblib")
    history = hgb.predict(np.concatenate((a["history"], a["context"]), axis=1))
    history = a["baseline"] + history * stats.residual_std + stats.residual_mean
    del model, a, hgb
    gc.collect()
    torch.cuda.empty_cache()
    return pred.astype(np.float32), history.astype(np.float32)


def normalized_state(physical, stats, valid):
    return (((physical - stats.lai_mean) / stats.lai_std) * valid).astype(np.float32)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_mib * 2**20 / total)
    root = OUTPUT / "smoke" / sha256(Path(__file__))[:10] if args.smoke else OUTPUT
    destination = root / "pipelines" / args.crop / f"origin_{args.origin}" / f"seed_{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    spec = {**vars(args), "implementation_sha256": sha256(Path(__file__)),
            "conditions": [list(c) for c in CONDITIONS], "anchor": "fixed HGB for every crop",
            "readout_years": [args.origin - 12, args.origin - 1],
            "interpretation": "retrospective robustness and workflow control, not an untouched test"}
    with (destination / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = destination / "config.json"
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError("Existing pipeline has a different specification")
        save_json(spec, config)
        if (destination / "complete.json").exists():
            return
        started = time.monotonic()
        raw = RawInputs(args.crop)
        indices = split_indices(raw.years, args.origin)
        indices["train"] = np.flatnonzero((raw.years >= args.origin - 12) & (raw.years < args.origin))
        if args.smoke:
            # Keep rows from every forward window so all boundary code is tested.
            indices["train"] = np.concatenate([np.flatnonzero((raw.years >= s) & (raw.years <= e))[:32]
                                                for _, _, s, e in forward_windows(args.origin)])
            indices["validation"] = indices["validation"][:64]
            indices["test"] = indices["test"][:64]
        final_dir, fitted, final_fit = fit_upstream(raw, args.origin - 1, args.origin, args.seed, args, root)
        arrays = {s: raw.arrays(ix, fitted) for s, ix in indices.items()}
        stats = fitted[0]
        lineage = {"full": str(final_dir), "forward_folds": []}
        for split, ix in indices.items():
            p, h = upstream_predict(raw, ix, final_dir, fitted, args.state_batch)
            arrays[split]["in_sample_lai"] = normalized_state(p, stats, arrays[split]["relative_valid"])
            arrays[split]["forward_lai"] = arrays[split]["in_sample_lai"].copy()
            arrays[split]["history_base"] = ((h - arrays[split]["baseline"] - stats.residual_mean)
                                               / stats.residual_std).astype(np.float32)
        full_climo = training_climatology({"train": raw.arrays(final_fit, fitted),
                                           "validation": arrays["validation"], "test": arrays["test"]})
        for split in ("validation", "test"):
            arrays[split]["climatology_lai"] = full_climo[split]
        del full_climo
        train = arrays["train"]
        train["climatology_lai"] = np.full_like(train["previous_lai"], np.nan)
        covered = np.zeros(len(indices["train"]), dtype=bool)
        for fit_end, val_year, start, end in forward_windows(args.origin):
            positions = np.flatnonzero((train["year"] >= start) & (train["year"] <= end))
            if not len(positions):
                raise ValueError("Empty forward prediction block")
            held = indices["train"][positions]
            assert fit_end < val_year < int(raw.years[held].min())
            fold_dir, fold_fitted, fold_fit = fit_upstream(raw, fit_end, val_year, args.seed, args, root)
            p, h = upstream_predict(raw, held, fold_dir, fold_fitted, args.state_batch)
            train["forward_lai"][positions] = normalized_state(p, stats, train["relative_valid"][positions])
            train["history_base"][positions] = (h - train["baseline"][positions] - stats.residual_mean) / stats.residual_std
            climo = training_climatology({"train": raw.arrays(fold_fit, fold_fitted),
                                         "held": raw.arrays(held, fold_fitted)})["held"]
            physical = climo * fold_fitted[0].lai_std + fold_fitted[0].lai_mean
            train["climatology_lai"][positions] = normalized_state(physical, stats, train["relative_valid"][positions])
            covered[positions] = True
            lineage["forward_folds"].append({"checkpoint": str(fold_dir), "fit_end": fit_end,
                                            "validation_year": val_year, "prediction_years": [start, end],
                                            "predicted_samples": len(positions), "source_hash": index_hash(raw.source[held])})
        if not covered.all() or not np.isfinite(train["climatology_lai"]).all():
            raise RuntimeError("Incomplete forward cache")
        meta = {"normalization": asdict(stats), "lineage": lineage,
                "source_manifest": raw.source_manifest,
                "sample_hashes": {s: index_hash(a["source_indices"]) for s, a in arrays.items()},
                "uniform_drop_probability": float(np.mean(.5 * (1 - reliability(train["crop_coverage"]))))}
        save_json(meta, destination / "input_manifest.json")
        state_rows = []
        for split, a in arrays.items():
            np.savez_compressed(destination / f"{split}_state_cache.npz", **{k: a[k] for k in
                                ("source_indices", "year", "row", "col", "target_lai", "target_lai_valid",
                                 "in_sample_lai", "forward_lai", "climatology_lai", "history_base")})
            for name in ("in_sample_lai", "forward_lai", "previous_lai", "climatology_lai"):
                mask = a["target_lai_valid"] > 0
                error = (a[name] - a["target_lai"])[mask].astype(np.float64) * stats.lai_std
                state_rows.append({"split": split, "state": name, "rmse": float(np.sqrt(np.mean(error**2))),
                                   "valid_values": int(mask.sum())})
        write_csv(state_rows, destination / "state_errors.csv")
        del raw
        gc.collect()
        for name, model, source, climate in CONDITIONS:
            out = destination / name
            if (out / "test_metrics.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            key = {"in_sample": "in_sample_lai", "forward": "forward_lai",
                   "previous": "previous_lai", "climatology": "climatology_lai"}[source]
            for a in arrays.values():
                a["state"] = a[key]
            set_seed(args.seed)
            settings = SimpleNamespace(crop=args.crop, seed=args.seed, model=model, state=source, climate=climate,
                                       gate="bce", moddrop="coverage", epochs=args.readout_epochs,
                                       patience=args.readout_patience, batch_size=4096)
            save_json({**vars(settings), "source": key, "input_manifest": str(destination / "input_manifest.json")}, out / "config.json")
            predictions, training = train_neural(settings, arrays, meta, out)
            metrics = {}
            for split, p in predictions.items():
                a = arrays[split]
                h = a["baseline"] + a["history_base"] * stats.residual_std + stats.residual_mean
                np.savez_compressed(out / f"{split}_predictions.npz", **p, history_prediction=h,
                                    **{k: a[k] for k in ("target", "source_indices", "year", "row", "col", "crop_coverage")})
                metrics[split] = regression_metrics(a["target"], p["prediction"])
                metrics[split]["history_rmse"] = regression_metrics(a["target"], h)["rmse"]
            save_json({"crop": args.crop, "seed": args.seed, "origin": args.origin, "condition": name,
                       "metrics": metrics, "training": training}, out / "test_metrics.json")
            gc.collect()
            torch.cuda.empty_cache()
        save_json({"conditions": [c[0] for c in CONDITIONS], "elapsed_seconds": time.monotonic() - started,
                   "sample_sizes": {s: len(a["target"]) for s, a in arrays.items()}}, destination / "complete.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--seed", type=int, choices=(42, 45, 48), required=True)
    p.add_argument("--origin", type=int, choices=(2004, 2008, 2012), required=True)
    p.add_argument("--state-epochs", type=int, default=40)
    p.add_argument("--state-patience", type=int, default=6)
    p.add_argument("--state-batch", type=int, default=1024)
    p.add_argument("--readout-epochs", type=int, default=30)
    p.add_argument("--readout-patience", type=int, default=5)
    p.add_argument("--memory-mib", type=int, default=4096)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
