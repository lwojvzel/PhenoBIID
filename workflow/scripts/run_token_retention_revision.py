"""Forward-only LAI/NDVI token-bottleneck experiments, isolated from old runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from torch import nn

from dual_remote_data import extract
from dual_remote_diagnostics import DirectRemoteYield, prepare_direct
from dual_remote_state import DualRemoteYield, INPUTS, YIELD_INPUTS, MODES
from forward_protocol_revision import RawInputs, forward_windows, split_indices, index_hash
from multimodal_baseline import regression_metrics, set_seed, write_csv
from prepare_pku_ndvi import OUT as NDVI_ROOT
from review_revision_data import ROOT, CROPS, sha256
from run_crossfit_readout_revision import expert
from run_dual_remote_state import loader, state_metrics, rmse, assign_states, STATE_LABELS, YIELD_LABELS
from run_review_revision_parallel import atomic_json
from token_retention_state import RetainedDynamics, STRATEGIES, state_lengths

RESULT = ROOT / "benchmark/results/token_retention_v1"
CODE_FILES = ("token_retention_state.py", "run_token_retention_revision.py", "dual_remote_state.py",
              "dual_remote_data.py", "dual_remote_diagnostics.py", "forward_protocol_revision.py",
              "biid_world_model.py", "review_revision_data.py", "run_dual_remote_state.py",
              "run_input_matched_direct_yield.py", "run_crossfit_readout_revision.py")


class RemoteInputs(RawInputs):
    def __init__(self, crop):
        super().__init__(crop)
        root = RESULT / "raw_cache" / crop
        root.mkdir(parents=True, exist_ok=True)
        expected = {"source_hash": index_hash(self.source),
                    "ndvi_manifest": sha256(NDVI_ROOT / "manifest.json"),
                    "extraction_code": sha256(ROOT / "scripts/dual_remote_data.py")}
        with (root / "cache.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = root / "complete.json"
            if marker.exists():
                if json.loads(marker.read_text())["contract"] != expected:
                    raise RuntimeError("Raw NDVI cache provenance changed")
            else:
                a = {"year": self.years,
                     "row": np.asarray(self.cache["row"])[self.source],
                     "col": np.asarray(self.cache["col"])[self.source],
                     "source_month": np.asarray(self.world["source_month"]),
                     "relative_valid": np.asarray(self.world["relative_valid"])}
                target, _ = extract(a)
                previous, quality = extract(a, previous=True)
                for name, array in (("target", target), ("previous", previous), ("quality", quality)):
                    np.save(root / f"{name}.npy", array)
                atomic_json(marker, {"contract": expected,
                                     "sha256": {n: sha256(root / f"{n}.npy") for n in ("target", "previous", "quality")}})
        self.ndvi = {n: np.load(root / f"{n}.npy", mmap_mode="r") for n in ("target", "previous", "quality")}
        self.source_manifest["ndvi_raw_cache"] = sha256(root / "complete.json")

    def normalization(self, indices):
        values = np.asarray(self.ndvi["target"][indices], np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError("No training NDVI observations")
        mean, std = float(values[finite].mean()), max(float(values[finite].std()), 1e-6)
        return (*super().normalization(indices), {"mean": mean, "std": std})

    def arrays(self, indices, fitted):
        a = super().arrays(indices, fitted[:3])
        stats = fitted[3]
        for period in ("target", "previous"):
            values = np.asarray(self.ndvi[period][indices])
            valid = np.isfinite(values) & a["relative_valid"].astype(bool)
            a[f"{period}_ndvi_valid"] = valid.astype(np.float32)
            a[f"{period}_ndvi"] = np.where(valid, (values - stats["mean"]) / stats["std"], 0).astype(np.float32)
        a["previous_ndvi_quality"] = np.where(a["previous_ndvi_valid"] > 0, self.ndvi["quality"][indices], 0).astype(np.float32)
        return a


def metadata(fitted):
    return {"normalization": asdict(fitted[0]), "ndvi": {"normalization": fitted[3]}}


@torch.no_grad()
def predict(model, arrays, kind, batch=512):
    model.eval()
    keys = INPUTS if kind == "state" else YIELD_INPUTS
    result = {}
    for raw in loader(arrays, keys, batch):
        b = {k: t.cuda() for k, t in zip(keys, raw)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        for key, value in out.items():
            result.setdefault(key, []).append(value.float().cpu().numpy())
    joined = {k: np.concatenate(v) for k, v in result.items()}
    if not all(np.isfinite(v).all() for v in joined.values()):
        raise FloatingPointError("Nonfinite predictions")
    return joined


def fit(model, arrays, meta, destination, args, kind):
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
    inputs = INPUTS if kind == "state" else YIELD_INPUTS
    labels = STATE_LABELS if kind == "state" else YIELD_LABELS
    keys = (*inputs, *labels)
    batch = 1024 if kind == "state" else 4096
    micro = args.state_micro if kind == "state" else 512
    epochs = args.state_epochs if kind == "state" else args.readout_epochs
    patience = args.state_patience if kind == "state" else args.readout_patience
    best, stale, best_epoch, state, rows = float("inf"), 0, 0, None, []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count, epoch_start = 0., 0, time.monotonic()
        for raw in loader(arrays["train"], keys, batch, True):
            full, n = dict(zip(keys, raw)), len(raw[0])
            optimizer.zero_grad(set_to_none=True)
            if kind == "state":
                masses = {p: (full[f"target_{p}_valid"] * full["relative_weight"]).sum().cuda() for p in model.products}
                n_products = sum(float(mass) > 0 for mass in masses.values())
            for start in range(0, n, micro):
                b = {k: v[start:start + micro].cuda() for k, v in full.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model({k: b[k] for k in inputs})
                    if kind == "state":
                        loss = sum(((out[p].float() - b[f"target_{p}"])**2 * b[f"target_{p}_valid"] * b["relative_weight"]).sum() / masses[p].clamp_min(1e-8) for p in model.products) / max(n_products, 1)
                    else:
                        loss = (nn.functional.mse_loss(out["prediction"].float(), b["target_residual"]) + .25 * nn.functional.mse_loss(out["candidate"].float(), b["target_residual"])) * (len(b[inputs[0]]) / n)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                total += float(loss.detach()) * n
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            count += n
        values = predict(model, arrays["validation"], kind)
        score = (float(np.mean([v["normalized_rmse"] for v in state_metrics(arrays["validation"], values, meta).values()]))
                 if kind == "state" else rmse(arrays["validation"]["target_residual"], values["prediction"]))
        rows.append(dict(epoch=epoch, training_loss=total / count, validation_selection=score,
                         seconds=time.monotonic() - epoch_start))
        print(f"[{kind}] {destination} epoch={epoch} validation={score:.6f} seconds={rows[-1]['seconds']:.1f}", flush=True)
        write_csv(rows, destination / "training_history.csv")
        if score < best - 1e-6:
            best, stale, best_epoch = score, 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if state is None:
        raise RuntimeError("No valid checkpoint selected")
    torch.save(state, destination / "model_best.pt")
    model.load_state_dict(state)
    training = dict(best_epoch=best_epoch, validation_selection=best, elapsed_seconds=time.monotonic() - started,
                    parameters=sum(p.numel() for p in model.parameters()), effective_batch=batch, micro_batch=micro,
                    peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    return model, training


def model_for(args):
    return RetainedDynamics(args.mode, args.strategy, biid_scale=args.biid_scale)


def prepare_upstream(raw, fit_end, val_year, args, root):
    fit_ix, val_ix = np.flatnonzero(raw.years <= fit_end), np.flatnonzero(raw.years == val_year)
    if not len(fit_ix) or not len(val_ix) or fit_end >= val_year:
        raise ValueError("Invalid upstream temporal boundary")
    if args.smoke:
        fit_ix, val_ix = fit_ix[:128], val_ix[:64]
    fitted = raw.normalization(fit_ix)
    destination = root / "upstream" / f"fit_{fit_end}_val_{val_year}"
    destination.mkdir(parents=True, exist_ok=True)
    expected = {"fit_hash": index_hash(raw.source[fit_ix]), "val_hash": index_hash(raw.source[val_ix]),
                "fit_years": [int(raw.years[fit_ix].min()), fit_end], "validation_year": val_year,
                "normalization": metadata(fitted)}
    marker = destination / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text())["contract"] != expected:
            raise RuntimeError("Upstream data contract mismatch")
    else:
        set_seed(args.seed)
        arrays = {"train": raw.arrays(fit_ix, fitted), "validation": raw.arrays(val_ix, fitted)}
        model, training = fit(model_for(args), arrays, metadata(fitted), destination, args, "state")
        scores = state_metrics(arrays["validation"], predict(model, arrays["validation"], "state"), metadata(fitted))
        atomic_json(marker, {"contract": expected, "training": training, "validation": scores,
                             "checkpoint_sha256": sha256(destination / "model_best.pt")})
        del model, arrays
        gc.collect()
        torch.cuda.empty_cache()
    return destination, fitted, fit_ix


def predict_physical(raw, indices, fitted, destination, args):
    model = model_for(args).cuda()
    model.load_state_dict(torch.load(destination / "model_best.pt", map_location="cpu", weights_only=True))
    a = raw.arrays(indices, fitted)
    values = predict(model, a, "state")
    for p in values:
        mean, std = ((fitted[0].lai_mean, fitted[0].lai_std) if p == "lai" else (fitted[3]["mean"], fitted[3]["std"]))
        values[p] = values[p] * std + mean
    del model, a
    gc.collect()
    torch.cuda.empty_cache()
    return values


def attach_states(a, values, fitted):
    for p in ("lai", "ndvi"):
        mean, std = ((fitted[0].lai_mean, fitted[0].lai_std) if p == "lai" else (fitted[3]["mean"], fitted[3]["std"]))
        a[f"predicted_{p}"] = (((values[p] - mean) / std) * a["relative_valid"]).astype(np.float32) if p in values else np.zeros_like(a["previous_lai"])


def history_checkpoint(raw, fitted, fit_ix, fit_end, val_year, args, root):
    # History contains no state features, so all token/product variants share it.
    shared = root / "history" / args.crop / f"origin_{args.origin}" / f"seed_{args.seed}" / f"fit_{fit_end}_val_{val_year}"
    shared.mkdir(parents=True, exist_ok=True)
    expected = {"fit_hash": index_hash(raw.source[fit_ix]), "normalization": asdict(fitted[0]),
                "source_manifest": raw.source_manifest, "expert_code": sha256(ROOT / "scripts/run_crossfit_readout_revision.py")}
    with (shared / "fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = shared / "complete.json"
        if marker.exists():
            if json.loads(marker.read_text())["contract"] != expected:
                raise RuntimeError("History expert provenance mismatch")
        else:
            a = raw.arrays(fit_ix, fitted)
            hgb = expert()
            hgb.fit(np.concatenate((a["history"], a["context"]), 1), a["target_residual"])
            joblib.dump(hgb, shared / "history.joblib")
            atomic_json(marker, {"contract": expected, "sha256": sha256(shared / "history.joblib")})
    return shared / "history.joblib"


def history_physical(raw, indices, fitted, checkpoint):
    a = raw.arrays(indices, fitted)
    hgb = joblib.load(checkpoint)
    value = hgb.predict(np.concatenate((a["history"], a["context"]), 1))
    return (a["baseline"] + value * fitted[0].residual_std + fitted[0].residual_mean).astype(np.float32)


def readout(model, arrays, meta, out, args, contract):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = out / "config.json"
        if config.exists() and json.loads(config.read_text()) != contract:
            raise RuntimeError(f"Readout provenance mismatch: {out}")
        atomic_json(config, contract)
        if (out / "metrics.json").exists():
            return
        set_seed(args.seed)
        model, training = fit(model, arrays, meta, out, args, "yield")
        scores = {}
        stats = meta["normalization"]
        for split in ("validation", "test"):
            a = arrays[split]
            values = predict(model, a, "yield")
            for key in ("prediction", "candidate"):
                values[key] = a["baseline"] + values[key] * stats["residual_std"] + stats["residual_mean"]
            history = a["baseline"] + a["history_base"] * stats["residual_std"] + stats["residual_mean"]
            np.savez_compressed(out / f"{split}_predictions.npz", **values, history_prediction=history,
                                **{k: a[k] for k in ("target", "source_indices", "row", "col", "year", "crop_coverage")})
            scores[split] = {**regression_metrics(a["target"], values["prediction"]), "history_rmse": rmse(a["target"], history)}
        atomic_json(out / "metrics.json", {"training": training, "scores": scores,
                                           "checkpoint_sha256": sha256(out / "model_best.pt")})
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_mib * 2**20 / total)
    code_hashes = {name: sha256(ROOT / "scripts" / name) for name in CODE_FILES}
    signature = sha256(Path(__file__))[:10] + sha256(ROOT / "scripts/token_retention_state.py")[:10]
    base = RESULT / "smoke" / signature if args.smoke else RESULT
    root = base / "pipelines" / args.crop / f"origin_{args.origin}" / args.mode / args.strategy / args.biid_scale / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    spec = {**vars(args), "code_hashes": code_hashes, "state_lengths": state_lengths(args.mode, args.strategy),
            "training_states": "three forward-only four-year blocks; upstream normalization refitted per block",
            "history": "shared HGB; forward-only training predictions", "target_observations_in_forward": False,
            "selection": "validation only; already seen historical test years are retrospective robustness",
            "yield_head": "unchanged DualRemoteYield; scalar LAI/NDVI trajectories, not latent trajectories"}
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = root / "config.json"
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError("Pipeline specification changed; use a new versioned result root")
        atomic_json(config, spec)
        if (root / "complete.json").exists():
            return
        started = time.monotonic()
        raw = RemoteInputs(args.crop)
        indices = split_indices(raw.years, args.origin)
        indices["train"] = np.flatnonzero((raw.years >= args.origin - 12) & (raw.years < args.origin))
        if args.smoke:
            indices["train"] = np.concatenate([np.flatnonzero((raw.years >= s) & (raw.years <= e))[:32]
                                                for _, _, s, e in forward_windows(args.origin)])
            indices["validation"], indices["test"] = indices["validation"][:64], indices["test"][:64]
        final_dir, fitted, fit_ix = prepare_upstream(raw, args.origin - 1, args.origin, args, root)
        arrays = {s: raw.arrays(ix, fitted) for s, ix in indices.items()}
        hpath = history_checkpoint(raw, fitted, fit_ix, args.origin - 1, args.origin, args, base)
        lineage = {"outer": str(final_dir), "outer_history": str(hpath), "folds": []}
        for split, ix in indices.items():
            a = arrays[split]
            if split == "train":
                for p in ("lai", "ndvi"):
                    a[f"predicted_{p}"] = np.full_like(a["previous_lai"], np.nan)
                a["history_base"] = np.full(len(ix), np.nan, np.float32)
            else:
                attach_states(a, predict_physical(raw, ix, fitted, final_dir, args), fitted)
                history = history_physical(raw, ix, fitted, hpath)
                a["history_base"] = ((history - a["baseline"] - fitted[0].residual_mean) / fitted[0].residual_std).astype(np.float32)
        train = arrays["train"]
        covered = np.zeros(len(indices["train"]), bool)
        for fit_end, val_year, start, end in forward_windows(args.origin):
            positions = np.flatnonzero((train["year"] >= start) & (train["year"] <= end))
            held = indices["train"][positions]
            if not len(held) or not fit_end < val_year < int(raw.years[held].min()):
                raise ValueError("Invalid forward block")
            fold_dir, fold_fit, fit_ix = prepare_upstream(raw, fit_end, val_year, args, root)
            values = predict_physical(raw, held, fold_fit, fold_dir, args)
            temporary = {"previous_lai": train["previous_lai"][positions], "relative_valid": train["relative_valid"][positions]}
            attach_states(temporary, values, fitted)
            for p in ("lai", "ndvi"):
                train[f"predicted_{p}"][positions] = temporary[f"predicted_{p}"]
            hp = history_checkpoint(raw, fold_fit, fit_ix, fit_end, val_year, args, base)
            physical_history = history_physical(raw, held, fold_fit, hp)
            train["history_base"][positions] = (physical_history - train["baseline"][positions] - fitted[0].residual_mean) / fitted[0].residual_std
            covered[positions] = True
            lineage["folds"].append(dict(fit_end=fit_end, validation_year=val_year, prediction_years=[start, end],
                                          sample_hash=index_hash(raw.source[held]), checkpoint=str(fold_dir), history=str(hp)))
        if not covered.all() or not all(np.isfinite(train[k]).all() for k in ("predicted_lai", "predicted_ndvi", "history_base")):
            raise RuntimeError("Incomplete forward cache")
        meta = {**metadata(fitted), "lineage": lineage, "source_manifest": raw.source_manifest,
                "sample_hashes": {s: index_hash(a["source_indices"]) for s, a in arrays.items()}}
        atomic_json(root / "input_manifest.json", meta)
        products = (args.mode,) if args.mode in ("lai", "ndvi") else ("lai", "ndvi")
        state_scores = {}
        for split, a in arrays.items():
            np.savez_compressed(root / f"{split}_state_cache.npz", **{k: a[k] for k in
                                ("predicted_lai", "predicted_ndvi", "target_lai", "target_lai_valid", "target_ndvi",
                                 "target_ndvi_valid", "previous_lai", "previous_ndvi", "source_indices", "year", "row", "col", "history_base")})
            state_scores[split] = {source: state_metrics(a, {p: a[f"{source}_{p}"] for p in products}, meta)
                                   for source in ("predicted", "previous")}
        atomic_json(root / "state_metrics.json", state_scores)
        del raw
        gc.collect()
        references = {}
        family = args.mode if args.mode in ("lai", "ndvi") else "both"
        common = base / "readout_controls" / args.crop / f"origin_{args.origin}" / family / f"seed_{args.seed}"
        shared_contract = {"sample_hashes": meta["sample_hashes"], "normalization": meta["normalization"],
                           "ndvi": meta["ndvi"], "family": family, "seed": args.seed,
                           "epochs": args.readout_epochs, "patience": args.readout_patience,
                           "history": {"outer": lineage["outer_history"], "folds": [f["history"] for f in lineage["folds"]]},
                           "code_hashes": code_hashes}
        for source in ("previous", "predicted"):
            for a in arrays.values():
                assign_states(a, source)
            for climate in (False, True):
                key = f"{source}__{'full' if climate else 'strict'}"
                out = common / key if source == "previous" else root / key
                contract = {**shared_contract, "source": source, "climate": climate}
                if source == "predicted":
                    contract["state_spec"] = spec
                    contract["state_manifest_sha256"] = sha256(root / "input_manifest.json")
                set_seed(args.seed)
                readout(DualRemoteYield(args.mode, climate), arrays, meta, out, args, contract)
                references[key] = str(out)
        direct = {s: prepare_direct(a, family) for s, a in arrays.items()}
        set_seed(args.seed)
        out = common / "direct_gru"
        readout(DirectRemoteYield("gru"), direct, meta, out, args, {**shared_contract, "source": "previous", "architecture": "direct_gru"})
        references["direct_gru"] = str(out)
        atomic_json(root / "complete.json", {"elapsed_seconds": time.monotonic() - started,
                                             "readouts": references, "state_fits": 4,
                                             "sample_sizes": {s: len(a["target"]) for s, a in arrays.items()}})
        print(f"[COMPLETE] {root}", flush=True)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--strategy", choices=STRATEGIES, required=True)
    p.add_argument("--biid-scale", choices=("original", "length_matched"), default="original")
    p.add_argument("--seed", type=int, choices=(42, 45, 48), default=42)
    p.add_argument("--origin", type=int, choices=(2004, 2008, 2012), default=2012)
    p.add_argument("--state-epochs", type=int, default=30)
    p.add_argument("--state-patience", type=int, default=5)
    p.add_argument("--state-micro", type=int, default=128)
    p.add_argument("--readout-epochs", type=int, default=30)
    p.add_argument("--readout-patience", type=int, default=5)
    p.add_argument("--memory-mib", type=int, default=4096)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if min(a.state_micro, a.state_epochs, a.readout_epochs, a.state_patience, a.readout_patience) < 1:
        p.error("Training budgets must be positive")
    return a


if __name__ == "__main__":
    run(arguments())
