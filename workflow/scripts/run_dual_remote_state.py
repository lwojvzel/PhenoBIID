"""Two-product rollout and matched predicted/previous-state yield readouts."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from dual_remote_data import load
from dual_remote_state import DualRemoteDynamics, DualRemoteYield, INPUTS, YIELD_INPUTS, MODES
from multimodal_baseline import regression_metrics, set_seed, write_csv
from prepare_pku_ndvi import save_json, digest
from review_revision_data import ROOT, CROPS
from run_biid_world_model import TensorBatchLoader

RESULT = ROOT / "benchmark/results/dual_remote_state_v1"
STATE_LABELS = ("target_lai", "target_lai_valid", "target_ndvi", "target_ndvi_valid", "relative_weight")
YIELD_LABELS = ("target_residual", "baseline")


def readout_sources(mode):
    return ("predicted", "previous", "mixed_lai", "mixed_ndvi") if mode in ("joint", "dual") else ("predicted", "previous")


def assign_states(a, source):
    for product in ("lai", "ndvi"):
        field = source if source in ("predicted", "previous") else ("predicted" if source == f"mixed_{product}" else "previous")
        a[f"state_{product}"] = a[f"{field}_{product}"]


def loader(a, keys, batch, shuffle=False):
    return TensorBatchLoader([torch.from_numpy(np.asarray(a[k], dtype=np.float32)) for k in keys], batch, shuffle)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, np.float64) - np.asarray(p, np.float64))**2)))


@torch.no_grad()
def predict(model, a, kind):
    model.eval()
    keys = INPUTS if kind == "state" else YIELD_INPUTS
    result = {}
    for raw in loader(a, keys, 512):
        b = {k: t.cuda() for k, t in zip(keys, raw)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        for key, value in out.items():
            result.setdefault(key, []).append(value.float().cpu().numpy())
    joined = {k: np.concatenate(v) for k, v in result.items()}
    if not all(np.isfinite(v).all() for v in joined.values()):
        raise FloatingPointError("Nonfinite prediction")
    return joined


def state_metrics(a, predictions, meta):
    result = {}
    for p, prediction in predictions.items():
        mask = a[f"target_{p}_valid"] > 0
        scale = meta["normalization"]["lai_std"] if p == "lai" else meta["ndvi"]["normalization"]["std"]
        result[p] = dict(normalized_rmse=rmse(a[f"target_{p}"][mask], prediction[mask]),
                         physical_rmse=rmse(a[f"target_{p}"][mask], prediction[mask]) * scale,
                         valid_slots=int(mask.sum()))
    return result


def fit(model, arrays, meta, destination, args, kind):
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
    inputs = INPUTS if kind == "state" else YIELD_INPUTS
    labels = STATE_LABELS if kind == "state" else YIELD_LABELS
    keys = (*inputs, *labels)
    batch = 1024 if kind == "state" else 4096
    micro = (256 if args.mode == "lai" else 512) if kind == "state" else 1024
    training = loader(arrays["train"], keys, batch, True)
    best, stale, best_epoch, state, rows = float("inf"), 0, 0, None, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.monotonic()
        total, count = 0., 0
        for raw in training:
            full = dict(zip(keys, raw))
            n = len(raw[0])
            optimizer.zero_grad(set_to_none=True)
            if kind == "state":
                masses = {p: (full[f"target_{p}_valid"] * full["relative_weight"]).sum().cuda() for p in model.products}
                n_products = sum(float(mass) > 0 for mass in masses.values())
            for start in range(0, n, micro):
                b = {k: v[start:start+micro].cuda() for k, v in full.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model({k: b[k] for k in inputs})
                    if kind == "state":
                        loss = sum(((out[p].float() - b[f"target_{p}"])**2 * b[f"target_{p}_valid"] * b["relative_weight"]).sum() / masses[p].clamp_min(1e-8) for p in model.products) / max(n_products, 1)
                    else:
                        loss = (nn.functional.mse_loss(out["prediction"].float(), b["target_residual"]) + .25 * nn.functional.mse_loss(out["candidate"].float(), b["target_residual"])) * (len(b[inputs[0]]) / n)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                total += float(loss.detach()) * n
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            count += n
        val = predict(model, arrays["validation"], kind)
        if kind == "state":
            metrics = state_metrics(arrays["validation"], val, meta)
            score = float(np.mean([m["normalized_rmse"] for m in metrics.values()]))
        else:
            score = rmse(arrays["validation"]["target_residual"], val["prediction"])
        row = dict(epoch=epoch, training_loss=total/count, validation_selection=score, seconds=time.monotonic()-epoch_start)
        rows.append(row)
        print(f"[{kind}] {args.crop}/{args.mode}/{destination.name} epoch={epoch} val={score:.6f} sec={row['seconds']:.1f}", flush=True)
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        write_csv(rows, destination / "training_history.csv")
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError("No selected checkpoint")
    torch.save(state, destination / "model_best.pt")
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, validation_selection=best,
                       elapsed_seconds=time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                       effective_batch=batch, micro_batch=micro)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.16)
    set_seed(args.seed)
    root = RESULT / ("smoke" if args.smoke else "pipelines") / args.crop / args.mode / f"seed_{args.seed}"
    marker = root / "complete.json"
    if marker.exists():
        existing = json.loads((root / "config.json").read_text())
        if existing["epochs"] != args.epochs or existing["patience"] != args.patience:
            raise RuntimeError("Existing run has a different training budget")
        return
    arrays, meta = load(args.crop, args.seed, needs_ndvi=args.mode != "lai")
    if args.smoke:
        arrays = {s: {k: v[:64].copy() for k, v in a.items()} for s, a in arrays.items()}
    root.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "input_manifest": meta,
              "selection": "2012 validation; average normalized state RMSE, then yield RMSE",
              "training_states": "frozen in-sample; not OOF", "target_observations_in_forward": False,
              "state_products": [args.mode] if args.mode in ("lai", "ndvi") else ["lai", "ndvi"],
              "implementation_hashes": {n: digest(ROOT / "scripts" / n) for n in ("dual_remote_state.py", "dual_remote_data.py", "run_dual_remote_state.py")}}
    save_json(root / "config.json", config)
    destination = root / "dynamics"
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "metrics.json").exists():
        model, training = fit(DualRemoteDynamics(args.mode), arrays, meta, destination, args, "state")
        scores = {}
        for split, a in arrays.items():
            predictions = predict(model, a, "state")
            saved = {f"prediction_{k}": v for k, v in predictions.items()}
            saved.update({k: a[k] for k in ("source_indices", "row", "col", "year")})
            np.savez_compressed(destination / f"{split}_predictions.npz", **saved)
            scores[split] = state_metrics(a, predictions, meta)
        save_json(destination / "metrics.json", {**training, "scores": scores})
        del model
        torch.cuda.empty_cache()
    for split, a in arrays.items():
        with np.load(destination / f"{split}_predictions.npz") as d:
            np.testing.assert_array_equal(a["source_indices"], d["source_indices"])
            for product in ("lai", "ndvi"):
                a[f"predicted_{product}"] = d[f"prediction_{product}"] if f"prediction_{product}" in d else np.zeros_like(a["previous_lai"])
    summaries = {}
    for source in readout_sources(args.mode):
        for climate in (False, True):
            key = f"{source}__{'climate' if climate else 'strict'}"
            destination = root / key
            destination.mkdir(parents=True, exist_ok=True)
            if (destination / "metrics.json").exists():
                continue
            for a in arrays.values():
                assign_states(a, source)
            set_seed(args.seed)
            model, training = fit(DualRemoteYield(args.mode, climate), arrays, meta, destination, args, "yield")
            scores = {}
            for split in ("validation", "test"):
                a = arrays[split]
                values = predict(model, a, "yield")
                stats = meta["normalization"]
                for k in ("prediction", "candidate"):
                    values[k] = a["baseline"] + values[k] * stats["residual_std"] + stats["residual_mean"]
                history = a["baseline"] + a["history_base"] * stats["residual_std"] + stats["residual_mean"]
                np.savez_compressed(destination / f"{split}_predictions.npz", **values, history_prediction=history,
                                    **{k: a[k] for k in ("target", "source_indices", "row", "col", "year", "crop_coverage")})
                scores[split] = {**regression_metrics(a["target"], values["prediction"]),
                                 "history_rmse": rmse(a["target"], history)}
            save_json(destination / "metrics.json", {**training, "scores": scores,
                      "state_source": source, "terminal_climate": climate,
                      "state_weight_sha256": digest(root / "dynamics/model_best.pt") if source != "previous" else None})
            summaries[key] = scores
            del model
            torch.cuda.empty_cache()
    save_json(marker, {"complete": True, "crop": args.crop, "mode": args.mode,
                       "seed": args.seed, "state_fits": 1, "readout_fits": 2*len(readout_sources(args.mode))})


if __name__ == "__main__":
    main()
