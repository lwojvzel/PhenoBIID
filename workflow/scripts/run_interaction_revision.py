"""Predeclared interaction replacements and LAI input diagnostics."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from biid_world_model import WorldNormalizationStats, lai_metrics
from interaction_revision import KINDS, InteractionDynamics, replace_interactions
from review_revision_data import CROPS, ROOT, RESULT_ROOT, load_shared, sha256
from multimodal_baseline import save_json, set_seed, write_csv, regression_metrics
from run_biid_world_model import make_loader, train_dynamics, predict_dynamics
from run_review_revision import RevisionModel, train_neural, rmse

NEW_ROOT = ROOT / "benchmark/results/review_revision_v3"
INPUT_MODES = ("no_lai", "previous_lai", "observed_lai")
READOUT_ONLY = tuple("readout_only_" + kind for kind in KINDS[1:])


def save_yield(predictions, training, arrays, meta, destination, config):
    scores = {}
    stats = meta["normalization"]
    for split, values in predictions.items():
        a = arrays[split]
        anchor = a["baseline"] + a["history_base"] * stats["residual_std"] + stats["residual_mean"]
        np.savez_compressed(destination / f"{split}_predictions.npz", **values,
                            **{k: a[k] for k in ("target", "source_indices", "row", "col", "year")},
                            history_prediction=anchor)
        m = regression_metrics(a["target"], values["prediction"])
        m["history_rmse"] = rmse(a["target"], anchor)
        m["gain_over_history_percent"] = 100 * (1 - m["rmse"] / m["history_rmse"])
        scores[split] = m
    save_json({**config, **training, "metrics": scores}, destination / "test_metrics.json")


def state_predictions(kind, arrays, meta, destination, args):
    stats = WorldNormalizationStats(**meta["normalization"])
    marker = destination / "test_metrics.json"
    if marker.exists():
        predictions = {}
        for split, a in arrays.items():
            with np.load(destination / f"{split}_predictions.npz") as data:
                np.testing.assert_array_equal(a["source_indices"], data["source_indices"])
                predictions[split] = data["prediction"]
        return predictions
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = InteractionDynamics(kind).to(device)
    loaders = {s: make_loader(a, 1024, s == "train", True) for s, a in arrays.items()}
    provenance = {"kind": kind, "crop": args.crop, "seed": args.seed,
                  "epochs": args.epochs, "patience": args.patience, "batch_size": 1024,
                  "micro_batch_size": 256, "gradient_accumulation": "exact masked-loss mass; effective batch 1024",
                  "lr": 3e-4, "weight_decay": 1e-4, "input_manifest": meta,
                  "module_sha256": sha256(Path(__file__).with_name("interaction_revision.py"))}
    start = time.monotonic()
    if kind == "original" and not args.smoke:
        original = RESULT_ROOT / "dynamics_controls" / args.crop / "full" / f"seed_{args.seed}"
        detail = json.loads((original / "test_metrics.json").read_text())
        assert args.epochs == 40 and args.patience == 6
        weights = torch.load(original / "dynamics_best.pt", map_location="cpu", weights_only=True)
        epoch, score = detail["best_epoch"], detail["validation_selection_rmse"]
        provenance["reused_checkpoint"] = str(original / "dynamics_best.pt")
        provenance["reused_sha256"] = sha256(original / "dynamics_best.pt")
        provenance["micro_batch_size"] = 1024
        provenance["gradient_accumulation"] = "none in reused original fit; effective batch 1024"
    else:
        weights, history, epoch, score = train_dynamics(
            model, loaders["train"], loaders["validation"], arrays["validation"],
            stats, device, args.epochs, args.patience, 3e-4, 1e-4, micro_batch_size=256)
        write_csv(history, destination / "training_history.csv")
    model.load_state_dict(weights)
    torch.save(weights, destination / "dynamics_best.pt")
    save_json(provenance, destination / "config.json")
    predictions, metrics = {}, {}
    for split, a in arrays.items():
        # Training predictions must preserve input order, unlike the training loader.
        pred = predict_dynamics(model, make_loader(a, 1024, False, True), "biid_climate", device)
        if not np.isfinite(pred).all(): raise FloatingPointError("Nonfinite state prediction")
        predictions[split] = pred
        np.savez_compressed(destination / f"{split}_predictions.npz", prediction=pred,
                            target=a["target_lai"], valid=a["target_lai_valid"],
                            **{k: a[k] for k in ("source_indices", "year", "row", "col")})
        if split != "train":
            metrics[split] = lai_metrics(a["target_lai"], pred, a["target_lai_valid"], a["relative_weight"], stats)
    save_json({"kind": kind, "crop": args.crop, "seed": args.seed,
               "best_epoch": epoch, "validation_rmse": score, "metrics": metrics,
               "parameters": sum(p.numel() for p in model.parameters()),
               "elapsed_seconds": time.monotonic() - start}, marker)
    del model, weights, loaders
    torch.cuda.empty_cache()
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=CROPS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kind", choices=KINDS + INPUT_MODES + READOUT_ONLY, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); set_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(0.16)
    root = NEW_ROOT / ("smoke" if args.smoke else "pipelines") / args.crop / args.kind / f"seed_{args.seed}"
    if (root / "test_metrics.json").exists(): return
    arrays, meta = load_shared(args.crop, args.seed)
    if args.smoke:
        arrays = {s: {k: v[:args.smoke_samples].copy() for k, v in a.items()} for s, a in arrays.items()}
    root.mkdir(parents=True, exist_ok=True)
    if args.kind in INPUT_MODES:
        for a in arrays.values():
            a["state"] = a["previous_lai"].copy()
            if args.kind == "no_lai":
                a["previous_lai"] = np.zeros_like(a["previous_lai"])
                a["previous_lai_valid"] = np.zeros_like(a["previous_lai_valid"])
            elif args.kind == "observed_lai":
                a["previous_lai"] = a["target_lai"].copy()
                a["previous_lai_valid"] = a["target_lai_valid"].copy()
        specs = [("gru", "original", False)]
    elif args.kind in READOUT_ONLY:
        interaction = args.kind.removeprefix("readout_only_")
        for split, a in arrays.items():
            cached = NEW_ROOT / "pipelines" / args.crop / "original" / f"seed_{args.seed}" / "dynamics" / f"{split}_predictions.npz"
            with np.load(cached) as data:
                indices = data["source_indices"][:len(a["source_indices"])] if args.smoke else data["source_indices"]
                np.testing.assert_array_equal(a["source_indices"], indices)
                a["state"] = data["prediction"][:len(indices)]
        specs = [("fusion", interaction, climate) for climate in (False, True)]
    else:
        predictions = state_predictions(args.kind, arrays, meta, root / "dynamics", args)
        for split, a in arrays.items(): a["state"] = predictions[split]
        interactions = ("original",) if args.kind == "original" else ("original", args.kind)
        specs = [("fusion", interaction, climate) for interaction in interactions for climate in (False, True)]
    summaries = {}
    for model_kind, interaction, climate in specs:
        key = f"{model_kind}__{interaction}__{'climate' if climate else 'state_only'}"
        destination = root / key
        if (destination / "test_metrics.json").exists():
            summaries[key] = json.loads((destination / "test_metrics.json").read_text()); continue
        set_seed(args.seed)
        destination.mkdir(parents=True, exist_ok=True)
        options = SimpleNamespace(crop=args.crop, seed=args.seed, model=model_kind,
                                  state=args.kind, climate=climate, gate="no_bce", moddrop="none",
                                  epochs=1 if args.smoke else 30, patience=5,
                                  batch_size=1024 if interaction != "original" else 4096,
                                  gradient_accumulation=4 if interaction != "original" else 1)
        model = RevisionModel(model_kind, 15, 5, climate, "no_bce")
        if model_kind == "fusion": replace_interactions(model.network, interaction)
        config = {**vars(options), "interaction": interaction,
                  "input_manifest": meta, "observed_target_lai_diagnostic": args.kind == "observed_lai",
                  "selection": "2012 validation; development-only global evaluation",
                  "training_states": "frozen in-sample, not OOF", "pipeline_kind": args.kind}
        config["state_transition_kind"] = "original" if args.kind in READOUT_ONLY else args.kind
        save_json(config, destination / "config.json")
        start = time.monotonic()
        values, training = train_neural(options, arrays, meta, destination, model_override=model)
        training["elapsed_seconds"] = time.monotonic() - start
        save_yield(values, training, arrays, meta, destination, config)
        summaries[key] = json.loads((destination / "test_metrics.json").read_text())
        del model; torch.cuda.empty_cache()
    save_json({"crop": args.crop, "seed": args.seed, "kind": args.kind,
               "completed_readouts": list(summaries)}, root / "test_metrics.json")


if __name__ == "__main__": main()
