"""Head-only fits on immutable forward LAI or frozen LAI/NDVI predictions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from biid_world_model import WorldNormalizationStats
from dual_remote_data import load as load_remote
from dual_remote_state import MODES
from forward_protocol_revision import RawInputs, index_hash
from review_revision_data import ROOT, CROPS, sha256
from run_dual_remote_state import assign_states, RESULT as REMOTE_RESULT
from run_history_multimodal_baselines import build_causal_history_features
from run_review_revision_parallel import atomic_json
from run_token_retention_revision import readout
from yield_head_redesign import HEADS, RedesignedYieldHead
from dual_remote_diagnostics import DirectRemoteYield, prepare_direct

RESULT = ROOT / "benchmark/results/yield_head_redesign_v1"
CODE_FILES = ("yield_head_redesign.py", "run_yield_head_redesign.py", "run_token_retention_revision.py",
              "dual_remote_state.py", "dual_remote_data.py", "forward_protocol_revision.py", "biid_world_model.py",
              "run_history_multimodal_baselines.py", "review_revision_data.py", "dual_remote_diagnostics.py",
              "run_input_matched_direct_yield.py", "run_dual_remote_state.py")


def load_forward(crop, seed, origin):
    source = ROOT / "benchmark/results/review_revision_v4/pipelines" / crop / f"origin_{origin}" / f"seed_{seed}"
    if not (source / "complete.json").exists():
        raise FileNotFoundError(f"Incomplete frozen forward source: {source}")
    meta = json.loads((source / "input_manifest.json").read_text())
    stats = WorldNormalizationStats(**meta["normalization"])
    raw = RawInputs(crop)
    history, baseline = build_causal_history_features(raw.cache, stats.target_mean, stats.target_std)
    arrays, hashes = {}, {}
    for split in ("train", "validation", "test"):
        path = source / f"{split}_state_cache.npz"
        with np.load(path, allow_pickle=False) as cache:
            ix = np.searchsorted(raw.source, cache["source_indices"])
            if np.any(ix >= len(raw.source)):
                raise ValueError("Unknown source index")
            np.testing.assert_array_equal(raw.source[ix], cache["source_indices"])
            a = raw.arrays(ix, (stats, history, baseline))
            for key in ("source_indices", "year", "row", "col", "target_lai", "target_lai_valid"):
                np.testing.assert_array_equal(a[key], cache[key])
            a["predicted_lai"] = cache["forward_lai"].copy()
            a["history_base"] = cache["history_base"].copy()
            for key in ("predicted_ndvi", "previous_ndvi", "previous_ndvi_valid", "previous_ndvi_quality", "target_ndvi", "target_ndvi_valid"):
                a[key] = np.zeros_like(a["previous_lai"])
            arrays[split] = a
        hashes[split] = sha256(path)
    lineage = [Path(meta["lineage"]["full"]), *[Path(f["checkpoint"]) for f in meta["lineage"]["forward_folds"]]]
    weights = {str(p / "dynamics_best.pt"): sha256(p / "dynamics_best.pt") for p in lineage}
    meta = {**meta, "ndvi": {"used": False}, "frozen_cache_hashes": hashes,
            "frozen_weight_hashes": weights, "source_directory": str(source),
            "track": "forward_lai", "history_protocol": "shared HGB and forward-only training histories",
            "state_training_protocol": "forward only; no state fitting in this experiment"}
    return arrays, meta


def load_frozen_remote(crop, mode):
    arrays, meta = load_remote(crop, 42, needs_ndvi=mode != "lai")
    source = REMOTE_RESULT / "pipelines" / crop / mode / "seed_42/dynamics"
    if not (source / "metrics.json").exists():
        raise FileNotFoundError(f"Missing frozen remote dynamics: {source}")
    hashes = {}
    for split, a in arrays.items():
        path = source / f"{split}_predictions.npz"
        with np.load(path, allow_pickle=False) as cache:
            for key in ("source_indices", "year", "row", "col"):
                np.testing.assert_array_equal(a[key], cache[key])
            for p in ("lai", "ndvi"):
                a[f"predicted_{p}"] = cache[f"prediction_{p}"] if f"prediction_{p}" in cache else np.zeros_like(a["previous_lai"])
        hashes[split] = sha256(path)
    meta = {**meta, "frozen_cache_hashes": hashes,
            "frozen_weight_hashes": {str(source / "model_best.pt"): sha256(source / "model_best.pt")},
            "source_directory": str(source), "track": "frozen_remote",
            "state_training_protocol": "frozen in-sample train predictions, not forward OOF",
            "state_seed": 42, "history_protocol": "existing crop-specific anchors; not the forward HGB track"}
    return arrays, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--track", choices=("forward_lai", "frozen_remote"), required=True)
    p.add_argument("--mode", choices=MODES, default="lai")
    p.add_argument("--head", choices=HEADS, required=True)
    p.add_argument("--seed", type=int, choices=(42, 45, 48), default=42)
    p.add_argument("--origin", type=int, choices=(2004, 2008, 2012), default=2012)
    p.add_argument("--readout-epochs", type=int, default=30)
    p.add_argument("--readout-patience", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.track == "forward_lai" and args.mode != "lai":
        p.error("The existing forward cache contains LAI only")
    if args.track == "frozen_remote" and (args.seed != 42 or args.origin != 2012):
        p.error("Frozen remote screen reuses the registered seed-42, origin-2012 product cache")
    if min(args.readout_epochs, args.readout_patience) < 1:
        p.error("Positive readout budgets required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_per_process_memory_fraction(4096 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    code_hashes = {name: sha256(ROOT / "scripts" / name) for name in CODE_FILES}
    signature = sha256(Path(__file__))[:10] + sha256(ROOT / "scripts/yield_head_redesign.py")[:10]
    base = RESULT / "smoke" / signature if args.smoke else RESULT
    root = base / "pipelines" / args.track / args.crop / f"origin_{args.origin}" / args.mode / args.head / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    spec = {**vars(args), "code_hashes": code_hashes, "state_fits": 0,
            "loss": "yield MSE + 0.25 candidate MSE; no extra gate BCE or modality dropout for any head",
            "effective_batch": 4096, "micro_batch": 512, "lr": .0003,
            "weight_decay": .0001, "dropout": .1, "interaction_layers": 2,
            "selection": "validation yield RMSE only; all historical evaluations retained"}
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = root / "config.json"
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError("Head experiment specification changed")
        atomic_json(config, spec)
        if (root / "complete.json").exists():
            return
        started = time.monotonic()
        arrays, meta = (load_forward(args.crop, args.seed, args.origin) if args.track == "forward_lai"
                        else load_frozen_remote(args.crop, args.mode))
        if args.smoke:
            arrays = {s: {k: v[:64].copy() for k, v in a.items()} for s, a in arrays.items()}
        meta["sample_hashes"] = {s: index_hash(a["source_indices"]) for s, a in arrays.items()}
        atomic_json(root / "input_manifest.json", meta)
        outputs = {}
        for source in ("predicted", "previous"):
            for a in arrays.values():
                assign_states(a, source)
            for climate in (False, True):
                key = f"{source}__{'full' if climate else 'strict'}"
                out = root / key
                torch.manual_seed(args.seed)
                model = RedesignedYieldHead(args.mode, args.head, climate)
                contract = {"pipeline": spec, "state_source": source, "climate": climate,
                            "manifest_sha256": sha256(root / "input_manifest.json")}
                readout(model, arrays, meta, out, args, contract)
                outputs[key] = str(out)
                del model
                torch.cuda.empty_cache()
        family = args.mode if args.mode in ("lai", "ndvi") else "both"
        direct = {s: prepare_direct(a, family) for s, a in arrays.items()}
        out = base / "direct_controls" / args.track / args.crop / f"origin_{args.origin}" / family / f"seed_{args.seed}"
        # joint/dual use identical past inputs and anchors, so share this fit.
        direct_contract = {"track": args.track, "crop": args.crop, "origin": args.origin, "family": family,
                           "seed": args.seed, "epochs": args.readout_epochs, "patience": args.readout_patience,
                           "code_hashes": code_hashes, "sample_hashes": meta["sample_hashes"],
                           "normalization": meta["normalization"], "architecture": "direct_gru",
                           "history_hashes": {s: hashlib.sha256(np.asarray(a["history_base"], dtype="<f4").tobytes()).hexdigest()
                                              for s, a in arrays.items()}}
        torch.manual_seed(args.seed)
        readout(DirectRemoteYield("gru"), direct, meta, out, args, direct_contract)
        outputs["direct_gru"] = str(out)
        for path, expected in meta["frozen_weight_hashes"].items():
            if sha256(Path(path)) != expected:
                raise RuntimeError("A frozen source checkpoint changed during the head experiment")
        atomic_json(root / "complete.json", {"state_fits": 0, "readouts": outputs,
                                             "source_weights_unchanged": True, "elapsed_seconds": time.monotonic() - started,
                                             "sample_sizes": {s: len(a["target"]) for s, a in arrays.items()}})
        print(f"[HEAD COMPLETE] {root}", flush=True)


if __name__ == "__main__":
    main()
