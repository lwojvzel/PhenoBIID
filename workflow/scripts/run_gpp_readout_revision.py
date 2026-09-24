"""Matched annual-product prior diagnostics, separate from slotwise state learning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from biid_yield_fusion import YieldFusionModel
from review_revision_data import CROPS, ROOT, load_shared, sha256
from multimodal_baseline import save_json, set_seed
from run_interaction_revision import NEW_ROOT, save_yield
from run_review_revision import LABEL_KEYS, RevisionModel, train_neural, train_tree
from prepare_glass_annual_gpp_revision import OUTPUT as GPP_ROOT

SOURCES = ("none", "previous", "previous_no_lai", "observed")


def extract_gpp(arrays):
    previous, current = {}, {}
    for split, a in arrays.items():
        previous[split] = np.full(len(a["year"]), np.nan, dtype=np.float32)
        current[split] = np.full_like(previous[split], np.nan)
        for year in np.unique(a["year"]):
            mask = a["year"] == year
            rows, cols = a["row"][mask], a["col"][mask]
            for offset, output in ((-1, previous), (0, current)):
                grid = np.load(GPP_ROOT / f"gpp_annual_{int(year) + offset}.npy", mmap_mode="r")
                output[split][mask] = grid[rows, cols]
    mean = float(np.nanmean(previous["train"]))
    std = max(float(np.nanstd(previous["train"])), 1e-6)
    def encode(values):
        return {s:np.stack((np.where(np.isfinite(v), (v - mean) / std, 0), np.isfinite(v)), -1).astype(np.float32) for s,v in values.items()}
    metadata = {"training_previous_gpp_mean":mean, "training_previous_gpp_std":std,
                "previous_valid_fraction":{s:float(np.isfinite(v).mean()) for s,v in previous.items()},
                "current_valid_fraction":{s:float(np.isfinite(v).mean()) for s,v in current.items()},
                "normalization":"previous-year GPP from train samples only; missing=0 with indicator",
                "source_manifest_sha256":sha256(GPP_ROOT / "manifest.json")}
    return encode(previous), encode(current), metadata


class GPPFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = YieldFusionModel("biid_query_reliability_climate_coverage_moddrop_gate")
        self.gpp_projection = nn.Linear(2, 128, bias=False)

    def forward(self, b):
        if any(k in b for k in LABEL_KEYS): raise ValueError("Labels must not enter ordinary yield forward")
        return self.network(b["state"], b["relative_valid"], b["history"][:, :15], b["context"],
                            b["history_base"], b["previous_lai"], b["weather"], b["crop_coverage"],
                            extra_context=self.gpp_projection(b["history"][:, 15:]))


def execute(crop, seed, model_kind, smoke=False):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if model_kind != "lightgbm": torch.cuda.set_per_process_memory_fraction(0.16)
    root = NEW_ROOT / ("gpp_smoke" if smoke else "gpp") / crop / model_kind / f"seed_{seed}"
    if (root / "test_metrics.json").exists(): return
    arrays, meta = load_shared(crop, seed)
    previous, current, gpp_meta = extract_gpp(arrays)
    for split, a in arrays.items():
        state_path = NEW_ROOT / "pipelines" / crop / "original" / f"seed_{seed}" / "dynamics" / f"{split}_predictions.npz"
        with np.load(state_path) as data:
            np.testing.assert_array_equal(a["source_indices"], data["source_indices"])
            a["state"] = data["prediction"]
    done = []
    for source in SOURCES:
        destination = root / source
        if (destination / "test_metrics.json").exists(): done.append(source); continue
        data = {s:dict(a) for s,a in arrays.items()}
        for split, a in data.items():
            extra = current[split] if source == "observed" else previous[split]
            if source == "none": extra = np.zeros_like(extra)
            a["history"] = np.concatenate((a["history"], extra), axis=1)
            if source == "previous_no_lai":
                for key in ("previous_lai", "previous_lai_valid", "state"):
                    a[key] = np.zeros_like(a[key])
            if smoke: data[split] = {k:v[:128].copy() for k,v in a.items()}
        set_seed(seed)
        destination.mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(crop=crop, seed=seed, model=model_kind, state=source,
                               climate=True, gate="no_bce", moddrop="none", epochs=1 if smoke else 30,
                               patience=5, batch_size=4096)
        config = {**vars(args), "gpp":gpp_meta, "input_manifest":meta,
                  "target_product_diagnostic_only":source == "observed",
                  "gpp_role":"annual context prior at yield readout; no slotwise GPP evolution or supervision",
                  "yield_model_scope":"global development; not independent confirmation"}
        save_json(config, destination / "config.json")
        start = time.monotonic()
        if model_kind == "lightgbm":
            predictions, training = train_tree(args, data, meta, destination)
        else:
            model = GPPFusion() if model_kind == "fusion" else RevisionModel("gru", 17, 5)
            predictions, training = train_neural(args, data, meta, destination, model_override=model)
            del model; torch.cuda.empty_cache()
        training["elapsed_seconds"] = time.monotonic() - start
        save_yield(predictions, training, data, meta, destination, config)
        done.append(source)
        print(f"[GPP READOUT] {crop}/{model_kind}/{seed}/{source} complete", flush=True)
    save_json({"crop":crop,"seed":seed,"model":model_kind,"completed":done},root / "test_metrics.json")


def tree_worker(crop):
    return subprocess.run([sys.executable, __file__, "--crop", crop, "--seed", "42", "--model", "lightgbm"],
                          env={**os.environ, "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"2"}, check=True).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop",choices=CROPS)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model",choices=("gru","fusion","lightgbm"),default="gru")
    p.add_argument("--all-trees",action="store_true")
    p.add_argument("--smoke",action="store_true")
    a = p.parse_args()
    if a.all_trees:
        with ProcessPoolExecutor(max_workers=4) as executor: list(executor.map(tree_worker,CROPS))
    else:
        if a.crop is None: p.error("--crop required")
        execute(a.crop,a.seed,a.model,a.smoke)


if __name__ == "__main__": main()
