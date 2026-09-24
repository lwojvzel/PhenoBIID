"""Direct tree regression on frozen forecast trajectories, with no learned fusion."""
import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

from forward_protocol_revision import index_hash
from multimodal_baseline import regression_metrics
from review_revision_data import ROOT, CROPS, sha256
from run_probabilistic_yield_heads import memory_guard
from run_review_revision_parallel import atomic_json
from run_yield_head_redesign import load_forward, CODE_FILES as SOURCE_CODE
from simple_concat_yield import INPUTS, VARIANTS, ENGINES, make_features

RESULT = ROOT / "benchmark/results/simple_concat_yield_v1"
CODE_FILES = tuple(dict.fromkeys(("simple_concat_yield.py", "run_simple_concat_yield.py", *SOURCE_CODE)))


class MemoryGuard(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if epoch % 100 == 0:
            memory_guard()
        return False


def build_model(engine, seed, threads, smoke=False):
    trees = 20 if smoke else 1200
    if engine == "hgb":
        return HistGradientBoostingRegressor(loss="squared_error", learning_rate=.05,
            max_iter=20 if smoke else 200, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1., early_stopping=False, random_state=seed)
    if engine == "lightgbm":
        return lgb.LGBMRegressor(objective="regression", n_estimators=trees, learning_rate=.03,
            num_leaves=31, min_child_samples=40, subsample=.9, subsample_freq=1,
            colsample_bytree=.9, reg_lambda=1., random_state=seed, n_jobs=threads,
            verbosity=-1, deterministic=True, force_col_wise=True)
    if engine == "xgboost":
        return xgb.XGBRegressor(objective="reg:squarederror", n_estimators=trees, learning_rate=.03,
            max_depth=6, min_child_weight=8., subsample=.9, colsample_bytree=.9,
            reg_lambda=1., tree_method="hist", device="cuda", early_stopping_rounds=60,
            eval_metric="rmse", random_state=seed, n_jobs=threads, callbacks=[MemoryGuard()])
    raise ValueError(engine)


def fit_model(model, engine, features, targets):
    if engine == "lightgbm":
        model.fit(features["train"], targets["train"], eval_set=[(features["validation"], targets["validation"])],
                  eval_metric="rmse", callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    elif engine == "xgboost":
        memory_guard()
        model.fit(features["train"], targets["train"], eval_set=[(features["validation"], targets["validation"])], verbose=False)
    else:
        model.fit(features["train"], targets["train"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", choices=CROPS, required=True)
    p.add_argument("--engine", choices=ENGINES, required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--origin", type=int, default=2012)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    variants = args.variants.split(",")
    if not set(variants).issubset(VARIANTS) or len(variants) != len(set(variants)) or args.threads < 1:
        p.error("Unknown/duplicate variant or invalid thread count")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = RESULT / ("smoke" if args.smoke else "pipelines") / args.crop / f"origin_{args.origin}" / args.engine / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    spec = {**{k: v for k, v in vars(args).items() if k != "variants"},
            "code_hashes": {f: sha256(ROOT / "scripts" / f) for f in CODE_FILES},
            "versions": {f: importlib.metadata.version(f) for f in ("numpy", "scikit-learn", "lightgbm", "xgboost")},
            "target": "annual yield standardized with actual head-training samples; no trend residual target",
            "head_weather": False, "head_static_context": False, "history_expert_input": False,
            "state_fits": 0, "selection": "validation only; test predictions never used for fitting/early stopping"}
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = root / "config.json"
        if config.exists() and json.loads(config.read_text()) != spec:
            raise RuntimeError(f"Configuration or code changed: {root}")
        atomic_json(config, spec)
        if all((root / v / "metrics.json").exists() for v in variants):
            return
        arrays, meta = load_forward(args.crop, args.seed, args.origin)
        if args.smoke:
            arrays = {s: {k: v[:256 if s == "train" else 128].copy() for k, v in a.items()} for s, a in arrays.items()}
        mean = float(arrays["train"]["target"].astype(np.float64).mean())
        std = max(float(arrays["train"]["target"].astype(np.float64).std()), 1e-6)
        targets = {s: ((a["target"] - mean) / std).astype(np.float32) for s, a in arrays.items()}
        manifest = {**meta, "sample_hashes": {s: index_hash(a["source_indices"]) for s, a in arrays.items()},
                    "head_target_mean": mean, "head_target_std": std,
                    "sample_sizes": {s: len(a["target"]) for s, a in arrays.items()}}
        atomic_json(root / "input_manifest.json", manifest)
        for variant in variants:
            out = root / variant
            out.mkdir(parents=True, exist_ok=True)
            if (out / "metrics.json").exists():
                continue
            features = {}
            for split, a in arrays.items():
                features[split], names = make_features({k: a[k] for k in INPUTS}, variant)
            model = build_model(args.engine, args.seed, args.threads, args.smoke)
            started = time.monotonic()
            fit_model(model, args.engine, features, targets)
            training_seconds = time.monotonic() - started
            if args.engine == "xgboost":
                # CPU prediction avoids a CPU-input/CUDA-model mismatch; trees are unchanged.
                model.set_params(device="cpu", callbacks=None)
                model.save_model(out / "model.ubj")
                weight = out / "model.ubj"
                steps = int(model.best_iteration) + 1
            else:
                weight = out / "model.joblib"
                joblib.dump(model, weight)
                steps = int(model.n_iter_) if args.engine == "hgb" else int(model.best_iteration_ or model.n_estimators)
            scores = {}
            for split in ("validation", "test"):
                a = arrays[split]
                prediction = np.asarray(model.predict(features[split]), dtype=np.float64) * std + mean
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite direct yield predictions")
                stats = meta["normalization"]
                anchor = a["baseline"].astype(np.float64) + a["history_base"] * stats["residual_std"] + stats["residual_mean"]
                scores[split] = {**regression_metrics(a["target"], prediction),
                                 "frozen_history_rmse": regression_metrics(a["target"], anchor)["rmse"]}
                np.savez_compressed(out / f"{split}_predictions.npz", prediction=prediction, frozen_history_prediction=anchor,
                                    **{k: a[k] for k in ("target", "source_indices", "row", "col", "year")})
            atomic_json(out / "metrics.json", dict(scores=scores, feature_names=names, feature_dim=len(names),
                training_seconds=training_seconds, selected_trees=steps, weight=str(weight), weight_sha256=sha256(weight),
                training_device="CUDA" if args.engine == "xgboost" else "CPU", variant=variant))
            print(f"[CONCAT] {args.crop} {args.engine} {variant} features={len(names)} val={scores['validation']['rmse']:.6f} test={scores['test']['rmse']:.6f}", flush=True)
            del model, features
            gc.collect()
        for path, expected in meta["frozen_weight_hashes"].items():
            if sha256(Path(path)) != expected:
                raise RuntimeError("Frozen state model changed")
        if all((root / v / "metrics.json").exists() for v in VARIANTS):
            atomic_json(root / "complete.json", dict(variants=list(VARIANTS), source_weights_unchanged=True, state_fits=0))


if __name__ == "__main__":
    main()
