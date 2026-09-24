#!/usr/bin/env python3
"""Rebuild yield-only summary CSVs from every completed run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmark/results/yield_only_baselines"
CROPS = ("maize", "rice", "soybean", "wheat")
CLASSICAL_METHODS = (
    "global_mean",
    "grid_climatology",
    "previous_year",
    "rolling_mean_3",
    "rolling_mean_5",
    "exponential_smoothing",
    "linear_trend",
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "huber",
    "linear_svr",
    "approximate_knn",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
)
NEURAL_METHODS = ("mlp", "rnn", "lstm", "gru", "tcn", "transformer")


def _seed_sort_key(seed: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(seed):012d}")
    except (TypeError, ValueError):
        return (1, str(seed))


def collect_rows(methods: Iterable[str]) -> list[dict[str, Any]]:
    allowed = set(methods)
    rows: list[dict[str, Any]] = []
    for crop in CROPS:
        crop_root = RESULTS_ROOT / crop
        if not crop_root.exists():
            continue
        for method_root in sorted(path for path in crop_root.iterdir() if path.is_dir()):
            method = method_root.name
            if method not in allowed:
                continue
            for run_dir in sorted(path for path in method_root.iterdir() if path.is_dir()):
                metrics_path = run_dir / "test_metrics.json"
                config_path = run_dir / "config.json"
                predictions_path = run_dir / "test_predictions.npz"
                if not (metrics_path.exists() and config_path.exists() and predictions_path.exists()):
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
                seed = config.get("seed", run_dir.name.removeprefix("seed_"))
                rows.append({"crop": crop, "method": method, "seed": seed, **metrics})
    return sorted(
        rows,
        key=lambda row: (
            CROPS.index(str(row["crop"])),
            str(row["method"]),
            _seed_sort_key(row["seed"]),
        ),
    )


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise RuntimeError(f"No completed runs found for {output.name}")
    preferred = ["crop", "method", "seed"]
    remaining = sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred + remaining)
        writer.writeheader()
        writer.writerows(rows)


def rebuild_summary(kind: str) -> tuple[Path, int]:
    if kind == "classical":
        methods = CLASSICAL_METHODS
        filename = "classical_baselines.csv"
    elif kind == "neural":
        methods = NEURAL_METHODS
        filename = "neural_baselines.csv"
    else:
        raise ValueError(f"Unsupported summary kind: {kind}")
    rows = collect_rows(methods)
    output = RESULTS_ROOT / "summary" / filename
    write_rows(rows, output)
    return output, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("classical", "neural", "all"), default="all"
    )
    args = parser.parse_args()
    kinds = ("classical", "neural") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        output, count = rebuild_summary(kind)
        print(f"{kind}: {count} completed runs -> {output}")


if __name__ == "__main__":
    main()
