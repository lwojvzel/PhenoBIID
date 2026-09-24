"""Recompute the main table from all annual, seed-wise published scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"world": "PhenoBIID", "lightgbm": "LightGBM", "random_forest": "Random Forest"}


def rebuild(folder=None):
    folder = Path(folder or ROOT / "data/reference/annual")
    frame = pd.concat([pd.read_csv(folder / name) for name in ("paired_10_30_50.csv", "paired_70.csv")],
                      ignore_index=True)
    keys = ["crop", "model", "percent", "seed", "year"]
    if frame[keys + ["rmse"]].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Missing or duplicate annual scores")
    if not np.isfinite(frame.rmse).all() or (frame.rmse < 0).any():
        raise ValueError("Invalid RMSE")
    config = json.loads((ROOT / "configs/paper_protocol.json").read_text())
    expected = pd.MultiIndex.from_product([
        config["crops"], list(METHODS), [10, 30, 50, 70], config["seeds"], config["target_years"]
    ], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError("Incomplete or unexpected crop/method/cutoff/seed/year coverage")
    means = frame.groupby(keys[:-1], as_index=False).rmse.mean()
    summary = means.groupby(keys[:3], as_index=False).agg(rmse=("rmse", "mean"), seed_sd=("rmse", "std"))
    summary["method"] = summary.model.map(METHODS)
    summary["unobserved_fraction"] = summary.percent / 100
    return summary[["unobserved_fraction", "method", "crop", "rmse", "seed_sd"]]


def verify(summary):
    expected = pd.read_csv(ROOT / "data/reference/main_yield_rmse.csv")
    keys = ["unobserved_fraction", "method", "crop"]
    compared = summary.merge(expected, on=keys, suffixes=("_recomputed", "_paper"),
                             how="outer", validate="one_to_one", indicator=True)
    if not (compared._merge == "both").all():
        raise ValueError("Main-table conditions differ")
    for metric in ("rmse", "seed_sd"):
        if not np.allclose(compared[f"{metric}_recomputed"], compared[f"{metric}_paper"],
                           atol=0.00005001, rtol=0):
            raise ValueError(f"Main-table {metric} differs beyond four-decimal rounding")
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = rebuild()
    verify(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    print("Verified 48 main-table cells from 1,872 annual scores, including sample SD across seeds.")


if __name__ == "__main__":
    main()
