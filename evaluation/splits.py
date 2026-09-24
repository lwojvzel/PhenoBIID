"""Chronological sample partitions for the three published temporal blocks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = Path(__file__).resolve().parents[1] / "configs/paper_protocol.json"


def partition(years, cutoff, protocol=None):
    protocol = protocol or json.loads(PROTOCOL.read_text())
    if str(cutoff) not in protocol["temporal_blocks"]:
        raise ValueError(f"Unknown refit cutoff: {cutoff}")
    years = np.asarray(years, dtype=float)
    if years.ndim != 1 or not np.isfinite(years).all() or (years != np.floor(years)).any():
        raise ValueError("Expected a finite one-dimensional array of integer years")
    return {
        "inner_train": years <= cutoff - 2,
        "inner_validation": (years > cutoff - 2) & (years <= cutoff),
        "final_train": years <= cutoff,
        "evaluation": np.isin(years, protocol["temporal_blocks"][str(cutoff)]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="CSV containing year; row order is retained")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.samples)
    config = json.loads(PROTOCOL.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for cutoff in map(int, config["temporal_blocks"]):
        masks = partition(frame.year.to_numpy(), cutoff, config)
        np.savez_compressed(args.output / f"cutoff_{cutoff}.npz",
                            **{name: np.flatnonzero(mask) for name, mask in masks.items()})
    print("Saved positional sample indices for all three temporal blocks.")


if __name__ == "__main__":
    main()
