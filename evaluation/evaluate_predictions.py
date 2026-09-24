#!/usr/bin/env python3
"""Score grid-level annual yield predictions under the paper protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import annual_rmse, mean_annual_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual-output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    summary = mean_annual_rmse(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    if args.annual_output:
        args.annual_output.parent.mkdir(parents=True, exist_ok=True)
        annual_rmse(frame).to_csv(args.annual_output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
