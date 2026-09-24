#!/usr/bin/env python3
"""Check machine-readable manuscript tables for protocol consistency."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    main_table = pd.read_csv(ROOT / "data/reference/main_yield_rmse.csv")
    crops = {"maize", "rice", "soybean", "wheat"}
    windows = {0.1, 0.3, 0.5, 0.7}
    assert set(main_table.crop) == crops
    assert set(main_table.unobserved_fraction) == windows
    assert (main_table.groupby(["unobserved_fraction", "crop"]).size() == 3).all()
    winners = main_table.loc[main_table.groupby(["unobserved_fraction", "crop"]).rmse.idxmin()]
    assert set(winners.method) == {"PhenoBIID"}

    direct = pd.read_csv(ROOT / "data/reference/direct_baselines_seed42.csv")
    assert direct.method.nunique() == 7
    for crop in crops:
        assert direct.loc[direct[crop].idxmin(), "method"] == "PhenoBIID"

    vegetation = pd.read_csv(ROOT / "data/reference/vegetation_rmse.csv")
    alternatives = ["gru", "no_observed_feedback", "historical_mean", "previous_trajectory", "last_visible"]
    assert len(vegetation) == 15
    assert (vegetation.biid.to_numpy()[:, None] < vegetation[alternatives].to_numpy()).all()
    print("Reference tables passed: main yield, direct baselines, and vegetation completion.")


if __name__ == "__main__":
    main()
