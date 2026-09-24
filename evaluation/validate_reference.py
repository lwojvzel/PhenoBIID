#!/usr/bin/env python3
"""Check machine-readable manuscript tables for protocol consistency."""

from pathlib import Path

import pandas as pd
import numpy as np

from rebuild_main_table import rebuild, verify


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    verify(rebuild())
    main_table = pd.read_csv(ROOT / "data/reference/main_yield_rmse.csv")
    crops = {"maize", "rice", "soybean", "wheat"}
    windows = {0.1, 0.3, 0.5, 0.7}
    assert set(main_table.crop) == crops
    assert set(main_table.unobserved_fraction) == windows
    assert (main_table.groupby(["unobserved_fraction", "crop"]).size() == 3).all()
    assert not main_table.duplicated(["unobserved_fraction", "crop", "method"]).any()
    assert np.isfinite(main_table[["rmse", "seed_sd"]]).all().all()
    assert (main_table[["rmse", "seed_sd"]] >= 0).all().all()

    direct = pd.read_csv(ROOT / "data/reference/direct_baselines_seed42.csv")
    assert direct.method.nunique() == 7
    assert not direct.method.duplicated().any()
    assert np.isfinite(direct[sorted(crops)]).all().all()
    assert (direct[sorted(crops)] >= 0).all().all()
    annual = pd.read_csv(ROOT / 'data/reference/annual/direct_seed42.csv')
    names = {'ridge': 'Ridge', 'random_forest': 'Random Forest', 'lightgbm': 'LightGBM',
             'gru': 'GRU', 'transformer': 'Transformer', 'cnn_rnn': 'CNN-RNN (adapted)',
             'world': 'PhenoBIID'}
    selected = annual[annual.percent.eq(10) & annual.model.isin(names)].copy()
    assert not selected.duplicated(['crop', 'model', 'year']).any()
    assert (selected.groupby(['crop', 'model']).year.nunique() == 13).all()
    recomputed = selected.groupby(['model', 'crop']).rmse.mean().unstack()
    recomputed.index = recomputed.index.map(names)
    ordered = direct.set_index('method')
    np.testing.assert_allclose(recomputed.loc[ordered.index, sorted(crops)],
                               ordered[sorted(crops)], atol=0.00005001, rtol=0)

    vegetation = pd.read_csv(ROOT / "data/reference/vegetation_rmse.csv")
    alternatives = ["gru", "no_observed_feedback", "historical_mean", "previous_trajectory", "last_visible"]
    assert len(vegetation) == 15
    assert np.isfinite(vegetation[["biid", *alternatives]]).all().all()
    assert (vegetation[["biid", *alternatives]] >= 0).all().all()
    print("Main and direct tables numerically reconstructed; vegetation table schema validated.")


if __name__ == "__main__":
    main()
