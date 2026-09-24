"""Metrics used by CropDynamicsBench."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"crop", "year", "target", "prediction"}
CONDITION_COLUMNS = ("method", "seed", "unobserved_fraction", "product")


def validate_predictions(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction table is empty")
    numeric = frame[["year", "target", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("year, target, and prediction must be finite")


def annual_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute spatial RMSE independently for each crop-year."""
    validate_predictions(frame)
    groups = [column for column in CONDITION_COLUMNS if column in frame.columns]
    groups += ["crop", "year"]
    scored = frame.assign(squared_error=(frame.prediction - frame.target) ** 2)
    result = scored.groupby(groups, as_index=False).agg(
        mse=("squared_error", "mean"), samples=("squared_error", "size")
    )
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result[[*groups, "samples", "rmse"]]


def mean_annual_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    """Average annual RMSE with equal weight for each year."""
    annual = annual_rmse(frame)
    groups = [column for column in CONDITION_COLUMNS if column in annual.columns]
    groups += ["crop"]
    return annual.groupby(groups, as_index=False).agg(
        years=("year", "nunique"), mean_annual_rmse=("rmse", "mean")
    )


def relative_reduction(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return 100 * (1 - candidate RMSE / reference RMSE)."""
    left = mean_annual_rmse(candidate).rename(columns={"mean_annual_rmse": "candidate_rmse"})
    right = mean_annual_rmse(reference).rename(columns={"mean_annual_rmse": "reference_rmse"})
    keys = [column for column in (*CONDITION_COLUMNS, "crop") if column in left.columns and column in right.columns]
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if not np.array_equal(merged.years_candidate, merged.years_reference):
        raise ValueError("Candidate and reference year counts differ")
    merged["rmse_reduction_percent"] = 100 * (1 - merged.candidate_rmse / merged.reference_rmse)
    return merged
