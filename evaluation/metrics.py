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
    groups = ["crop", *[c for c in CONDITION_COLUMNS if c in frame]]
    if frame[groups].isna().any().any():
        raise ValueError("Crop and condition columns must not contain missing values")
    numeric = frame[["year", "target", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("year, target, and prediction must be finite")
    if not np.equal(numeric[:, 0], np.floor(numeric[:, 0])).all():
        raise ValueError("year must be an integer")


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
    validate_predictions(candidate)
    validate_predictions(reference)
    for frame in (candidate, reference):
        if "method" in frame and frame.method.nunique(dropna=False) != 1:
            raise ValueError("Compare one candidate method with one reference method")
    conditions = [c for c in CONDITION_COLUMNS if c != "method"]
    if {c for c in conditions if c in candidate} != {c for c in conditions if c in reference}:
        raise ValueError("Candidate and reference condition columns differ")
    keys = [c for c in conditions if c in candidate] + ["crop"]
    identity = keys + ["year"]
    for column in ("row", "col", "source_indices"):
        if (column in candidate) != (column in reference):
            raise ValueError("Candidate and reference spatial identity columns differ")
        if column in candidate:
            identity.append(column)
    # Without coordinates, records are paired in their supplied within-year order.
    def paired(frame):
        values = frame[identity + ["target"]].copy()
        values["_occurrence"] = values.groupby(identity, dropna=False).cumcount()
        return values.sort_values(identity + ["_occurrence"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(paired(candidate), paired(reference), check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("Candidate and reference years, sample identities, or targets differ") from error
    left = mean_annual_rmse(candidate).rename(columns={"mean_annual_rmse": "candidate_rmse"})
    right = mean_annual_rmse(reference).rename(columns={"mean_annual_rmse": "reference_rmse"})
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if merged.empty or (merged.reference_rmse <= 0).any():
        raise ValueError("Relative reduction needs matched groups and positive reference RMSE")
    merged["rmse_reduction_percent"] = 100 * (1 - merged.candidate_rmse / merged.reference_rmse)
    return merged
