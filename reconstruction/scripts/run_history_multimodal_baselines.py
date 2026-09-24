from __future__ import annotations
import numpy as np
from multimodal_baseline import GRID_WIDTH
HISTORY_FEATURE_NAMES = (
    "yield_lag_1_normalized",
    "yield_lag_2_normalized",
    "yield_lag_3_normalized",
    "yield_lag_4_normalized",
    "yield_lag_5_normalized",
    "yield_lag_1_observed",
    "yield_lag_2_observed",
    "yield_lag_3_observed",
    "yield_lag_4_observed",
    "yield_lag_5_observed",
    "historical_mean_normalized",
    "historical_trend_prediction_normalized",
    "historical_std_normalized",
    "historical_trend_slope_normalized",
    "historical_observation_count_fraction",
)

def build_causal_history_features(
    cache: dict[str, np.ndarray],
    target_mean: float,
    target_std: float,
    lag_years: int = 5,
    min_trend_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build features using only yields strictly before each sample's year."""
    if lag_years != 5:
        raise ValueError("The frozen feature schema currently expects five lag years.")

    years = np.asarray(cache["year"], dtype=np.int64)
    rows = np.asarray(cache["row"], dtype=np.int64)
    cols = np.asarray(cache["col"], dtype=np.int64)
    targets = np.asarray(cache["target"], dtype=np.float64)
    keys = rows * GRID_WIDTH + cols
    n_grid = 360 * GRID_WIDTH
    n_samples = targets.size

    features = np.zeros((n_samples, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    baseline = np.full(n_samples, target_mean, dtype=np.float32)

    count = np.zeros(n_grid, dtype=np.int16)
    sum_y = np.zeros(n_grid, dtype=np.float64)
    sum_yy = np.zeros(n_grid, dtype=np.float64)
    sum_t = np.zeros(n_grid, dtype=np.float64)
    sum_tt = np.zeros(n_grid, dtype=np.float64)
    sum_ty = np.zeros(n_grid, dtype=np.float64)
    lag_buffers = [np.full(n_grid, np.nan, dtype=np.float32) for _ in range(lag_years)]
    lag_buffer_years = np.full(lag_years, -1, dtype=np.int64)
    first_year = int(years.min())
    total_year_span = max(int(years.max()) - first_year, 1)

    for year in sorted(np.unique(years)):
        sample_idx = np.flatnonzero(years == year)
        sample_keys = keys[sample_idx]
        sample_count = count[sample_keys].astype(np.float64)

        for lag in range(1, lag_years + 1):
            expected_year = int(year) - lag
            slot = expected_year % lag_years
            if lag_buffer_years[slot] == expected_year:
                lag_values = lag_buffers[slot][sample_keys].astype(np.float64)
                observed = np.isfinite(lag_values)
                features[sample_idx, lag - 1] = np.where(
                    observed,
                    (lag_values - target_mean) / target_std,
                    0.0,
                )
                features[sample_idx, lag_years + lag - 1] = observed.astype(np.float32)

        historical_mean = np.divide(
            sum_y[sample_keys],
            sample_count,
            out=np.full(sample_idx.size, target_mean, dtype=np.float64),
            where=sample_count > 0,
        )
        denominator = (
            sample_count * sum_tt[sample_keys]
            - np.square(sum_t[sample_keys])
        )
        slope = np.divide(
            sample_count * sum_ty[sample_keys]
            - sum_t[sample_keys] * sum_y[sample_keys],
            denominator,
            out=np.zeros(sample_idx.size, dtype=np.float64),
            where=(
                (sample_count >= min_trend_points)
                & (np.abs(denominator) > 1.0e-12)
            ),
        )
        time_value = float(int(year) - first_year)
        intercept = np.divide(
            sum_y[sample_keys] - slope * sum_t[sample_keys],
            sample_count,
            out=np.full(sample_idx.size, target_mean, dtype=np.float64),
            where=sample_count > 0,
        )
        trend_prediction = np.where(
            sample_count >= min_trend_points,
            intercept + slope * time_value,
            historical_mean,
        )
        trend_prediction = np.maximum(trend_prediction, 0.0)
        historical_variance = np.divide(
            sum_yy[sample_keys],
            sample_count,
            out=np.zeros(sample_idx.size, dtype=np.float64),
            where=sample_count > 0,
        ) - np.square(historical_mean)
        historical_std = np.sqrt(np.maximum(historical_variance, 0.0))

        features[sample_idx, 10] = (historical_mean - target_mean) / target_std
        features[sample_idx, 11] = (trend_prediction - target_mean) / target_std
        features[sample_idx, 12] = historical_std / target_std
        features[sample_idx, 13] = slope / target_std
        features[sample_idx, 14] = sample_count / total_year_span
        baseline[sample_idx] = trend_prediction.astype(np.float32)

        current_time = float(int(year) - first_year)
        current_targets = targets[sample_idx]
        count[sample_keys] += 1
        sum_y[sample_keys] += current_targets
        sum_yy[sample_keys] += np.square(current_targets)
        sum_t[sample_keys] += current_time
        sum_tt[sample_keys] += current_time * current_time
        sum_ty[sample_keys] += current_time * current_targets

        current_slot = int(year) % lag_years
        lag_buffers[current_slot].fill(np.nan)
        lag_buffers[current_slot][sample_keys] = current_targets.astype(np.float32)
        lag_buffer_years[current_slot] = int(year)

    return features, baseline
