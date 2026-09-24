#!/usr/bin/env python3
"""External-label pilot on the official CY-Bench sample data.

This script intentionally keeps the information boundary identical between the
direct and factorized yield models. Both see historical yields, previous-season
FAPAR, target-season weather, and spatial/year context. The factorized model
first predicts target-season FAPAR and only passes cross-fitted state predictions
to its yield readout; observed target-season FAPAR is never a yield input.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Data/external/CYBench/sample_data"
RESULT_ROOT = ROOT / "benchmark/results/cybench_sample_external_v1"
TABLE_ROOT = ROOT / "visualize/paper_experiments/review_revision_20260904"
CROPS = ("maize", "wheat")
COUNTRIES = ("ES", "NL")
WEATHER_COLUMNS = ("tmin", "tmax", "prec", "rad", "tavg", "et0", "vpd", "cwb")
SEEDS = (42, 45, 48)
SLOTS = 12


def season_bounds(year: int, sos: float, eos: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    sos_day = max(1, min(366, int(round(sos))))
    eos_day = max(1, min(366, int(round(eos))))
    start_year = year if sos_day <= eos_day else year - 1
    start = pd.Timestamp(start_year, 1, 1) + pd.Timedelta(days=sos_day - 1)
    end = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=eos_day - 1)
    return start, end


def binned_values(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    left = int(np.searchsorted(dates, np.datetime64(start), side="left"))
    right = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    if right <= left:
        return None
    selected = frame.iloc[left:right]
    duration = max(float((end - start).days + 1), 1.0)
    progress = (selected["date"] - start).dt.total_seconds().to_numpy() / 86400.0
    bins = np.floor(progress / duration * SLOTS).astype(np.int64)
    bins = np.clip(bins, 0, SLOTS - 1)
    values = np.full((SLOTS, len(columns)), np.nan, dtype=np.float64)
    valid = np.zeros(SLOTS, dtype=np.float64)
    raw = selected.loc[:, columns].to_numpy(dtype=np.float64)
    for slot in range(SLOTS):
        mask = bins == slot
        if mask.any():
            slot_value = np.nanmean(raw[mask], axis=0)
            if np.isfinite(slot_value).any():
                values[slot] = slot_value
                valid[slot] = 1.0
    return values, valid


def interpolate_slots(values: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    output = values.astype(np.float64, copy=True)
    slots = np.arange(SLOTS)
    for channel in range(output.shape[1]):
        finite = np.isfinite(output[:, channel]) & (valid > 0)
        if finite.sum() < 3:
            return None
        output[:, channel] = np.interp(slots, slots[finite], output[finite, channel])
    return output


def history_features(history: pd.DataFrame, adm_id: str, year: int) -> np.ndarray | None:
    prior = history[(history["adm_id"] == adm_id) & (history["harvest_year"] < year)]
    prior = prior.sort_values("harvest_year").tail(5)
    if len(prior) < 2:
        return None
    values = np.full(5, np.nan, dtype=np.float64)
    mask = np.zeros(5, dtype=np.float64)
    recent = prior["yield"].to_numpy(dtype=np.float64)[::-1]
    values[: len(recent)] = recent
    mask[: len(recent)] = 1.0
    years = prior["harvest_year"].to_numpy(dtype=np.float64)
    yields = prior["yield"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(years, yields, 1)
    trend = intercept + slope * year
    summary = np.array(
        [yields[-1], yields.mean(), yields.std(ddof=0), trend, slope],
        dtype=np.float64,
    )
    return np.concatenate((values, mask, summary))


def read_grouped(path: Path, value_columns: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, usecols=("adm_id", "date", *value_columns), dtype={"adm_id": str})
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    return {
        str(adm): group.sort_values("date").reset_index(drop=True)
        for adm, group in frame.groupby("adm_id", sort=False)
    }


def build_crop_table(crop: str, force: bool, data_root=DATA_ROOT,
                     result_root=RESULT_ROOT, countries=COUNTRIES) -> dict[str, np.ndarray]:
    cache_path = result_root / crop / "aligned_samples.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            return {key: np.asarray(cached[key]) for key in cached.files}

    rows: list[dict[str, Any]] = []
    for country_index, country in enumerate(countries):
        directory = data_root / crop / country
        yields = pd.read_csv(directory / f"yield_{crop}_{country}.csv", dtype={"adm_id": str})
        yields = yields[np.isfinite(yields['yield']) & (yields['yield'] > 0)].copy()
        calendar = pd.read_csv(directory / f"crop_calendar_{crop}_{country}.csv", dtype={"adm_id": str})
        locations = pd.read_csv(directory / f"location_{crop}_{country}.csv", dtype={"adm_id": str})
        fpar = read_grouped(directory / f"fpar_{crop}_{country}.csv", ("fpar",))
        weather = read_grouped(
            directory / f"meteo_{crop}_{country}.csv", WEATHER_COLUMNS
        )
        calendar_map = calendar.set_index("adm_id")[["sos", "eos"]].to_dict("index")
        location_map = locations.set_index("adm_id")[["latitude", "longitude"]].to_dict("index")
        for label in yields.to_dict("records"):
            adm_id = str(label["adm_id"])
            year = int(label["harvest_year"])
            if year < 2002 or adm_id not in calendar_map or adm_id not in location_map:
                continue
            if adm_id not in fpar or adm_id not in weather:
                continue
            history = history_features(yields, adm_id, year)
            if history is None:
                continue
            cal = calendar_map[adm_id]
            current_start, current_end = season_bounds(year, cal["sos"], cal["eos"])
            previous_start, previous_end = season_bounds(year - 1, cal["sos"], cal["eos"])
            current_state_raw = binned_values(
                fpar[adm_id], current_start, current_end, ("fpar",)
            )
            previous_state_raw = binned_values(
                fpar[adm_id], previous_start, previous_end, ("fpar",)
            )
            climate_raw = binned_values(
                weather[adm_id], current_start, current_end, WEATHER_COLUMNS
            )
            if current_state_raw is None or previous_state_raw is None or climate_raw is None:
                continue
            current_state = interpolate_slots(*current_state_raw)
            previous_state = interpolate_slots(*previous_state_raw)
            climate = interpolate_slots(*climate_raw)
            if current_state is None or previous_state is None or climate is None:
                continue
            location = location_map[adm_id]
            latitude = float(location["latitude"])
            longitude = float(location["longitude"])
            context = np.array(
                [
                    latitude / 90.0,
                    math.sin(math.radians(longitude)),
                    math.cos(math.radians(longitude)),
                    (year - 2000.0) / 25.0,
                    float(country_index),
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "adm_id": adm_id,
                    "country": country,
                    "year": year,
                    "target": float(label["yield"]),
                    "history": history,
                    "context": context,
                    "previous_state": previous_state[:, 0] / 100.0,
                    "previous_state_valid": previous_state_raw[1],
                    "target_state": current_state[:, 0] / 100.0,
                    "target_state_valid": current_state_raw[1],
                    "climate": climate,
                    "climate_valid": climate_raw[1],
                }
            )
    if not rows:
        raise RuntimeError(f"No aligned CY-Bench samples for {crop}")
    arrays = {
        "adm_id": np.asarray([row["adm_id"] for row in rows]),
        "country": np.asarray([row["country"] for row in rows]),
        "year": np.asarray([row["year"] for row in rows], dtype=np.int16),
        "target": np.asarray([row["target"] for row in rows], dtype=np.float32),
        "history": np.stack([row["history"] for row in rows]).astype(np.float32),
        "context": np.stack([row["context"] for row in rows]).astype(np.float32),
        "previous_state": np.stack([row["previous_state"] for row in rows]).astype(np.float32),
        "previous_state_valid": np.stack([row["previous_state_valid"] for row in rows]).astype(np.float32),
        "target_state": np.stack([row["target_state"] for row in rows]).astype(np.float32),
        "target_state_valid": np.stack([row["target_state_valid"] for row in rows]).astype(np.float32),
        "climate": np.stack([row["climate"] for row in rows]).astype(np.float32),
        "climate_valid": np.stack([row["climate_valid"] for row in rows]).astype(np.float32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def fill_from_train(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    output = []
    for values in (train, *others):
        output.append(np.where(np.isfinite(values), values, median).astype(np.float32))
    return tuple(output)


def state_features(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            data["previous_state"],
            data["previous_state_valid"],
            data["climate"].reshape(len(data["year"]), -1),
            data["climate_valid"],
            data["context"],
        ),
        axis=1,
    )


def history_features_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate((data["history"], data["context"]), axis=1)


def fit_state_model(x: np.ndarray, y: np.ndarray, seed: int) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.7,
        min_samples_leaf=3,
        n_jobs=4,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def cross_fitted_state(
    x: np.ndarray, y: np.ndarray, years: np.ndarray, seed: int
) -> np.ndarray:
    unique_years = np.unique(years)
    splitter = GroupKFold(n_splits=min(5, len(unique_years)))
    prediction = np.full_like(y, np.nan, dtype=np.float32)
    for fold, (fit_index, held_index) in enumerate(splitter.split(x, groups=years)):
        model = fit_state_model(x[fit_index], y[fit_index], seed + fold)
        prediction[held_index] = model.predict(x[held_index]).astype(np.float32)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Cross-fitted state predictions are incomplete")
    return prediction


def fit_yield_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(validation_x, validation_y)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse_mean_percent": float(100.0 * rmse / np.mean(target)),
        "r2": float(1.0 - np.sum(np.square(error)) / np.sum(np.square(target - target.mean()))),
        "pearson": float(np.corrcoef(target, prediction)[0, 1]),
    }


def run_crop(crop: str, force: bool) -> list[dict[str, Any]]:
    data = build_crop_table(crop, force)
    years = data["year"]
    split = {
        "train": np.flatnonzero(years <= 2014),
        "validation": np.flatnonzero((years >= 2015) & (years <= 2016)),
        "test": np.flatnonzero(years >= 2017),
    }
    if any(len(index) == 0 for index in split.values()):
        raise RuntimeError(f"Empty temporal split for {crop}: { {k: len(v) for k,v in split.items()} }")
    state_x = state_features(data)
    history_x = history_features_matrix(data)
    direct_x = np.concatenate((history_x, state_x), axis=1)
    train, validation, test = split["train"], split["validation"], split["test"]
    state_x_train, state_x_validation, state_x_test = fill_from_train(
        state_x[train], state_x[validation], state_x[test]
    )
    history_train, history_validation, history_test = fill_from_train(
        history_x[train], history_x[validation], history_x[test]
    )
    direct_train, direct_validation, direct_test = fill_from_train(
        direct_x[train], direct_x[validation], direct_x[test]
    )
    target = data["target"]
    target_state = data["target_state"]
    rows: list[dict[str, Any]] = []
    run_dir = RESULT_ROOT / crop
    run_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        started = time.time()
        state_oof = cross_fitted_state(
            state_x_train, target_state[train], years[train], seed
        )
        state_model = fit_state_model(state_x_train, target_state[train], seed)
        state_validation = state_model.predict(state_x_validation).astype(np.float32)
        state_test = state_model.predict(state_x_test).astype(np.float32)
        state_rmse = float(np.sqrt(np.mean(np.square(state_test - target_state[test]))))
        state_climatology = np.mean(target_state[train], axis=0, keepdims=True)
        state_clim_rmse = float(
            np.sqrt(np.mean(np.square(state_climatology - target_state[test])))
        )

        world_state_train = np.concatenate(
            (history_train, state_oof, state_oof - data["previous_state"][train]), axis=1
        )
        world_state_validation = np.concatenate(
            (history_validation, state_validation, state_validation - data["previous_state"][validation]), axis=1
        )
        world_state_test = np.concatenate(
            (history_test, state_test, state_test - data["previous_state"][test]), axis=1
        )
        augmented_train = np.concatenate((direct_train, state_oof), axis=1)
        augmented_validation = np.concatenate((direct_validation, state_validation), axis=1)
        augmented_test = np.concatenate((direct_test, state_test), axis=1)
        feature_sets = {
            "history_only": (history_train, history_validation, history_test),
            "direct_same_input": (direct_train, direct_validation, direct_test),
            "factorized_state_readout": (
                world_state_train, world_state_validation, world_state_test
            ),
            "factorized_state_augmented": (
                augmented_train, augmented_validation, augmented_test
            ),
        }
        predictions: dict[str, np.ndarray] = {}
        for method, (fit_x, val_x, test_x) in feature_sets.items():
            model = fit_yield_model(
                fit_x, target[train], val_x, target[validation], seed
            )
            prediction = model.predict(test_x).astype(np.float32)
            predictions[method] = prediction
            metrics = regression_metrics(target[test], prediction)
            rows.append(
                {
                    "crop": crop,
                    "seed": seed,
                    "method": method,
                    **metrics,
                    "state_rmse": state_rmse,
                    "state_skill_vs_climatology": 1.0 - state_rmse**2 / state_clim_rmse**2,
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                    "elapsed_seconds": float(time.time() - started),
                }
            )
            joblib.dump(model, run_dir / f"{method}_seed{seed}.joblib", compress=3)
        joblib.dump(state_model, run_dir / f"state_model_seed{seed}.joblib", compress=3)
        np.savez_compressed(
            run_dir / f"test_predictions_seed{seed}.npz",
            year=years[test],
            adm_id=data["adm_id"][test],
            country=data["country"][test],
            target=target[test],
            target_state=target_state[test],
            predicted_state=state_test,
            **predictions,
        )
    metadata = {
        "crop": crop,
        "source": "official WUR-AI/CY-Bench sample_data repository",
        "label": "subnational statistics, independent of GDHY",
        "state_variable": "JRC FAPAR",
        "input_boundary": "historical yields + previous-season FAPAR + full target-season AgERA5 weather + context",
        "yield_input_uses_observed_target_state": False,
        "phenology_alignment": "12 equal-progress bins between CY-Bench SOS and EOS",
        "state_training_predictions": "five-fold GroupKFold by harvest year",
        "train_years": "<=2014",
        "validation_years": "2015-2016",
        "test_years": ">=2017",
        "sample_counts": {name: int(len(index)) for name, index in split.items()},
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=(*CROPS, "all"), default="all")
    parser.add_argument("--force-alignment", action="store_true")
    args = parser.parse_args()
    crops = CROPS if args.crop == "all" else (args.crop,)
    all_rows: list[dict[str, Any]] = []
    for crop in crops:
        rows = run_crop(crop, args.force_alignment)
        pd.DataFrame(rows).to_csv(RESULT_ROOT / crop / "metrics_by_seed.csv", index=False)
        all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary = (
        frame.groupby(["crop", "method"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            nrmse_mean_percent=("nrmse_mean_percent", "mean"),
            r2_mean=("r2", "mean"),
            state_rmse=("state_rmse", "mean"),
            state_skill_vs_climatology=("state_skill_vs_climatology", "mean"),
        )
    )
    direct = summary[summary["method"] == "direct_same_input"][["crop", "rmse_mean"]]
    direct = direct.rename(columns={"rmse_mean": "direct_rmse"})
    summary = summary.merge(direct, on="crop", how="left")
    summary["gain_vs_direct_percent"] = 100.0 * (
        summary["direct_rmse"] - summary["rmse_mean"]
    ) / summary["direct_rmse"]
    output = TABLE_ROOT / "table_cybench_sample_external_validation.csv"
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
