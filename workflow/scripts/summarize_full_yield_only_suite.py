#!/usr/bin/env python3
"""Summarize the complete yield-only baseline suite against PhenoBIID."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = ROOT / "benchmark/results/yield_only_baselines"
OLD_ROOT = ROOT / "benchmark/results/multimodal_main"
WORLD_ROOT = ROOT / "benchmark/results/biid_fixed_protocol_validation_ensemble_v1"
OUTPUT = ROOT / "visualize/paper_experiments"
FIGURES = ROOT / "visualize/paper_figures"
CROPS = ("maize", "rice", "soybean", "wheat")
SEEDS = (101, 102, 103)
WORLD_HISTORY_EXPERTS = {
    "maize": "MLP",
    "rice": "HGB + MLP ensemble",
    "soybean": "HGB + MLP ensemble",
    "wheat": "HGB",
}

METHODS = (
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
    "hgb_mlp_ensemble",
    "mlp",
    "rnn",
    "lstm",
    "tcn",
    "gru",
    "transformer",
    "world_model",
)
LABELS = {
    "global_mean": "Global mean",
    "grid_climatology": "Grid climatology",
    "previous_year": "Previous year",
    "rolling_mean_3": "3-year moving mean",
    "rolling_mean_5": "5-year moving mean",
    "exponential_smoothing": "Exponential smoothing",
    "linear_trend": "Per-grid linear trend",
    "linear_regression": "Linear regression",
    "ridge": "Ridge regression",
    "lasso": "Lasso regression",
    "elastic_net": "Elastic Net",
    "huber": "Huber regression",
    "linear_svr": "Linear SVR",
    "approximate_knn": "Approximate KNN",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "HGB history expert",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "hgb_mlp_ensemble": "HGB + MLP ensemble",
    "mlp": "MLP",
    "rnn": "RNN",
    "lstm": "LSTM",
    "tcn": "TCN",
    "gru": "GRU",
    "transformer": "Transformer",
    "world_model": "PhenoBIID state-constrained ensemble",
}
FAMILIES = {
    **{method: "Statistical" for method in METHODS[:7]},
    **{method: "Linear and robust" for method in METHODS[7:13]},
    "approximate_knn": "Neighborhood",
    **{method: "Tree ensemble" for method in METHODS[14:20]},
    "hgb_mlp_ensemble": "Prediction ensemble",
    **{method: "Neural sequence" for method in METHODS[21:27]},
    "world_model": "World model",
}
DETERMINISTIC = set(METHODS[:7])
CLASSICAL = set(METHODS[7:20])
NEURAL = set(METHODS[21:26])


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(target.astype(np.float64) - prediction.astype(np.float64))
            )
        )
    )


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = target.astype(np.float64)
    residual = np.square(target64 - prediction.astype(np.float64)).sum()
    total = np.square(target64 - target64.mean()).sum()
    return float(1.0 - residual / total)


def load_standard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {
            key: np.asarray(values[key]).copy()
            for key in ("year", "row", "col", "y_true", "y_pred")
        }


def load_world(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {
            "year": np.asarray(values["year"]).copy(),
            "row": np.asarray(values["row"]).copy(),
            "col": np.asarray(values["col"]).copy(),
            "y_true": np.asarray(values["target"]).copy(),
            "y_pred": np.asarray(values["prediction"]).copy(),
        }


def assert_aligned(reference: dict[str, np.ndarray], value: dict[str, np.ndarray]) -> None:
    for key in ("year", "row", "col"):
        if not np.array_equal(reference[key], value[key]):
            raise RuntimeError(f"Prediction arrays are not aligned on {key}.")
    if not np.allclose(reference["y_true"], value["y_true"], atol=1.0e-6):
        raise RuntimeError("Prediction arrays do not share targets.")


def ensemble(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    for value in values[1:]:
        assert_aligned(values[0], value)
    result = {key: values[0][key] for key in ("year", "row", "col", "y_true")}
    result["y_pred"] = np.mean([value["y_pred"] for value in values], axis=0)
    return result


def prediction_path(crop: str, method: str, seed: int | str) -> Path:
    if method == "gru":
        return (
            OLD_ROOT
            / crop
            / "gru_history_yield"
            / f"seed_{seed}"
            / "test_predictions.npz"
        )
    if method == "hist_gradient_boosting":
        return (
            OLD_ROOT
            / crop
            / "hgb_history_yield"
            / "seed_42"
            / "test_predictions.npz"
        )
    return NEW_ROOT / crop / method / f"seed_{seed}" / "test_predictions.npz"


def load_crop(crop: str) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        if method == "world_model":
            predictions[method] = load_world(
                WORLD_ROOT / crop / "test_predictions.npz"
            )
        elif method in DETERMINISTIC:
            predictions[method] = load_standard(
                prediction_path(crop, method, "deterministic")
            )
        elif method in CLASSICAL:
            predictions[method] = load_standard(prediction_path(crop, method, 42))
        elif method == "hgb_mlp_ensemble":
            hgb = load_standard(prediction_path(crop, "hist_gradient_boosting", 42))
            mlp = ensemble(
                [
                    load_standard(prediction_path(crop, "mlp", seed))
                    for seed in SEEDS
                ]
            )
            assert_aligned(hgb, mlp)
            predictions[method] = {
                **{key: hgb[key] for key in ("year", "row", "col", "y_true")},
                "y_pred": 0.5 * (hgb["y_pred"] + mlp["y_pred"]),
            }
        else:
            predictions[method] = ensemble(
                [
                    load_standard(prediction_path(crop, method, seed))
                    for seed in SEEDS
                ]
            )
    for value in predictions.values():
        assert_aligned(predictions["world_model"], value)
    return predictions


def cluster_interval(
    target: np.ndarray,
    reference: np.ndarray,
    proposed: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    iterations: int = 5000,
) -> tuple[float, float]:
    grid = row.astype(np.int64) * 720 + col.astype(np.int64)
    _unique, inverse = np.unique(grid, return_inverse=True)
    clusters = int(inverse.max()) + 1
    reference_sse = np.bincount(
        inverse,
        weights=np.square(reference.astype(np.float64) - target.astype(np.float64)),
        minlength=clusters,
    )
    proposed_sse = np.bincount(
        inverse,
        weights=np.square(proposed.astype(np.float64) - target.astype(np.float64)),
        minlength=clusters,
    )
    counts = np.bincount(inverse, minlength=clusters).astype(np.float64)
    rng = np.random.default_rng(20260820 + clusters)
    gains = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 100):
        stop = min(iterations, start + 100)
        sampled = rng.integers(0, clusters, size=(stop - start, clusters))
        sample_count = counts[sampled].sum(axis=1)
        reference_rmse = np.sqrt(reference_sse[sampled].sum(axis=1) / sample_count)
        proposed_rmse = np.sqrt(proposed_sse[sampled].sum(axis=1) / sample_count)
        gains[start:stop] = 100.0 * (reference_rmse - proposed_rmse) / reference_rmse
    return float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    for crop in CROPS:
        values = load_crop(crop)
        target = values["world_model"]["y_true"]
        target_mean = float(np.abs(target.astype(np.float64).mean()))
        scores: dict[str, float] = {}
        for method in METHODS:
            score = rmse(target, values[method]["y_pred"])
            scores[method] = score
            rows.append(
                {
                    "crop": crop,
                    "family": FAMILIES[method],
                    "method": method,
                    "method_label": LABELS[method],
                    "input": (
                        "historical_yield"
                        if method != "world_model"
                        else "historical_yield+climate+predicted_lai"
                    ),
                    "seeds": (
                        "deterministic"
                        if method in DETERMINISTIC
                        else "42"
                        if method in CLASSICAL
                        else "42 + 101/102/103"
                        if method == "hgb_mlp_ensemble"
                        else "/".join(map(str, SEEDS))
                    ),
                    "rmse": score,
                    "nrmse": score / target_mean,
                    "r2": r2_score(target, values[method]["y_pred"]),
                    "n_samples": int(target.size),
                    "world_history_expert": (
                        WORLD_HISTORY_EXPERTS[crop]
                        if method == "world_model"
                        else ""
                    ),
                }
            )
        yield_only = {method: scores[method] for method in METHODS[:-1]}
        best_method = min(yield_only, key=yield_only.get)
        best_rmse = yield_only[best_method]
        proposed_rmse = scores["world_model"]
        gain = 100.0 * (best_rmse - proposed_rmse) / best_rmse
        low, high = cluster_interval(
            target,
            values[best_method]["y_pred"],
            values["world_model"]["y_pred"],
            values["world_model"]["row"],
            values["world_model"]["col"],
        )
        comparison.append(
            {
                "crop": crop,
                "best_yield_only_method": best_method,
                "best_yield_only_label": LABELS[best_method],
                "best_yield_only_rmse": best_rmse,
                "world_model_rmse": proposed_rmse,
                "world_history_expert": WORLD_HISTORY_EXPERTS[crop],
                "gain_over_best_yield_only_percent": gain,
                "gain_ci_low": low,
                "gain_ci_high": high,
                "n_compared_yield_only_methods": len(yield_only),
                "n_samples": int(target.size),
            }
        )
    long_table = pd.DataFrame(rows)
    wide = long_table.pivot(index=["family", "method", "method_label"], columns="crop", values="rmse").reset_index()
    crop_ranks = long_table.pivot(index="method", columns="crop", values="rmse").rank(axis=0)
    wide["average_rank"] = wide["method"].map(crop_ranks.mean(axis=1))
    wide = wide.sort_values(["average_rank", "family", "method_label"])
    return long_table, wide, pd.DataFrame(comparison)


def plot_heatmap(long_table: pd.DataFrame) -> None:
    matrix = long_table.pivot(index="method", columns="crop", values="nrmse").loc[list(METHODS)]
    labels = [LABELS[method] for method in METHODS]
    data = matrix.loc[:, list(CROPS)].to_numpy()
    fig, ax = plt.subplots(figsize=(7.8, 10.0), constrained_layout=True)
    image = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0.12, vmax=0.45)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(
                col,
                row,
                f"{data[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="#16202A" if data[row, col] < 0.34 else "white",
            )
    ax.set_xticks(np.arange(len(CROPS)), [crop.capitalize() for crop in CROPS])
    ax.set_yticks(np.arange(len(METHODS)), labels)
    ax.set_title("Causal yield-only baselines and the proposed model", loc="left", fontweight="bold")
    ax.set_xlabel("Crop")
    ax.set_ylabel("")
    for boundary in (6.5, 12.5, 13.5, 19.5, 20.5, 26.5):
        ax.axhline(boundary, color="white", linewidth=2.0)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.65, pad=0.02)
    colorbar.set_label("NRMSE (lower is better)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURES / f"figC03_full_yield_only_suite.{suffix}", bbox_inches="tight")
    fig.savefig(FIGURES / "figC03_full_yield_only_suite.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gain(comparison: pd.DataFrame) -> None:
    x = np.arange(len(CROPS))
    gains = comparison["gain_over_best_yield_only_percent"].to_numpy()
    low = gains - comparison["gain_ci_low"].to_numpy()
    high = comparison["gain_ci_high"].to_numpy() - gains
    fig, ax = plt.subplots(figsize=(7.8, 3.6), constrained_layout=True)
    colors = np.where(gains >= 0.0, "#B24C3E", "#457B9D")
    ax.bar(x, gains, 0.58, color=colors)
    ax.errorbar(x, gains, yerr=np.vstack((low, high)), fmt="none", ecolor="#26313B", capsize=3, linewidth=0.9)
    for index, gain in enumerate(gains):
        offset = high[index] + 0.12 if gain >= 0.0 else -(low[index] + 0.12)
        ax.text(index, gain + offset, f"{gain:+.2f}%", ha="center", va="bottom" if gain >= 0 else "top", fontsize=8, fontweight="bold")
    ax.set_xticks(x, [crop.capitalize() for crop in CROPS])
    ax.set_ylabel("RMSE reduction (%)")
    ax.set_title("Gain over the strongest of 27 yield-only baselines", loc="left", fontweight="bold")
    ax.axhline(0.0, color="#26313B", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURES / f"figC02_world_model_vs_yield_only.{suffix}", bbox_inches="tight")
    fig.savefig(FIGURES / "figC02_world_model_vs_yield_only.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    long_table, wide_table, comparison = build_tables()
    long_table.to_csv(OUTPUT / "table_yield_only_baseline_suite_long.csv", index=False)
    wide_table.to_csv(OUTPUT / "table_yield_only_baseline_suite_wide.csv", index=False)
    comparison.to_csv(OUTPUT / "table_world_model_vs_full_yield_only_suite.csv", index=False)
    plot_heatmap(long_table)
    plot_gain(comparison)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
