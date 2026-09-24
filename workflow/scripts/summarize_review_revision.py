"""Aggregate every completed matched experiment without selecting test winners."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_spatial_block_bootstrap import block_bootstrap
from review_revision_data import CROPS, RESULT_ROOT, ROOT, SEEDS
from run_review_revision_parallel import configurations, experiment_id

OUTPUT = ROOT / "visualize/paper_experiments/review_revision_v2"
FIGURES = ROOT / "visualize/paper_figures"
FULL = "fusion__predicted__climate__coverage__bce"
STATE = "fusion__predicted__state_only__coverage__bce"
DIRECT = ("lightgbm", "mlp", "gru", "transformer")


def label(key):
    direct = {"lightgbm": "Direct LightGBM", "mlp": "Direct MLP", "gru": "Direct GRU", "transformer": "Direct Transformer", "multitask_gru": "Multitask GRU", "trajectory_mlp": "State-anomaly MLP", "trajectory_gru": "State-anomaly GRU"}
    if key in direct:
        return direct[key]
    _, source, climate, drop, gate = key.split("__")
    states = {"predicted": "Predicted", "previous": "Previous", "climatology": "Climatology", "constant": "Constant"}
    return f"{states[source]} state / {'+ weather' if climate == 'climate' else 'no weather'} / {drop} / {gate}"


def ensemble(crop, key, split):
    values = []
    for seed in SEEDS:
        with np.load(RESULT_ROOT / crop / key / f"seed_{seed}" / f"{split}_predictions.npz") as data:
            values.append({k: data[k] for k in data.files})
    for other in values[1:]:
        for field in ("target", "source_indices", "year", "row", "col"):
            np.testing.assert_array_equal(values[0][field], other[field])
    result = values[0].copy()
    for field in ("prediction", "history_prediction", "candidate", "gate"):
        result[field] = np.mean([v[field].astype(float) for v in values], axis=0)
    return result


def rmse(y, p):
    return float(np.sqrt(np.mean((y.astype(float)-p.astype(float))**2)))


def paired(crop, name, reference, model):
    for field in ("target", "source_indices", "row", "col"):
        np.testing.assert_array_equal(reference[field], model[field])
    rows = []
    for scale in (10, 20):
        point, low, high, blocks = block_bootstrap(model["target"].astype(float), reference["prediction"].astype(float), model["prediction"].astype(float), model["row"], model["col"], scale, 20260905+CROPS.index(crop)*100+scale)
        rows.append(dict(crop=crop, comparison=name, block_degrees=scale, gain_percent=point, ci_low=low, ci_high=high, n_blocks=blocks))
    return rows


def figures(table, comparisons):
    selected = (*DIRECT, "multitask_gru", STATE, FULL, "trajectory_mlp", "trajectory_gru")
    plot = table[table.experiment.isin(selected)].pivot(index="experiment", columns="crop", values="gain_over_history_percent").reindex(index=selected, columns=CROPS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9.2, 5.8), layout="constrained")
    data = plot.to_numpy()
    limit = max(1., float(np.nanmax(np.abs(data))))
    image = ax.imshow(data, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), [c.title() for c in CROPS])
    ax.set_yticks(range(len(selected)), [label(k).replace(" / coverage / bce", "") for k in selected])
    for i in range(data.shape[0]):
        for j in range(4):
            if np.isfinite(data[i,j]):
                ax.text(j, i, f"{data[i,j]:+.2f}%", ha="center", va="center", color="white" if abs(data[i,j]) > .6*limit else "black")
    fig.colorbar(image, ax=ax, label="RMSE reduction over the identical history anchor (%)")
    ax.set_title("Matched inputs and anchors: development comparison", loc="left", fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig_revision_v2_matched_models.{suffix}", dpi=220)
    plt.close(fig)
    subset = comparisons[(comparisons.block_degrees == 10) & comparisons.comparison.str.startswith("State-only predicted vs")]
    if len(subset):
        fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.5), sharey=True, layout="constrained")
        names = ("previous", "climatology", "constant")
        for ax, crop in zip(axes, CROPS):
            ax.axvline(0, color="0.5", linewidth=.8)
            for i, source in enumerate(names):
                row = subset[(subset.crop == crop) & (subset.comparison == f"State-only predicted vs {source}")]
                if len(row):
                    r = row.iloc[0]
                    ax.plot([r.ci_low, r.ci_high], [i,i], color="#287c73", linewidth=2)
                    ax.scatter(r.gain_percent, i, color="#287c73", s=30, zorder=3)
            ax.set_title(crop.title())
            ax.set_yticks(range(3), ["Previous LAI", "Climatology", "Constant state"])
            ax.set_xlabel("Predicted-state gain (%)")
            ax.grid(axis="x", alpha=.15)
        fig.suptitle("Is the forecast state useful? Matched readout, 95% spatial-block intervals")
        for suffix in ("png", "pdf"):
            fig.savefig(FIGURES / f"fig_revision_v2_state_necessity.{suffix}", dpi=220)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    keys = [experiment_id(s) for s in configurations("all")]
    all_rows, summary, missing, loaded = [], [], [], {}
    for crop in CROPS:
        for key in keys:
            paths = [RESULT_ROOT / crop / key / f"seed_{s}" / "test_metrics.json" for s in SEEDS]
            for seed, path in zip(SEEDS, paths):
                if not path.exists():
                    missing.append(f"{crop}/{key}/{seed}")
                    continue
                d = json.loads(path.read_text())
                all_rows.append(dict(crop=crop, experiment=key, seed=seed, validation_rmse=d["metrics"]["validation"]["rmse"], **d["metrics"]["test"], elapsed_seconds=d["elapsed_seconds"]))
            if not all(path.exists() for path in paths):
                continue
            test, val = ensemble(crop, key, "test"), ensemble(crop, key, "validation")
            loaded[crop,key] = test
            history_error = rmse(test["target"], test["history_prediction"])
            error = rmse(test["target"], test["prediction"])
            runs = [row for row in all_rows if row['crop'] == crop and row['experiment'] == key]
            summary.append(dict(crop=crop, experiment=key, label=label(key), ensemble_rmse=error, mean_seed_rmse=float(np.mean([r['rmse'] for r in runs])), std_seed_rmse=float(np.std([r['rmse'] for r in runs], ddof=1)), history_rmse=history_error, gain_over_history_percent=100*(history_error-error)/history_error, positive_seeds=sum(r['gain_over_history_percent'] > 0 for r in runs), validation_rmse=rmse(val['target'], val['prediction'])))
    pd.DataFrame(all_rows).to_csv(OUTPUT / "all_runs.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(OUTPUT / "ensemble_results.csv", index=False)
    (OUTPUT / "completion.json").write_text(json.dumps(dict(completed=len(all_rows), expected=len(keys)*12, missing=missing), indent=2))
    if args.require_complete and missing:
        raise RuntimeError(f"{len(missing)} experiments are still missing")
    comparisons = []
    selected = []
    for crop in CROPS:
        candidates = table[(table.crop == crop) & table.experiment.isin(DIRECT)] if len(table) else table
        if len(candidates) == 4 and (crop,FULL) in loaded:
            best = candidates.sort_values(['validation_rmse','experiment']).iloc[0]
            selected.append(dict(crop=crop, selected_direct=best.experiment, validation_rmse=best.validation_rmse, direct_rmse=best.ensemble_rmse, full_rmse=rmse(loaded[crop,FULL]['target'],loaded[crop,FULL]['prediction'])))
            comparisons += paired(crop, "Full vs validation-selected direct", loaded[crop,best.experiment], loaded[crop,FULL])
        for pathway in ("state_only", "climate"):
            pred = f"fusion__predicted__{pathway}__coverage__bce"
            for source in ("previous", "climatology", "constant"):
                other = f"fusion__{source}__{pathway}__coverage__bce"
                if (crop,pred) in loaded and (crop,other) in loaded:
                    prefix = "State-only" if pathway == "state_only" else "Full"
                    comparisons += paired(crop, f"{prefix} predicted vs {source}", loaded[crop,other], loaded[crop,pred])
            for drop in ("none", "uniform", "shuffled"):
                other = f"fusion__predicted__{pathway}__{drop}__bce"
                if (crop,pred) in loaded and (crop,other) in loaded:
                    comparisons += paired(crop, f"{pathway} coverage vs {drop}", loaded[crop,other], loaded[crop,pred])
    comp = pd.DataFrame(comparisons)
    comp.to_csv(OUTPUT / "paired_spatial_intervals.csv", index=False)
    pd.DataFrame(selected).to_csv(OUTPUT / "validation_selected_direct.csv", index=False)
    if len(table):
        print(table[['crop','experiment','ensemble_rmse','gain_over_history_percent','positive_seeds']].to_string(index=False))
    if len(comp):
        figures(table, comp)
    print(f"[SUMMARY] {len(all_rows)}/{len(keys)*12}; missing={len(missing)}", flush=True)


if __name__ == "__main__":
    main()
