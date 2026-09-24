"""Replay named historical baselines without changing frozen experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import summarize_full_yield_only_suite as legacy

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "benchmark/results/inseason_signal_match_v1/pipelines"
REFERENCES = ROOT / "benchmark/results/inseason_ndvi_reuse_v1/pipelines"
IDENTITY = ("year", "row", "col")
YEARS = (2002, 2003, 2004, 2006, 2007, 2008, *range(2010, 2017))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annual_rmse(labels, prediction):
    target = np.asarray(labels["target"], dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or prediction.shape != target.shape or not len(target):
        raise ValueError("Predictions must match a nonempty one-dimensional target")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite target or prediction; sample filtering is forbidden")
    year = np.asarray(labels["year"])
    if year.shape != target.shape:
        raise ValueError("Year shape differs from target")
    return {int(y): float(np.sqrt(np.mean((target[year == y] - prediction[year == y]) ** 2)))
            for y in np.unique(year)}


def aligned_prediction(path, labels, key, target_key="target"):
    with np.load(path) as values:
        for name in IDENTITY:
            np.testing.assert_array_equal(labels[name], values[name], err_msg=str(path))
        np.testing.assert_allclose(labels["target"], values[target_key], rtol=0, atol=1e-6,
                                   err_msg=str(path))
        prediction = values[key]
    annual_rmse(labels, prediction)
    return prediction


def history_name(path):
    parts = Path(path).parts
    if "linear_state_yield_v1" in parts:
        return "Ridge residual, lambda=" + parts[-1].removeprefix("penalty_")
    if "neighbor_state_readout_v1" in parts:
        return "ModernNCA residual" if parts[-2] == "modern" else "Linear-NCA residual"
    if "forward_anchor_readout_v1" in parts:
        head = "TabM" if parts[-2] == "tabm" else "MLP"
        mode = "forward" if parts[-3] == "forward" else "in-sample"
        return f"{head} residual ({mode}, 1993+)"
    if "numeric_embedding_readout_v1" in parts:
        return "TabM residual (PLE)"
    if "neural_process_readout_v1" in parts:
        head = "TabM" if parts[-2] == "tabm" else "MLP"
        return f"{head} residual (full window)"
    if "task_aligned_world_v1" in parts:
        return "Lag-feature MLP"
    raise ValueError("Unregistered historical model: " + str(path))


def tex_name(name, wrap=False):
    name = name.replace("lambda=", r"$\lambda=$")
    if wrap and " (" in name:
        return r"\makecell[l]{" + name.replace(" (", r"\\(", 1) + "}"
    return name


def build_history_tables(table, recipes, summary, out):
    rows, records, sources, selected_records, selection_rows = [], [], {}, [], []
    inventory = None
    for crop in recipes:
        cells = []
        for origin in (2004, 2008, 2012):
            reference = REFERENCES / crop / f"origin_{origin}/seed_42/history_reference.json"
            spec = json.loads(reference.read_text())
            sources[str(reference)] = digest(reference)
            names = sorted(history_name(c["path"]) for c in spec["candidates"])
            if len(set(names)) != 15 or (inventory is not None and names != inventory):
                raise ValueError("Expected the same fifteen distinct configurations")
            inventory = names
            selected = spec["selected"]
            cells.append(tex_name(history_name(selected["path"]), wrap=True))
            selected_records.append(dict(crop=crop, origin=origin,
                                         model=history_name(selected["path"]), **selected))
            for split in (("validation", "test") if origin == 2012 else ("validation",)):
                label_file = CURRENT / crop / f"origin_{origin}/seed_42/{split}_labels.npz"
                with np.load(label_file) as values:
                    labels = {k: values[k] for k in (*IDENTITY, "target", "strong")}
                sources[str(label_file)] = digest(label_file)
                for candidate in spec["candidates"]:
                    path = Path(candidate["path"]) / f"{split}_predictions.npz"
                    prediction = aligned_prediction(path, labels, candidate["key"])
                    sources[str(path)] = digest(path)
                    if split == "validation" and sources[str(path)] != candidate["prediction_sha256"]:
                        raise ValueError("Historical candidate changed: " + str(path))
                    scores = annual_rmse(labels, prediction)
                    if split == "validation":
                        np.testing.assert_allclose(np.mean(list(scores.values())), candidate["score"],
                                                   rtol=0, atol=1e-10)
                    is_selected = candidate["path"] == selected["path"]
                    if is_selected:
                        np.testing.assert_array_equal(prediction, labels["strong"])
                    for year, error in scores.items():
                        records.append(dict(crop=crop, origin=origin, split=split, year=year,
                                            model=history_name(candidate["path"]), rmse=error,
                                            selected=is_selected, path=str(path), key=candidate["key"]))
        selection_rows.append(crop.title() + " & " + " & ".join(cells) + r" \\")
    annual = pd.DataFrame(records)
    if annual.duplicated(["crop", "model", "year"]).any():
        raise ValueError("Overlapping annual history evaluations")
    for _, group in annual.groupby(["crop", "model"]):
        if tuple(sorted(group.year)) != YEARS:
            raise ValueError("Incomplete thirteen-year historical support")
    scores = annual.groupby(["model", "crop"]).rmse.mean().unstack("crop")
    for model in inventory:
        rows.append(tex_name(model) + " & " + " & ".join(f"{scores.loc[model, c]:.4f}"
                                                         for c in recipes) + r" \\")
    selected_scores = annual[annual.selected].groupby("crop").rmse.mean()
    for crop in recipes:
        r = summary[summary.crop.eq(crop) & summary.recipe.eq(recipes[crop])
                    & summary["mode"].eq("biid") & np.isclose(summary.requested_fraction, .1)]
        if len(r) != 1:
            raise ValueError("Missing primary seasonal score")
        np.testing.assert_allclose(selected_scores[crop], r.iloc[0].strong_rmse, rtol=0, atol=1e-10)
    rows.extend([r"\midrule", r"Selected baseline (Table~\ref{tab:inseason_history}) & " +
                 " & ".join(f"{selected_scores[c]:.4f}" for c in recipes) + r" \\"])
    table("inseason_history_table.tex",
          "Named baselines used as the denominator of the 13-year gain. Each is selected "
          "from 15 configurations on its three-year development block; the last choice "
          "is also used for 2013--2016. Residual and training-window variants are defined "
          "in Appendix~\\ref{app:inseason_baselines}.",
          "tab:inseason_history", "llll",
          r"Crop & 2002--2004 & 2006--2008 & 2010--2016", selection_rows)
    table("inseason_history_candidates_table.tex",
          "All 15 retained history-input configurations on the 13-year cohort. Entries "
          "are mean annual yield RMSE (t/ha). These are configurations, not 15 independent "
          "model families; the selected-baseline row changes model only between the "
          "blocks in Table~\\ref{tab:inseason_history}.",
          "tab:inseason_history_candidates", "lrrrr",
          r"Historical model configuration & Maize & Rice & Soybean & Wheat", rows)
    annual.to_csv(out / "history_candidates_annual.csv", index=False)
    scores.to_csv(out / "history_candidates_rmse.csv")
    pd.DataFrame(selected_records).to_csv(out / "history_references.csv", index=False)
    return dict(sources=sources, inventory=inventory, selected=selected_records,
                configurations=15, crop_origin_configurations=180,
                evaluation_files=len(annual.groupby(["crop", "origin", "split", "model"])),
                years_per_crop=13, selected_predictions_replayed=True)


def legacy_predictions(crop, labels, sources, configs):
    def load(method, seed):
        path = legacy.prediction_path(crop, method, seed)
        prediction = aligned_prediction(path, labels, "y_pred", "y_true")
        sources[str(path)] = digest(path)
        config_file = path.parent / "config.json"
        config = json.loads(config_file.read_text())
        for key, expected in (("train_years", [1981, 2011]), ("validation_years", [2012, 2012]),
                              ("test_years", [2013, 2016])):
            if config[key] != expected:
                raise ValueError(f"Unexpected legacy {key}: {config_file}")
        sources[str(config_file)] = digest(config_file)
        configs.append(dict(crop=crop, method=method, seed=seed, path=str(config_file),
                            **{k: config[k] for k in ("train_years", "validation_years", "test_years")}))
        return prediction

    values = {}
    for method in legacy.METHODS[:-1]:
        if method in legacy.DETERMINISTIC:
            values[method] = load(method, "deterministic")
        elif method in legacy.CLASSICAL:
            values[method] = load(method, 42)
        elif method == "hgb_mlp_ensemble":
            continue
        else:
            values[method] = np.mean([load(method, seed) for seed in legacy.SEEDS], axis=0)
    values["hgb_mlp_ensemble"] = .5 * (values["hist_gradient_boosting"] + values["mlp"])
    if len(values) != 27:
        raise ValueError("The legacy comparison must retain all 27 baselines")
    return values


def build_legacy_tables(table, recipes, out):
    annual, sources, configs, comparisons, pooled, controls, cutoffs = [], {}, [], [], [], [], []
    names = dict(legacy.LABELS, hist_gradient_boosting="Histogram gradient boosting")
    for crop, recipe in recipes.items():
        root = CURRENT / crop / "origin_2012/seed_42"
        label_file = root / "test_labels.npz"
        with np.load(label_file) as values:
            labels = {k: values[k] for k in (*IDENTITY, "target")}
        identities = np.column_stack([labels[k] for k in IDENTITY])
        if len(np.unique(identities, axis=0)) != len(identities):
            raise ValueError("Duplicate seasonal evaluation identities")
        if tuple(np.unique(labels["year"])) != (2013, 2014, 2015, 2016):
            raise ValueError("Legacy support must be the common four-year period")
        sources[str(label_file)] = digest(label_file)
        prediction_file = root / f"test_{recipe}_biid_010.npz"
        with np.load(prediction_file) as values:
            proposed = values["prediction"]
        sources[str(prediction_file)] = digest(prediction_file)
        predictions = legacy_predictions(crop, labels, sources, configs)
        predictions["inseason"] = proposed
        crop_scores = {}
        pooled_scores = {}
        for method, prediction in predictions.items():
            scores = annual_rmse(labels, prediction)
            crop_scores[method] = scores
            pooled_scores[method] = legacy.rmse(labels["target"], prediction)
            pooled.append(dict(crop=crop, method=method, rmse=pooled_scores[method],
                               mean_annual_rmse=np.mean(list(scores.values()))))
            for year, error in scores.items():
                annual.append(dict(crop=crop, method=method, year=year, rmse=error,
                                   samples=int((labels["year"] == year).sum())))
        best = min(legacy.METHODS[:-1], key=pooled_scores.get)
        paired = [100 * (1 - crop_scores["inseason"][y] / crop_scores[best][y])
                  for y in (2013, 2014, 2015, 2016)]
        comparisons.append(dict(crop=crop, best_method=best, best_label=names[best],
                                baseline_rmse=pooled_scores[best],
                                proposed_rmse=pooled_scores["inseason"],
                                rmse_gain=100 * (1 - pooled_scores["inseason"] / pooled_scores[best]),
                                mean_annual_gain=np.mean(paired),
                                positive_years=int(np.sum(np.array(paired) > 0)),
                                stronger_than=int(sum(pooled_scores["inseason"] < pooled_scores[m]
                                                      for m in legacy.METHODS[:-1]))))
        control = dict(crop=crop, recipe=recipe, biid=pooled_scores["inseason"])
        for name, filename in (("climatology", f"test_{recipe}_climatology_010.npz"),
                               ("observed", f"test_{recipe}_biid_000.npz")):
            file = root / filename
            with np.load(file) as values:
                prediction = values["prediction"]
            annual_rmse(labels, prediction)
            control[name] = legacy.rmse(labels["target"], prediction)
            sources[str(file)] = digest(file)
        control["gain_over_climatology"] = 100 * (1 - control["biid"] / control["climatology"])
        controls.append(control)
        for percent in range(0, 101, 10):
            for mode in ("biid", "climatology"):
                file = root / f"test_{recipe}_{mode}_{percent:03d}.npz"
                with np.load(file) as values:
                    prediction = values["prediction"]
                annual_rmse(labels, prediction)
                error = legacy.rmse(labels["target"], prediction)
                sources[str(file)] = digest(file)
                cutoffs.append(dict(crop=crop, recipe=recipe, percent=percent, mode=mode,
                                    rmse=error, baseline=best, baseline_label=names[best],
                                    baseline_rmse=pooled_scores[best],
                                    gain=100 * (1 - error / pooled_scores[best])))
    annual = pd.DataFrame(annual)
    if len(annual) != 4 * 28 * 4 or annual.duplicated(["crop", "method", "year"]).any():
        raise ValueError("Incomplete full-baseline evaluation")
    scores = pd.DataFrame(pooled).pivot(index="method", columns="crop", values="rmse")
    annual_scores = annual.groupby(["method", "crop"]).rmse.mean().unstack("crop")
    prior_table = ROOT / "visualize/paper_experiments/table_yield_only_baseline_suite_long.csv"
    prior = pd.read_csv(prior_table)
    prior = prior[prior.method.isin(legacy.METHODS[:-1])]
    if len(prior) != 108 or prior.duplicated(["method", "crop"]).any():
        raise ValueError("Incomplete original 27-baseline score table")
    differences = [abs(scores.loc[r.method, r.crop] - r.rmse) for r in prior.itertuples()]
    if max(differences) > 1e-10:
        raise ValueError("Original baseline predictions/ensembles were not reproduced")
    sources[str(prior_table)] = digest(prior_table)
    rows, annual_rows = [], []
    groups = {0: "Statistical rules (7)", 7: "Linear and robust regression (6)",
              13: "Neighborhood and tree models (7)", 20: "Prediction ensemble (1)",
              21: "Neural predictors (6)"}
    for i, method in enumerate(legacy.METHODS[:-1]):
        if i in groups:
            if i:
                rows.append(r"\midrule")
            rows.append(r"\multicolumn{5}{l}{\textit{" + groups[i] + r"}} \\")
        cells = []
        for crop in recipes:
            value = f"{scores.loc[method, crop]:.4f}"
            if scores.loc[method, crop] == scores[crop].min():
                value = r"\textbf{" + value + "}"
            cells.append(value)
        rows.append(names[method] + " & " + " & ".join(cells) + r" \\")
        annual_rows.append(names[method] + " & " + " & ".join(f"{annual_scores.loc[method, c]:.4f}"
                                                               for c in recipes) + r" \\")
    proposed_cells = [r"\textbf{" + f"{scores.loc['inseason', c]:.4f}" + "}"
                      if scores.loc['inseason', c] == scores[c].min()
                      else f"{scores.loc['inseason', c]:.4f}" for c in recipes]
    rows.extend([r"\midrule", r"\method{} (10\% suffix) & " +
                 " & ".join(proposed_cells) + r" \\"])
    annual_rows.extend([r"\midrule", r"\method{} (10\% suffix) & " +
                        " & ".join(f"{annual_scores.loc['inseason', c]:.4f}" for c in recipes) + r" \\"])
    table("inseason_27_baselines_table.tex",
          "Test yield RMSE on identical 2013--2016 grid--years (t/ha; lower is better). "
          "All 27 historical baselines are shown; bold denotes the lowest error per crop. "
          "\\method{} uses the fixed crop-specific signals at a 10\\% unobserved suffix. "
          "Fitting windows and seeds are specified in Section~\\ref{sec:exp_protocol}.",
          "tab:inseason_27_baselines", "lrrrr",
          r"Model & Maize & Rice & Soybean & Wheat", rows)
    table("inseason_27_annual_table.tex",
          "Equal-year sensitivity of the 27-baseline test comparison. Entries average "
          "the four annual RMSEs from 2013--2016 (t/ha), rather than pooling their squared "
          "errors as in Table~\\ref{tab:inseason_27_baselines}. Predictions are unchanged.",
          "tab:inseason_27_annual", "lrrrr",
          r"Model & Maize & Rice & Soybean & Wheat", annual_rows)
    comparison_rows = [r["crop"].title() + " & " + r["best_label"] +
                       f" & {r['baseline_rmse']:.4f} & {r['proposed_rmse']:.4f} & {r['rmse_gain']:+.2f} " +
                       r"\\" for r in comparisons]
    table("inseason_test_comparison_table.tex",
          "Test improvement relative to the explicitly named, lowest-RMSE member of "
          "the 27-baseline library. Gain uses pooled 2013--2016 RMSE. The minimum "
          "baseline is a retrospective comparator, not a test-selected component of our model.",
          "tab:inseason_test_comparison", "llrrr",
          r"Crop & Best of 27 baselines & Baseline RMSE & \method{} RMSE & Gain (\%)", comparison_rows)
    control_rows = [r["crop"].title() + " & " +
                    " & ".join(f"{r[k]:.4f}" for k in ("observed", "climatology", "biid")) +
                    f" & {r['gain_over_climatology']:+.2f} " + r"\\" for r in controls]
    table("inseason_test_completion_table.tex",
          "Test completion controls with the same crop-specific yield readout. Entries "
          "are pooled 2013--2016 RMSE (t/ha). Fully observed trajectories are a diagnostic "
          "reference; the other columns share the same 10\\% cutoff. Gain compares BIID "
          "directly with climatological completion, not with a historical yield baseline.",
          "tab:inseason_test_completion", "lrrrr",
          r"Crop & Fully observed & Climatology & BIID & Gain vs. climatology (\%)", control_rows)
    annual.to_csv(out / "baseline27_annual.csv", index=False)
    scores.reindex([*legacy.METHODS[:-1], "inseason"]).to_csv(out / "baseline27_rmse.csv")
    annual_scores.reindex([*legacy.METHODS[:-1], "inseason"]).to_csv(out / "baseline27_mean_annual_rmse.csv")
    pd.DataFrame(comparisons).to_csv(out / "baseline27_comparison.csv", index=False)
    pd.DataFrame(controls).to_csv(out / "test_completion_controls.csv", index=False)
    pd.DataFrame(cutoffs).to_csv(out / "test_cutoff_curves.csv", index=False)
    return dict(sources=sources, configurations=configs, comparisons=comparisons,
                models=27, years=[2013, 2014, 2015, 2016], all_sample_identities_matched=True,
                target_absolute_tolerance=1e-6, independent_test=False, matched_training=False,
                neural_prediction_ensemble_seeds=list(legacy.SEEDS), classical_seed=42,
                seasonal_seed=42, seasonal_suffix=.1,
                ranking="Pooled test RMSE; post-hoc minimum is descriptive only",
                annual_rmse_sensitivity_reported=True,
                original_baseline_rmse_replay_max_error=max(differences))
