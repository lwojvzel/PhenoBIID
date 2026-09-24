"""Matched state baselines and anomaly diagnostics on thirteen evaluation years."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, CACHE, BLOCKS, YEARS, RECIPES, load, partition
from inseason_nested_common import LABELS, RATIOS, folder, verify
from inseason_nested_features import local_climate
from forecast_bridge_data import fit_stats
from ndvi_tail_replacement import tail_mask
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'visualize/paper_experiments/inseason_state_transfer_v1'
NAMES = dict(biid='BIID', climatology='Grid-month climatology',
             previous='Previous trajectory', persistence='Last visible observation')


def persistence(observed, previous, active, tail, climate):
    visible = active & ~tail & np.isfinite(observed)
    slot = np.arange(12)[None, :]
    last = np.where(visible, slot, -1).max(1)
    old_last = np.where(active & np.isfinite(previous), slot, -1).max(1)
    rows = np.arange(len(last))
    value = np.where(last >= 0, observed[rows, np.maximum(last, 0)],
        np.where(old_last >= 0, previous[rows, np.maximum(old_last, 0)], np.nan))
    return np.where(np.isfinite(value[:, None]), value[:, None], climate)


def diagnostic(truth, prediction, climate):
    if not len(truth) or not np.isfinite(prediction).all():
        raise ValueError('Empty or nonfinite matched state population')
    error = prediction-truth
    anomaly, forecast = truth-climate, prediction-climate
    variance = float(anomaly.var())
    forecast_variance = float(forecast.var())
    correlation = float(np.corrcoef(anomaly, forecast)[0, 1]) if min(variance, forecast_variance) > 1e-16 else float('nan')
    return dict(slots=len(truth), mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
        climatology_mse=float(np.mean(anomaly**2)), anomaly_correlation=correlation,
        anomaly_std_ratio=float(np.sqrt(forecast_variance/variance)) if variance > 0 else float('nan'),
        anomaly_bias=float(np.mean(error)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, sources = [], {}
    for crop, recipe in RECIPES.items():
        raw, _ = load(crop)
        sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
        for cutoff in BLOCKS:
            group = partition(raw, cutoff)
            fit, ix = group['full_fit'], group['evaluation']
            active = raw['relative_valid'][ix] > 0
            state = folder('states', crop, cutoff)
            verify(state)
            sources[str(state / 'complete.json')] = sha256(state / 'complete.json')
            with np.load(state / 'full_identities.npz') as saved:
                np.testing.assert_array_equal(saved['raw_indices'], ix)
                for key in LABELS:
                    np.testing.assert_array_equal(saved[key], raw[key][ix])
            stats = fit_stats(raw, fit)
            for product in recipe.split('_'):
                climate = local_climate(raw, fit, ix, product, stats[product]['mean']).astype(float)
                truth = np.asarray(raw[f'observed_{product}'][ix], dtype=float)
                previous = np.asarray(raw[f'previous_{product}'][ix], dtype=float)
                for ratio in RATIOS:
                    tail = tail_mask(active, ratio)
                    valid = tail & active & np.isfinite(truth)
                    path = state / f'full_prefix_{round(100*ratio):02d}.npz'
                    with np.load(path) as saved:
                        predictions = dict(biid=saved[product].astype(float), climatology=climate,
                            previous=np.where(np.isfinite(previous), previous, climate),
                            persistence=persistence(truth, previous, active, tail, climate))
                    sources[str(path)] = sha256(path)
                    for year in BLOCKS[cutoff]:
                        selected = valid & (raw['year'][ix] == year)[:, None]
                        for method, prediction in predictions.items():
                            records.append(dict(crop=crop, product=product, cutoff=cutoff,
                                percent=round(100*ratio), year=year, method=method,
                                **diagnostic(truth[selected], prediction[selected], climate[selected])))
        print('[STATE DIAGNOSTICS]', crop, flush=True)
    annual = pd.DataFrame(records)
    annual.to_csv(OUT / 'annual.csv', index=False)
    rows = []
    for (crop, product, percent, method), group in annual.groupby(['crop', 'product', 'percent', 'method']):
        if tuple(sorted(group.year)) != YEARS or len(group) != 13:
            raise ValueError('Incomplete state baseline years')
        rows.append(dict(crop=crop, product=product, percent=percent, method=method,
            mean_annual_rmse=float(group.rmse.mean()),
            equal_year_mse_skill=100*(1-group.mse.mean()/group.climatology_mse.mean()),
            mean_annual_anomaly_correlation=float(group.anomaly_correlation.mean()),
            mean_annual_anomaly_std_ratio=float(group.anomaly_std_ratio.mean()),
            better_than_climatology_years=int((group.mse < group.climatology_mse).sum()),
            hidden_observed_slots=int(group.slots.sum())))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'scores.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.1))
    fig.subplots_adjust(left=.07, right=.98, bottom=.17, top=.94, hspace=.45, wspace=.38)
    for ax, (crop, product) in zip(axes.flat, [(c, p) for c, r in RECIPES.items() for p in r.split('_')]):
        for method, color in (('biid', '#217c69'), ('previous', '#b5643c'), ('persistence', '#3d70a3')):
            part = frame[frame.crop.eq(crop) & frame['product'].eq(product) & frame.method.eq(method)].sort_values('percent')
            relative = np.sqrt(1-part.equal_year_mse_skill/100)
            ax.plot(part.percent, relative, 'o-', color=color, markersize=4, label=NAMES[method])
        ax.axhline(1, color='.3', linewidth=1, linestyle='--')
        ax.set_title(crop.title()+' | '+product.upper(), fontsize=11)
        ax.set_xticks([10, 30, 50])
        ax.set_xlabel('Unobserved suffix (%)')
        ax.set_ylabel('State RMS error / climatology')
        ax.set_yscale('log')
        ax.set_yticks([.75, 1, 2, 4], ['0.75', '1', '2', '4'])
        ax.set_ylim(.7, 5)
        ax.minorticks_off()
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes.flat[-1].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'state_baselines.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    atomic_json(OUT / 'audit.json', dict(sources=sources, code_sha256=sha256(ROOT / 'scripts/analyze_inseason_state_transfer.py'),
        years=YEARS, state_seed=42, crop_recipes=RECIPES,
        matched_mask='Active hidden slots with finite true target; identical across all four state methods',
        normalization='Physical product units; each evaluation block uses full-fit grid/month climatology',
        persistence_fallback='Last finite active previous-trajectory slot, then fitted grid/month climatology',
        anomaly_correlation='Within-year Pearson of predicted and true deviations from fitted grid/month climatology; constant forecast undefined',
        no_terminal_yield_mechanism_claim=True))
    print(frame[frame.method.eq('biid')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
