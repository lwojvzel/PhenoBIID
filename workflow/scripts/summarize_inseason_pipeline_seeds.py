"""Describe every registered pipeline seed without changing paper scores."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, BLOCKS, YEARS, RECIPES
from inseason_nested_common import LABELS, hashes, verify
from review_revision_data import sha256
from run_inseason_pipeline_seed_inference import CODE, root_for, OUT as RESULTS
from run_review_revision_parallel import atomic_json
from verify_inseason_pipeline_outputs import CONDITIONS, measure

OUT = ROOT / 'visualize/paper_experiments/inseason_pipeline_seeds_v1'
SEEDS = (42, 45, 48)
CROPS = tuple(RECIPES)
NAMES = dict(maize='Maize | GPP', rice='Rice | NDVI', soybean='Soybean | NDVI', wheat='Wheat | NDVI + GPP')


def summarize_annual(frame):
    records = []
    for (crop, seed, percent, mode), group in frame.groupby(['crop', 'seed', 'percent', 'mode'], sort=True):
        if tuple(sorted(group.year)) != YEARS or group.reference.nunique() != 1:
            raise ValueError('Incomplete years or changing historical comparator')
        rmse, baseline = float(group.rmse.mean()), float(group.reference_rmse.mean())
        records.append(dict(crop=crop, seed=int(seed), percent=int(percent), mode=mode,
            mean_annual_rmse=rmse, reference_rmse=baseline, gain=100*(1-rmse/baseline),
            mean_yearwise_gain=float(group.gain.mean()), positive_years=int((group.gain > 0).sum()),
            reference=group.reference.iloc[0]))
    single = pd.DataFrame(records)
    summaries = []
    for (crop, percent, mode), group in single.groupby(['crop', 'percent', 'mode'], sort=True):
        if tuple(sorted(group.seed)) != SEEDS:
            raise ValueError('All three registered seeds must be retained')
        if group.reference.nunique() != 1 or not np.allclose(group.reference_rmse, group.reference_rmse.iloc[0], rtol=0, atol=1e-12):
            raise ValueError('Historical comparator changed across seeds')
        summaries.append(dict(crop=crop, percent=int(percent), mode=mode,
            rmse_mean=float(group.mean_annual_rmse.mean()), rmse_sd=float(group.mean_annual_rmse.std(ddof=1)),
            gain_mean=float(group.gain.mean()), gain_sd=float(group.gain.std(ddof=1)),
            minimum_seed_gain=float(group.gain.min()), maximum_seed_gain=float(group.gain.max()),
            positive_seeds=int((group.gain > 0).sum()),
            positive_years_min=int(group.positive_years.min()), positive_years_max=int(group.positive_years.max()),
            reference=group.reference.iloc[0]))
    return single, pd.DataFrame(summaries)


def ensemble_prediction(predictions):
    values = np.stack(predictions)
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError('Expected three finite, paired prediction vectors')
    return values.mean(axis=0)


def ensembles(sources):
    records = []
    for crop in CROPS:
        for cutoff in BLOCKS:
            for seed in SEEDS:
                root = root_for(crop, cutoff, seed)
                verify(root, hashes(CODE))
                sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
        for percent, mode in CONDITIONS:
            all_labels, all_predictions, all_reference = [], [], []
            for cutoff in BLOCKS:
                candidates, labels = [], None
                for seed in SEEDS:
                    root = root_for(crop, cutoff, seed)
                    with np.load(root / f'tail_{percent:03d}_{mode}.npz') as saved:
                        current = {k: saved[k] for k in LABELS}
                        if labels is not None:
                            for k in LABELS:
                                np.testing.assert_array_equal(current[k], labels[k])
                        labels = current
                        candidates.append(saved['prediction'])
                all_labels.append(labels)
                all_predictions.append(ensemble_prediction(candidates))
                with np.load(root_for(crop, cutoff, 42) / 'labels.npz') as f:
                    all_reference.append(f['strong'])
            labels = {k: np.concatenate([a[k] for a in all_labels]) for k in LABELS}
            score = measure(labels['target'], np.concatenate(all_predictions), labels['year'])
            ref = measure(labels['target'], np.concatenate(all_reference), labels['year'])
            gain = [100*(1-value/ref['per_year_rmse'][year]) for year, value in score['per_year_rmse'].items()]
            records.append(dict(crop=crop, percent=percent, mode=mode,
                mean_annual_rmse=score['mean_annual_rmse'], reference_rmse=ref['mean_annual_rmse'],
                gain=100*(1-score['mean_annual_rmse']/ref['mean_annual_rmse']),
                positive_years=int((np.array(gain) > 0).sum()), mean_yearwise_gain=float(np.mean(gain))))
    return pd.DataFrame(records)


def plots(annual, single, summary, ensemble):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    colors = ('#0072B2', '#D55E00', '#009E73')
    main = single[(single.percent == 10) & (single['mode'] == 'biid')]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(4)
    for seed, shift, color in zip(SEEDS, (-.2, 0, .2), colors):
        values = main[main.seed == seed].set_index('crop').loc[list(CROPS), 'gain']
        ax.scatter(x+shift, values, color=color, s=45, label=f'Random seed {seed}', zorder=3)
    mean = summary[(summary.percent == 10) & (summary['mode'] == 'biid')].set_index('crop').loc[list(CROPS)]
    ax.errorbar(x+.32, mean.gain_mean, yerr=mean.gain_sd, fmt='s', color='#333333',
                capsize=3, markersize=4, label='Mean and sample SD', zorder=4)
    ax.axhline(0, color='#666666', linewidth=.8)
    ax.set_xticks(x, [NAMES[c] for c in CROPS])
    ax.set_ylabel('Mean annual RMSE reduction versus named history (%)')
    ax.set_title('Nominal 10% unobserved suffix | all registered pipeline seeds')
    ax.grid(axis='y', alpha=.2)
    ax.legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.14), frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'seed_comparison.{ext}', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for crop, ax in zip(CROPS, axes.flat):
        for mode, color, label in [('biid', '#D55E00', 'BIID suffix'), ('climatology', '#0072B2', 'Climatological suffix')]:
            group = summary[(summary.crop == crop) & (summary['mode'] == mode)].sort_values('percent')
            xx, yy, sd = group.percent.to_numpy(), group.gain_mean.to_numpy(), group.gain_sd.to_numpy()
            ax.plot(xx, yy, '-o', color=color, label=label, markersize=4)
            ax.fill_between(xx, yy-sd, yy+sd, color=color, alpha=.13)
        observed = summary[(summary.crop == crop) & (summary['mode'] == 'observed')].iloc[0]
        ax.errorbar([0], [observed.gain_mean], yerr=[observed.gain_sd], fmt='D',
                    color='#444444', markersize=4, capsize=3, label='Full observations')
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.set_title(NAMES[crop])
        ax.set_xticks([0, 10, 30, 50])
        ax.set_xlabel('Nominal unobserved suffix (%)')
        ax.set_ylabel('RMSE reduction versus named history (%)')
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Fixed-recipe seed sensitivity | lines: means; bands: across-seed SD', fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, .95))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'seed_lead_sensitivity.{ext}', bbox_inches='tight')
    plt.close(fig)

    main = annual[(annual.percent == 10) & (annual['mode'] == 'biid')]
    limit = max(1., float(np.ceil(np.abs(main.gain).max())))
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.3), sharex=True, layout='constrained')
    for seed, ax in zip(SEEDS, axes):
        matrix = main[main.seed == seed].pivot(index='crop', columns='year', values='gain').loc[list(CROPS), list(YEARS)].to_numpy()
        pic = ax.imshow(matrix, cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
        for i in range(4):
            for j in range(13):
                ax.text(j, i, f'{matrix[i, j]:+.1f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(matrix[i, j]) > .58*limit else '#222222')
        ax.set_yticks(np.arange(4), [c.title() for c in CROPS])
        ax.set_title(f'Random seed {seed}', loc='left', fontsize=10)
    axes[-1].set_xticks(np.arange(13), YEARS)
    axes[-1].set_xlabel('Evaluation year')
    fig.colorbar(pic, ax=list(axes), shrink=.8, label='Annual RMSE reduction versus named history (%)')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'seed_annual_gains.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    audit_file = RESULTS / 'inference_all_verification.json'
    audit = json.loads(audit_file.read_text())
    if (not audit['passed'] or audit['groups'] != 36 or audit['annual_rows'] != 1092 or
            audit['verifier_sha256'] != sha256(ROOT / 'scripts/verify_inseason_pipeline_outputs.py') or
            audit['pipeline_code_sha256'] != hashes(CODE)):
        raise ValueError('Full three-seed output verification required')
    annual_file = Path(audit['annual_csv'])
    if sha256(annual_file) != audit['annual_sha256']:
        raise ValueError('Verified annual scores changed')
    annual = pd.read_csv(annual_file, float_precision='round_trip')
    single, summary = summarize_annual(annual)
    if len(single) != 84 or len(summary) != 28:
        raise ValueError('Incomplete three-seed summaries')
    sources = {str(audit_file): sha256(audit_file), str(annual_file): sha256(annual_file)}
    ensemble = ensembles(sources)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('annual', annual), ('per_seed', single), ('across_seeds', summary), ('ensemble', ensemble)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    plots(annual, single, summary, ensemble)
    atomic_json(OUT / 'audit.json', dict(passed=True, seeds=SEEDS, crops=CROPS,
        annual_rows=len(annual), per_seed_rows=len(single), across_seed_rows=len(summary),
        ensemble_rows=len(ensemble), sources=sources, code_sha256=sha256(Path(__file__)),
        negative_seeds_retained=True, sample_sd_ddof=1, ensemble_separate_from_seed_mean=True,
        historical_reference='Fixed named comparator from the original 13-year table',
        paired_direct_repeats_completed=False, original_paper_modified=False,
        files={p.name: sha256(p) for p in OUT.iterdir() if p.suffix in ('.csv', '.png', '.pdf')}))
    main_rows = single[(single.percent == 10) & (single['mode'] == 'biid')]
    print(main_rows.to_string(index=False), flush=True)
    print(summary[(summary.percent == 10) & (summary['mode'] == 'biid')].to_string(index=False), flush=True)
    print('Ensemble (separate, not a single-seed result):', flush=True)
    print(ensemble[(ensemble.percent == 10) & (ensemble['mode'] == 'biid')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
