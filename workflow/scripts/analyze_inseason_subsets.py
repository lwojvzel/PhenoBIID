"""Fixed calendar/area sensitivity without changing main evaluation identities."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_crop_area_fraction import grid_area_hectares
from inseason_13year_data import ROOT, BLOCKS, YEARS, RECIPES, load, partition
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from summarize_inseason_nested import collect

OUT = ROOT / 'visualize/paper_experiments/inseason_support_sensitivity_v1'
LABELS = {'all': 'All cells', 'area_low': 'Crop share < 1%',
          'area_mid': 'Crop share 1-5%', 'area_high': 'Crop share >= 5%',
          'contiguous': 'One active interval', 'disjoint': 'Disjoint intervals',
          'year_round': 'Year-round support'}


def subsets(raw, ix):
    active = raw['relative_valid'][ix] > 0
    months = raw['source_month'][ix].astype(int)
    gap = np.any((np.diff(months, axis=1) > 1) & active[:, :-1] & active[:, 1:], axis=1)
    share = raw['crop_coverage'][ix]
    return dict(all=np.ones(len(ix), dtype=bool), area_low=share < .01,
        area_mid=(share >= .01) & (share < .05), area_high=share >= .05,
        contiguous=~gap & (active.sum(1) < 12), disjoint=gap, year_round=active.sum(1) == 12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, annual, sources = [], [], {}
    for crop in RECIPES:
        labels, predicted, reference = collect(crop, ('observed',), 42, sources, [])
        raw, _ = load(crop, verify=False)
        ix = np.concatenate([partition(raw, end)['evaluation'] for end in BLOCKS])
        groups = subsets(raw, ix)
        area = raw['crop_coverage'][ix]*grid_area_hectares(-89.75+.5*raw['row'][ix])
        truth, years = labels['target'], labels['year']
        methods = dict(original_frozen=predicted['original_frozen', 0, 10, 'biid'],
            observed_all=predicted['observed_all', 0, 10, 'biid'],
            direct_lightgbm=predicted['direct_lightgbm', 10, 10, 'prefix'])
        base = predicted['library_reference', 0, 0, 'history']
        for group, selected in groups.items():
            for weighting in (('grid_equal', 'growing_area_proxy') if group == 'all' else ('grid_equal',)):
                weight = area if weighting == 'growing_area_proxy' else np.ones(len(ix))
                for method, prediction in methods.items():
                    measured = []
                    for year in YEARS:
                        use = selected & (years == year)
                        if not use.any() or weight[use].sum() <= 0:
                            continue
                        candidate_rmse = float(np.sqrt(np.average((prediction[use]-truth[use])**2, weights=weight[use])))
                        baseline_rmse = float(np.sqrt(np.average((base[use]-truth[use])**2, weights=weight[use])))
                        item = dict(crop=crop, subset=group, weighting=weighting, method=method,
                            year=year, rows=int(use.sum()), rmse=candidate_rmse, reference_rmse=baseline_rmse)
                        measured.append(item)
                        annual.append(item)
                    if not measured:
                        raise ValueError('Empty prespecified sensitivity subset')
                    a = np.array([r['rmse'] for r in measured])
                    b = np.array([r['reference_rmse'] for r in measured])
                    rows.append(dict(crop=crop, subset=group, weighting=weighting, method=method,
                        reference=reference.baseline_label, years=len(a), rows=int(selected.sum()),
                        mean_annual_rmse=float(a.mean()), mean_annual_reference_rmse=float(b.mean()),
                        score_gain=100*(1-a.mean()/b.mean()), annual_gain=float(np.mean(100*(1-a/b))),
                        positive_years=int((a < b).sum()), minimum_rows_per_year=min(r['rows'] for r in measured)))
        print('[SUPPORT SENSITIVITY]', crop, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'scores.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT / 'annual.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.8))
    fig.subplots_adjust(left=.19, right=.98, bottom=.12, top=.94, hspace=.35, wspace=.63)
    for ax, crop in zip(axes.flat, RECIPES):
        for offset, method, title, color in ((-.18, 'original_frozen', 'Original frozen pipeline', '#237d6b'),
                                            (.18, 'direct_lightgbm', 'Direct LightGBM', '#b7683d')):
            part = frame[frame.crop.eq(crop) & frame.method.eq(method) & frame.weighting.eq('grid_equal')].set_index('subset').loc[list(LABELS)]
            ax.barh(np.arange(len(LABELS))+offset, part.score_gain, height=.32, label=title, color=color)
        labels = []
        for name, label in LABELS.items():
            row = part.loc[name]
            if row.years < 13:
                label += f'\n({int(row.years)} years; n={int(row.rows)})'
            elif row.minimum_rows_per_year < 20:
                label += f'\n(min yearly n={int(row.minimum_rows_per_year)})'
            labels.append(label)
        ax.set_yticks(np.arange(len(LABELS)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='.3', linewidth=.8)
        ax.set_title(crop.title(), fontsize=11)
        ax.set_xlabel('Equal-year RMSE reduction (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'support_sensitivity.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    atomic_json(OUT / 'audit.json', dict(sources=sources,
        code_sha256=sha256(ROOT / 'scripts/analyze_inseason_subsets.py'),
        thresholds=(.01, .05), years=YEARS, no_new_model_training=True,
        no_main_cohort_filter_change=True, no_per_subset_reference_reselection=True,
        calendar_classes_not_verified_season_types=True,
        area_weight='Clipped maximum-monthly crop-share proxy multiplied by spherical 0.5-degree cell area',
        geographic_generalization_claim=False))
    print(frame[frame.subset.eq('all')][['crop', 'method', 'weighting', 'score_gain', 'positive_years']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
