"""Paired seed tables and figures; never replace missing seeds with single runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, YEARS, RECIPES
from inseason_direct_seed_protocol import OUT as DIRECT, MODELS, SEEDS
from inseason_nested_common import hashes
from review_revision_data import sha256
from run_inseason_direct_seed import CODE
from run_review_revision_parallel import atomic_json

NAMES = dict(world='World-model pipeline', lightgbm='LightGBM (direct)', random_forest='Random Forest (direct)')
COLORS = dict(world='#0072B2', lightgbm='#009E73', random_forest='#D55E00')


def summarize(frame):
    keys = ['crop', 'seed', 'model', 'percent', 'year']
    if frame.duplicated(keys).any() or not np.isfinite(frame.rmse).all() or (frame.rmse <= 0).any():
        raise ValueError('Invalid or duplicate annual scores')
    singles, pairs = [], []
    for (crop, percent), group in frame.groupby(['crop', 'percent']):
        expected = {(seed, model, year) for seed in SEEDS for model in ('world', *MODELS) for year in YEARS}
        if set(zip(group.seed, group.model, group.year)) != expected:
            raise ValueError('Every paired seed, family, and year is required')
        for (seed, model), part in group.groupby(['seed', 'model']):
            singles.append(dict(crop=crop, seed=seed, model=model, percent=percent, mean_annual_rmse=float(part.rmse.mean())))
        for seed in SEEDS:
            annual = group[group.seed == seed].pivot(index='year', columns='model', values='rmse').loc[list(YEARS)]
            for model in MODELS:
                pairs.append(dict(crop=crop, seed=seed, reference=model, percent=percent,
                    world_rmse=float(annual.world.mean()), direct_rmse=float(annual[model].mean()),
                    rmse_reduction=float((annual[model]-annual.world).mean()),
                    gain=100*(1-float(annual.world.mean()/annual[model].mean())),
                    positive_years=int((annual.world < annual[model]).sum())))
    single, paired = pd.DataFrame(singles), pd.DataFrame(pairs)
    scores, comparison = [], []
    for (crop, model, percent), group in single.groupby(['crop', 'model', 'percent']):
        scores.append(dict(crop=crop, model=model, percent=percent,
            rmse_mean=float(group.mean_annual_rmse.mean()), rmse_sd=float(group.mean_annual_rmse.std(ddof=1))))
    for (crop, model, percent), group in paired.groupby(['crop', 'reference', 'percent']):
        comparison.append(dict(crop=crop, reference=model, percent=percent,
            reduction_mean=float(group.rmse_reduction.mean()), reduction_sd=float(group.rmse_reduction.std(ddof=1)),
            gain_ratio_of_seed_means=100*(1-float(group.world_rmse.mean()/group.direct_rmse.mean())),
            mean_seed_gain=float(group.gain.mean()), seed_gain_sd=float(group.gain.std(ddof=1)),
            positive_seeds=int((group.gain > 0).sum()), minimum_seed_gain=float(group.gain.min()),
            positive_years_min=int(group.positive_years.min()), positive_years_max=int(group.positive_years.max())))
    return single, pd.DataFrame(scores), paired, pd.DataFrame(comparison)


def figures(folder, single, scores, paired, annual):
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for crop, ax in zip(RECIPES, axes.flat):
        part = single[(single.crop == crop) & (single.percent == 10)]
        means = scores[(scores.crop == crop) & (scores.percent == 10)].set_index('model')
        for pos, model in enumerate(('random_forest', 'lightgbm', 'world')):
            values = part[part.model == model].sort_values('seed').mean_annual_rmse.to_numpy()
            ax.scatter(pos+np.array([-.15, 0, .15]), values, color=COLORS[model], s=24)
            ax.errorbar(pos+.26, means.loc[model].rmse_mean, yerr=means.loc[model].rmse_sd,
                color='#333333', fmt='s', markersize=3, capsize=3)
        ax.set_xticks([0, 1, 2], ['Random Forest', 'LightGBM', 'World model'])
        ax.set_title(crop.title()+' | '+RECIPES[crop].upper().replace('_', ' + '))
        ax.set_ylabel('Mean annual RMSE (t/ha), lower is better')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Paired full-pipeline seeds at a 10% unobserved suffix', fontsize=11)
    fig.text(.5, .02, 'Colored points: seeds 42, 45, 48; black square and whisker: mean and sample SD', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'paired_primary_rmse.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for crop, ax in zip(RECIPES, axes.flat):
        for model in MODELS:
            group = paired[(paired.crop == crop) & (paired.reference == model)]
            result = group.groupby('percent').gain.agg(['mean', 'std'])
            xx, yy, sd = result.index.to_numpy(), result['mean'].to_numpy(), result['std'].to_numpy()
            ax.errorbar(xx, yy, yerr=sd, fmt='o-', capsize=4, color=COLORS[model], label=NAMES[model])
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set_title(crop.title())
        ax.set_xticks(sorted(paired.percent.unique()))
        ax.set_xlabel('Nominal unobserved suffix (%)')
        ax.set_ylabel('World-model RMSE reduction vs direct (%)')
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Paired gain across all registered seeds | mean and sample SD', fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'paired_lead_gains.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(stage):
    file = DIRECT / f'{stage}_verification.json'
    audit = json.loads(file.read_text())
    percents = (10,) if stage == 'primary' else (10, 30, 50)
    expected_conditions = 4*3*3*2*len(percents)
    if (not audit['passed'] or audit['conditions'] != expected_conditions or tuple(audit['seeds']) != SEEDS or
            audit['training_code_sha256'] != hashes(CODE) or not audit['same_seed_world_labels_matched'] or
            audit['verifier_sha256'] != sha256(ROOT / 'scripts/verify_inseason_direct_seed_outputs.py')):
        raise ValueError('Complete independently verified paired direct seeds required')
    for filename, expected in {**audit['completed_sources'], **audit['output_sha256']}.items():
        if sha256(ROOT / filename) != expected:
            raise ValueError('Verified direct output changed')
    world_file = ROOT / 'benchmark/results/inseason_pipeline_seeds_v1/inference_all_verification.json'
    world_audit = json.loads(world_file.read_text())
    if not world_audit['passed'] or world_audit['groups'] != 36 or sha256(Path(world_audit['annual_csv'])) != world_audit['annual_sha256']:
        raise ValueError('Verified world seed scores required')
    world = pd.read_csv(world_audit['annual_csv'])
    world = world[(world['mode'] == 'biid') & world.percent.isin(percents)].assign(model='world')
    direct = pd.read_csv(DIRECT / f'{stage}_verified_annual.csv')
    columns = ['crop', 'seed', 'model', 'percent', 'year', 'rmse']
    annual = pd.concat([world[columns], direct[columns]], ignore_index=True)
    if set(annual.crop) != set(RECIPES) or set(annual.percent) != set(percents):
        raise ValueError('Incomplete crop or lead matrix')
    single, scores, paired, comparison = summarize(annual)
    folder = ROOT / 'visualize/paper_experiments/inseason_paired_seeds_v1' / stage
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in [('annual', annual), ('per_seed', single), ('mean_sd', scores),
            ('paired_per_seed', paired), ('paired_summary', comparison)]:
        frame.to_csv(folder / f'{name}.csv', index=False)
    figures(folder, single, scores, paired, annual)
    atomic_json(folder / 'audit.json', dict(passed=True, stage=stage, seeds=SEEDS, percents=percents,
        source_sha256={str(file): sha256(file), str(world_file): sha256(world_file)},
        code_sha256=sha256(Path(__file__)), negative_results_retained=True,
        standard_deviation_not_confidence_interval=True, paper_modified=False,
        file_sha256={p.name: sha256(p) for p in folder.iterdir() if p.suffix in ('.png', '.pdf', '.csv')}))
    print(scores[scores.percent == 10].to_string(index=False), flush=True)
    print(comparison[comparison.percent == 10].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('primary', 'all'), required=True)
    main(parser.parse_args().stage)
