"""Dataset support, product availability, and actual thirteen-year footprint."""
import json

import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES, load, partition
from multimodal_baseline import load_coordinates
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from summarize_inseason_13year import write_table

OUT = ROOT / 'visualize/paper_experiments/inseason_dataset_card_v1'


def describe(raw, take):
    active = raw['relative_valid'][take] > 0
    month = raw['source_month'][take].astype(int)
    gaps = np.any((np.diff(month, axis=1) > 1) & active[:, 1:] & active[:, :-1], axis=1)
    count = active.sum(1)
    first = np.where(active, month, 12).min(1)
    last = np.where(active, month, -1).max(1)
    coverage = raw['crop_coverage'][take]
    result = dict(rows=len(take), cells=len(np.unique(raw['row'][take].astype(int)*720+raw['col'][take])),
        years=len(np.unique(raw['year'][take])), active_slots=int(active.sum()), median_active_slots=float(np.median(count)),
        contiguous_partial_percent=100*float(np.mean(~gaps & (count < 12))),
        disjoint_percent=100*float(gaps.mean()), year_round_percent=100*float(np.mean(count == 12)),
        january_december_gap_percent=100*float(np.mean(gaps & (first == 0) & (last == 11))),
        median_crop_area_percent=100*float(np.median(coverage)),
        fraction_below_one_percent=100*float(np.mean(coverage < .01)),
        fraction_above_ten_percent=100*float(np.mean(coverage >= .1)))
    for product in ('lai', 'ndvi', 'gpp'):
        for prefix in ('observed', 'previous'):
            result[f'{prefix}_{product}_available_percent'] = 100*float(np.isfinite(raw[f'{prefix}_{product}'][take])[active].mean())
    result['weather_available_percent'] = 100*float(np.isfinite(raw['weather'][take])[active].mean())
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, periods, yearly, sources = [], [], [], {}
    lat, lon = load_coordinates()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), subplot_kw=dict(projection=ccrs.Robinson()))
    fig.subplots_adjust(left=.03, right=.99, bottom=.14, top=.94, hspace=.20, wspace=.04)
    for ax, crop in zip(axes.flat, RECIPES):
        raw, _ = load(crop)
        sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
        evaluation = np.concatenate([partition(raw, end)['evaluation'] for end in BLOCKS])
        for scope, take in (('record', np.arange(len(raw['year']))), ('evaluation', evaluation)):
            summaries.append(dict(crop=crop, scope=scope, **describe(raw, take)))
        for end in BLOCKS:
            for split, take in partition(raw, end).items():
                periods.append(dict(crop=crop, cutoff=end, split=split,
                    years=';'.join(map(str, np.unique(raw['year'][take]))), rows=len(take),
                    common_readout_rows=int(np.sum(raw['year'][take] >= 1993))))
        for year in np.unique(raw['year']):
            take = np.flatnonzero(raw['year'] == year)
            yearly.append(dict(crop=crop, year=int(year), **describe(raw, take)))
        counts = np.zeros((360, 720), np.int16)
        np.add.at(counts, (raw['row'][evaluation], raw['col'][evaluation]), 1)
        if counts.max() > 13:
            raise ValueError('Duplicate evaluation cell/year')
        np.save(OUT / f'{crop}_evaluation_year_count.npy', counts)
        grid = np.where(counts > 0, counts.astype(float), np.nan)
        im = ax.pcolormesh(lon, lat, grid, transform=ccrs.PlateCarree(), shading='nearest',
                          cmap='viridis', vmin=1, vmax=13, rasterized=True)
        ax.coastlines(resolution='110m', linewidth=.35, color='.3')
        ax.set_global()
        ax.set_title(crop.title(), fontsize=11)
    color_ax = fig.add_axes([.3, .07, .4, .025])
    fig.colorbar(im, cax=color_ax, orientation='horizontal', ticks=[1, 4, 7, 10, 13],
                 label='Number of evaluation years with an eligible sample')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cohort_footprint.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'support.csv', index=False)
    pd.DataFrame(periods).to_csv(OUT / 'periods.csv', index=False)
    pd.DataFrame(yearly).to_csv(OUT / 'annual_support.csv', index=False)
    selected = summary[summary.scope.eq('evaluation')].set_index('crop')
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for i, (product, color) in enumerate(zip(('lai', 'ndvi', 'gpp'), ('#b16635', '#3476a5', '#287c69'))):
        values = selected[f'observed_{product}_available_percent'].to_numpy()
        bars = ax.bar(np.arange(4)+(i-1)*.25, values, width=.23, color=color, label=product.upper())
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set_xticks(np.arange(4), [c.title() for c in RECIPES])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Finite active-slot observations (%)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'product_availability.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    rows = []
    for crop, r in selected.iterrows():
        values = [r[f'observed_{p}_available_percent'] for p in ('lai', 'ndvi', 'gpp')]
        values += [r.median_crop_area_percent, r.contiguous_partial_percent, r.disjoint_percent]
        rows.append(crop.title()+' & '+' & '.join(f'{v:.1f}' for v in values)+r' \\')
    write_table('inseason_dataset_support_table.tex',
        'Data support in the main thirteen-year cohort (percent). LAI, NDVI, and GPP report finite '
        'values among crop-active slots, not satellite retrieval accuracy. Crop share is the median '
        'maximum-monthly growing-area/grid-area proxy. One interval excludes year-round support; '
        'disjoint means separated active natural months, not verified multiple harvests.',
        'tab:inseason_dataset_support', 'Crop & LAI & NDVI & GPP & Crop share & One interval & Disjoint',
        'lrrrrrr', rows)
    rows = []
    for crop in RECIPES:
        for end in BLOCKS:
            sub = {x['split']: x for x in periods if x['crop'] == crop and x['cutoff'] == end}
            rows.append(crop.title()+f' & {end} & '+ ' & '.join(f'{sub[s]["rows"]:,}' for s in
                ('inner_fit', 'inner_validation', 'full_fit', 'evaluation'))+r' \\')
    write_table('inseason_dataset_partitions_table.tex',
        'Grid--year counts for the fixed temporal blocks. Inner validation joins the full fitting '
        'window after duration selection. Counts repeat samples across successive expanding fitting windows.',
        'tab:inseason_dataset_partitions', 'Crop & Fit end & Inner fit & Inner validation & Full fit & Evaluation',
        'lrrrrr', rows)
    for name in ('inseason_dataset_partitions_table.tex', 'inseason_dataset_support_table.tex'):
        path = ROOT / 'Paper/iclr2027/sections' / name
        path.write_text(path.read_text().replace('scripts/summarize_inseason_13year.py',
            'scripts/build_inseason_dataset_card.py').replace(r'\begin{table}[t]', r'\begin{table}[htbp]'))
    atomic_json(OUT / 'audit.json', dict(sources=sources, code_sha256=sha256(ROOT / 'scripts/build_inseason_dataset_card.py'),
        no_new_sample_filter=True, evaluation_years=sorted(y for group in BLOCKS.values() for y in group),
        calendar_classes_not_ground_truth_seasons=True, product_availability_not_crop_purity=True))
    print(selected[['rows', 'cells', 'median_crop_area_percent', 'contiguous_partial_percent', 'disjoint_percent',
                    'observed_lai_available_percent', 'observed_ndvi_available_percent', 'observed_gpp_available_percent']].to_string())


if __name__ == '__main__':
    main()
