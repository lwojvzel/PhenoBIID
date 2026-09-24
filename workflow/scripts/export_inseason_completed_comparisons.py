"""Recompute completed comparison summaries and export manuscript tables only."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from summarize_inseason_paired_seeds import summarize as global_summary
from summarize_cybench_inseason import summarize as regional_summary

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / 'visualize/paper_experiments/inseason_paired_seeds_v1/all'
REGIONAL = ROOT / 'visualize/paper_experiments/cybench_inseason_v1'
OUT = ROOT / 'visualize/paper_experiments/inseason_completed_comparisons_20260909'


def validate(folder, sources, regional=False):
    path = folder / 'audit.json'
    audit = json.loads(path.read_text())
    assert audit['passed']
    for filename, digest in audit['source_sha256'].items():
        if filename not in sources:
            assert sha256(Path(filename)) == digest, filename
        else:
            assert sources[filename] == digest, filename
        sources[filename] = digest
    for filename, digest in audit['files' if regional else 'file_sha256'].items():
        target = folder / filename
        assert sha256(target) == digest, target
        sources[str(target)] = digest
    sources[str(path)] = sha256(path)
    script = 'summarize_cybench_inseason.py' if regional else 'summarize_inseason_paired_seeds.py'
    assert sha256(ROOT / 'scripts' / script) == audit['script_sha256' if regional else 'code_sha256']
    sources[str(ROOT / 'scripts' / script)] = sha256(ROOT / 'scripts' / script)


def compare_frames(actual, expected, keys):
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    validate(GLOBAL, sources)
    validate(REGIONAL, sources, regional=True)
    global_frames = global_summary(pd.read_csv(GLOBAL / 'annual.csv'))
    regional_frames = regional_summary(pd.read_csv(REGIONAL / 'annual.csv'))
    names = ('per_seed', 'mean_sd', 'paired_per_seed', 'paired_summary')
    for folder, frames, country in ((GLOBAL, global_frames, False), (REGIONAL, regional_frames, True)):
        for name, frame in zip(names, frames):
            dimensions = [key for key in ('crop', 'country', 'model', 'reference', 'percent', 'seed') if key in frame]
            compare_frames(frame, pd.read_csv(folder / f'{name}.csv'), dimensions)
    direct_audit = json.loads((ROOT / 'benchmark/results/inseason_direct_seeds_v1/all_verification.json').read_text())
    assert direct_audit['conditions'] == 216 and direct_audit['prediction_rows_replayed'] == 8005878
    assert direct_audit['maximum_weight_replay_error'] == 0 and direct_audit['same_seed_world_labels_matched']
    paired = global_frames[-1]
    assert len(paired) == 24 and (paired.gain_ratio_of_seed_means > 0).all()
    assert (paired.positive_seeds == 3).sum() == 22
    lines = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Paired direct-model comparisons at all three cutoffs. Gain is the percentage reduction in the ratio of seed-mean annual RMSE; both direct families and all seeds are retained. Positive years give the range across seeds.}',
        r'\label{tab:inseason_paired_all_leads}',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Crop & Direct reference & Suffix (\%) & Gain (\%) & Positive seeds & Positive years \\', r'\midrule']
    for row in paired.itertuples():
        model = 'LightGBM' if row.reference == 'lightgbm' else 'Random Forest'
        years = f'{row.positive_years_min}--{row.positive_years_max}/13'
        lines.append(f'{row.crop.title()} & {model} & {row.percent} & {row.gain_ratio_of_seed_means:+.2f} & {row.positive_seeds}/3 & {years}'+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (OUT / 'global_leads_table.tex').write_text('\n'.join(lines)+'\n')
    region = regional_frames[-1]
    region = region[region.model == 'biid']
    assert len(region) == 72
    primary = region[region.percent == 10]
    assert (primary[primary.reference == 'historical_expert'].gain_mean > 0).sum() == 5
    lines = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Independent-label paired gains at all cutoffs (\%). For each seed, divide mean annual RMSE by that of the named reference, then average gains across seeds. Positive values favor world-model completion. All countries and cutoffs are retained.}',
        r'\label{tab:regional_independent_leads}', r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'Crop & Country & Suffix (\%) & History & Direct RF & Direct LGBM & Climatology \\', r'\midrule']
    for (crop, country, percent), frame in region.groupby(['crop', 'country', 'percent']):
        values = frame.set_index('reference').gain_mean
        numbers = [f'{values[ref]:+.2f}' for ref in ('historical_expert', 'direct_random_forest', 'direct_lightgbm', 'climatology')]
        lines.append(f'{crop.title()} & {country} & {percent} & '+' & '.join(numbers)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (OUT / 'regional_leads_table.tex').write_text('\n'.join(lines)+'\n')
    for path in OUT.glob('*_table.tex'):
        sources[str(path)] = sha256(path)
    sources[str(Path(__file__))] = sha256(Path(__file__))
    atomic_json(OUT / 'audit.json', dict(passed=True,
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        global_configurations=216, global_comparisons=24, regional_comparisons=72,
        summaries_recomputed_from_annual_rows=True, negative_results_retained=True,
        sources=sources, new_fits=0))
    print(f'Completed evidence verified: {len(sources)} sources; 24 global and 72 regional contrasts.')


if __name__ == '__main__':
    main()
