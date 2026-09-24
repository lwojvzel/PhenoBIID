"""Country-specific external-label tables with every registered seed and lead."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cybench_inseason_model_data import ROOT, RESULT, RECIPES, SEEDS, PERCENTS
from cybench_label_protocol import BLOCKS
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from verify_cybench_inseason_inference import verify_record as verify_inference
from verify_cybench_inseason_readouts import verify_record as verify_readouts

OUT = ROOT / 'visualize/paper_experiments/cybench_inseason_v1'
NAMES = {
    'latest_available': 'Latest available yield', 'mean_last_three': 'Past 3-yield mean',
    'mean_last_five': 'Past 5-yield mean', 'past_mean': 'Past yield mean',
    'past_linear_trend': 'Past yield trend', 'history_ridge': 'Ridge (history)',
    'history_random_forest': 'Random Forest (history)', 'history_lightgbm': 'LightGBM (history)',
    'historical_expert': 'Validation-selected history', 'direct_random_forest': 'Random Forest (direct)',
    'direct_lightgbm': 'LightGBM (direct)', 'climatology': 'Climatological completion',
    'biid': 'World-model completion', 'observed': 'Full observed diagnostic',
}
CN = {'maize': '玉米', 'wheat': '小麦', 'DE': '德国', 'FR': '法国', 'PL': '波兰'}
COUNTRIES = ('DE', 'FR', 'PL')
MAIN_MODELS = tuple(name for name in NAMES if name != 'observed')
YEARS = tuple(year for years in BLOCKS.values() for year in years)


def expected_years(country):
    return set(BLOCKS[2009] if country == 'PL' else YEARS)


def validate_annual(annual):
    keys = ['crop', 'country', 'seed', 'model', 'percent', 'year']
    if annual.duplicated(keys).any():
        raise ValueError('Duplicate external annual result')
    if not np.isfinite(annual[['rmse', 'mae']].to_numpy()).all() or (annual.rmse <= 0).any():
        raise ValueError('Nonfinite or nonpositive RMSE')
    year_to_cutoff = {year: cutoff for cutoff, years in BLOCKS.items() for year in years}
    if not np.array_equal(annual.year.map(year_to_cutoff), annual.cutoff):
        raise ValueError('Evaluation year assigned to the wrong fitted cutoff')
    expected_groups = {(crop, country, seed, model, percent)
        for crop in RECIPES for country in COUNTRIES for seed in SEEDS
        for percent in PERCENTS for model in MAIN_MODELS}
    expected_groups |= {(crop, country, seed, 'observed', 0)
        for crop in RECIPES for country in COUNTRIES for seed in SEEDS}
    actual_groups = set(annual[keys[:-1]].itertuples(index=False, name=None))
    if actual_groups != expected_groups:
        raise ValueError('Every registered country, seed, route and lead is required')
    for key, frame in annual.groupby(keys[:-1]):
        if set(frame.year) != expected_years(key[1]):
            raise ValueError('Incomplete country-specific year coverage')
        if (frame.samples <= 0).any():
            raise ValueError('Empty regional evaluation cell')
    common = annual.groupby(['crop', 'country', 'year']).samples.nunique()
    if not (common == 1).all():
        raise ValueError('Methods do not share the canonical regional cohort')


def summarize(annual):
    validate_annual(annual)
    singles = annual.groupby(['crop', 'country', 'seed', 'model', 'percent']).agg(
        rmse=('rmse', 'mean'), mae=('mae', 'mean'), years=('year', 'nunique'), samples=('samples', 'sum')).reset_index()
    scores = singles.groupby(['crop', 'country', 'model', 'percent']).agg(
        rmse_mean=('rmse', 'mean'), rmse_sd=('rmse', 'std'), mae_mean=('mae', 'mean'),
        seeds=('seed', 'nunique'), years=('years', 'min'), samples=('samples', 'min')).reset_index()
    rows = []
    contrasts = [('biid', ref) for ref in ('historical_expert', 'direct_random_forest', 'direct_lightgbm', 'climatology')]
    contrasts += [(model, 'historical_expert') for model in ('direct_random_forest', 'direct_lightgbm', 'climatology')]
    for (crop, country, seed, percent), frame in annual[annual.percent > 0].groupby(['crop', 'country', 'seed', 'percent']):
        pivot = frame.pivot(index='year', columns='model', values='rmse')
        for model, reference in contrasts:
            target, baseline = pivot[model], pivot[reference]
            rows.append(dict(crop=crop, country=country, seed=seed, percent=percent,
                model=model, reference=reference, rmse=float(target.mean()), reference_rmse=float(baseline.mean()),
                gain=float(100*(1-target.mean()/baseline.mean())),
                positive_years=int((target < baseline).sum()), years=len(target)))
    pairs = pd.DataFrame(rows)
    paired = pairs.groupby(['crop', 'country', 'percent', 'model', 'reference']).agg(
        gain_mean=('gain', 'mean'), gain_sd=('gain', 'std'), minimum_seed_gain=('gain', 'min'),
        positive_seeds=('gain', lambda values: int((values > 0).sum())),
        positive_years_min=('positive_years', 'min'), positive_years_max=('positive_years', 'max'),
        years=('years', 'min')).reset_index()
    return singles, scores, pairs, paired


def load_annual(sources):
    hist_path = RESULT / 'history_summary/annual.csv'
    simple_path = ROOT / 'benchmark/results/cybench_inseason_labels_v1/simple_history_annual.csv'
    direct_path = RESULT / 'readout_all_summary/annual.csv'
    world_path = RESULT / 'inference_summary/annual.csv'
    for path in (hist_path, simple_path, direct_path, world_path):
        sources[str(path)] = sha256(path)
    history = pd.read_csv(hist_path)
    history['model'] = history['model'].map({
        'ridge': 'history_ridge', 'random_forest': 'history_random_forest',
        'lightgbm': 'history_lightgbm', 'validation_selected_expert': 'historical_expert'})
    simple = pd.read_csv(simple_path).rename(columns={'method': 'model'})
    simple = pd.concat([simple.assign(seed=seed) for seed in SEEDS], ignore_index=True)
    historical = pd.concat([history, simple], ignore_index=True)
    historical = pd.concat([historical.assign(percent=p) for p in PERCENTS], ignore_index=True)
    direct = pd.read_csv(direct_path)
    direct = direct[direct.route == 'direct'].copy()
    direct['model'] = 'direct_'+direct['model']
    world = pd.read_csv(world_path).rename(columns={'mode': 'model'})
    columns = ['crop', 'cutoff', 'country', 'seed', 'model', 'percent', 'year', 'samples', 'rmse', 'mae']
    return pd.concat([historical[columns], direct[columns], world[columns]], ignore_index=True)


def figures(scores, pairs, annual):
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    selected = ('historical_expert', 'direct_random_forest', 'direct_lightgbm', 'climatology', 'biid', 'observed')
    labels = ['Selected\nhistory', 'Direct\nforest', 'Direct\nLightGBM', 'Climatology', 'World\nmodel', 'Observed\ndiagnostic']
    colors = ['#666666', '#D55E00', '#009E73', '#8C6BB1', '#0072B2', '#888888']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, crop in enumerate(RECIPES):
        for j, country in enumerate(COUNTRIES):
            ax = axes[i, j]
            part = scores[(scores.crop == crop) & (scores.country == country) & scores.percent.isin([0, 10])].set_index('model')
            for x, (method, color) in enumerate(zip(selected, colors)):
                row = part.loc[method]
                ax.errorbar(x, row.rmse_mean, yerr=row.rmse_sd, fmt='o', color=color, capsize=3)
            ax.axvline(4.5, color='#aaaaaa', linestyle=':', linewidth=.8)
            ax.set_xticks(range(6), labels, fontsize=7)
            ax.set_title(f'{crop.title()} | {country} | {len(expected_years(country))} years')
            ax.set_ylabel('Mean annual RMSE (t/ha)')
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Independent regional yields | nominal 10% hidden suffix', fontsize=12)
    fig.text(.5, .015, 'Mean and sample SD across seeds 42, 45, 48. Complete observations are a separate diagnostic.', ha='center')
    fig.tight_layout(rect=(0, .04, 1, .96))
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'regional_primary_rmse.{extension}', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, crop in enumerate(RECIPES):
        for j, country in enumerate(COUNTRIES):
            ax = axes[i, j]
            for method, color in zip(selected[1:5], colors[1:5]):
                frame = pairs[(pairs.crop == crop) & (pairs.country == country) &
                    (pairs.model == method) & (pairs.reference == 'historical_expert')]
                means = frame.groupby('percent').gain.agg(['mean', 'std'])
                ax.errorbar(means.index, means['mean'], yerr=means['std'], fmt='o-', capsize=3,
                    color=color, label=NAMES[method])
            ax.axhline(0, color='#555555', linewidth=.8)
            ax.set_xticks(PERCENTS)
            ax.set_title(f'{crop.title()} | {country}')
            ax.set_xlabel('Nominal hidden suffix (%)')
            ax.set_ylabel('RMSE reduction vs selected history (%)')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Paired lead comparison under supplied weather | all registered seeds', fontsize=12)
    fig.tight_layout(rect=(0, .055, 1, .96))
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'regional_lead_gains.{extension}', dpi=200)
    plt.close(fig)

    matrix = np.full((6, len(YEARS)), np.nan)
    labels = []
    for i, (crop, country) in enumerate((c, k) for c in RECIPES for k in COUNTRIES):
        part = annual[(annual.crop == crop) & (annual.country == country) & (annual.percent == 10)]
        pivot = part.pivot(index=['seed', 'year'], columns='model', values='rmse')
        gains = (100*(1-pivot.biid/pivot.historical_expert)).groupby('year').mean()
        for year, value in gains.items():
            matrix[i, YEARS.index(year)] = value
        labels.append(f'{crop.title()} | {country}')
    limit = max(float(np.nanmax(abs(matrix))), 1.)
    cmap = plt.get_cmap('BrBG').copy()
    cmap.set_bad('#eeeeee')
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(len(YEARS)), YEARS)
    ax.set_yticks(range(6), labels)
    for i in range(6):
        for j in range(len(YEARS)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:+.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(matrix[i, j]) > .65*limit else '#222222')
    fig.colorbar(im, ax=ax, label='Mean seed annual RMSE reduction (%)')
    ax.set_title('World-model completion vs validation-selected history | nominal 10% suffix')
    ax.set_xlabel('Evaluation year; gray cells have no registered Polish evaluation cohort')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'regional_annual_gains.{extension}', dpi=200)
    plt.close(fig)


def write_tables(scores, paired):
    columns = [(crop, country) for crop in RECIPES for country in COUNTRIES]
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize',
        r'\caption{Independent regional statistical yields. Mean annual RMSE (t/ha), averaged over three seeds; parentheses denote seed SD. The main comparison hides a nominal 10\% suffix. Germany and France have 13 evaluation years; Poland has seven. Complete observations are a separate diagnostic.}',
        r'\label{tab:regional_independent_yield}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
        'Method & '+ ' & '.join(f'{crop.title()} {country}' for crop, country in columns)+r' \\', r'\midrule']
    md = ['# 独立区域标签：条件季中验证结果', '',
        '本结果在区域统计标签上重新拟合模型，不是全球权重零样本迁移，也不是CY-Bench官方截断未来天气的协议。全部使用给定实际ERA5-Land；固定玉米GPP、小麦NDVI+GPP。', '',
        '主表为每个国家先逐年计算空间RMSE，再对年份平均，最后报告三个种子的均值和样本标准差。德国、法国各13个登记年份；波兰只有最后7年。历史五种固定公式在种子间复用，Ridge是确定性方法，不能把重复副本当作独立随机性证据。', '',
        '历史参照既包含全部八种固定算法，也单列仅由前置验证确定的专家。后者是前三种学习型方法的选择策略，不是第九种新算法。没有按外部结果剔除方法或种子。', '',
        '## 主结果：名义10%未观测后缀', '',
        '| 方法 | '+ ' | '.join(f'{CN[c]}·{CN[k]}' for c, k in columns)+' |',
        '|---|'+'---:|'*6]
    for model in NAMES:
        if model == 'observed':
            lines.append(r'\midrule')
        values = []
        for crop, country in columns:
            row = scores[(scores.crop == crop) & (scores.country == country) & (scores.model == model) &
                (scores.percent == (0 if model == 'observed' else 10))].iloc[0]
            values.append(f'{row.rmse_mean:.3f} ({row.rmse_sd:.3f})')
        lines.append(NAMES[model]+' & '+' & '.join(values)+r' \\')
        md.append('| '+NAMES[model]+' | '+' | '.join(values)+' |')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    (OUT / 'primary_table.tex').write_text('\n'.join(lines)+'\n')
    md.extend(['', '## 世界模型相对每种强参照的配对结果', '',
        '收益是各种子年均RMSE比值的下降百分比，再对种子取均值；不是实际增产，也不是置信区间。下表全部保留正负数。', '',
        '| 作物·国家 | 未观测比例 | 参照 | 收益均值% | 种子标准差 | 正收益种子 | 胜出年数范围 |',
        '|---|---:|---|---:|---:|---:|---|'])
    for row in paired[paired.model == 'biid'].itertuples():
        md.append(f'| {CN[row.crop]}·{CN[row.country]} | {row.percent}% | {NAMES[row.reference]} | {row.gain_mean:+.2f} | {row.gain_sd:.2f} | {row.positive_seeds}/3 | {row.positive_years_min}至{row.positive_years_max}/{row.years} |')
    md.extend(['', '## 图怎么读', '',
        '主误差图按作物和国家分六个面板，点越低越好，竖线是种子标准差。完整真实轨迹位于分隔线右侧，只是信息参照，不能算季中成绩。', '',
        f'![各路线绝对误差]({OUT / "regional_primary_rmse.png"})', '',
        '提前量图中每条线都相对同一个前置验证历史专家。横轴隐藏更多生长槽，纵轴大于0才优于该历史参照。线间差别用于判断直接预测、常态和状态预测的增量，不能只看世界模型是否在0以上。', '',
        f'![各提前量的配对收益]({OUT / "regional_lead_gains.png"})', '',
        '年度热力图显示三个种子的年度收益均值，正数更好、负数更差；灰色为波兰未登记评价的年份。完整逐种子值在CSV中，不能把均值图当成每个种子都改善。', '',
        f'![全部登记年份]({OUT / "regional_annual_gains.png"})', '',
        '## 状态到产量的解释边界', '',
        '状态误差按产品、国家、年份、种子与提前量单独保存，只评价隐藏且有有效监督的区域槽。NDVI与GPP单位不同，不混成一个原始RMSE。小麦为双产品联合读出，不能把同一个小麦产量差值分别归因于某一个产品。', '',
        '区域聚合为行政区面积乘有效支持，不是作物种植面积加权。德国冬小麦与法国、波兰普通小麦口径不同。未验证水稻、大豆或未见国家；不把这些结果并入全球GDHY主分数。', '',
        '全部权重位于项目benchmark/results/cybench_inseason_models_v1的states、history、heads目录；推理在inference，国家年度分数与核验在同目录inference_summary及inference_verification.json。此脚本只汇总固定实验，不选模型、不修改论文。'])
    (OUT / 'results.md').write_text('\n'.join(md)+'\n')


def main():
    inference = verify_inference()
    readouts = verify_readouts(False, 'all')
    history_path = RESULT / 'history_verification.json'
    label_path = ROOT / 'benchmark/results/cybench_inseason_labels_v1/label_verification.json'
    history, labels = [json.loads(p.read_text()) for p in (history_path, label_path)]
    assert history['passed'] and history['models'] == 54 and labels['passed']
    sources = {}
    for record in (inference, readouts, history):
        for filename, digest in record['sources'].items():
            if filename not in sources:
                assert sha256(Path(filename)) == digest, filename
            sources[filename] = digest
    for path in (history_path, label_path, RESULT / 'inference_verification.json', RESULT / 'readout_all_verification.json'):
        sources[str(path)] = sha256(path)
    simple = ROOT / 'benchmark/results/cybench_inseason_labels_v1/simple_history_annual.csv'
    manifest_path = ROOT / 'benchmark/cache/cybench_inseason_labels_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha256(manifest_path) == labels['manifest_sha256']
    assert sha256(ROOT / 'scripts/verify_cybench_label_protocol.py') == labels['code_sha256']
    assert sha256(simple) == manifest['simple_score_files'][str(simple)]
    sources[str(manifest_path)] = sha256(manifest_path)
    annual = load_annual(sources)
    singles, scores, pairs, paired = summarize(annual)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('annual', annual), ('per_seed', singles), ('mean_sd', scores),
                        ('paired_per_seed', pairs), ('paired_summary', paired)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    state = pd.read_csv(RESULT / 'inference_summary/state_errors.csv')
    state_single = state.groupby(['crop', 'product', 'country', 'seed', 'mode', 'percent']).agg(
        rmse=('rmse', 'mean'), mae=('mae', 'mean'), years=('year', 'nunique')).reset_index()
    state_single.to_csv(OUT / 'state_per_seed.csv', index=False)
    state_single.groupby(['crop', 'product', 'country', 'mode', 'percent']).agg(
        rmse_mean=('rmse', 'mean'), rmse_sd=('rmse', 'std'), years=('years', 'min')).reset_index().to_csv(OUT / 'state_mean_sd.csv', index=False)
    figures(scores, pairs, annual)
    write_tables(scores, paired)
    files = {p.name: sha256(p) for p in OUT.iterdir() if p.suffix in ('.csv', '.png', '.pdf', '.md', '.tex')}
    atomic_json(OUT / 'audit.json', dict(passed=True, source_sha256=sources, files=files,
        script_sha256=sha256(Path(__file__)), all_registered_seeds_and_years_retained=True,
        common_country_year_sample_counts_verified=True, annual_rows=len(annual),
        per_seed_rows=len(singles), paired_rows=len(pairs), global_paper_modified=False))
    print(paired[(paired.model == 'biid') & (paired.percent == 10)].to_string(index=False))


if __name__ == '__main__':
    main()
