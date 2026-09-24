"""Audited equal-year comparison of nested yield readouts and frozen controls."""
import argparse
from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, CACHE, BLOCKS, YEARS, RECIPES, load, partition
from inseason_nested_common import LABELS, RATIOS, folder, verify, branches
from inseason_13year_uncertainty import historical
from review_revision_data import sha256
from run_inseason_13year_direct import root_for as direct_root, CODE as DIRECT_CODE
from run_ndvi_signal_permutation import check_files
from run_inseason_direct_baselines import score
from run_inseason_nested_readout import SUITES, outputs_for
from run_review_revision_parallel import atomic_json
from summarize_inseason_13year import annual_gain

OUT = ROOT / 'visualize/paper_experiments/inseason_nested_world_v1'
OLD = ROOT / 'visualize/paper_experiments/inseason_13year_v1'
CN = dict(maize='玉米', rice='水稻', soybean='大豆', wheat='小麦')


def checked_prediction(path, labels, sources):
    with np.load(path) as saved:
        for key in LABELS:
            np.testing.assert_array_equal(saved[key], labels[key])
        prediction = saved['prediction'].copy()
    if prediction.shape != labels['target'].shape or not np.isfinite(prediction).all():
        raise ValueError(f'Invalid prediction: {path}')
    sources[str(path)] = sha256(path)
    return prediction


def audited_root(root, sources):
    verify(root)
    path = root / 'complete.json'
    sources[str(path)] = sha256(path)
    config = root / 'config.json'
    if config.exists():
        for source, digest in json.loads(config.read_text()).get('sources', {}).items():
            if sha256(ROOT / source) != digest:
                raise ValueError(f'Changed upstream model: {source}')
            sources[source] = digest


def collect(crop, suites, seed, sources, registry):
    raw, _ = load(crop)
    sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
    compare_path = OLD / 'comparisons.csv'
    sources[str(compare_path)] = sha256(compare_path)
    reference = pd.read_csv(compare_path).set_index('crop').loc[crop]
    collected, labels = defaultdict(list), {k: [] for k in LABELS}
    for cutoff in BLOCKS:
        ix = partition(raw, cutoff)['evaluation']
        part = {k: raw[k][ix] for k in LABELS}
        for key in LABELS:
            labels[key].append(part[key])
        collected['library_reference', 0, 0, 'history'].append(
            historical(crop, cutoff, reference.baseline, part, sources))
        for percent in (10, 30, 50):
            path = CACHE / crop / f'world_{cutoff}_{RECIPES[crop]}_biid_{percent:03d}.npy'
            collected['original_frozen', 0, percent, 'biid'].append(np.load(path))
            sources[str(path)] = sha256(path)
        for model in ('lightgbm', 'gru', 'transformer'):
            root = direct_root(crop, cutoff, model)
            marker = json.loads((root / 'complete.json').read_text())
            check_files(root, marker['files'])
            if marker['code_sha256'] != {n: sha256(ROOT / 'scripts' / n) for n in DIRECT_CODE}:
                raise ValueError('Changed direct-model implementation')
            sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
            for percent in (10, 30, 50):
                path = root / f'tail_{percent:02d}/evaluation_predictions.npz'
                collected['direct_'+model, percent, percent, 'prefix'].append(
                    checked_prediction(path, part, sources))
        root = folder('history', crop, cutoff, seed)
        audited_root(root, sources)
        with np.load(root / 'full/predictions.npz') as saved:
            pos = np.searchsorted(saved['raw_indices'], ix)
            np.testing.assert_array_equal(saved['raw_indices'][pos], ix)
            for key in LABELS:
                np.testing.assert_array_equal(saved[key][pos], part[key])
            own = np.zeros(len(ix))
            for name in ('trend', 'mlp', *(['tabm'] if crop == 'soybean' else [])):
                collected['trained_'+name, 0, 0, 'history'].append(saved[name][pos])
            for branch, weight in branches(crop):
                own += weight*saved[branch][pos]
            collected['own_anchor', 0, 0, 'history'].append(own)
        for stage in ('inner', 'full'):
            for weight in sorted((root / stage).glob('*/model.pt')):
                metadata = weight.parent / 'training.json'
                if not metadata.exists():
                    metadata = weight.parent / 'metrics.json'
                registry.append(dict(crop=crop, cutoff=cutoff, seed=seed, stage=stage,
                    component='history', condition=weight.parent.name, training_percent=0,
                    weight=str(weight), weight_sha256=sha256(weight), metadata=str(metadata),
                    complete_manifest=str(root / 'complete.json')))
        for suite in suites:
            root = folder('readout_'+suite, crop, cutoff, seed)
            audited_root(root, sources)
            for condition in SUITES[suite]:
                for ratio in ((0.,) if condition.startswith('observed') else RATIOS):
                    train = round(100*ratio)
                    dest = root / condition / f'train_tail_{train:02d}'
                    for issue, completion in outputs_for(condition, ratio):
                        percent = round(100*issue)
                        path = dest / f'tail_{percent:03d}_{completion}.npz'
                        collected[condition, train, percent, completion].append(
                            checked_prediction(path, part, sources))
                    for stage in ('inner', 'full'):
                        for branch, _ in branches(crop):
                            weight = dest / stage / branch / 'model.joblib'
                            meta = weight.parent / 'training.json'
                            record = json.loads(meta.read_text())
                            if record['selected_trees'] < 1 or max(record['training_years']) > cutoff:
                                raise ValueError('Invalid terminal training provenance')
                            registry.append(dict(crop=crop, cutoff=cutoff, seed=seed, stage=stage,
                                component='yield_readout', condition=condition,
                                training_percent=train, branch=branch, weight=str(weight),
                                weight_sha256=sha256(weight), metadata=str(meta),
                                selected_trees=record['selected_trees'],
                                training_rows=record['training_rows'],
                                complete_manifest=str(root / 'complete.json')))
    labels = {k: np.concatenate(v) for k, v in labels.items()}
    if tuple(np.unique(labels['year'])) != YEARS:
        raise ValueError('Missing registered evaluation years')
    identity = np.rec.fromarrays([labels[k] for k in ('year', 'row', 'col')])
    if len(np.unique(identity)) != len(identity):
        raise ValueError('Duplicated evaluation identity')
    for key, chunks in collected.items():
        if len(chunks) != len(BLOCKS):
            raise ValueError(f'Partial model cannot enter summary: {key}')
    return labels, {k: np.concatenate(v) for k, v in collected.items()}, reference


def grouped_score(labels, prediction):
    if tuple(np.unique(labels['year'])) != YEARS:
        raise ValueError('A main score must contain all thirteen years')
    if prediction.shape != labels['target'].shape or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite or misaligned main prediction')
    return score(labels['target'], prediction, labels['year'])


def plots(frame, dest, adapted):
    styles = [('observed_all', 'biid', 'Original-window observed training', '#717171'),
              ('observed_common', 'biid', 'Common-window observed training', '#ba663c')]
    if adapted:
        styles += [('biid', 'biid', 'Forward BIID trajectory training', '#217d6b'),
                   ('climatology', 'climatology', 'Climatology trajectory training', '#396f9c'),
                   ('prefix', 'prefix', 'Observed-prefix training', '#9d6688')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3))
    fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.94, hspace=.43, wspace=.29)
    for ax, crop in zip(axes.flat, RECIPES):
        part = frame[frame.crop.eq(crop)]
        for name, mode, label, color in styles:
            selected = part[part.condition.eq(name) & part.completion.eq(mode) & part.percent.gt(0)].sort_values('percent')
            ax.plot(selected.percent, selected.library_score_gain, 'o-', markersize=4, label=label, color=color)
        ax.axhline(0, color='.25', linestyle='--', linewidth=1)
        ax.set_title(crop.title()+' | '+RECIPES[crop].upper().replace('_', ' + '), fontsize=11)
        ax.set_xticks([10, 30, 50])
        ax.set_xlabel('Nominal unobserved suffix (%)')
        ax.set_ylabel('Equal-year RMSE reduction (%)')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False, fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(dest / f'nested_readout_comparison.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    if not adapted:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.9))
    fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.94, hspace=.43, wspace=.29)
    for ax, crop in zip(axes.flat, RECIPES):
        for mode, label, color in (('biid', 'BIID completion', '#217d6b'),
                                   ('climatology', 'Climatology replacement', '#ba663c'),
                                   ('prefix', 'No future completion', '#396f9c')):
            sub = frame[frame.crop.eq(crop) & frame.condition.eq('biid') & frame.completion.eq(mode)].sort_values('percent')
            ax.plot(sub.percent, sub.mean_annual_rmse, 'o-', label=label, color=color, markersize=4)
        ax.set_title(crop.title(), fontsize=11)
        ax.set_xticks([10, 30, 50])
        ax.set_xlabel('Nominal unobserved suffix (%)')
        ax.set_ylabel('Thirteen-year mean RMSE (t/ha)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False, fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(dest / f'same_readout_completion.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def report(frame, dest, adapted):
    text = ['# 十三年主实验：前置验证重拟合结果', '',
        '主评价固定为2002–2004、2006–2008、2010–2016，共13年。以下每个RMSE均先逐年计算，再对13年等权平均；不是对三个时间块等权。', '',
        '## 这次改变了什么', '',
        '保留玉米GPP、水稻NDVI、大豆NDVI、小麦NDVI+GPP及原状态结构；历史专家和产量树只用评价之前的内部年份选训练长度，再重拟合。历史专家不允许初始化模型参选。状态权重复用并核验训练截止。原冻结结果另列，没有覆盖。', '',
        '完整观测训练头分别使用原始窗口和1993+公共窗口。预测适配比较只使用公共窗口，历史专家、输入维度、树预算和统计规则不变。头训练行的历史预测仍来自in-sample历史专家，因此本实验没有同时解决历史残差训练与推理分布差异。', '',
        '## 10%名义后缀结果', '',
        '表中“收益”是相对具名最强历史库参照的年均RMSE降幅，正值更好。正年份也必须查看；原冻结世界模型保留原来的开发选型来源，不应当成新前置验证模型。', '',
        '| 作物 | 模型/训练方式 | 年均RMSE | 收益 | 正年份 |',
        '|---|---|---:|---:|---:|']
    names = [('library_reference', 'history', '最强历史库参照'),
             ('own_anchor', 'history', '本次训练并冻结的历史锚点'),
             ('original_frozen', 'biid', '原冻结世界模型'),
             ('direct_lightgbm', 'prefix', '直接LightGBM'),
             ('observed_all', 'biid', '完整观测训练，原窗口'),
             ('observed_common', 'biid', '完整观测训练，公共窗口')]
    if adapted:
        names += [('prefix', 'prefix', '仅前缀训练'), ('climatology', 'climatology', '常态补全训练'),
                  ('biid', 'biid', '前向BIID预测轨迹训练')]
    for crop in RECIPES:
        for name, mode, label in names:
            part = frame[frame.crop.eq(crop) & frame.condition.eq(name) & frame.completion.eq(mode)
                         & frame.percent.eq(0 if mode == 'history' else 10)]
            if len(part) != 1:
                raise ValueError(f'Nonunique report row: {crop}/{name}/{mode}')
            row = part.iloc[0]
            text.append(f'| {CN[crop]} | {label} | {row.mean_annual_rmse:.4f} | {row.library_score_gain:+.2f}% | {row.library_positive_years}/13 |')
    text += ['', '具体历史参照：', '']
    for crop in RECIPES:
        row = frame[frame.crop.eq(crop)].iloc[0]
        text.append(f'- {CN[crop]}：{row.library_reference_label}。')
    text += ['', '## 如何看图', '',
        f'![不同训练输入方式]({dest}/nested_readout_comparison.png)', '',
        '横轴越大，未观测、需要补全的活动槽越多；不是统一天数。纵轴是相对同一具名历史参照的收益，超过零线才胜过该参照。灰色与橙色之差主要检验删除早期头训练数据的影响；其它曲线与橙色比较时样本一致，用来检验让训练输入适配推理条件是否有用。不同作物的纵轴范围可能不同，应读数值而不是只看曲线高度。', '']
    if adapted:
        text += [f'![固定预测适配头的补全替换]({dest}/same_readout_completion.png)', '',
            '这张图固定BIID预测轨迹训练得到的产量头，只改变评价时的未来补全。纵轴越低越好；BIID低于常态才说明该头确实从动态预测获得超过静态常态的收益。这与第一张图分别重新训练各条件的实验不同。', '']
        text += ['## 结果判读', '']
        for crop in RECIPES:
            sub = frame[frame.crop.eq(crop) & frame.percent.eq(10)]
            adapted_row = sub[sub.condition.eq('biid') & sub.completion.eq('biid')].iloc[0]
            obs = sub[sub.condition.eq('observed_common') & sub.completion.eq('biid')].iloc[0]
            climo = sub[sub.condition.eq('biid') & sub.completion.eq('climatology')].iloc[0]
            text.append(f'- {CN[crop]}：预测适配后的年均RMSE为{adapted_row.mean_annual_rmse:.4f}，'
                f'相对公共窗口观测训练头变化{100*(adapted_row.mean_annual_rmse/obs.mean_annual_rmse-1):+.2f}%；'
                f'相对同头常态替换的降幅{100*(1-adapted_row.mean_annual_rmse/climo.mean_annual_rmse):+.2f}%。'
                f'相对历史库参照收益{adapted_row.library_score_gain:+.2f}%，{adapted_row.library_positive_years}/13年更好。')
    text += ['', '## 权重与审计', '',
        f'- 新实验权重总目录：`{ROOT}/benchmark/results/inseason_nested_world_v1/pipelines/`。',
        f'- 每个内部选择/完整重拟合权重的路径、SHA256、训练元数据见 `{dest}/checkpoint_registry.csv`。',
        f'- 逐年结果：`{dest}/annual.csv`；全部条件：`{dest}/scores.csv`；源文件核验：`{dest}/audit.json`。',
        '- 当前固定seed42；不是多种子结果。作物配方继承前期探索，因此前置训练窗口并不能消除配方选择的历史影响。',
        '- 这些实验用于诊断并保留完整记录；在模型选择、机制与重复性证据齐全前，不把某一条最优曲线直接宣布为最终投稿结论。', '']
    (dest / '结果说明.md').write_text('\n'.join(text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--observed-only', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    suites = ('observed',) if args.observed_only else ('observed', 'adapted')
    dest = OUT / ('observed_only' if args.observed_only else 'complete') / f'seed_{args.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    rows, annual, sources, registry = [], [], {}, []
    for crop in RECIPES:
        labels, predictions, reference = collect(crop, suites, args.seed, sources, registry)
        historical_ref = predictions['library_reference', 0, 0, 'history']
        anchor = predictions['own_anchor', 0, 0, 'history']
        for (condition, train, percent, completion), prediction in predictions.items():
            metric = grouped_score(labels, prediction)
            lib = annual_gain(labels['target'], prediction, historical_ref, labels['year'])
            own = annual_gain(labels['target'], prediction, anchor, labels['year'])
            spec = dict(crop=crop, seed=args.seed, condition=condition, training_percent=train,
                        percent=percent, completion=completion)
            rows.append(dict(**spec, **metric, library_reference=reference.baseline,
                library_reference_label=reference.baseline_label,
                library_score_gain=lib['annual_mean_rmse_gain'],
                library_annual_gain=lib['mean_annual_gain'], library_positive_years=lib['positive_years'],
                anchor_score_gain=own['annual_mean_rmse_gain'], anchor_positive_years=own['positive_years']))
            for y, rmse in metric['per_year_rmse'].items():
                annual.append(dict(**spec, year=int(y), rmse=rmse))
        print('[NESTED SUMMARY]', crop, len(predictions), 'complete prediction configurations', flush=True)
    frame = pd.DataFrame(rows).drop(columns='per_year_rmse')
    frame.to_csv(dest / 'scores.csv', index=False)
    pd.DataFrame(annual).to_csv(dest / 'annual.csv', index=False)
    pd.DataFrame(registry).to_csv(dest / 'checkpoint_registry.csv', index=False)
    plots(frame, dest, not args.observed_only)
    report(frame, dest, not args.observed_only)
    atomic_json(dest / 'audit.json', dict(sources=sources, code_sha256=sha256(ROOT / 'scripts/summarize_inseason_nested.py'),
        suites=suites, seed=args.seed, years=YEARS, equal_year_aggregation=True,
        exact_prediction_identity=True, all_registered_conditions_retained=True,
        state_weights_reused=True, historical_recipe_selection_inherited=True))
    print(frame[frame.percent.eq(10) & frame.completion.eq('biid')][
        ['crop', 'condition', 'mean_annual_rmse', 'library_score_gain', 'library_positive_years']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
