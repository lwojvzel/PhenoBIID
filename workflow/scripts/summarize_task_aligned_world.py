"""Compare the complete joint-state pilot with validation-selected controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multimodal_baseline import regression_metrics
from review_revision_data import ROOT, CROPS, sha256
from run_review_revision_parallel import atomic_json
from run_task_aligned_world import RESULT
from task_aligned_world import VARIANTS

OUT = ROOT / 'visualize/paper_experiments/task_aligned_world_v1'
FIG = ROOT / 'visualize/task_aligned_world_v1'
LABELS = dict(history_mlp='History MLP', direct_gru='Direct climate GRU', biid_joint='Joint BIID',
              slot_observable='Full slots: predicted LAI', slot_joint='Full slots: LAI + latent state',
              forcing_joint='Full slots: decomposed forcing')


def aligned(first, second):
    for split in ('validation', 'test'):
        with np.load(Path(first) / f'{split}_predictions.npz') as a, np.load(Path(second) / f'{split}_predictions.npz') as b:
            for key in ('source_indices', 'target', 'year', 'row', 'col'):
                np.testing.assert_array_equal(a[key], b[key])


def read_pilot():
    rows, years = [], []
    for path in sorted((RESULT / 'pipelines').glob('*/origin_2012/*/seed_42/metrics.json')):
        m = json.loads(path.read_text()); config = json.loads((path.parent / 'config.json').read_text())
        for name, digest in config['code_hashes'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise ValueError(f'Training source changed: {name}')
        if sha256(Path(m['weight'])) != m['weight_sha256']:
            raise ValueError(f'Weight changed: {m["weight"]}')
        row = {k: m[k] for k in ('crop', 'variant', 'seed', 'origin', 'selected_epoch', 'state_constraint_met', 'parameters', 'seconds', 'peak_allocated_mib', 'weight')}
        row['directory'] = str(path.parent)
        for split, name in (('validation', 'val'), ('test', 'test')):
            row[name] = m['scores'][split]['rmse']
            row[f'{name}_lai'] = m['state_scores'][split]['rmse']
            row[f'{name}_persistence'] = m['state_scores'][split]['persistence_rmse']
            with np.load(path.parent / f'{split}_predictions.npz') as p:
                score = regression_metrics(p['target'], p['prediction'])
                for metric in ('rmse', 'mae', 'r2', 'nrmse'):
                    np.testing.assert_allclose(score[metric], m['scores'][split][metric], atol=1e-7, rtol=0)
                for year in np.unique(p['year']):
                    keep = p['year'] == year
                    years.append(dict(crop=m['crop'], variant=m['variant'], split=split, year=int(year),
                                      **regression_metrics(p['target'][keep], p['prediction'][keep])))
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = {(crop, variant) for crop in CROPS for variant in VARIANTS}
    if len(rows) != 24 or set(zip(frame.crop, frame.variant)) != expected:
        raise RuntimeError(f'Pilot incomplete: {len(rows)}/24')
    return frame, pd.DataFrame(years)


def summarize():
    frame, years = read_pilot()
    library = pd.read_csv(ROOT / 'visualize/paper_experiments/stable_remote_v1/stage_C_metrics.csv')
    library = library[(library.origin == 2012) & (library.seed == 42)]
    controls = []
    for crop in CROPS:
        a = frame[frame.crop == crop]
        history = a[a.variant == 'history_mlp'].iloc[0]
        direct = a[a.variant == 'direct_gru'].iloc[0]
        same_window = library[(library.crop == crop) & (library.condition == 'history') & (library.window == 0)]
        options = [dict(val=history.val, test=history.test, directory=history.directory, model='history_mlp')]
        options += [dict(val=r.val, test=r.test, directory=r.directory, model=r.engine) for r in same_window.itertuples()]
        strongest = sorted(options, key=lambda r: (r['val'], r['model']))[0]
        all_windows = library[(library.crop == crop) & (library.condition == 'history')].sort_values(['val', 'engine', 'window']).iloc[0]
        metadata = library[(library.crop == crop) & (library.condition == 'metadata') & (library.window == 0)].sort_values(['val', 'engine']).iloc[0]
        record = dict(crop=crop, history_mlp_val=history.val, history_mlp_test=history.test,
            direct_val=direct.val, direct_test=direct.test, strong_val=strongest['val'], strong_test=strongest['test'],
            strong_model=strongest['model'], strong_directory=strongest['directory'],
            all_window_val=all_windows.val, all_window_test=all_windows.test, all_window_directory=all_windows.directory,
            metadata_val=metadata.val, metadata_test=metadata.test, metadata_directory=metadata.directory)
        for directory in (strongest['directory'], all_windows.directory, metadata.directory, direct.directory):
            aligned(history.directory, directory)
        for row in a.itertuples():
            aligned(history.directory, row.directory)
        controls.append(record)
    paired = frame.merge(pd.DataFrame(controls), on='crop', validate='many_to_one')
    for ref in ('history_mlp', 'strong', 'all_window', 'metadata', 'direct'):
        for split in ('val', 'test'):
            paired[f'{ref}_{split}_gain'] = 100 * (1 - paired[split] / paired[f'{ref}_{split}'])
    paired['lai_test_gain'] = 100 * (1 - paired.test_lai / paired.test_persistence)
    ranking = []
    for variant, a in paired[~paired.variant.isin(('history_mlp', 'direct_gru'))].groupby('variant'):
        ratio = a.val / np.minimum(a.strong_val, a.direct_val)
        ranking.append(dict(variant=variant, state_constraint_met=bool(a.state_constraint_met.all()),
            worst_crop_validation_ratio=float(ratio.max()), macro_validation_ratio=float(ratio.mean()),
            crop_validation_ratios=dict(zip(a.crop, ratio))))
    ranking.sort(key=lambda r: (not r['state_constraint_met'], r['worst_crop_validation_ratio'], r['macro_validation_ratio'], r['variant']))
    selection = dict(selected=ranking[0], ranking=ranking, controls=controls,
        criterion='Validation only: state constraint first, then worst crop ratio against stronger of full-history library history and direct climate GRU, then macro ratio. No test ranking.',
        passes_pilot=bool(ranking[0]['state_constraint_met'] and ranking[0]['worst_crop_validation_ratio'] < 1),
        limitation='Single origin and seed, used development years. Passing this pilot is not stable four-crop confirmation.')
    destination = RESULT / 'pilot_validation_selection.json'
    if destination.exists() and json.loads(destination.read_text()) != selection:
        raise ValueError('Frozen pilot selection changed')
    atomic_json(destination, selection)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'pilot_metrics.csv', index=False)
    years.to_csv(OUT / 'pilot_per_year.csv', index=False)
    paired.to_csv(OUT / 'pilot_paired.csv', index=False)
    frame[['crop', 'variant', 'weight', 'directory']].to_csv(OUT / 'pilot_checkpoints.csv', index=False)
    metrics = ('strong_test_gain', 'direct_test_gain', 'lai_test_gain')
    titles = ('Yield vs. strong history', 'Yield vs. direct climate GRU', 'LAI vs. previous-season state')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained', sharey=True)
    for ax, metric, title in zip(axes, metrics, titles):
        matrix = paired.pivot(index='variant', columns='crop', values=metric).reindex(index=VARIANTS, columns=CROPS)
        if metric == 'lai_test_gain':
            matrix.loc[['history_mlp', 'direct_gru']] = np.nan
        limit = max(2., float(np.nanmax(np.abs(matrix.to_numpy()))))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_title(title, fontsize=11); ax.set_xticks(range(4), [c.title() for c in CROPS], fontsize=9)
        ax.set_yticks(range(6), [LABELS[v] for v in VARIANTS], fontsize=9)
        for i in range(6):
            for j in range(4):
                value = matrix.iloc[i, j]
                text = f'{value:+.2f}' if np.isfinite(value) else 'N/A'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                        color='white' if np.isfinite(value) and abs(value) > limit * .7 else 'black')
        fig.colorbar(im, ax=ax, location='bottom', shrink=.85, label='RMSE reduction (%)')
    fig.suptitle('Joint state / yield pilot: 2013-2016 evaluation, seed 42', fontsize=13)
    for extension in ('png', 'pdf'):
        fig.savefig(FIG / f'pilot_comparisons.{extension}', dpi=200)
    plt.close(fig)
    fields = ['crop', 'variant', 'val', 'test', 'strong_test_gain', 'direct_test_gain', 'lai_test_gain', 'selected_epoch', 'state_constraint_met']
    report = ['# 共同训练世界模型：24 组首轮结果', '',
        '## 在做什么', '', '输入历史产量、去年 LAI、目标季气象与固定上下文，预测本季各物候槽 LAI 和最终年度产量。真实本季 LAI 只用于训练监督，不作为推理输入。历史 MLP 与直接气象 GRU 是两个不同用途的对照。', '',
        '训练 1982--2009，验证 2010--2012，评价 2013--2016。固定种子 42。所有条件同样格点和标签，强历史模型由同协议六类树配置及 MLP 的验证误差选择，不按测试成绩挑选。单起点不等于跨年稳健。', '',
        '## 图怎么读', '', f'![共同训练对比]({FIG / "pilot_comparisons.png"})', '',
        '每行是一种结构，每列一个作物。左图看产量是否超过强历史模型，中图看是否超过直接输入气象的 GRU，右图看 LAI 演化是否优于直接沿用去年。正数表示误差降低；各面板色标独立，应读数字而不是跨面板比较颜色深浅。', '',
        '如果右图为正而左图为负，说明预测植被的能力还没有转化为产量优势。左图为正、中图为负，则仍不能证明显式植被演化优于相同可用信息的直接学习。后两者不同，不能只展示 LAI 改善来论证最终任务。', '',
        '## 完整结果', '', paired[fields].to_markdown(index=False, floatfmt='.5f'), '',
        '## 冻结的验证选择', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
        '只有验证约束与四作物增量共同通过，才将候选扩展到固定三个种子与三个时间起点。未通过时保留本轮结果，并根据验证信息检查状态监督权重、读出能力和输入表示。不能把评价期最有利的作物分别拼接成一个方法。', '',
        '## 权重及统计', '', f'- 权重索引：{OUT / "pilot_checkpoints.csv"}',
        f'- 配对比较：{OUT / "pilot_paired.csv"}', f'- 逐年误差：{OUT / "pilot_per_year.csv"}',
        '- 重放审计由独立脚本执行，不以成功汇总替代模型可复现检查。']
    path = ROOT / 'Paper/task/共同训练世界模型_24组结果_20260906.md'
    path.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(paired[fields].to_string(index=False))
    print(json.dumps(dict(selection=selection['selected'], passes_pilot=selection['passes_pilot'], report=str(path)), indent=2))


if __name__ == '__main__':
    summarize()
