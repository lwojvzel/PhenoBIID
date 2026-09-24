"""Report every task-priority outcome against frozen, input-matched controls."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gru_task_priority import MODES
from multimodal_baseline import regression_metrics
from review_revision_data import ROOT, CROPS, sha256
from run_gru_task_priority import RESULT
from run_review_revision_parallel import atomic_json
from summarize_task_aligned_world import aligned, OUT as PILOT_OUT

OUT = ROOT / 'visualize/paper_experiments/gru_task_priority_v1'
FIG = ROOT / 'visualize/gru_task_priority_v1'
LABELS = dict(direct_continued='Direct GRU: continued training', joint_one='World GRU: LAI weight 1',
              joint_tenth='World GRU: LAI weight 0.1', joint_hundredth='World GRU: LAI weight 0.01',
              yield_priority='World GRU: yield-priority gradients')


def collect():
    rows, years = [], []
    for crop in CROPS:
        for mode in MODES:
            root = RESULT / 'pipelines' / crop / 'origin_2012' / mode / 'seed_42'
            m = json.loads((root / 'metrics.json').read_text()); spec = json.loads((root / 'config.json').read_text())
            for name, digest in spec['code_hashes'].items():
                if sha256(ROOT / 'scripts' / name) != digest:
                    raise ValueError(f'Changed training source: {name}')
            from pathlib import Path
            if sha256(Path(m['weight'])) != m['weight_sha256']:
                raise ValueError('Changed weight')
            row = {k: m[k] for k in ('crop', 'mode', 'seed', 'origin', 'selected_epoch', 'state_constraint_met', 'parameters', 'seconds', 'weight')}
            row['directory'] = str(root)
            for split, short in (('validation', 'val'), ('test', 'test')):
                row[short] = m['scores'][split]['rmse']
                row[f'{short}_lai'] = m['state_scores'][split]['rmse']
                row[f'{short}_persistence'] = m['state_scores'][split]['persistence_rmse']
                with np.load(root / f'{split}_predictions.npz') as p:
                    actual = regression_metrics(p['target'], p['prediction'])
                    for metric in ('rmse', 'mae', 'r2', 'nrmse'):
                        np.testing.assert_allclose(actual[metric], m['scores'][split][metric], atol=1e-7, rtol=0)
                    for year in np.unique(p['year']):
                        keep = p['year'] == year
                        years.append(dict(crop=crop, mode=mode, split=split, year=int(year),
                                          **regression_metrics(p['target'][keep], p['prediction'][keep])))
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    frame, years = collect()
    reference = pd.read_csv(PILOT_OUT / 'pilot_paired.csv')
    controls = []
    for crop in CROPS:
        old = reference[(reference.crop == crop) & (reference.variant == 'direct_gru')].iloc[0]
        continued = frame[(frame.crop == crop) & (frame['mode'] == 'direct_continued')].iloc[0]
        best = min([dict(val=old.val, test=old.test, directory=old.directory),
                    dict(val=continued.val, test=continued.test, directory=continued.directory)], key=lambda a: (a['val'], a['directory']))
        controls.append(dict(crop=crop, strong_val=old.strong_val, strong_test=old.strong_test,
            strong_directory=old.strong_directory, strong_model=old.strong_model,
            direct_val=best['val'], direct_test=best['test'], direct_directory=best['directory']))
        for row in frame[frame.crop == crop].itertuples():
            aligned(row.directory, old.strong_directory); aligned(row.directory, best['directory'])
    paired = frame.merge(pd.DataFrame(controls), on='crop', validate='many_to_one')
    for ref in ('strong', 'direct'):
        for split in ('val', 'test'):
            paired[f'{ref}_{split}_gain'] = 100 * (1 - paired[split] / paired[f'{ref}_{split}'])
    paired['lai_test_gain'] = 100 * (1 - paired.test_lai / paired.test_persistence)
    ranking = []
    for mode, a in paired[paired['mode'] != 'direct_continued'].groupby('mode'):
        ratios = a.val / np.minimum(a.strong_val, a.direct_val)
        ranking.append(dict(mode=mode, state_constraint_met=bool(a.state_constraint_met.all()),
            worst_crop_validation_ratio=float(ratios.max()), macro_validation_ratio=float(ratios.mean()),
            crop_validation_ratios=dict(zip(a.crop, ratios))))
    ranking.sort(key=lambda r: (not r['state_constraint_met'], r['worst_crop_validation_ratio'], r['macro_validation_ratio'], r['mode']))
    selection = dict(selected=ranking[0], ranking=ranking, controls=controls,
        passes_pilot=bool(ranking[0]['state_constraint_met'] and ranking[0]['worst_crop_validation_ratio'] < 1),
        criterion='State fidelity constraint, then worst-crop validation ratio against stronger of matched history and validation-selected direct/continued GRU. Evaluation scores do not select candidates.',
        scope='Single-seed development pilot, not a new independent evaluation')
    path = RESULT / 'pilot_validation_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Frozen selection changed')
    atomic_json(path, selection)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'pilot_metrics.csv', index=False); paired.to_csv(OUT / 'pilot_paired.csv', index=False)
    years.to_csv(OUT / 'pilot_per_year.csv', index=False)
    frame[['crop', 'mode', 'weight']].to_csv(OUT / 'pilot_checkpoints.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained', sharey=True)
    for ax, metric, title in zip(axes, ('strong_test_gain', 'direct_test_gain', 'lai_test_gain'),
        ('Yield vs. strong history', 'Yield vs. best validation-selected direct GRU', 'LAI vs. previous-season state')):
        matrix = paired.pivot(index='mode', columns='crop', values=metric).reindex(index=MODES, columns=CROPS)
        if metric == 'lai_test_gain':
            matrix.loc['direct_continued'] = np.nan
        limit = max(2., float(np.nanmax(np.abs(matrix.to_numpy()))))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_title(title, fontsize=10); ax.set_xticks(range(4), [c.title() for c in CROPS], fontsize=9)
        ax.set_yticks(range(5), [LABELS[m] for m in MODES], fontsize=9)
        for i in range(5):
            for j in range(4):
                v = matrix.iloc[i, j]
                ax.text(j, i, f'{v:+.2f}' if np.isfinite(v) else 'N/A', ha='center', va='center', fontsize=9,
                    color='white' if np.isfinite(v) and abs(v) > limit*.7 else 'black')
        fig.colorbar(im, ax=ax, location='bottom', shrink=.85, label='RMSE reduction (%)')
    fig.suptitle('Recurrent world extension: task-weight and gradient pilot, seed 42', fontsize=13)
    for extension in ('png', 'pdf'):
        fig.savefig(FIG / f'pilot_comparisons.{extension}', dpi=200)
    plt.close(fig)
    fields = ['crop', 'mode', 'val', 'test', 'strong_test_gain', 'direct_test_gain', 'lai_test_gain', 'selected_epoch', 'state_constraint_met']
    document = ['# GRU 世界模型与产量优先优化：20 组结果', '',
        '## 模型与实验', '', '沿用 W2 的 1982--2009 训练、2010--2012 验证、2013--2016 评价。五种设置均使用种子 42，并从同作物的直接气象 GRU 开始；直接继续训练是预算对照。世界模型加入可监督的 LAI 解码与下一槽预测反馈，历史产量只进入最终产量头。', '',
        '状态损失分别为 1、0.1、0.01；另一个设置用 0.1 并投影冲突的 LAI 辅助梯度，使其不直接抵消同一 batch 的产量梯度。该局部约束不保证泛化改进。每个模型的训练曲线、梯度记录、权重和预测均保留。', '',
        '## 逐格读图', '', f'![GRU 优化对照]({FIG / "pilot_comparisons.png"})', '',
        '行是五种训练设置，列是四作物。左：能否优于强历史基线。中：相同气象信息下，显式世界模型是否优于直接 GRU。右：是否真的改善了植被预测。正数更好，负数退步，右图的直接基线没有 LAI 预测任务，因此标为 N/A。不同面板的色标独立。', '',
        '## 所有结果', '', paired[fields].to_markdown(index=False, floatfmt='.5f'), '',
        '## 验证选择', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
        '这是固定单种子、单时间起点的诊断，不足以独立证明四作物稳定收益。只在验证候选通过后扩展固定种子与起点，不能按评价收益替换个别作物的候选。', '',
        '## 文件', '', f'- 权重：{OUT / "pilot_checkpoints.csv"}', f'- 每年结果：{OUT / "pilot_per_year.csv"}',
        f'- 所有配对统计：{OUT / "pilot_paired.csv"}']
    report = ROOT / 'Paper/task/GRU产量优先_20组结果_20260906.md'
    report.write_text('\n'.join(document) + '\n', encoding='utf-8')
    print(paired[fields].to_string(index=False))
    print(json.dumps(dict(selection=selection['selected'], passes_pilot=selection['passes_pilot'], report=str(report)), indent=2))


if __name__ == '__main__':
    main()
