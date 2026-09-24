"""Fixed-penalty, validation-year-balanced shrinkage toward the history anchor."""
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
from summarize_task_aligned_world import OUT as TASK_OUT, LABELS as TASK_LABELS, aligned
from summarize_gru_task_priority import OUT as GRU_OUT, LABELS as GRU_LABELS

RESULT = ROOT / 'benchmark/results/validated_world_anchor_v1'
OUT = ROOT / 'visualize/paper_experiments/validated_world_anchor_v1'
FIG = ROOT / 'visualize/validated_world_anchor_v1'
PENALTY = .1


def coefficient(target, history, component, year):
    moments = []
    for current in np.unique(year):
        keep = year == current
        error = target[keep].astype(float) - history[keep]
        delta = component[keep].astype(float) - history[keep]
        scale = max(float(np.mean(error**2)), 1e-12)
        moments.append((float(np.mean(error * delta)) / scale, float(np.mean(delta**2)) / scale))
    if not moments or not np.isfinite(moments).all():
        raise ValueError('Invalid calibration data')
    numerator, denominator = np.mean(moments, 0)
    return float(np.clip(numerator / (denominator + PENALTY), 0, 1))


def leave_year_out(target, history, component, year):
    prediction = np.empty_like(target, dtype=float); coefficients = {}
    if len(np.unique(year)) < 2:
        raise ValueError('At least two calibration years required')
    for current in np.unique(year):
        keep = year == current
        value = coefficient(target[~keep], history[~keep], component[~keep], year[~keep])
        prediction[keep] = history[keep] + value * (component[keep] - history[keep])
        coefficients[str(int(current))] = value
    return prediction, coefficients


def main():
    task = pd.read_csv(TASK_OUT / 'pilot_paired.csv')
    gru = pd.read_csv(GRU_OUT / 'pilot_paired.csv')
    rows = []; years = []
    for stage, frame, id_column, labels in (('task', task, 'variant', TASK_LABELS), ('gru', gru, 'mode', GRU_LABELS)):
        for r in frame.to_dict('records'):
            name = r[id_column]; candidate = f'{stage}__{name}'
            root = RESULT / 'pipelines' / r['crop'] / candidate
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(r['directory']); anchor = Path(r['strong_directory'])
            aligned(directory, anchor)
            with np.load(directory / 'validation_predictions.npz') as p, np.load(anchor / 'validation_predictions.npz') as h:
                weight = coefficient(p['target'], h['prediction'], p['prediction'], p['year'])
                held, held_weights = leave_year_out(p['target'], h['prediction'], p['prediction'], p['year'])
                held_ratios = []
                for year in np.unique(p['year']):
                    keep = p['year'] == year
                    num = regression_metrics(p['target'][keep], held[keep])['rmse']
                    den = regression_metrics(p['target'][keep], h['prediction'][keep])['rmse']
                    held_ratios.append(num / den)
                loo_ratio = float(np.mean(held_ratios))
            config = dict(candidate=candidate, crop=r['crop'], coefficient=weight, leave_year_out_coefficients=held_weights,
                penalty=PENALTY, calibration_years=[2010, 2011, 2012], anchor=str(anchor), component=str(directory),
                source_prediction_sha256={f'{kind}_{split}': sha256(path / f'{split}_predictions.npz')
                    for kind, path in (('component', directory), ('anchor', anchor)) for split in ('validation', 'test')},
                code_sha256=sha256(Path(__file__)),
                objective='Equal validation-year mean of normalized MSE + 0.1 lambda^2, lambda in [0,1]; no evaluation values used for calibration',
                scope='Development calibration: source checkpoints already selected on these validation years; leave-year-out calibrator is not fully nested model validation')
            if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != config:
                raise ValueError('Frozen calibration changed')
            atomic_json(root / 'config.json', config)
            score = {}
            for split in ('validation', 'test'):
                with np.load(directory / f'{split}_predictions.npz') as p, np.load(anchor / f'{split}_predictions.npz') as h:
                    prediction = h['prediction'] + weight * (p['prediction'] - h['prediction'])
                    score[split] = regression_metrics(p['target'], prediction)
                    score[split]['history_rmse'] = regression_metrics(p['target'], h['prediction'])['rmse']
                    score[split]['gain_percent'] = 100 * (1 - score[split]['rmse'] / score[split]['history_rmse'])
                    np.savez_compressed(root / f'{split}_predictions.npz', prediction=prediction, history_prediction=h['prediction'],
                        component_prediction=p['prediction'], **{k: p[k] for k in ('target', 'source_indices', 'row', 'col', 'year')})
                    for year in np.unique(p['year']):
                        keep = p['year'] == year
                        metric = regression_metrics(p['target'][keep], prediction[keep])
                        base = regression_metrics(p['target'][keep], h['prediction'][keep])['rmse']
                        years.append(dict(crop=r['crop'], candidate=candidate, split=split, year=int(year),
                                          **metric, history_rmse=base, gain_percent=100*(1-metric['rmse']/base)))
            world = name not in ('history_mlp', 'direct_gru', 'direct_continued')
            row = dict(crop=r['crop'], candidate=candidate, label=labels[name], world=world,
                state_constraint_met=bool(r['state_constraint_met']), coefficient=weight,
                leave_year_out_mean_ratio=loo_ratio, val=score['validation']['rmse'], test=score['test']['rmse'],
                val_gain=score['validation']['gain_percent'], test_gain=score['test']['gain_percent'],
                lai_test_gain=r['lai_test_gain'], directory=str(root), component_directory=str(directory), anchor_directory=str(anchor))
            rows.append(row); atomic_json(root / 'metrics.json', dict(**row, scores=score))
    frame = pd.DataFrame(rows); yearly = pd.DataFrame(years)
    if len(frame) != 44:
        raise ValueError('Expected all 24 joint and 20 recurrent pilot candidates')
    direct = frame[frame.candidate.isin(('task__direct_gru', 'gru__direct_continued'))].sort_values(['leave_year_out_mean_ratio', 'candidate']).groupby('crop').head(1)
    direct = direct[['crop', 'candidate', 'test', 'leave_year_out_mean_ratio']].rename(columns={k: 'direct_'+k for k in ('candidate', 'test', 'leave_year_out_mean_ratio')})
    frame = frame.merge(direct, on='crop', validate='many_to_one')
    frame['direct_test_gain'] = 100 * (1-frame.test/frame.direct_test)
    ranking = []
    for candidate, a in frame[frame.world].groupby('candidate'):
        ratios = a.leave_year_out_mean_ratio / np.minimum(a.direct_leave_year_out_mean_ratio, 1.)
        eligible = bool(a.state_constraint_met.all() and (a.coefficient > 1e-8).all())
        ranking.append(dict(candidate=candidate, eligible=eligible, worst_crop_validation_ratio=float(ratios.max()),
                            macro_validation_ratio=float(ratios.mean()), crop_validation_ratios=dict(zip(a.crop, ratios))))
    ranking.sort(key=lambda r: (not r['eligible'], r['worst_crop_validation_ratio'], r['macro_validation_ratio'], r['candidate']))
    selection = dict(selected=ranking[0], ranking=ranking, penalty=PENALTY, direct_controls=direct.to_dict('records'),
        passes_pilot=bool(ranking[0]['eligible'] and ranking[0]['worst_crop_validation_ratio'] < 1),
        criterion='State eligibility and positive coefficient for all crops; minimize worst crop leave-calibration-year-out ratio against history/direct controls; evaluation excluded')
    path = RESULT / 'pilot_validation_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Frozen calibration selection changed')
    atomic_json(path, selection)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'pilot_metrics.csv', index=False); yearly.to_csv(OUT / 'pilot_per_year.csv', index=False)
    order = list(frame.candidate.unique()); labels = frame.drop_duplicates('candidate').set_index('candidate').label
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), layout='constrained', sharey=True)
    for ax, field, title in zip(axes, ('test_gain', 'direct_test_gain', 'coefficient'),
        ('Yield vs. history anchor', 'Yield vs. calibrated direct GRU', 'Validation-calibrated candidate weight')):
        matrix = frame.pivot(index='candidate', columns='crop', values=field).reindex(index=order, columns=CROPS)
        limit = 1 if field == 'coefficient' else max(2., float(np.max(np.abs(matrix.to_numpy()))))
        im = ax.imshow(matrix, cmap='Blues' if field == 'coefficient' else 'RdYlGn',
                       vmin=0 if field == 'coefficient' else -limit, vmax=limit, aspect='auto')
        ax.set_title(title, fontsize=10); ax.set_xticks(range(4), [c.title() for c in CROPS], fontsize=9)
        ax.set_yticks(range(len(order)), [labels[k] for k in order], fontsize=9)
        for i in range(len(order)):
            for j in range(4):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{value:.3f}' if field == 'coefficient' else f'{value:+.2f}', ha='center', va='center',
                        fontsize=9, color='white' if abs(value) > limit*.7 else 'black')
        fig.colorbar(im, ax=ax, location='bottom', shrink=.85, label='Weight' if field == 'coefficient' else 'RMSE reduction (%)')
    fig.suptitle('Fixed-penalty history anchoring: all development candidates, seed 42', fontsize=13)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'pilot_comparisons.{ext}', dpi=200)
    plt.close(fig)
    fields = ['crop', 'candidate', 'coefficient', 'leave_year_out_mean_ratio', 'test_gain', 'direct_test_gain', 'state_constraint_met']
    report = ['# 低自由度历史锚定：44 个校准对照', '',
        '这不是 44 个重新训练的状态网络，而是对 W2/W4 全部候选分别拟合一个验证期标量。最终产量为历史预测加权融合候选预测；候选内部仍保留原有气象到植被到产量路径。', '',
        '每年先按历史均方误差归一，再等权求三个验证年的目标；固定惩罚 0.1，将权重向历史锚点收缩，并限制在 0 到 1。2013--2016 评价期的真实产量、真实遥感与误差不用于选择权重。所有直接 GRU 也做完全同样的校准，以防把普通集成收益归功于世界模型。', '',
        f'![全部校准比较]({FIG / "pilot_comparisons.png"})', '',
        '左图是相对强历史锚点的产量提升；中图是相对校准后的直接气象 GRU；右图是验证期确定的候选权重。权重为零意味着完全退回历史，不能算植被产生了增量。左图为正、中图为负，仍不足以证明显式植被演化优于直接使用气象。不同面板色标独立。', '',
        frame[fields].to_markdown(index=False, floatfmt='.5f'), '', '## 验证选择', '',
        '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
        '校准还做了留一验证年检查，但上游权重已经用这些验证年早停，所以不能称为严格嵌套的独立验证。通过本轮后仍需固定种子、跨时间起点和空间聚类区间确认。', '',
        f'每年结果：{OUT / "pilot_per_year.csv"}', f'校准参数及两路权重来源：{RESULT / "pipelines"}']
    target = ROOT / 'Paper/task/历史锚定校准_44组结果_20260906.md'
    target.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(frame[fields].to_string(index=False))
    print(json.dumps(selection['selected'], indent=2))


if __name__ == '__main__':
    main()
