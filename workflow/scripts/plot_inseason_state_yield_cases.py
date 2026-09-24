"""Render all registered cases and generate their exact-value reading guide."""
import calendar
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from build_inseason_state_yield_cases import ROOT, OUT
from inseason_state_yield_cases import PRODUCTS, PRIMARY, PERCENTS
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

COLORS = dict(biid='#d56b18', climatology='#087fb1', observed_suffix='#19896b',
              historical_reference='#777777', truth='#202020')
CN = dict(maize='玉米', rice='水稻', soybean='大豆', wheat='小麦')
METHODS = ('biid', 'climatology', 'observed_suffix', 'historical_reference')
LABELS = ('BIID suffix', 'Climatological suffix', 'Observed suffix*', 'Historical reference')
HISTORY_LABELS = {'maize': 'TabM residual\n(full window)',
                  'rice': 'MLP residual\n(1993+ in-sample)',
                  'soybean': 'TabM residual (PLE)', 'wheat': 'Linear-NCA residual'}


def render():
    audit = json.loads((OUT / 'export_audit.json').read_text())
    if not audit['passed'] or audit['new_fits'] != 0:
        raise ValueError('Verified fixed-model case exports required')
    for name, digest in audit['files'].items():
        if sha256(OUT / name) != digest:
            raise ValueError('Case data changed after export')
    cases = json.loads((OUT / 'cases.json').read_text())
    lines = pd.read_csv(OUT / 'trajectories.csv')
    scores = pd.read_csv(OUT / 'yield_predictions.csv')
    states = pd.read_csv(OUT / 'state_scores.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(8.0, 10.3))
    grid = fig.add_gridspec(4, 3, width_ratios=(1, 1, 1.05),
        left=.09, right=.985, top=.88, bottom=.08, hspace=.93, wspace=1.18)
    for row, case in enumerate(cases):
        crop = case['crop']
        products = PRODUCTS[crop]
        name = 'Soybean' if crop == 'soybean' else crop.title()
        for j, product in enumerate(products):
            ax = fig.add_subplot(grid[row, j] if len(products) == 2 else grid[row, :2])
            frame = lines[(lines.crop == crop) & (lines['product'] == product) & (lines.percent == PRIMARY)]
            frame = frame.sort_values('slot')
            x, hidden = frame.slot.to_numpy(), frame.hidden.to_numpy(bool)
            observed = ~hidden
            boundary = x[hidden][0] - .5
            ax.axvspan(boundary, x[-1] + .35, color='#eeeeee', zorder=0)
            ax.axvline(boundary, color='#888888', lw=.8, ls=':')
            ax.plot(x, frame.truth, color=COLORS['truth'], ls='--', lw=1.3, marker='.', ms=5)
            ax.plot(x[observed], frame.truth.to_numpy()[observed], color=COLORS['truth'],
                lw=1.8, marker='o', ms=4, zorder=4)
            # Include the last visible point solely to connect the suffix lines.
            segment = hidden.copy()
            segment[np.flatnonzero(observed)[-1]] = True
            for mode in ('biid', 'climatology'):
                ax.plot(x[segment], frame[mode].to_numpy()[segment],
                    color=COLORS[mode], marker='s' if mode == 'biid' else '^', ms=4, lw=1.6)
            ax.set_xlim(.65, x[-1] + .35)
            ticks = np.arange(len(x)) if len(products) == 1 else np.unique(np.r_[np.arange(0, len(x), 2), len(x)-1])
            ax.set_xticks(x[ticks], [f'{x[k]}\n{calendar.month_abbr[int(frame.source_month.iloc[k])]}' for k in ticks], fontsize=7)
            ax.set_xlabel('Slot / source month', fontsize=8)
            ax.set_ylabel('NDVI' if product == 'ndvi' else 'GPP (gC m$^{-2}$ day$^{-1}$)', fontsize=8)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='y', alpha=.18)
            title = f'{name} | {product.upper()}'
            subtitle = f'{case["latitude"]:.2f}, {case["longitude"]:.2f}; {hidden.sum()}/{len(x)} hidden'
            ax.set_title(title + '\n' + subtitle, fontsize=8, loc='left', pad=8)
        ax = fig.add_subplot(grid[row, 2])
        values = scores[(scores.crop == crop) & (scores.percent == PRIMARY)].set_index('mode')
        target = float(values.target.iloc[0])
        predictions = values.loc[list(METHODS), 'prediction'].to_numpy()
        extent = np.r_[predictions, target]
        margin = max(float(np.ptp(extent))*.22, .08)
        ax.set_xlim(extent.min()-margin, extent.max()+margin)
        ax.axvline(target, color=COLORS['truth'], ls='--', lw=1.2, zorder=1)
        for k, mode in enumerate(METHODS):
            pred = float(values.loc[mode, 'prediction'])
            ax.scatter(pred, k, color=COLORS[mode], s=35, zorder=3,
                marker='D' if mode == 'observed_suffix' else 'o')
            ax.annotate(f'{pred:.3f}', (pred, k), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=7, color=COLORS[mode])
        ax.set_yticks(range(4), (*LABELS[:3], HISTORY_LABELS[crop]), fontsize=7)
        ax.set_ylim(3.6, -.7)
        ax.set_xlabel('Yield (t/ha)', fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        lo, hi = ax.get_xlim()
        ax.set_xticks([tick for tick in ax.get_xticks() if lo <= tick <= hi])
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(axis='x', alpha=.15)
        ax.set_title(f'Frozen yield readout\nTruth: {target:.3f} t/ha', fontsize=8, loc='left', pad=8)
    fig.suptitle('From vegetation completion to yield: fixed 2010 cases', fontsize=12, y=.98)
    handles = [Line2D([0], [0], color=COLORS['truth'], marker='o', label='Visible observations'),
        Line2D([0], [0], color=COLORS['truth'], ls='--', label='Hidden truth (diagnostic)'),
        Line2D([0], [0], color=COLORS['biid'], marker='s', label='BIID completion'),
        Line2D([0], [0], color=COLORS['climatology'], marker='^', label='Training climatology')]
    fig.legend(handles=handles, ncol=2, loc='upper center', bbox_to_anchor=(.5, .96), frameon=False, fontsize=8)
    fig.text(.5, .027, '* Observed suffix: same cutoff metadata; diagnostic, not a forecast or guaranteed upper bound.',
        ha='center', fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        b = artist.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
            outside.append(artist.get_text())
    if outside:
        raise ValueError(f'Text outside figure: {outside}')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'state_to_yield_cases.{ext}', dpi=220, facecolor='white')
    plt.close(fig)
    document = ['---', 'title: "真实植被轨迹如何影响产量：四作物案例读图报告"',
        'date: "2026年9月9日"', 'lang: zh-CN', 'fontsize: 11pt',
        'geometry: [a4paper, margin=20mm]', 'mainfont: Noto Serif CJK SC',
        'sansfont: Noto Sans CJK SC', 'monofont: DejaVu Sans Mono',
        'header-includes:', '  - \\usepackage{xurl}', '  - \\usepackage{float}',
        '  - \\floatplacement{figure}{H}', '  - \\XeTeXlinebreaklocale "zh"',
        '  - \\XeTeXlinebreakskip=0pt plus 1pt', '---', '', '# 这张图在做什么', '',
        '同一个历史专家、同一组已训练产量树、相同给定气象和相同可见前缀，只替换未来植被部分。'
        '图展示这次替换怎样改变最后产量。它是世界模型过程的真实样本解释，不是新增训练，也不替代十三年总体成绩。', '',
        '每作物一个样本，均为2010年、拟合截止2009年、seed42；先按身份哈希锁定，再读取个体预测。'
        '没有按误差、收益、产量或植被数值筛选。四个案例不是全球统计代表，也不保证都有改善。', '',
        f'![固定四个格点，名义30%后缀。左边是植被，右边是同头产量；灰色背景为未观测槽。]({OUT / "state_to_yield_cases.png"})'
        '{width=100% height=210mm}', '', '# 怎么看', '',
        '1. 每行一个作物。小麦左侧有两张状态图，NDVI与GPP使用各自单位，不能直接比较原始误差大小。',
        '2. 左侧黑色实线是已经观测的前缀；灰底内的黑色虚线是事后真实值，橙线是BIID预测，蓝线是训练期常态。'
        '曲线越靠近黑线，该样本的状态预测越准确。横轴标活动槽与其原自然月，不是已核实的距收获天数。',
        '3. 右侧黑色竖虚线是真实产量，点离它越近，产量绝对误差越小。橙点和蓝点分别对应两种补全。'
        '绿色菱形使用真实后缀，是诊断参照；灰点是原主表中具名强历史参照。',
        '4. 比较橙蓝曲线，再比较橙蓝产量点：生长曲线更准不保证产量点更近。看右侧时，应比较距离，不是点越右越好。',
        '5. 真实后缀诊断保留当前截点的支持/缺失元数据，只替换轨迹。它不同于0%完整观测条件；'
        '真实信息也不保证一个冻结模型输出的误差必然最小。', '', '# 精确数值与逐案例解释', '']
    for case in cases:
        crop = case['crop']
        subset = scores[(scores.crop == crop) & (scores.percent == PRIMARY)].set_index('mode')
        bs, cs = float(subset.loc['biid', 'absolute_error']), float(subset.loc['climatology', 'absolute_error'])
        document += [f'## {CN[crop]}', '',
            f'格点(row={case["row"]}, col={case["col"]})，中心纬度{case["latitude"]:.2f}、经度{case["longitude"]:.2f}。'
            f'历史参照为{case["historical_reference"]}，不是本次临时挑选的模型。', '',
            f'真实产量为{case["target_yield"]:.4f}吨/公顷。', '',
            '| 输入后缀或参照 | 产量预测 | 绝对误差 |', '|---|---:|---:|']
        names = ('BIID预测后缀', '训练常态后缀', '真实后缀诊断', '固定历史参照')
        for mode, name in zip(METHODS, names):
            r = subset.loc[mode]
            document.append(f'| {name} | {r.prediction:.4f} | {r.absolute_error:.4f} |')
        document += ['', '| 状态信号 | BIID后缀RMSE | 常态后缀RMSE | 有效后缀槽 |', '|---|---:|---:|---:|']
        for product in PRODUCTS[crop]:
            rows = states[(states.crop == crop) & (states['product'] == product) & (states.percent == PRIMARY)].set_index('mode')
            b, c = rows.loc['biid'], rows.loc['climatology']
            document.append(f'| {product.upper()} | {b.state_rmse:.4f} | {c.state_rmse:.4f} | {int(b.valid_hidden_slots)} |')
        change = '减少' if bs < cs else ('增加' if bs > cs else '不变，差值为')
        document += ['', f'**本例说明：** 用BIID替换常态后，产量绝对误差{change}{abs(bs-cs):.4f}吨/公顷。'
            '上方状态表和产量表需要分别读，不把状态RMSE下降直接当作产量误差下降。'
            '这是单个格点年的现象，不代表该作物的总体收益。', '']
    document += ['# 相同四格点的其他截点', '',
        '保持同一批格点，记录10%、30%、50%后缀。下表均为产量绝对误差，吨/公顷，越小越好；不是十三年RMSE。', '',
        '| 作物 | 名义隐藏比例 | 实际隐藏槽 | BIID | 常态 | 真实后缀诊断 | 历史参照 |', '|---|---:|---:|---:|---:|---:|---:|']
    for crop in PRODUCTS:
        for percent in PERCENTS:
            frame = scores[(scores.crop == crop) & (scores.percent == percent)].set_index('mode')
            r = frame.iloc[0]
            document.append(f'| {CN[crop]} | {percent}% | {int(r.hidden_slots)}/{int(r.active_slots)} | ' +
                ' | '.join(f'{frame.loc[m, "absolute_error"]:.4f}' for m in METHODS) + ' |')
    document += ['', '# 可以支持什么，不能支持什么', '',
        '可以直观看到：同一冻结产量头对不同补全轨迹的反应，以及两种状态产品和产量单位的区别。'
        '这补充了世界模型“状态演化到终端读出”的过程解释。', '',
        '不能由四例证明全球提升、因果效应或BIID普遍优于常态。总体结论仍应依据全样本同头消融、'
        '四作物十三年主比较和配对多种子结果；本图没有修改它们。', '', '# 文件与复现', '',
        f'输出根目录：`{OUT}`。', '',
        '`selection.json`保存先锁定的身份；`trajectories.csv`保存逐槽值；`yield_predictions.csv`保存48项产量值；'
        '`state_scores.csv`保存30项状态误差；四个`*_case.npz`保存输入特征。'
        '`export_audit.json`记录原模型重放，`verification.json`在独立验证完成后生成。', '',
        '图为`state_to_yield_cases.png/pdf`；`cases_appendix.tex`为待统一接入论文的英文附录片段。'
        '本报告与原41页汇报、原45页论文分开保存。', '']
    report = ROOT / 'Paper/task/状态到产量_四作物真实案例读图报告_20260909.md'
    report.write_text('\n'.join(document), encoding='utf-8')
    appendix = r'''\subsection{Illustrative state-to-yield cases}
\label{app:inseason_state_yield_cases}
Figure~\ref{fig:inseason_state_yield_cases} connects vegetation trajectories
to terminal predictions at four fixed grid--years. For each crop, the case
is the smallest SHA256 identity among eligible 2010 samples with at least
four active slots; neither targets nor prediction errors enter this rule.
The original seed-42 model fitted through 2009 is evaluated at a nominal
30\% hidden suffix. We keep the historical anchor, readout weights,
weather, prefix and cutoff metadata fixed while replacing the suffix by
BIID predictions, training climatology, or observed values. The last is
a diagnostic, not a forecast or a guaranteed performance upper bound.
The examples illustrate how differences in state trajectories propagate
through the same readout; aggregate evidence remains the full-cohort
completion comparisons, rather than these four cases.

\begin{figure}[p]
 \centering
 \includegraphics[width=\linewidth]{../../visualize/paper_experiments/inseason_state_yield_cases_v1/state_to_yield_cases.pdf}
 \caption{Fixed grid--year cases linking vegetation completion (left) to
 frozen-head yield predictions (right). Gray shading marks the hidden
 suffix; black dashed lines show hidden vegetation or yield truth.
 Observed-suffix readouts retain the same cutoff metadata.}
 \label{fig:inseason_state_yield_cases}
\end{figure}
'''
    (OUT / 'cases_appendix.tex').write_text(appendix, encoding='utf-8')
    paths = [report, OUT / 'state_to_yield_cases.png', OUT / 'state_to_yield_cases.pdf', OUT / 'cases_appendix.tex']
    atomic_json(OUT / 'plot_audit.json', dict(passed=True, figures=1, panels=9, text_outside_figure=outside,
        primary_percent=PRIMARY, all_registered_cases_displayed=True,
        source_export_sha256=sha256(OUT / 'export_audit.json'),
        script_sha256=sha256(Path(__file__)), files={str(p): sha256(p) for p in paths}))
    print(f'[CASE FIGURE READY] {OUT / "state_to_yield_cases.pdf"}')


if __name__ == '__main__':
    render()
