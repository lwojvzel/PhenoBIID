"""Paired state and terminal evidence from fixed crop yield readouts."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, RECIPES, BLOCKS, YEARS, load, partition
from inseason_nested_common import LABELS, verify
from inseason_state_controls import METHODS
from run_inseason_state_controls import root_for, RATIOS
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from summarize_inseason_six_direct import complete_score
from summarize_inseason_13year import annual_gain

OUT = ROOT / 'visualize/paper_experiments/inseason_state_controls_v1'
TEX = ROOT / 'Paper/iclr2027/sections'
NAMES = dict(biid='BIID + observed feedback', gru='GRU + observed feedback',
    no_feedback='BIID without observed feedback', climatology='Grid-month climatology',
    previous='Previous trajectory', persistence='Last visible observation')
CN = dict(maize='玉米',rice='水稻',soybean='大豆',wheat='小麦')


def aggregate_states(frame):
    rows = []
    for keys, group in frame.groupby(['crop','product','percent','method']):
        if tuple(sorted(group.year)) != YEARS:
            raise ValueError('A state condition must contain exactly thirteen years')
        rows.append(dict(zip(['crop','product','percent','method'],keys),
            mean_annual_rmse=float(group.rmse.mean()), mean_annual_anomaly_correlation=float(group.anomaly_correlation.mean()),
            equal_year_mse_skill=100*(1-group.mse.mean()/group.climatology_mse.mean()),
            better_than_climatology_years=int((group.mse < group.climatology_mse).sum())))
    return pd.DataFrame(rows)


def collect():
    rows, annual, paired, state, sources, registry = [], [], [], [], {}, []
    for crop in RECIPES:
        raw, _ = load(crop)
        labels = {k:np.concatenate([raw[k][partition(raw,end)['evaluation']] for end in BLOCKS]) for k in LABELS}
        outputs = {(round(100*r),method):[] for r in RATIOS for method in METHODS}
        for cutoff in BLOCKS:
            root = root_for(crop,cutoff)
            verify(root)
            if json.loads((root / 'complete.json').read_text())['smoke']:
                raise ValueError('Smoke cannot enter paper summary')
            sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
            provenance = json.loads((root / 'provenance.json').read_text())
            for path,digest in provenance['sources'].items():
                if sha256(ROOT / path) != digest:
                    raise ValueError('Changed upstream state/readout source')
                sources[path] = digest
            for record in provenance['state_weights']:
                registry.append(dict(crop=crop,cutoff=cutoff,**record))
            part = pd.read_csv(root / 'state_annual.csv')
            if set(part.method) != set(METHODS) or set(part['percent']) != {10,30,50}:
                raise ValueError('Incomplete state controls')
            state.append(part)
            ix = partition(raw,cutoff)['evaluation']
            for (percent,method), pieces in outputs.items():
                path = root / f'tail_{percent:02d}_{method}.npz'
                with np.load(path) as saved:
                    for key in LABELS:
                        np.testing.assert_array_equal(saved[key],raw[key][ix])
                    pieces.append(saved['prediction'].copy())
        outputs = {key:np.concatenate(value) for key,value in outputs.items()}
        for (percent,method), prediction in outputs.items():
            metrics = complete_score(labels,prediction)
            rows.append(dict(crop=crop,percent=percent,method=method,pooled_rmse=metrics['pooled_rmse'],
                mean_annual_rmse=metrics['mean_annual_rmse']))
            annual.extend(dict(crop=crop,percent=percent,method=method,year=int(year),rmse=value)
                for year,value in metrics['per_year_rmse'].items())
            if method != 'biid':
                paired.append(dict(crop=crop,percent=percent,reference=method,
                    **annual_gain(labels['target'],outputs[percent,'biid'],prediction,labels['year'])))
    states = pd.concat(state,ignore_index=True)
    aggregate = aggregate_states(states)
    if len(aggregate) != 5*3*6:
        raise ValueError('Missing crop-product state conditions')
    return pd.DataFrame(rows),pd.DataFrame(annual),pd.DataFrame(paired),states,aggregate,sources,registry


def figure(frame, methods=METHODS, output_dir=OUT):
    colors = dict(biid='#1c7c68',gru='#3267a3',no_feedback='#985d91',
        climatology='#ba6338',previous='#c19827',persistence='#777777')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(10.5,7.6))
    fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.94,hspace=.43,wspace=.3)
    for ax,crop in zip(axes.flat,RECIPES):
        for method in methods:
            values = frame[frame.crop.eq(crop) & frame.method.eq(method)].sort_values('percent')
            ax.plot(values.percent,values.mean_annual_rmse,'o-',color=colors[method],label=NAMES[method],markersize=4)
        ax.set_title(crop.title()+' | '+RECIPES[crop].upper().replace('_',' + '),fontsize=11)
        ax.set_xticks([10,30,50])
        ax.set_xlabel('Nominal unobserved suffix (%)')
        ax.set_ylabel('Mean annual yield RMSE (t/ha)')
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(output_dir / f'same_head_completions.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def four_rule_figure():
    output = ROOT / 'visualize/paper_experiments/checklist_revision_20260909/state_controls'
    output.mkdir(parents=True, exist_ok=True)
    source = OUT / 'yield_scores.csv'
    source_hash = sha256(source)
    frame = pd.read_csv(source)
    methods = ('biid', 'climatology', 'previous', 'persistence')
    selected = frame[frame.method.isin(methods)].copy()
    expected = {(crop, percent, method) for crop in RECIPES
                for percent in (10, 30, 50) for method in methods}
    actual = set(selected[['crop', 'percent', 'method']].itertuples(index=False, name=None))
    if actual != expected or len(selected) != len(expected):
        raise ValueError('Expected exactly four rules at three cutoffs for four crops')
    if not np.isfinite(selected.mean_annual_rmse).all():
        raise ValueError('Invalid yield RMSE')
    figure(selected, methods=methods, output_dir=output)
    selected.to_csv(output / 'yield_scores.csv', index=False)
    replay = pd.read_csv(output / 'yield_scores.csv')
    pd.testing.assert_frame_equal(selected.reset_index(drop=True), replay,
                                  check_exact=False, rtol=1e-14, atol=0)
    if sha256(source) != source_hash:
        raise ValueError('Original results changed')
    atomic_json(output / 'figure_revision.json', dict(
        sources={str(source): source_hash}, methods=list(methods),
        labels=[NAMES[m] for m in methods], crop_cutoff_points=len(selected),
        metric='thirteen-year mean annual yield RMSE (t/ha)', seed=42,
        new_training=0, original_six_rule_results_retained=True,
        files={str(output / name): sha256(output / name) for name in
               ('yield_scores.csv', 'same_head_completions.png', 'same_head_completions.pdf')},
        generator_sha256=sha256(ROOT / 'scripts/summarize_inseason_state_controls.py')))
    print('Saved four-rule figure:', output, flush=True)


def export(frame,paired,states):
    def save_table(file,caption,label,columns,header,lines):
        (TEX / file).write_text('% Generated by scripts/summarize_inseason_state_controls.py\n'
            '\\begin{table}[htbp]\n\\centering\\small\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{columns}}}\n\\toprule\n'
            +header+r' \\'+'\n\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    lines = []
    for percent in (10,30,50):
        sub = frame[frame.percent.eq(percent)].pivot(index='method',columns='crop',values='mean_annual_rmse')
        for method in METHODS:
            lines.append(' & '.join([f'{percent}\\%',NAMES[method]]+[f'{sub.loc[method,c]:.4f}' for c in RECIPES])+r' \\')
        if percent != 50:
            lines.append(r'\midrule')
    save_table('inseason_core_completion_table.tex',
        'Six completion rules evaluated by identical frozen crop-specific yield heads. '
        'All entries are thirteen-year mean annual yield RMSE (t/ha). No observed feedback '
        'removes prefix assimilation from the BIID recurrence but preserves the true prefix in the terminal input. '
        'This is a fixed-weight intervention, not a retrained no-feedback model.',
        'tab:inseason_core_completion','llrrrr','Suffix & Completion & Maize & Rice & Soybean & Wheat',lines)
    lines = []
    for crop,recipe in RECIPES.items():
        for product in recipe.split('_'):
            for percent in (10,30,50):
                sub = states[states.crop.eq(crop) & states['product'].eq(product) & states.percent.eq(percent)].set_index('method')
                lines.append(' & '.join([crop.title(),product.upper(),f'{percent}\\%']+
                    [f'{sub.loc[method,"mean_annual_rmse"]:.4f}' for method in METHODS])+r' \\')
    save_table('inseason_core_state_table.tex',
        'State errors for the matched completion rules. Values are mean annual hidden-slot RMSE, '
        'in NDVI units or GPP gC m$^{-2}$ day$^{-1}$. No obs. removes BIID observed-prefix feedback. '
        'The same finite-target hidden slots are scored for all methods; state units are not pooled across products.',
        'tab:inseason_core_state','lllrrrrrr',
        'Crop & State & Suffix & BIID & GRU & No obs. & Clim. & Previous & Last',lines)
    lines = []
    for (crop,product,percent),group in states.groupby(['crop','product','percent']):
        sub = group.set_index('method')
        cells = [crop.title(),product.upper(),f'{percent}\\%']
        cells += [f'{sub.loc[m,"mean_annual_anomaly_correlation"]:.3f}'
            for m in ('biid','gru','no_feedback','previous')]
        cells += [f'{int(sub.loc[m,"better_than_climatology_years"])}/13' for m in ('biid','gru')]
        lines.append(' & '.join(cells)+r' \\')
    save_table('inseason_core_anomaly_table.tex',
        'State anomaly fidelity after subtracting training-period grid--month climatology. '
        'Correlations are averaged over years. The final two columns count years with lower '
        'state RMSE than climatology. Climatology has undefined zero-anomaly correlation.',
        'tab:inseason_core_anomaly','lllrrrrrr',
        'Crop & State & Suffix & BIID & GRU & No obs. & Previous & BIID wins & GRU wins',lines)
    report = ['# 十三年同头补全与状态转移结果','',
        '这批实验固定历史专家、产量头、观测前缀、气象和样本，只改变未来植被的补全方式。GRU是重新训练的另一种状态结构；无观测反馈是原BIID权重的推理干预，两者不能混称同一种重训练消融。','',
        '## 10%后缀的产量误差','', '| 补全方式 | 玉米 | 水稻 | 大豆 | 小麦 |','|---|---:|---:|---:|---:|']
    sub = frame[frame.percent.eq(10)].pivot(index='method',columns='crop',values='mean_annual_rmse')
    for method in METHODS:
        report.append('| '+NAMES[method]+' | '+' | '.join(f'{sub.loc[method,c]:.4f}' for c in RECIPES)+' |')
    report += ['', '单位为吨/公顷，13个年度RMSE等权平均。每个作物的原BIID和常态结果已逐样本重放核验。','',
        f'![同头六种补全方式]({OUT}/same_head_completions.png)','',
        '怎么看：横轴越右，未观测后缀越长；纵轴越低产量越准。每一作物内所有曲线通过同一产量头，因此曲线差异比“世界模型对历史基线”的差异更直接检验补全机制。','',
        '## BIID与GRU及反馈干预','']
    for crop in RECIPES:
        for percent in (10,30,50):
            row = paired[paired.crop.eq(crop) & paired.percent.eq(percent)].set_index('reference')
            report.append(f'- {CN[crop]}、{percent}%后缀：BIID相对GRU年均产量RMSE降幅'
                f'{row.loc["gru","annual_mean_rmse_gain"]:+.2f}%，'
                f'{int(row.loc["gru","positive_years"])}/13年更好；相对无观测反馈降幅'
                f'{row.loc["no_feedback","annual_mean_rmse_gain"]:+.2f}%。')
    report += ['', '## 证据范围','',
        '状态和产量分别计分。NDVI与GPP单位不同，比较误差时应看同一产品、同一截点。较低状态误差不自动意味着较低产量误差。完整状态误差、去常态异常相关和常态skill均保存。', '',
        '主时间范围仍为2002–2004、2006–2008、2010–2016，seed42。原产量头保留原来的开发选型来源；本批没有更换为新前置验证产量头，也没有训练新主模型。', '',
        '后续仍需补训练层面的气象驱动、终端天气直连、元数据及数据组织对照，不把这次补全替换说成所有消融已完成。', '',
        f'- 产量结果：`{OUT}/yield_scores.csv`；逐年结果：`{OUT}/yield_annual.csv`。',
        f'- 状态结果：`{OUT}/state_scores.csv`；逐年结果：`{OUT}/state_annual.csv`。',
        f'- 状态权重索引：`{OUT}/state_checkpoint_registry.csv`。',
        f'- 含冻结产量头路径的完整来源核验：`{OUT}/audit.json`。']
    (OUT / '结果说明.md').write_text('\n'.join(report)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,annual,paired,state_annual,states,sources,registry = collect()
    for name,data in [('yield_scores',frame),('yield_annual',annual),('paired_gains',paired),
                      ('state_annual',state_annual),('state_scores',states)]:
        data.to_csv(OUT / (name+'.csv'),index=False)
    pd.DataFrame(registry).to_csv(OUT / 'state_checkpoint_registry.csv',index=False)
    figure(frame)
    export(frame,paired,states)
    atomic_json(OUT / 'audit.json',dict(sources=sources,
        generator_sha256=sha256(ROOT / 'scripts/summarize_inseason_state_controls.py'),
        crop_blocks=12,logical_yield_evaluations=216,new_yield_fits=0,
        new_state_fits=6,reused_gru_state_fits=9,seed=42,all_thirteen_years=True,
        original_biid_and_climatology_yield_replayed=True))
    print(frame[frame.percent.eq(10)].to_string(index=False),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--four-rule-figure', action='store_true',
                        help='Redraw four selected rules from saved scores without training')
    args = parser.parse_args()
    if args.four_rule_figure:
        four_rule_figure()
    else:
        main()
