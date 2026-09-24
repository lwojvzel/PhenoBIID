"""Paired direct-model evidence for weather, vegetation and shared quality."""
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT,CACHE,RECIPES,BLOCKS,load,partition
from inseason_nested_common import LABELS
from review_revision_data import sha256
from run_inseason_13year_direct import root_for as full_root,CODE as FULL_CODE
from run_inseason_modality_controls import root_for,verify,CONDITIONS
from inseason_modality_controls import ablate
from inseason_complete_inputs import flat_features
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from summarize_inseason_13year import annual_gain
from summarize_inseason_six_direct import complete_score

OUT = ROOT / 'visualize/paper_experiments/inseason_modality_controls_v1'
TEX = ROOT / 'Paper/iclr2027/sections'
NAMES = dict(full='Full direct LightGBM',no_remote='Without vegetation inputs',
    no_weather='Without weather inputs',no_shared_quality='Without shared product quality')
CN = dict(maize='玉米',rice='水稻',soybean='大豆',wheat='小麦')


def collect():
    rows,annual,paired,registry,sources = [],[],[],[],{}
    code = {name:sha256(ROOT / 'scripts' / name) for name in FULL_CODE}
    for crop in RECIPES:
        raw,_ = load(crop)
        sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
        labels = {key:np.concatenate([raw[key][partition(raw,cutoff)['evaluation']]
            for cutoff in BLOCKS]) for key in LABELS}
        predictions = {condition:[] for condition in ('full',*CONDITIONS)}
        for cutoff in BLOCKS:
            ix = partition(raw,cutoff)['evaluation']
            for condition in predictions:
                root = full_root(crop,cutoff,'lightgbm') if condition=='full' else root_for(crop,cutoff,condition)
                if condition=='full':
                    marker = json.loads((root / 'complete.json').read_text())
                    check_files(root,marker['files'])
                    if marker['code_sha256'] != code:
                        raise ValueError('Changed complete direct LightGBM source')
                else:
                    marker = verify(root)
                if marker['smoke']:
                    raise ValueError('Smoke cannot enter modality evidence')
                cfg = json.loads((root / 'config.json').read_text())
                if cfg['crop'] != crop or cfg['cutoff'] != cutoff or cfg['seed'] != 42:
                    raise ValueError('Invalid modality model identity')
                if condition!='full' and cfg['condition']!=condition:
                    raise ValueError('Incorrect input-removal condition')
                sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
                dest = root / 'tail_10' if condition=='full' else root
                path = dest / 'evaluation_predictions.npz'
                with np.load(path) as saved:
                    for key in LABELS:
                        np.testing.assert_array_equal(saved[key],raw[key][ix])
                    predictions[condition].append(saved['prediction'].copy())
                sources[str(path)] = sha256(path)
                removed = np.array([],dtype=int)
                if condition!='full':
                    products = len(RECIPES[crop].split('_'))
                    probe = dict(sequence=np.ones((1,12,38+7*products)),static=np.ones((1,21+24*products)))
                    removed = np.flatnonzero(flat_features(ablate(probe,condition))[0]==0)
                    fitted = joblib.load(dest / 'model.joblib')
                    importance = fitted.booster_.feature_importance(importance_type='split')
                    if np.any(importance[removed]!=0):
                        raise ValueError('Tree split references a removed feature')
                registry.append(dict(crop=crop,cutoff=cutoff,condition=condition,seed=42,
                    weight=str(dest / 'model.joblib'),sha256=sha256(dest / 'model.joblib'),
                    config=str(root / 'config.json'),metrics=str(dest / 'metrics.json'),
                    removed_feature_indices=json.dumps(removed.tolist()),masked_tree_splits_verified=True))
        predictions = {key:np.concatenate(value) for key,value in predictions.items()}
        for condition,prediction in predictions.items():
            metric = complete_score(labels,prediction)
            rows.append(dict(crop=crop,condition=condition,pooled_rmse=metric['pooled_rmse'],
                mean_annual_rmse=metric['mean_annual_rmse']))
            annual.extend(dict(crop=crop,condition=condition,year=int(year),rmse=value)
                for year,value in metric['per_year_rmse'].items())
            if condition!='full':
                paired.append(dict(crop=crop,removed=condition,
                    **annual_gain(labels['target'],predictions['full'],prediction,labels['year'])))
    if len(registry)!=48:
        raise ValueError('Expected 36 new and 12 reused direct modality models')
    return pd.DataFrame(rows),pd.DataFrame(annual),pd.DataFrame(paired),registry,sources


def export(frame,paired):
    values = frame.pivot(index='condition',columns='crop',values='mean_annual_rmse')
    lines = [' & '.join([NAMES[c]]+[f'{values.loc[c,crop]:.4f}' for crop in RECIPES])+r' \\'
        for c in ('full',*CONDITIONS)]
    (TEX / 'inseason_modality_table.tex').write_text('% Generated by scripts/summarize_inseason_modality_controls.py\n'
        '\\begin{table}[htbp]\n\\centering\\small\n'
        '\\caption{Input ablations of direct LightGBM at the 10\\% suffix. Entries are thirteen-year '
        'mean annual yield RMSE (t/ha). Each condition shares samples, normalization, historical anchor and '
        'candidate budgets. Removal is applied in both fitting and inference; heads are refitted.}\n'
        '\\label{tab:inseason_direct_modalities}\n\\begin{tabular}{lrrrr}\n\\toprule\n'
        'Input condition & Maize & Rice & Soybean & Wheat \\\\\n\\midrule\n'+
        '\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,ax = plt.subplots(figsize=(9.5,4.8))
    fig.subplots_adjust(left=.10,right=.98,bottom=.24,top=.93)
    centers = np.arange(4)
    for offset,(condition,color) in enumerate(zip(CONDITIONS,('#b7673d','#3267a3','#777777'))):
        sub = paired[paired.removed.eq(condition)].set_index('crop')
        ax.bar(centers+(offset-1)*.23,[sub.loc[c,'annual_mean_rmse_gain'] for c in RECIPES],
            width=.22,color=color,label=NAMES[condition])
    ax.axhline(0,color='.25',linewidth=.8)
    ax.set_xticks(centers,[c.title() for c in RECIPES])
    ax.set_ylabel('Full-input RMSE reduction (%)')
    ax.set_title('Direct LightGBM | nominal 10% unobserved suffix',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(OUT / f'direct_modality_value.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    report = ['# 直接LightGBM的模态价值：十三年结果','',
        '这组不是更换世界模型，而是在同一个直接回归模型、同样样本和选型预算下删输入并重新拟合。主截点为10%后缀。', '',
        '| 输入条件 | 玉米 | 水稻 | 大豆 | 小麦 |','|---|---:|---:|---:|---:|']
    for condition in ('full',*CONDITIONS):
        report.append('| '+NAMES[condition]+' | '+' | '.join(f'{values.loc[condition,c]:.4f}' for c in RECIPES)+' |')
    report += ['', '单位为吨/公顷；13个年度空间RMSE等权平均，越低越好。','',
        f'![直接回归的模态贡献]({OUT}/direct_modality_value.png)','',
        '怎么看：横轴是作物。每种柱颜色对应一种被删除的信息，柱高表示完整输入比该删除条件降低了多少RMSE。正柱说明保留它更好；负柱说明本实验中删掉它反而更好。不是“柱越高，被删除条件越好”。', '',
        '## 各条件和结论','']
    for condition in CONDITIONS:
        report.append(f'### {NAMES[condition]}')
        for crop in RECIPES:
            row = paired[paired.crop.eq(crop) & paired.removed.eq(condition)].iloc[0]
            report.append(f'- {CN[crop]}：完整输入相对该删除条件年均RMSE降幅{row.annual_mean_rmse_gain:+.2f}%，{int(row.positive_years)}/13年更好。')
    report += ['', '无遥感删掉所选产品值、异常、统计量、有效性、上一年质量和共享遥感质量字段；日历、覆盖、历史及天气仍保留。无气象删气象值及有效性；其他输入保留。无共享产品质量只删六个额外支持/质量字段，保留所选产品的有效性和上一年质量。', '',
        '本表检验输入对直接LightGBM的作用，不检验BIID的状态演化或冻结世界模型的终端天气直连。所有条件使用同样13年和样本，不能把较差条件归因于被重新筛过样本。','',
        '来源说明：原训练配置中的`quality_control`文字是“无共享产品质量”条件的解释，不是所有条件统一删除规则；实际规则由`condition`及已核验的`ablate`代码确定。原记录保持不变，权重索引另列实际删除的列，并逐一验证树没有在这些列上分裂。','',
        f'- 权重索引（36新增、12复用）：`{OUT}/checkpoint_registry.csv`。',
        f'- 全部误差、逐年结果、配对收益及来源：`{OUT}/scores.csv`、`annual.csv`、`paired_gains.csv`、`audit.json`。']
    (OUT / '结果说明.md').write_text('\n'.join(report)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,annual,paired,registry,sources = collect()
    for name,data in [('scores',frame),('annual',annual),('paired_gains',paired)]:
        data.to_csv(OUT / (name+'.csv'),index=False)
    pd.DataFrame(registry).to_csv(OUT / 'checkpoint_registry.csv',index=False)
    export(frame,paired)
    atomic_json(OUT / 'audit.json',dict(sources=sources,new_models=36,reused_models=12,
        all_thirteen_years=True,all_conditions_and_common_samples=True,seed=42,
        generator_sha256=sha256(ROOT / 'scripts/summarize_inseason_modality_controls.py')))
    print(frame.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
