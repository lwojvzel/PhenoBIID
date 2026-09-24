"""Aggregate rolling direct controls on the complete thirteen-year population."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES, load, partition
from run_inseason_13year_direct import root_for, CODE
from run_inseason_direct_baselines import score
from run_review_revision_parallel import atomic_json
from run_ndvi_signal_permutation import check_files
from review_revision_data import sha256
from summarize_inseason_13year import OUT, annual_gain, write_table

MODELS = ('lightgbm','gru','transformer')
NAMES = dict(lightgbm='Direct LightGBM',gru='Direct GRU',transformer='Direct Transformer',world='Frozen world-model readout')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows, annual, paired, sources,checkpoints = [], [], [], {},[]
    hashes = {n:sha256(ROOT/'scripts'/n) for n in CODE}
    for crop,recipe in RECIPES.items():
        raw,_ = load(crop)
        all_ix = np.concatenate([partition(raw,end)['evaluation'] for end in BLOCKS])
        truth, years = raw['target'][all_ix],raw['year'][all_ix]
        for model in MODELS:
            for end in BLOCKS:
                root=root_for(crop,end,model)
                marker=json.loads((root/'complete.json').read_text())
                check_files(root,marker['files'])
                if hashes != marker['code_sha256']:
                    raise ValueError('Changed rolling seasonal control')
                sources[str(root/'complete.json')]=sha256(root/'complete.json')
        for percent in (10,30,50):
            outputs={}
            for model in MODELS:
                chunks=[]
                for end in BLOCKS:
                    ix=partition(raw,end)['evaluation']
                    path=root_for(crop,end,model)/f'tail_{percent:02d}/evaluation_predictions.npz'
                    with np.load(path) as f:
                        for k in ('year','row','col','target','source_indices'):
                            np.testing.assert_array_equal(raw[k][ix],f[k])
                        chunks.append(f['prediction'])
                    weight=path.parent/('model.joblib' if model=='lightgbm' else 'model.pt')
                    checkpoints.append(dict(crop=crop,cutoff=end,model=model,percent=percent,
                        seed=42,weight=str(weight),metrics=str(path.parent/'metrics.json'),predictions=str(path)))
                outputs[model]=np.concatenate(chunks)
            outputs['world']=np.concatenate([np.load(CACHE/crop/f'world_{end}_{recipe}_biid_{percent:03d}.npy') for end in BLOCKS])
            for model,pred in outputs.items():
                metrics=score(truth,pred,years)
                rows.append(dict(crop=crop,percent=percent,model=model,**metrics))
                for y,error in metrics['per_year_rmse'].items():
                    annual.append(dict(crop=crop,percent=percent,model=model,year=int(y),rmse=error))
                if model!='world':
                    paired.append(dict(crop=crop,percent=percent,reference=model,
                        **annual_gain(truth,outputs['world'],pred,years)))
    frame=pd.DataFrame(rows)
    frame.drop(columns='per_year_rmse').to_csv(OUT/'direct.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'direct_annual.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'direct_gains.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(OUT/'direct_checkpoint_registry.csv',index=False)
    result=frame[frame.percent.eq(10)].pivot(index='model',columns='crop',values='mean_annual_rmse')
    best=result.loc[list(MODELS)].min()
    winners=result.loc[list(MODELS)].idxmin()
    winning_methods=(NAMES[winners.iloc[0]]+' for every crop' if winners.nunique()==1
        else ', '.join(NAMES[winners[c]]+' for '+c for c in RECIPES))
    wins=sum(result.loc['world',c]<best[c] for c in RECIPES)
    text=('At the 10\\% suffix, the direct-model minimum is attained by '
        +winning_methods+'. '
        'Their mean annual RMSEs are '+', '.join(f'{best[c]:.4f}' for c in RECIPES)+'. '
        'The frozen world-model pipeline is below the best direct error for '
        +('all four crops. ' if wins==4 else f'{wins} of the four crops. ')
        +'This minimum summarizes the reported candidates; it does not select or retrain the world model. '
        'All registered cutoffs are retained in Table~\\ref{tab:inseason_13year_direct_leads}; '
        'these aggregate advantages need not hold in every year.\n')
    (ROOT/'Paper/iclr2027/sections/inseason_13year_direct_findings.tex').write_text(text)
    for lead in (False,True):
        entries=[]
        for percent in ((10,30,50) if lead else (10,)):
            for model in (*MODELS,'world'):
                block=frame[frame.percent.eq(percent)&frame.model.eq(model)].set_index('crop')
                name=r'\method{} (frozen readout)' if model=='world' else NAMES[model]
                cells=([f'{percent}\\%'] if lead else [])+[name]+[f'{block.loc[c,"mean_annual_rmse"]:.4f}' for c in RECIPES]
                entries.append(' & '.join(cells)+r' \\')
        write_table('inseason_13year_direct_leads.tex' if lead else 'inseason_13year_direct_table.tex',
            'Direct seasonal prediction on the main thirteen-year cohort. Mean annual yield RMSE (t/ha); '
            'lower is better. Direct models share structured inputs, quality masking, and a causal-trend anchor. '
            'The world-model pipeline retains its crop-specific historical anchors and frozen readouts.',
            'tab:inseason_13year_direct_leads' if lead else 'tab:inseason_13year_direct',
            ('Suffix & ' if lead else '')+'Method & Maize & Rice & Soybean & Wheat',
            'llrrrr' if lead else 'lrrrr',entries)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.1,7.2))
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.94,hspace=.45,wspace=.30)
    colors=dict(lightgbm='#ba6338',gru='#3267a3',transformer='#985d91',world='#1c7c68')
    for ax,crop in zip(axes.flat,RECIPES):
        for model in (*MODELS,'world'):
            block=frame[frame.crop.eq(crop)&frame.model.eq(model)].sort_values('percent')
            ax.plot(block.percent,block.mean_annual_rmse,'o-',color=colors[model],label=NAMES[model],markersize=4)
        ax.set_title(crop.title()+' | '+RECIPES[crop].upper().replace('_',' + '),fontsize=10)
        ax.set_xticks([10,30,50])
        ax.set_xlabel('Nominal unobserved suffix (%)')
        ax.set_ylabel('Thirteen-year mean RMSE (t/ha)')
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    h,l=axes.flat[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.53,.015),fontsize=9)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'direct_comparison.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    atomic_json(OUT/'direct_audit.json',dict(sources=sources,final_models=108,all_years_replayed=True,
        complete_registered_methods_and_cutoffs=True,primary_metric='Mean of thirteen annual RMSEs',
        refits_before_evaluation=True,seed=42))
    print(frame[frame.percent.eq(10)].drop(columns='per_year_rmse').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
