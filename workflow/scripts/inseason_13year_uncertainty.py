"""Spatial-block uncertainty for the equal-year thirteen-year comparison."""
import json

import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES, load, partition
from run_inseason_13year_history import root_for, MODELS
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from review_revision_data import sha256
from summarize_inseason_13year import OUT, write_table
from inseason_13year_extended_history import PREFIX,extended_predictions


def interval(labels, candidate, reference, degrees, draws=1000):
    step=int(2*degrees)
    block=(labels['row'].astype(int)//step)*(720//step)+labels['col'].astype(int)//step
    _,inv=np.unique(block,return_inverse=True)
    nblocks=int(inv.max()+1)
    target=labels['target'].astype(float)
    ec,er,n=[],[],[]
    for year in np.unique(labels['year']):
        use=labels['year']==year
        ec.append(np.bincount(inv[use],weights=(candidate[use]-target[use])**2,minlength=nblocks))
        er.append(np.bincount(inv[use],weights=(reference[use]-target[use])**2,minlength=nblocks))
        n.append(np.bincount(inv[use],minlength=nblocks))
    ec,er,n=map(lambda a:np.array(a).T,(ec,er,n))
    rng=np.random.default_rng(42)
    annual,ratio=[],[]
    for start in range(0,draws,100):
        w=rng.multinomial(nblocks,np.full(nblocks,1./nblocks),size=min(100,draws-start))
        denom=w@n
        if np.any(denom<=0):
            raise ValueError('A spatial draw omitted an entire evaluation year')
        a,b=np.sqrt((w@ec)/denom),np.sqrt((w@er)/denom)
        annual.extend(100*(1-a/b).mean(1))
        ratio.extend(100*(1-a.mean(1)/b.mean(1)))
    low,high=np.quantile(annual,[.025,.975])
    rlow,rhigh=np.quantile(ratio,[.025,.975])
    return dict(degrees=degrees,blocks=nblocks,draws=draws,mean_annual_gain_low=float(low),
        mean_annual_gain_high=float(high),annual_mean_rmse_gain_low=float(rlow),annual_mean_rmse_gain_high=float(rhigh))


def historical(crop,end,method,labels,sources):
    if method.startswith(PREFIX):
        return extended_predictions(crop,end,labels,sources)[method]
    if method=='hgb_mlp_ensemble':
        return .5*(historical(crop,end,'hist_gradient_boosting',labels,sources)+historical(crop,end,'mlp',labels,sources))
    root=root_for(crop,end,method if method in MODELS else 'classical')
    marker=json.loads((root/'complete.json').read_text())
    check_files(root,marker['files'])
    path=root/method/'evaluation_predictions.npz'
    sources[str(path)]=sha256(path)
    with np.load(path) as f:
        for key in ('target','year','row','col','source_indices'):
            np.testing.assert_array_equal(labels[key],f[key])
        return f['prediction']


def main():
    compare=pd.read_csv(OUT/'comparisons.csv').set_index('crop')
    records,sources=[],{}
    for crop,recipe in RECIPES.items():
        raw,_=load(crop)
        labels={k:[] for k in ('target','year','row','col','source_indices')}
        candidates,references=[],[]
        for end in BLOCKS:
            ix=partition(raw,end)['evaluation']
            part={k:raw[k][ix] for k in labels}
            for k in labels:
                labels[k].append(part[k])
            references.append(historical(crop,end,compare.loc[crop,'baseline'],part,sources))
            path=CACHE/crop/f'world_{end}_{recipe}_biid_010.npy'
            sources[str(path)]=sha256(path)
            candidates.append(np.load(path))
        labels={k:np.concatenate(v) for k,v in labels.items()}
        candidate,reference=np.concatenate(candidates),np.concatenate(references)
        for degrees in (10,20):
            records.append(dict(crop=crop,baseline=compare.loc[crop,'baseline'],
                point_mean_annual_gain=compare.loc[crop,'mean_annual_gain'],
                point_annual_mean_rmse_gain=compare.loc[crop,'annual_mean_rmse_gain'],
                **interval(labels,candidate,reference,degrees)))
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'uncertainty.csv',index=False)
    rows=[r.crop.title()+f' & {r.degrees} & {r.point_mean_annual_gain:+.2f} & '
        +f'[{r.mean_annual_gain_low:+.2f}, {r.mean_annual_gain_high:+.2f}] & '
        +f'{r.blocks}'+r' \\' for r in frame.itertuples()]
    write_table('inseason_13year_uncertainty_table.tex',
        'Paired 95\\% spatial-block intervals for the mean annual RMSE reduction (\\%) '
        'at the 10\\% suffix. Each of 1,000 draws retains all thirteen years and both '
        'predictions within a sampled block. These intervals condition on fitted models '
        'and do not include model-selection or seed uncertainty.',
        'tab:inseason_13year_uncertainty','Crop & Block degrees & Gain (\\%) & 95\\% interval & Blocks','lrrrr',rows)
    atomic_json(OUT/'uncertainty_audit.json',dict(sources=sources,metric='Mean of paired annual RMSE reductions',
        resampling='Spatial blocks with all thirteen years retained',random_seed=42,
        fitted_seed_uncertainty=False,model_selection_uncertainty=False))
    print(frame.to_string(index=False))


if __name__=='__main__':
    main()
