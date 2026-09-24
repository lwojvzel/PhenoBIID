"""Forward-fitted history anchors and small, validation-selected state readouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from review_revision_data import load_shared, CROPS, SEEDS, RESULT_ROOT
from multimodal_baseline import save_json, regression_metrics

OUTPUT=RESULT_ROOT/'forward_history_readouts'
METHODS=('static_only','previous_state','climatology_state','predicted_state','weather_only','predicted_and_weather')
ALPHAS=(.1,1.,10.,100.,1000.,10000.)


def expert():
    return HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=42)


def anchors(crop,arrays):
    root=OUTPUT/crop;root.mkdir(parents=True,exist_ok=True)
    with (root/'anchor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'forward_anchors.npz'
        if path.exists():
            with np.load(path) as f:return {k:f[k] for k in f.files}
        x={s:np.concatenate((a['history'],a['context']),axis=1) for s,a in arrays.items()}
        year=arrays['train']['year'];y=arrays['train']['target_residual']
        output={'train':np.full(len(y),np.nan,dtype=np.float32)}
        folds=[]
        for start in (2002,2004,2006,2008,2010):
            fit=np.flatnonzero(year<start);held=np.flatnonzero((year>=start)&(year<=start+1))
            if not len(fit) or not len(held):raise RuntimeError('Empty forward fold')
            model=expert();model.fit(x['train'][fit],y[fit]);output['train'][held]=model.predict(x['train'][held])
            joblib.dump(model,root/f'history_before_{start}.joblib')
            folds.append(dict(train_end=start-1,prediction_years=[start,start+1],train_samples=len(fit),predicted_samples=len(held)))
            print(f'[FORWARD ANCHOR] {crop}/{start}',flush=True)
        model=expert();model.fit(x['train'],y);joblib.dump(model,root/'history_full.joblib')
        for split in ('validation','test'):output[split]=model.predict(x[split]).astype(np.float32)
        np.savez_compressed(path,**output)
        save_json(dict(crop=crop,folds=folds,full_expert_train_end=2011,states='Frozen in-sample training trajectories; only history predictions are forward-fitted'),root/'anchor_config.json')
        return output


def state_summary(state,valid):
    mask=valid>0;n=mask.sum(1).clip(1)
    mean=(state*mask).sum(1)/n
    maximum=np.where(mask,state,-np.inf).max(1);minimum=np.where(mask,state,np.inf).min(1)
    std=np.sqrt((((state-mean[:,None])**2)*mask).sum(1)/n)
    first=state[np.arange(len(state)),mask.argmax(1)]
    last=state[np.arange(len(state)),11-mask[:,::-1].argmax(1)]
    peak=np.where(mask,state,-np.inf).argmax(1)/11.
    change=np.diff(state,axis=1);pair=mask[:,1:]&mask[:,:-1]
    growth=(np.maximum(change,0)*pair).sum(1)/pair.sum(1).clip(1)
    return np.stack((mean,maximum,minimum,std,first,last,peak,growth),axis=1)


def features(a,anchor,method):
    q=a['crop_coverage'];r=.1+.9*q/(q+.05)
    pieces=[a['history'],a['context'],q[:,None],r[:,None],anchor[:,None]]
    if method in ('previous_state','climatology_state','predicted_state','predicted_and_weather'):
        source={'previous_state':'previous_lai','climatology_state':'climatology_lai'}.get(method,'predicted_lai')
        summary=state_summary(a[source],a['relative_valid'])
        pieces += [summary,summary-state_summary(a['previous_lai'],a['relative_valid'])]
    if method in ('weather_only','predicted_and_weather'):
        pieces.append((a['weather']*a['relative_valid'][...,None]).reshape(len(q),-1))
    return np.concatenate(pieces,axis=1).astype(np.float64)


def run(job):
    crop,seed=job;started=time.monotonic();root=OUTPUT/crop/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True)
    if (root/'test_metrics.json').exists():return
    arrays,meta=load_shared(crop,seed);anchor=anchors(crop,arrays);keep=np.isfinite(anchor['train'])
    stats=meta['normalization'];results=[]
    for method in METHODS:
        destination=root/method;destination.mkdir(parents=True,exist_ok=True)
        x={s:features(a,anchor[s],method) for s,a in arrays.items()}
        y=arrays['train']['target_residual'][keep]-anchor['train'][keep]
        val_y=arrays['validation']['target_residual']-anchor['validation']
        candidates=[];best=None
        for alpha in ALPHAS:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(x['train'][keep],y)
            score=float(np.sqrt(np.mean((model.predict(x['validation'])-val_y)**2)))
            candidates.append(dict(alpha=alpha,validation_residual_rmse=score))
            if best is None or score<best[0]:best=(score,alpha,model)
        model=best[2];joblib.dump(model,destination/'model_best.joblib');metrics={}
        for split in ('validation','test'):
            a=arrays[split];history=a['baseline']+anchor[split]*stats['residual_std']+stats['residual_mean']
            pred=history+model.predict(x[split])*stats['residual_std']
            m=regression_metrics(a['target'],pred);h=float(np.sqrt(np.mean((history-a['target'])**2)))
            metrics[split]={**m,'history_rmse':h,'gain_over_history_percent':100*(h-m['rmse'])/h}
            np.savez_compressed(destination/f'{split}_predictions.npz',target=a['target'],prediction=pred,history_prediction=history,source_indices=a['source_indices'],year=a['year'],row=a['row'],col=a['col'])
        save_json(dict(crop=crop,seed=seed,method=method,selected_alpha=best[1],candidates=candidates,history_readout_training_years=[2002,2011],training_samples=int(keep.sum()),input_features=x['test'].shape[1],state_summary=['mean','max','min','std','first','last','peak_position','positive_change'],summary_change='relative to previous LAI; no biological interpretation of latent variables',normalization=stats),destination/'config.json')
        record=dict(crop=crop,seed=seed,method=method,metrics=metrics);save_json(record,destination/'test_metrics.json');results.append(record)
        print('[FORWARD READOUT] '+json.dumps(record),flush=True)
    save_json(dict(crop=crop,seed=seed,results=results,elapsed_seconds=time.monotonic()-started),root/'test_metrics.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--crop',choices=CROPS);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.all:
        with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(run,[(c,s) for c in CROPS for s in SEEDS]))
    else:run((a.crop,a.seed))


if __name__=='__main__':main()
