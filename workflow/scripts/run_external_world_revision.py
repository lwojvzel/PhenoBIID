"""Independent administrative-label evaluation with cross-fitted BIID states."""

import argparse
import fcntl
import json
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from biid_world_model import CropWorldDynamics
from multimodal_baseline import regression_metrics, save_json, set_seed, write_csv
from review_revision_data import ROOT
from run_cybench_sample_external_validation import build_crop_table
from run_review_revision import train_neural, train_tree, rmse

DATA = ROOT / 'Data/external/CYBench/full_v1_10'
OUTPUT = ROOT / 'benchmark/results/review_revision_v2/external_cybench'
COUNTRIES = ('DE','FR','PL')


def load_data(crop):
    if not (DATA/'selected_download_complete.json').exists():
        raise RuntimeError('Official selected-country download is not complete')
    destination=OUTPUT/crop
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=build_crop_table(crop,False,data_root=DATA,result_root=OUTPUT,countries=COUNTRIES)
    indices={'train':np.flatnonzero(result['year']<=2016), 'validation':np.flatnonzero((result['year']>=2017)&(result['year']<=2018)), 'test':np.flatnonzero((result['year']>=2019)&(result['year']<=2023))}
    if min(map(len,indices.values()))<30:
        raise RuntimeError(f'Insufficient external samples: {[(k,len(v)) for k,v in indices.items()]}')
    return result,indices


def state_model(data,fit_index,predict_index,seed,epochs,destination,validation_index=None):
    set_seed(seed)
    device=torch.device('cuda')
    weather=data['climate']
    mean=weather[fit_index].mean(axis=(0,1),keepdims=True)
    std=weather[fit_index].std(axis=(0,1),keepdims=True).clip(1e-6)
    observed=data['target_state']
    valid=data['target_state_valid']>0
    lai_mean=float(observed[fit_index][valid[fit_index]].mean())
    lai_std=max(float(observed[fit_index][valid[fit_index]].std()),1e-6)
    weather=torch.as_tensor((weather-mean)/std,dtype=torch.float32)
    prev=torch.as_tensor((data['previous_state']-lai_mean)/lai_std,dtype=torch.float32)
    prev_valid=torch.as_tensor(data['previous_state_valid'],dtype=torch.float32)
    target=torch.as_tensor((observed-lai_mean)/lai_std,dtype=torch.float32)
    target_valid=torch.as_tensor(valid,dtype=torch.float32)
    context=torch.as_tensor(data['context'])
    history=torch.zeros(len(target),15)
    model=CropWorldDynamics('biid_climate',weather_variables=8).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,fused=True)
    scaler=torch.amp.GradScaler('cuda')
    def forward(idx):
        return model(weather[idx].to(device),prev[idx].to(device),prev_valid[idx].to(device),torch.ones(len(idx),12,device=device),history[idx].to(device),context[idx].to(device))[0]
    @torch.no_grad()
    def predict(index):
        model.eval()
        result=[]
        for start in range(0,len(index),256):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=forward(index[start:start+256])
            result.append(value.float().cpu().numpy()*lai_std+lai_mean)
        return np.concatenate(result)
    best=float('inf');best_epoch=epochs;stale=0;best_state=None;rows=[]
    for epoch in range(1,epochs+1):
        model.train()
        order=np.random.permutation(fit_index)
        total,count=0.,0
        for start in range(0,len(order),256):
            idx=order[start:start+256]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=forward(idx)
                mask=target_valid[idx].to(device)
                loss=(((pred-target[idx].to(device))**2)*mask).sum()/mask.sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*len(idx);count+=len(idx)
        row=dict(epoch=epoch,loss=total/count)
        if validation_index is not None:
            value=predict(validation_index)
            mask=valid[validation_index]
            score=rmse(observed[validation_index][mask],value[mask])
            row['validation_state_rmse']=score
            if score<best-1e-6:
                best,best_epoch,stale=score,epoch,0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        rows.append(row)
        print(f'[EXTERNAL STATE] {destination.name} epoch={epoch} {row}',flush=True)
        if validation_index is not None and stale>=6:break
    if best_state is not None:model.load_state_dict(best_state)
    prediction=predict(predict_index)
    destination.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),destination/'dynamics_best.pt')
    write_csv(rows,destination/'training_history.csv')
    save_json(dict(seed=seed,fit_indices=fit_index.tolist(),prediction_indices=predict_index.tolist(),weather_mean=mean.tolist(),weather_std=std.tolist(),state_mean=lai_mean,state_std=lai_std,selected_epoch=best_epoch),destination/'config.json')
    del model,optimizer,scaler
    torch.cuda.empty_cache()
    return prediction,best_epoch


def prepare_arrays(data,indices,state):
    train=indices['train']
    history=data['history'].copy()
    med=np.nanmedian(history[train],axis=0)
    med=np.where(np.isfinite(med),med,0.)
    history=np.where(np.isfinite(history),history,med)
    baseline=history[:,13].copy()
    mu=history[train].mean(0);sigma=history[train].std(0).clip(1e-6)
    mu[5:10]=0;sigma[5:10]=1
    normalized=(history-mu)/sigma
    residual=data['target']-baseline
    residual_mean=float(residual[train].mean());residual_std=max(float(residual[train].std()),1e-6)
    target=(residual-residual_mean)/residual_std
    x=np.concatenate((normalized,data['context']),axis=1)
    anchor=np.full(len(target),np.nan,dtype=np.float32)
    splitter=GroupKFold(n_splits=5)
    for fit,held in splitter.split(train,groups=data['year'][train]):
        model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)
        model.fit(x[train[fit]],target[train[fit]])
        anchor[train[held]]=model.predict(x[train[held]])
    model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)
    model.fit(x[train],target[train])
    other=np.concatenate((indices['validation'],indices['test']))
    anchor[other]=model.predict(x[other])
    climate=data['climate'];wm=climate[train].mean((0,1));ws=climate[train].std((0,1)).clip(1e-6)
    truth=data['target_state'];mask=data['target_state_valid']>0
    lm=float(truth[train][mask[train]].mean());ls=max(float(truth[train][mask[train]].std()),1e-6)
    arrays={}
    for split,ix in indices.items():
        arrays[split]={
            'weather':((climate[ix]-wm)/ws).astype(np.float32),
            'previous_lai':((data['previous_state'][ix]-lm)/ls).astype(np.float32),
            'previous_lai_valid':data['previous_state_valid'][ix].astype(np.float32),
            'relative_valid':np.ones((len(ix),12),dtype=np.float32),
            'history':normalized[ix].astype(np.float32),'context':data['context'][ix],
            'history_base':anchor[ix], 'crop_coverage':np.zeros(len(ix),dtype=np.float32),
            'state':((state[ix]-lm)/ls).astype(np.float32),
            'climatology_lai':np.zeros((len(ix),12),dtype=np.float32),
            'target_residual':target[ix].astype(np.float32),
            'target_lai':((truth[ix]-lm)/ls).astype(np.float32),'target_lai_valid':mask[ix].astype(np.float32),
            'target':data['target'][ix],'baseline':baseline[ix],'source_indices':ix,
        }
        if not np.isfinite(arrays[split]['state']).all():raise RuntimeError('Missing OOF/future state')
    meta=dict(normalization=dict(residual_mean=residual_mean,residual_std=residual_std,lai_mean=lm,lai_std=ls),uniform_drop_probability=0.,weather_mean=wm.tolist(),weather_std=ws.tolist(),history_mean=mu.tolist(),history_std=sigma.tolist(),coverage='unavailable; q=0 constant; modality dropout disabled',state='fPAR fraction, not GLASS LAI',training_predictions='five-fold cross-fitting over harvest years')
    return arrays,meta,model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop',choices=('maize','wheat'),required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data,indices=load_data(args.crop)
    root=OUTPUT/args.crop/f'seed_{args.seed}'
    root.mkdir(parents=True,exist_ok=True)
    if (root/'test_metrics.json').exists():return
    state_path=root/'cross_fitted_states.npz'
    if state_path.exists():
        with np.load(state_path) as f:state=f['state']
    else:
        state=np.full_like(data['target_state'],np.nan)
        predict_indices=np.concatenate((indices['validation'],indices['test']))
        value,epochs=state_model(data,indices['train'],predict_indices,args.seed,40,root/'dynamics_full',indices['validation'])
        state[predict_indices]=value
        train=indices['train']
        for fold,(fit,held) in enumerate(GroupKFold(n_splits=5).split(train,groups=data['year'][train])):
            value,_=state_model(data,train[fit],train[held],args.seed+fold+100,epochs,root/f'dynamics_oof_{fold}')
            state[train[held]]=value
        np.savez_compressed(state_path,state=state,**{k:data[k] for k in ('year','country','adm_id')})
    arrays,meta,history_model=prepare_arrays(data,indices,state)
    joblib.dump(history_model,root/'history_full.joblib')
    save_json(dict(crop=args.crop,seed=args.seed,countries=list(COUNTRIES),split_sizes={s:len(i) for s,i in indices.items()},split_years={s:np.unique(data['year'][i]).tolist() for s,i in indices.items()},input_metadata=meta,protocol='new independent administrative-label countries; within-country future-year holdout; not zero-shot geographic transfer'),root/'config.json')
    settings=[('lightgbm',False,'predicted'),('gru',False,'predicted'),('transformer',False,'predicted'),('multitask_gru',False,'predicted'),('fusion',False,'predicted'),('fusion',False,'previous'),('fusion',True,'predicted')]
    results=[]
    for name,climate,source in settings:
        key=name if name!='fusion' else f'fusion_{source}_{"climate" if climate else "state_only"}'
        destination=root/key;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'test_metrics.json').exists():
            results.append(json.loads((destination/'test_metrics.json').read_text()));continue
        set_seed(args.seed)
        current={s:{**a,'state':a['previous_lai'] if source=='previous' else a['state']} for s,a in arrays.items()}
        configuration=SimpleNamespace(model=name,climate=climate,gate='no_bce',moddrop='none',crop=args.crop,seed=args.seed,state=source,epochs=60,patience=10,batch_size=256)
        save_json(vars(configuration),destination/'config.json')
        predictions,details=train_tree(configuration,current,meta,destination) if name=='lightgbm' else train_neural(configuration,current,meta,destination)
        metrics={}
        for split,values in predictions.items():
            a=current[split];ix=indices[split]
            history=a['baseline']+a['history_base']*meta['normalization']['residual_std']+meta['normalization']['residual_mean']
            latitude=data['context'][ix,0]*90
            longitude=np.rad2deg(np.arctan2(data['context'][ix,1],data['context'][ix,2]))
            np.savez_compressed(destination/f'{split}_predictions.npz',**values,history_prediction=history,target=a['target'],source_indices=ix,country=data['country'][ix],adm_id=data['adm_id'][ix],year=data['year'][ix],latitude=latitude,longitude=longitude)
            m=regression_metrics(a['target'],values['prediction']);h=rmse(a['target'],history)
            metrics[split]={**m,'history_rmse':h,'gain_over_history_percent':100*(h-m['rmse'])/h}
        result=dict(method=key,seed=args.seed,crop=args.crop,metrics=metrics,training=details)
        save_json(result,destination/'test_metrics.json');results.append(result)
        print('[EXTERNAL RESULT] '+json.dumps(result),flush=True)
    save_json(dict(crop=args.crop,seed=args.seed,results=results),root/'test_metrics.json')


if __name__=='__main__':
    main()
