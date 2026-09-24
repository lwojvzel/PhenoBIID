"""Rolling observed-RS diagnostics with a three-year validation interval."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path

import numpy as np

from dual_remote_data import extract
from forward_protocol_revision import RawInputs, index_hash
from observed_remote_anomaly import seasonal_anomalies
from observed_remote_benchmark import INPUTS, make_features, trajectory_features
from prepare_pku_ndvi import OUT as NDVI_ROOT
from review_revision_data import ROOT, CROPS, SPLITS, sha256
from run_review_revision_parallel import atomic_json

CACHE=ROOT/'benchmark/cache/stable_remote_v1'
RESULT=ROOT/'benchmark/results/stable_remote_v1'
LOGS=ROOT/'benchmark/logs/stable_remote_v1'
ORIGINS=(2004,2008,2012)
WINDOWS=(0,12)
BASE_CONDITIONS=('history','metadata','lai_raw','lai_anomaly','lai_weighted')
EXTRA_CONDITIONS=('metadata_regional','lai_contrast','lai_regional','both_anomaly')
CONDITIONS=BASE_CONDITIONS+EXTRA_CONDITIONS


def split_rows(years,origin,window,val_span=3):
    years=np.asarray(years)
    train_end=origin-val_span
    start=int(years.min()) if window==0 else train_end-window+1
    masks=dict(train=(years>=start)&(years<=train_end),
               validation=(years>train_end)&(years<=origin),
               test=(years>origin)&(years<=origin+4))
    indices={s:np.flatnonzero(m) for s,m in masks.items()}
    if any(len(ix)==0 for ix in indices.values()): raise ValueError('Empty rolling split')
    return indices


def cache_root(crop,origin,window):
    return CACHE/crop/f'origin_{origin}__w{window}__v3'


def prepare(crop,origin,window):
    root=cache_root(crop,origin,window);root.mkdir(parents=True,exist_ok=True)
    with (root/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=root/'manifest.json'
        if marker.exists(): return root
        raw=RawInputs(crop);indices=split_rows(raw.years,origin,window)
        fitted=raw.normalization(indices['train']);arrays={};ndvi_stats=None
        for s in SPLITS:
            a=raw.arrays(indices[s],fitted)
            current,_=extract(a);previous,_=extract(a,True)
            mask=np.isfinite(current)&(a['relative_valid']>0)
            old_mask=np.isfinite(previous)&(a['relative_valid']>0)
            if s=='train':
                ndvi_stats=dict(mean=float(current[mask].mean(dtype=np.float64)),std=float(current[mask].std(dtype=np.float64)))
                if ndvi_stats['std']<1e-6:raise ValueError('Degenerate NDVI training statistics')
            a['observed_lai']=a['target_lai'];a['observed_lai_valid']=a['target_lai_valid']
            a['observed_ndvi']=np.where(mask,(current-ndvi_stats['mean'])/ndvi_stats['std'],0).astype(np.float32)
            a['observed_ndvi_valid']=mask.astype(np.float32)
            a['previous_ndvi']=np.where(old_mask,(previous-ndvi_stats['mean'])/ndvi_stats['std'],0).astype(np.float32)
            a['previous_ndvi_valid']=old_mask.astype(np.float32)
            keys=(*INPUTS,'previous_lai','previous_lai_valid','previous_ndvi','previous_ndvi_valid',
                  'target','target_residual','baseline','source_indices','row','col','year')
            arrays[s]={k:a[k] for k in keys}
            temp=root/f'{s}.tmp.npz';np.savez(temp,**arrays[s]);temp.replace(root/f'{s}.npz')
        meta=dict(crop=crop,origin=origin,window=window,val_span=3,normalization=asdict(fitted[0]),ndvi_normalization=ndvi_stats,
            protocol='Observed target-season remote sensing, rolling annual history, not prospective remote-state forecasting',
            sample_universe='Unchanged LAI-filtered world-cache rows; no crop-coverage threshold filtering',
            alignment='Existing ascending valid natural-month packing; no claim of repaired cross-year crop seasons',
            source_indices_hash=index_hash(raw.source),data_code_sha256=sha256(Path(__file__)),
            ndvi_manifest_sha256=sha256(NDVI_ROOT/'manifest.json'),
            splits={s:dict(n=len(a['target']),years=np.unique(a['year']).tolist(),
                index_sha256=index_hash(a['source_indices']),file_sha256=sha256(root/f'{s}.npz')) for s,a in arrays.items()})
        atomic_json(marker,meta)
        print(f'[CACHE] {crop} origin={origin} window={window}: '+str({s:m['n'] for s,m in meta['splits'].items()}),flush=True)
    return root


def load(crop,origin,window):
    root=cache_root(crop,origin,window)
    meta=json.loads((root/'manifest.json').read_text())
    arrays={}
    for s in SPLITS:
        with np.load(root/f'{s}.npz',allow_pickle=False) as f:arrays[s]={k:f[k] for k in f.files}
    return arrays,meta


def regional_context(arrays,anomalies):
    """Aggregate same-year observed RS, never regional yield labels."""
    sizes=[len(arrays[s]['target']) for s in SPLITS]
    boundaries=np.cumsum([0,*sizes])
    data={k:np.concatenate([arrays[s][k] for s in SPLITS]) for k in
          ('year','row','col','source_month','crop_coverage','relative_valid','observed_lai_valid')}
    value=np.concatenate([anomalies[s] for s in SPLITS])
    mask=(data['relative_valid']>0)&(data['observed_lai_valid']>0)
    outputs={s:[] for s in SPLITS};metadata={s:[] for s in SPLITS}
    for degrees in (10,20):
        width=360//degrees;nregions=(180//degrees)*width
        region=(data['row'].astype(np.int64)//(degrees*2))*width+data['col']//(degrees*2)
        key=((data['year'].astype(np.int64)-data['year'].min())*nregions+region)[:,None]*12+np.minimum(data['source_month'],11)
        weight=np.broadcast_to(np.clip(data['crop_coverage'],0,1)[:,None],mask.shape)
        length=int(key.max())+1
        sums=np.bincount(key[mask],weights=(value*weight)[mask],minlength=length)
        denom=np.bincount(key[mask],weights=weight[mask],minlength=length)
        counts=np.bincount(key[mask],minlength=length)
        unweighted=np.bincount(key[mask],weights=value[mask],minlength=length)
        fallback=np.divide(unweighted,counts,out=np.zeros_like(unweighted),where=counts>0)
        mean=np.divide(sums,denom,out=fallback,where=denom>0)
        slots=np.where(mask,mean[key],0).astype(np.float32)
        # These controls expose footprint/coverage, not vegetation values.
        footprint=np.stack((np.log1p(counts[key]),np.log1p(denom[key])),axis=-1)
        footprint=np.where(mask[...,None],footprint,0).reshape(len(value),-1).astype(np.float32)
        for i,s in enumerate(SPLITS):
            sl=slice(boundaries[i],boundaries[i+1])
            outputs[s].append(trajectory_features(slots[sl],mask[sl]))
            metadata[s].append(footprint[sl])
    return ({s:np.concatenate(v,axis=1) for s,v in outputs.items()},
            {s:np.concatenate(v,axis=1) for s,v in metadata.items()})


class Features:
    def __init__(self,arrays):
        self.arrays=arrays;self.anomalies={};self.regional=None

    def anomaly(self,p):
        if p not in self.anomalies:self.anomalies[p]=seasonal_anomalies(self.arrays,p)
        return self.anomalies[p]

    def build(self,condition):
        if condition not in CONDITIONS:raise ValueError(condition)
        if condition in ('lai_regional','metadata_regional') and self.regional is None:
            self.regional=regional_context(self.arrays,self.anomaly('lai'))
        result={};names=None
        for s,a in self.arrays.items():
            variant=('history' if condition=='history' else 'metadata' if condition.startswith('metadata') else
                     'observed_both' if condition=='both_anomaly' else 'observed_lai')
            x,_,current_names=make_features({k:a[k] for k in INPUTS},variant)
            chunks=[x]
            if condition not in ('history','metadata','lai_raw','metadata_regional'):
                products=('lai','ndvi') if condition=='both_anomaly' else ('lai',)
                for p in products:
                    valid=(a['relative_valid']>0)&(a[f'observed_{p}_valid']>0)
                    anomaly=self.anomaly(p)[s]
                    chunks.append(trajectory_features(anomaly,valid))
                    current_names.extend(f'{p}_local_anomaly_{i}' for i in range(18))
                    if condition=='lai_weighted':
                        for label,weight in (('area',np.clip(a['crop_coverage'],0,1)),('sqrt_area',np.sqrt(np.clip(a['crop_coverage'],0,1)))):
                            chunks.append(trajectory_features(anomaly*weight[:,None],valid))
                            current_names.extend(f'{label}_lai_anomaly_{i}' for i in range(18))
                    if condition=='lai_contrast':
                        paired=valid&(a['previous_lai_valid']>0)
                        delta=np.where(paired,a['observed_lai']-a['previous_lai'],0)
                        chunks.extend((trajectory_features(delta,paired),paired.astype(np.float32)))
                        current_names.extend(f'lai_year_change_{i}' for i in range(18))
                        current_names.extend(f'lai_paired_valid_{i}' for i in range(12))
            if condition in ('metadata_regional','lai_regional'):
                chunks.append(self.regional[1][s]);current_names.extend(f'regional_footprint_{i}' for i in range(48))
                if condition=='lai_regional':
                    chunks.append(self.regional[0][s]);current_names.extend(f'regional_lai_anomaly_{i}' for i in range(36))
            result[s]=np.concatenate(chunks,axis=1).astype(np.float32)
            if names is not None and names!=current_names:raise ValueError('Feature order differs across splits')
            names=current_names
            if result[s].shape[1]!=len(names) or not np.isfinite(result[s]).all():raise ValueError('Invalid design matrix')
        return result,names


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--crop',choices=CROPS,required=True)
    p.add_argument('--origin',type=int,choices=ORIGINS,required=True);p.add_argument('--window',type=int,choices=WINDOWS,required=True)
    a=p.parse_args();prepare(a.crop,a.origin,a.window)
