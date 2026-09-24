from __future__ import annotations
from dataclasses import asdict
import fcntl, json
from pathlib import Path
import numpy as np
from dual_remote_data import extract
from forward_protocol_revision import RawInputs, index_hash
from observed_remote_benchmark import INPUTS
from prepare_pku_ndvi import OUT as NDVI_ROOT
from review_revision_data import ROOT, CROPS, SPLITS, sha256
from run_review_revision_parallel import atomic_json
CACHE=ROOT/'benchmark/cache/stable_remote_v1'

ORIGINS=(2004,2008,2012)

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
