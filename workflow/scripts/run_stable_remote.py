"""Fit a reproducible rolling candidate on matched history/observed-RS features."""
import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from multimodal_baseline import regression_metrics,set_seed
from stable_remote_data import ROOT,RESULT,CROPS,ORIGINS,WINDOWS,CONDITIONS,BASE_CONDITIONS,Features,load
from stable_remote_models import ENGINES,build,fit
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

CODE=('stable_remote_data.py','stable_remote_models.py','run_stable_remote.py','observed_remote_anomaly.py',
      'observed_remote_benchmark.py','forward_protocol_revision.py','run_history_multimodal_baselines.py')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop',choices=CROPS,required=True);p.add_argument('--origin',type=int,choices=ORIGINS,required=True)
    p.add_argument('--window',type=int,choices=WINDOWS,required=True);p.add_argument('--engine',choices=ENGINES,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--conditions',default=','.join(BASE_CONDITIONS))
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    conditions=args.conditions.split(',')
    if not set(conditions).issubset(CONDITIONS) or len(set(conditions))!=len(conditions):p.error('Invalid conditions')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    root=RESULT/('smoke' if args.smoke else 'pipelines')/args.crop/f'origin_{args.origin}'/f'{args.engine}__w{args.window}'/f'seed_{args.seed}'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        arrays,meta=load(args.crop,args.origin,args.window)
        config={k:v for k,v in vars(args).items() if k!='conditions'}
        config.update(input_manifest=meta,code_hashes={f:sha256(ROOT/'scripts'/f) for f in CODE},
            versions={k:importlib.metadata.version(k) for k in ('scikit-learn','xgboost','lightgbm','catboost')},
            selection='Three-year validation only; no test scores used in fitting or early stopping')
        if (root/'config.json').exists() and json.loads((root/'config.json').read_text())!=config:raise RuntimeError('Run configuration changed')
        atomic_json(root/'config.json',config)
        if args.smoke:arrays={s:{k:v[:256 if s=='train' else 128] for k,v in a.items()} for s,a in arrays.items()}
        features=Features(arrays);norm=meta['normalization'];y={s:a['target_residual'] for s,a in arrays.items()}
        for condition in conditions:
            out=root/condition;out.mkdir(exist_ok=True)
            if (out/'metrics.json').exists():continue
            set_seed(args.seed);started=time.monotonic();x,names=features.build(condition)
            model=build(args.engine,args.seed,args.smoke);fit(model,args.engine,x,y)
            if args.engine=='xgb_smooth':
                model.set_params(device='cpu',callbacks=None);weight=out/'model.ubj';model.save_model(weight)
                selected=int(model.best_iteration)+1
            elif args.engine=='catboost':
                weight=out/'model.cbm';model.save_model(str(weight));selected=int(model.tree_count_)
            else:
                weight=out/'model.joblib';joblib.dump(model,weight)
                selected=int(model.n_iter_) if args.engine.startswith('hgb') else int(model.best_iteration_)
            scores={};per_year={}
            for s in ('validation','test'):
                a=arrays[s];prediction=a['baseline'].astype(float)+model.predict(x[s]).astype(float)*norm['residual_std']+norm['residual_mean']
                if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite output')
                scores[s]=regression_metrics(a['target'],prediction)
                per_year[s]={str(int(year)):regression_metrics(a['target'][a['year']==year],prediction[a['year']==year]) for year in np.unique(a['year'])}
                np.savez_compressed(out/f'{s}_predictions.npz',prediction=prediction,
                    **{k:a[k] for k in ('target','baseline','source_indices','year','row','col')})
            metric=dict(crop=args.crop,origin=args.origin,window=args.window,engine=args.engine,seed=args.seed,
                condition=condition,scores=scores,per_year=per_year,features=names,feature_dim=len(names),
                weight=str(weight),weight_sha256=sha256(weight),selected_trees=selected,seconds=time.monotonic()-started)
            atomic_json(out/'metrics.json',metric)
            print(f"[FIT] {args.crop} {args.origin} w{args.window} {args.engine} {condition} seed={args.seed} val={scores['validation']['rmse']:.6f} test={scores['test']['rmse']:.6f}",flush=True)
            del model,x;gc.collect()
        if all((root/c/'metrics.json').exists() for c in BASE_CONDITIONS):atomic_json(root/'base_complete.json',dict(conditions=list(BASE_CONDITIONS)))
        if all((root/c/'metrics.json').exists() for c in CONDITIONS):atomic_json(root/'all_complete.json',dict(conditions=list(CONDITIONS)))


if __name__=='__main__':main()
