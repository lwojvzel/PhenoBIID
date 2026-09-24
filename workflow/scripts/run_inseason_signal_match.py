"""Evaluate frozen NDVI/GPP crop heads over observable-prefix lead positions."""
import argparse
import fcntl
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from inseason_signal_matching import ROOT, RECIPES, load_group, remap_product, feature_matrix
from forecast_bridge_data import CROPS,ORIGINS
from forecast_bridge_state import ForecastState
from run_forecast_bridge_state import run_root as state_root
from run_ndvi_tail_replacement import forecast_prefix,yield_prediction
from inseason_ndvi_reuse import RATIOS,tail_mask,mix_trajectory
from inseason_calendar_audit import corrected_lead_times
from run_ndvi_signal_permutation import check_files
from crop_signal_history_reference import IDENTITY
from run_crop_signal_screen import yearly_rmse
from run_review_revision_parallel import atomic_json
from review_revision_data import sha256
from summarize_forecast_bridge import verify

RESULT = ROOT/'benchmark/results/inseason_signal_match_v1'
PLAN = ROOT/'Paper/task/季中遥感信号匹配_执行方案_20260908.md'
CODE = ('inseason_signal_matching.py','run_inseason_signal_match.py')


def run_root(crop,origin):
    return RESULT/'pipelines'/crop/f'origin_{origin}/seed_42'


def run(crop,origin):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    dest=run_root(crop,origin); dest.mkdir(parents=True,exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (dest/'complete.json').exists():
            verify(dest); return
        start=time.monotonic()
        raw,groups,scales,epochs,sources=load_group(crop,origin)
        state=state_root(crop,'gpp','biid',origin-3,42)
        marker=json.loads((state/'complete.json').read_text())
        check_files(state,marker['files'])
        if marker['smoke'] or marker['full_fit_cutoff']!=origin-3:
            raise ValueError('Incorrect GPP state training window')
        state_cfg=json.loads((state/'config.json').read_text())
        for name,digest in state_cfg['code_sha256'].items():
            if sha256(ROOT/'scripts'/name)!=digest:
                raise ValueError('GPP implementation changed')
        stats=json.loads((state/'normalization.json').read_text())
        stats=dict(stats,ndvi=stats['gpp'])
        model=ForecastState('biid').cuda().eval()
        model.load_state_dict(torch.load(state/'model.pt',map_location='cpu',weights_only=True))
        for file in (state/'complete.json',state/'model.pt',state/'normalization.json',PLAN):
            sources[str(file)]=sha256(file)
        files,rows=[],[]
        for split,g in groups.items():
            take=g['take']; labels=g['labels']; active=raw['relative_valid'][take]>0
            observed={p:raw[f'observed_{p}'][take] for p in ('ndvi','gpp')}
            zero=np.zeros_like(active)
            original={recipe:yield_prediction(g['heads'][recipe],feature_matrix(g['common'],g['observed_encodings'],zero,g['support'],recipe),crop) for recipe in RECIPES}
            if split=='validation':
                for recipe,file in g['observed_sources'].items():
                    with np.load(file) as f:
                        for k in IDENTITY:
                            np.testing.assert_array_equal(labels[k],f[k])
                        np.testing.assert_array_equal(original[recipe],f['prediction'])
                    sources[str(file)]=sha256(file)
            label_file=f'{split}_labels.npz'
            np.savez_compressed(dest/label_file,**labels,active=active,source_month=raw['source_month'][take],
                                **{f'observed_{r}':p for r,p in original.items()})
            files.append(label_file)
            strong=yearly_rmse(labels['target'],labels['strong'],labels['year'])
            obs_scores={r:yearly_rmse(labels['target'],p,labels['year']) for r,p in original.items()}
            gpp_raw=remap_product(raw,'gpp')
            climo={p:-g['encoders'][p].encode(np.full_like(observed[p],scales[p]['mean']),active,True)[:,-18:-6]*scales[p]['std']+scales[p]['mean'] for p in scales}
            prefix_cache={}
            for ratio in RATIOS:
                percent=round(100*ratio); tail=tail_mask(active,ratio)
                dates=corrected_lead_times(labels['year'],raw['source_month'][take],active,tail)
                tf=f'{split}_timeline_{percent:03d}.npz'
                np.savez_compressed(dest/tf,tail=tail,**dates); files.append(tf)
                ref=g['reference']
                ndvi_name=f'available_prefix_prefix_forecast_{percent:03d}.npz'
                with np.load(ref/ndvi_name) as f:
                    ndvi_reference=f['prediction']
                    ndvi_mixed=f['mixed_ndvi'] if 'mixed_ndvi' in f.files else None
                if ndvi_mixed is None:
                    with np.load(ref/f'trajectory_prefix_forecast_{percent:03d}.npz') as f:
                        ndvi_mixed=f['mixed_ndvi']
                nscore=yearly_rmse(labels['target'],ndvi_reference,labels['year'])
                key=tuple(np.unique(np.column_stack((active.sum(1),tail.sum(1))),axis=0).ravel())
                if ratio==0:
                    gpp_forecast=climo['gpp']
                elif key in prefix_cache:
                    gpp_forecast=prefix_cache[key]
                else:
                    gpp_forecast=forecast_prefix(model,gpp_raw,take,stats,observed['gpp'],tail,active)
                    prefix_cache[key]=gpp_forecast
                mixed=dict(ndvi=ndvi_mixed,gpp=mix_trajectory(observed['gpp'],gpp_forecast,tail))
                climatological={p:mix_trajectory(observed[p],climo[p],tail) for p in scales}
                for mode,values in (('biid',mixed),('climatology',climatological),
                                    ('ndvi_biid_gpp_climatology',dict(ndvi=ndvi_mixed,gpp=climatological['gpp']))):
                    enc={p:g['encoders'][p].encode(v,tail,True) for p,v in values.items()}
                    recipes=('ndvi_gpp',) if mode=='ndvi_biid_gpp_climatology' else RECIPES
                    for recipe in recipes:
                        x=feature_matrix(g['common'],enc,tail,g['support'],recipe)
                        pred=yield_prediction(g['heads'][recipe],x,crop)
                        if ratio==0:
                            np.testing.assert_array_equal(pred,original[recipe])
                        if recipe=='ndvi' and mode=='biid':
                            np.testing.assert_array_equal(pred,ndvi_reference)
                        if not np.isfinite(pred).all():
                            raise ValueError('Nonfinite yield prediction')
                        name=f'{split}_{recipe}_{mode}_{percent:03d}.npz'
                        np.savez_compressed(dest/name,prediction=pred);files.append(name)
                        scores=yearly_rmse(labels['target'],pred,labels['year'])
                        for year,rmse in scores.items():
                            ix=labels['year']==int(year)
                            r=dict(crop=crop,origin=origin,split=split,year=int(year),seed=42,recipe=recipe,mode=mode,
                                requested_fraction=ratio,rmse=rmse,strong_rmse=strong[year],ndvi_rmse=nscore[year],
                                observed_rmse=obs_scores[recipe][year],gain_strong=100*(1-rmse/strong[year]),
                                gain_ndvi=100*(1-rmse/nscore[year]),gain_observed=100*(1-rmse/obs_scores[recipe][year]),
                                samples=int(ix.sum()),actual_fraction=float((tail[ix].sum(1)/active[ix].sum(1)).mean()),
                                median_lead_days=float(np.median(dates['lead_days'][ix])))
                            for p in scales:
                                valid=tail[ix]&np.isfinite(observed[p][ix])
                                error=(values[p][ix]-observed[p][ix])[valid]**2
                                r[f'{p}_suffix_rmse']=float(np.sqrt(error.mean())) if len(error) else np.nan
                            rows.append(r)
                name=f'{split}_trajectories_{percent:03d}.npz'
                np.savez_compressed(dest/name,**mixed);files.append(name)
                print(f'[SIGNAL] {crop} {origin} {split} ratio={ratio:.1f}',flush=True)
        atomic_json(dest/'config.json',dict(crop=crop,origin=origin,seed=42,recipes=RECIPES,ratios=RATIOS,
            sources=sources,code_sha256={n:sha256(ROOT/'scripts'/n) for n in CODE},original_expert_epochs=epochs,
            information='Full weather condition; true prefix only; suffix support train-imputed',
            state_branches='Separate NDVI and GPP retain12 BIID; feature concatenation only',
            heads_retrained=False,independent_test=False,all_ndvi_predictions_replayed=True))
        pd.DataFrame(rows).to_csv(dest/'annual.csv',index=False)
        files += ['config.json','annual.csv']
        atomic_json(dest/'complete.json',dict(files={n:sha256(dest/n) for n in files},
            seconds=time.monotonic()-start,logical_evaluations=len(groups)*77,head_fits=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--crop',choices=CROPS,required=True)
    parser.add_argument('--origin',choices=ORIGINS,type=int,required=True)
    args=parser.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=4):
        run(args.crop,args.origin)
