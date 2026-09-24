"""Use no-weather state refits with the unchanged original yield readouts."""
import argparse
import fcntl
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from analyze_inseason_state_transfer import diagnostic
from forecast_bridge_state import batch_arrays
from inseason_13year_data import load as load_current, partition
from inseason_nested_common import LABELS, hashes, register, finish, verify
from inseason_no_weather_state import NoWeatherState, no_weather_prefix
from inseason_signal_matching import load_group, feature_matrix
from ndvi_tail_replacement import tail_mask, prefix_values, mix_trajectory
from review_revision_data import sha256
from run_forecast_bridge_state import tensors
from run_inseason_direct_baselines import score
from run_inseason_no_weather_state import ROOT, OUT, RECIPES, BLOCKS, verify_state
from run_inseason_state_controls import root_for as controls_root
from run_ndvi_tail_replacement import yield_prediction
from run_review_revision_parallel import atomic_json

RATIOS = (.1,.3,.5)
CODE = ('run_inseason_no_weather_eval.py','inseason_no_weather_state.py',
    'run_inseason_no_weather_state.py','inseason_signal_matching.py',
    'run_ndvi_tail_replacement.py','analyze_inseason_state_transfer.py','forecast_bridge_state.py')


def eval_root(crop, cutoff, smoke=False):
    return OUT / ('smoke_evaluation' if smoke else 'evaluation') / crop / f'cutoff_{cutoff}/seed_42'


@torch.no_grad()
def predict_prefix(model, raw, take, norm, product, tail):
    values, known = prefix_values(raw[f'observed_{product}'][take],
        raw['relative_valid'][take]>0, tail, norm[product]['mean'], norm[product]['std'])
    pieces = []
    for start in range(0,len(take),256):
        end = min(start+256,len(take))
        batch = tensors(batch_arrays(raw,take[start:end],norm,product),'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction = no_weather_prefix(model,batch,torch.as_tensor(values[start:end],device='cuda'),
                torch.as_tensor(known[start:end],device='cuda'))
        pieces.append(prediction.float().cpu().numpy())
    return np.concatenate(pieces)*norm[product]['std']+norm[product]['mean']


def run(crop, cutoff, smoke=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = eval_root(crop,cutoff,smoke)
    root.mkdir(parents=True,exist_ok=True)
    code = hashes(CODE)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root,code)
            return
        started = time.monotonic()
        raw,groups,scales,_,sources = load_group(crop,cutoff+3)
        expected,_ = load_current(crop)
        ix = partition(expected,cutoff)['evaluation']
        for key in LABELS:
            np.testing.assert_array_equal(np.concatenate([g['labels'][key] for g in groups.values()]),expected[key][ix])
        reference = controls_root(crop,cutoff)
        verify(reference)
        sources[str(reference / 'complete.json')] = sha256(reference / 'complete.json')
        register(root,dict(crop=crop,cutoff=cutoff,smoke=smoke,seed=42,code_sha256=code,
            recipe=RECIPES[crop],ratios=RATIOS,terminal_weather_unchanged=True,
            historical_and_terminal_weights_frozen=True,new_yield_fits=0))
        models,norms,weights = {},{},[]
        for product in RECIPES[crop].split('_'):
            state,norm = verify_state(crop,product,cutoff,smoke)
            model = NoWeatherState().cuda().eval()
            model.load_state_dict(torch.load(state / 'model.pt',map_location='cpu',weights_only=True))
            models[product],norms[product] = model,norm
            for name in ('model.pt','complete.json','config.json','normalization.json'):
                sources[str(state / name)] = sha256(state / name)
            weights.append(dict(product=product,path=str(state / 'model.pt'),sha256=sha256(state / 'model.pt')))
        labels = {key:[] for key in LABELS}
        outputs = {round(100*r):[] for r in RATIOS}
        state_rows = []
        for split,group in groups.items():
            all_take = group['take']
            local = np.arange(len(all_take))
            if smoke:
                local = np.concatenate([np.flatnonzero(raw['year'][all_take]==year)[:12]
                    for year in np.unique(raw['year'][all_take])])
            take = all_take[local]
            label = {key:group['labels'][key][local] for key in LABELS}
            for key in LABELS:
                labels[key].append(label[key])
            active,full_active = raw['relative_valid'][take]>0,raw['relative_valid'][all_take]>0
            climate = {}
            for product in models:
                constant = np.full_like(raw[f'observed_{product}'][all_take],scales[product]['mean'])
                encoded = group['encoders'][product].encode(constant,full_active,True)
                climate[product] = (-encoded[:,-18:-6]*scales[product]['std']+scales[product]['mean'])[local]
            for ratio in RATIOS:
                percent = round(100*ratio)
                tail,full_tail = tail_mask(active,ratio),tail_mask(full_active,ratio)
                encodings,trajectories = {},{}
                for product,model in models.items():
                    prediction = predict_prefix(model,raw,take,norms[product],product,tail)
                    truth = raw[f'observed_{product}'][take]
                    trajectories[product] = mix_trajectory(truth,prediction,tail)
                    for year in np.unique(label['year']):
                        mask = tail & np.isfinite(truth) & (label['year']==year)[:,None]
                        state_rows.append(dict(crop=crop,cutoff=cutoff,product=product,percent=percent,
                            method='no_weather',year=int(year),
                            **diagnostic(truth[mask],prediction[mask],climate[product][mask])))
                    if smoke:
                        full = np.array(raw[f'observed_{product}'][all_take],copy=True)
                        full[local] = trajectories[product]
                        encodings[product] = group['encoders'][product].encode(full,full_tail,True)[local]
                    else:
                        encodings[product] = group['encoders'][product].encode(trajectories[product],tail,True)
                features = feature_matrix(group['common'][local],encodings,tail,group['support'][local],RECIPES[crop])
                heads = [(model,config,base[local]) for model,config,base in group['heads'][RECIPES[crop]]]
                outputs[percent].append(yield_prediction(heads,features,crop))
                np.savez_compressed(root / f'{split}_trajectories_{percent:02d}.npz',**trajectories,**label)
                print(f'[NO WEATHER EVAL] {crop} {cutoff} {split} {percent}',flush=True)
        labels = {key:np.concatenate(value) for key,value in labels.items()}
        metrics = []
        for percent,pieces in outputs.items():
            prediction = np.concatenate(pieces)
            if not smoke:
                with np.load(reference / f'tail_{percent:02d}_biid.npz') as old:
                    for key in LABELS:
                        np.testing.assert_array_equal(old[key],labels[key])
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite no-weather terminal prediction')
            np.savez_compressed(root / f'tail_{percent:02d}.npz',prediction=prediction,**labels)
            metrics.append(dict(percent=percent,**score(labels['target'],prediction,labels['year'])))
        pd.DataFrame(state_rows).to_csv(root / 'state_annual.csv',index=False)
        atomic_json(root / 'metrics.json',metrics)
        atomic_json(root / 'provenance.json',dict(sources=sources,state_weights=weights,new_yield_fits=0))
        finish(root,code,smoke=smoke,seconds=time.monotonic()-started,terminal_weather_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop',choices=tuple(RECIPES),required=True)
    parser.add_argument('--cutoff',type=int,choices=tuple(BLOCKS),required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=2):
        run(args.crop,args.cutoff,args.smoke)
