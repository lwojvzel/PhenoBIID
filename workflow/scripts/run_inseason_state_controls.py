"""Six completion rules passed through identical retained crop yield heads."""
import argparse
import fcntl
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from analyze_inseason_state_transfer import persistence, diagnostic
from forecast_bridge_state import ForecastState, batch_arrays
from inseason_13year_data import load as load_current, partition
from inseason_nested_common import LABELS, hashes, register, finish, verify
from inseason_signal_matching import load_group, feature_matrix
from inseason_state_controls import ROOT, OUT, RECIPES, BLOCKS, METHODS, checked_state, gru_prefix_rollout
from ndvi_tail_replacement import tail_mask, prefix_values, mix_trajectory
from review_revision_data import sha256
from run_forecast_bridge_state import tensors
from run_inseason_direct_baselines import score
from run_ndvi_tail_replacement import yield_prediction
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json

CODE = ('run_inseason_state_controls.py', 'inseason_state_controls.py',
    'forecast_bridge_state.py', 'forecast_bridge_data.py', 'inseason_signal_matching.py',
    'run_ndvi_tail_replacement.py', 'analyze_inseason_state_transfer.py')
RATIOS = (.1,.3,.5)


def root_for(crop, cutoff, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_42'


@torch.no_grad()
def forecast(model, raw, take, norm, product, tail=None):
    outputs = []
    if tail is not None:
        values, known = prefix_values(raw[f'observed_{product}'][take],
            raw['relative_valid'][take] > 0, tail, norm[product]['mean'], norm[product]['std'])
    for start in range(0,len(take),256):
        end = min(start+256,len(take))
        batch = tensors(batch_arrays(raw,take[start:end],norm,product),'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if tail is None:
                output = model(batch)
            else:
                output = gru_prefix_rollout(model,batch,torch.as_tensor(values[start:end],device='cuda'),
                    torch.as_tensor(known[start:end],device='cuda'))
        outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs)*norm[product]['std']+norm[product]['mean']


def run(crop, cutoff, smoke=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    destination = root_for(crop, cutoff, smoke)
    destination.mkdir(parents=True,exist_ok=True)
    code = hashes(CODE)
    with (destination / 'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / 'complete.json').exists():
            verify(destination,code)
            return
        started = time.monotonic()
        raw, groups, scales, _, sources = load_group(crop,cutoff+3)
        original = ROOT / f'benchmark/results/inseason_signal_match_v1/pipelines/{crop}/origin_{cutoff+3}/seed_42'
        check_files(original,json.loads((original / 'complete.json').read_text())['files'])
        sources[str(original / 'complete.json')] = sha256(original / 'complete.json')
        expected, _ = load_current(crop)
        indices = partition(expected,cutoff)['evaluation']
        all_labels = {k:np.concatenate([g['labels'][k] for g in groups.values()]) for k in LABELS}
        for key in LABELS:
            np.testing.assert_array_equal(all_labels[key],expected[key][indices])
        register(destination,dict(crop=crop,cutoff=cutoff,seed=42,smoke=smoke,code_sha256=code,
            recipe=RECIPES[crop],methods=METHODS,ratios=RATIOS,
            heads_refitted=False,feedback_intervention='Fixed BIID weights; remove only observed-state feedback',
            gru_prefix='Same scalar-feedback rule with matched observable prefix',
            new_seasonal_information=False, original_selection_provenance_retained=True))
        models, norms, weights = {}, {}, []
        for product in RECIPES[crop].split('_'):
            for architecture in ('biid','gru'):
                root, norm, _ = checked_state(crop,product,architecture,cutoff)
                model = ForecastState(architecture).cuda().eval()
                model.load_state_dict(torch.load(root / 'model.pt',map_location='cpu',weights_only=True))
                models[product,architecture], norms[product,architecture] = model, norm
                for name in ('complete.json','model.pt','normalization.json'):
                    sources[str(root / name)] = sha256(root / name)
                weights.append(dict(product=product,architecture=architecture,path=str(root / 'model.pt'),
                    sha256=sha256(root / 'model.pt')))
        outputs = {(ratio,method):[] for ratio in RATIOS for method in METHODS}
        labels = {k:[] for k in LABELS}
        state_rows = []
        for split, group in groups.items():
            take = group['take']
            local = np.arange(len(take))
            if smoke:
                local = np.concatenate([np.flatnonzero(raw['year'][take] == year)[:12]
                    for year in np.unique(raw['year'][take])])
            take = take[local]
            label = {key:group['labels'][key][local] for key in LABELS}
            for key in LABELS:
                labels[key].append(label[key])
            active = raw['relative_valid'][take] > 0
            full_active = raw['relative_valid'][group['take']] > 0
            observed = {p:raw[f'observed_{p}'][take] for p in RECIPES[crop].split('_')}
            climate, no_feedback = {}, {}
            for product in observed:
                full_values = np.full_like(raw[f'observed_{product}'][group['take']],scales[product]['mean'])
                encoded = group['encoders'][product].encode(full_values,full_active,True)
                climate[product] = (-encoded[:,-18:-6]*scales[product]['std']+scales[product]['mean'])[local]
                no_feedback[product] = forecast(models[product,'biid'],raw,take,norms[product,'biid'],product)
            for ratio in RATIOS:
                percent = round(100*ratio)
                tail = tail_mask(active,ratio)
                with np.load(original / f'{split}_trajectories_{percent:03d}.npz') as saved:
                    biid = {p:saved[p][local] for p in observed}
                completed = {method:{} for method in METHODS}
                for product, truth in observed.items():
                    previous = raw[f'previous_{product}'][take]
                    candidates = dict(biid=biid[product], no_feedback=no_feedback[product],
                        climatology=climate[product], previous=np.where(np.isfinite(previous),previous,climate[product]),
                        persistence=persistence(truth,previous,active,tail,climate[product]),
                        gru=forecast(models[product,'gru'],raw,take,norms[product,'gru'],product,tail))
                    for method, value in candidates.items():
                        completed[method][product] = mix_trajectory(truth,value,tail)
                        for year in np.unique(label['year']):
                            mask = tail & np.isfinite(truth) & (label['year'] == year)[:,None]
                            state_rows.append(dict(crop=crop,cutoff=cutoff,product=product,percent=percent,
                                method=method,year=int(year),**diagnostic(truth[mask],value[mask],climate[product][mask])))
                for method, values in completed.items():
                    # Existing encoders refer to full-group identities; retain them for smoke replay.
                    full_tail = tail_mask(full_active,ratio)
                    encodings = {}
                    for product, mixed in values.items():
                        if smoke:
                            full = np.array(raw[f'observed_{product}'][group['take']],copy=True)
                            full[local] = mixed
                            encodings[product] = group['encoders'][product].encode(full,full_tail,True)[local]
                        else:
                            encodings[product] = group['encoders'][product].encode(mixed,tail,True)
                    features = feature_matrix(group['common'][local],encodings,tail,group['support'][local],RECIPES[crop])
                    heads = [(model,cfg,base[local]) for model,cfg,base in group['heads'][RECIPES[crop]]]
                    prediction = yield_prediction(heads,features,crop)
                    if method in ('biid','climatology'):
                        with np.load(original / f'{split}_{RECIPES[crop]}_{method}_{percent:03d}.npz') as saved:
                            np.testing.assert_array_equal(prediction,saved['prediction'][local])
                    outputs[ratio,method].append(prediction)
                print(f'[STATE CONTROLS] {crop} {cutoff} {split} suffix={ratio}',flush=True)
        labels = {k:np.concatenate(v) for k,v in labels.items()}
        measured = []
        for (ratio,method), pieces in outputs.items():
            prediction = np.concatenate(pieces)
            np.savez_compressed(destination / f'tail_{round(100*ratio):02d}_{method}.npz',prediction=prediction,**labels)
            measured.append(dict(percent=round(100*ratio),method=method,
                **score(labels['target'],prediction,labels['year'])))
        atomic_json(destination / 'metrics.json',measured)
        pd.DataFrame(state_rows).to_csv(destination / 'state_annual.csv',index=False)
        atomic_json(destination / 'provenance.json',dict(sources=sources,state_weights=weights,
            biid_and_climatology_original_predictions_exact=True,new_yield_fits=0))
        finish(destination,code,smoke=smoke,seconds=time.monotonic()-started,
            six_completion_rules=True,all_temporal_identities_verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop',choices=tuple(RECIPES),required=True)
    parser.add_argument('--cutoff',type=int,choices=tuple(BLOCKS),required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=2):
        run(args.crop,args.cutoff,args.smoke)
