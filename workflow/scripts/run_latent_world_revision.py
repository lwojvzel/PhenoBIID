"""Observable-plus-latent world-model candidates with a LAI validation guard."""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from biid_world_model import CropWorldDynamics
from multimodal_baseline import regression_metrics, save_json, set_seed, write_csv
from review_revision_data import CROPS, RESULT_ROOT, load_shared
from run_biid_yield_fusion import load_state
from run_review_revision import direct_features, make_batches, split_batch, rmse


class LatentWorldReadout(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        self.dynamics = CropWorldDynamics("biid_climate")
        self.dynamics.load_state_dict(load_state(checkpoint, device), strict=False)
        self.projection = nn.Linear(130, 128)
        self.temporal = nn.GRU(128, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(151, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, b):
        lai, latent = self.dynamics(b['weather'], b['previous_lai'], b['previous_lai_valid'], b['relative_valid'], b['history'], b['context'], return_trajectory=True)
        tokens = self.projection(torch.cat((latent, lai[...,None], (lai-b['previous_lai'])[...,None]), dim=-1))
        valid = b['relative_valid']
        tokens, _ = self.temporal(tokens * valid[...,None])
        pool = (tokens*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        _, static, _ = direct_features(b)
        return b['history_base'] + self.head(torch.cat((pool, static), dim=-1)).squeeze(-1), lai


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval()
    predictions, states = [], []
    squared, count = 0., 0.
    for raw in loader:
        b, labels = split_batch(raw, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction, lai = model(b)
        physical = labels['baseline'] + prediction.float()*stats['residual_std'] + stats['residual_mean']
        predictions.append(physical.cpu().numpy())
        states.append(lai.float().cpu().numpy())
        mask = labels['target_lai_valid']
        squared += float((((lai.float()-labels['target_lai'])**2)*mask).sum())
        count += float(mask.sum())
    return np.concatenate(predictions), np.concatenate(states), (squared/max(count,1))**.5 * stats['lai_std']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=CROPS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--mode', choices=('frozen','joint_01','joint_1','joint_10'), required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    root = RESULT_ROOT/'latent_smoke' if args.smoke else RESULT_ROOT/'latent_candidates'
    destination = root/args.crop/args.mode/f'seed_{args.seed}'
    if (destination/'test_metrics.json').exists():
        return
    arrays, meta = load_shared(args.crop,args.seed)
    for s,a in arrays.items():
        a['state'] = a['predicted_lai']
        if args.smoke:
            arrays[s] = {k:v[:64].copy() for k,v in a.items()}
    device = torch.device('cuda')
    model = LatentWorldReadout(meta['dynamics_checkpoint'], device).to(device)
    loaders = {s:make_batches(a,1024,s=='train') for s,a in arrays.items()}
    stats = meta['normalization']
    destination.mkdir(parents=True,exist_ok=True)
    save_json({**vars(args), 'input_manifest':meta, 'terminal_raw_climate':False, 'readout':'pooled latent trajectory + predicted LAI + LAI change + identical history/q/context', 'state_precision_guard':'validation LAI RMSE <= 1.10 * pretrained validation LAI RMSE', 'learning_rate_readout':3e-4,'learning_rate_dynamics':1e-5,'warmup_frozen_epochs':3,'batch_size':1024,'selection':'validation yield RMSE subject to LAI guard; no test selection'}, destination/'config.json')
    dynamics_parameters = list(model.dynamics.parameters())
    readout_parameters = [p for n,p in model.named_parameters() if not n.startswith('dynamics.')]
    optimizer = torch.optim.AdamW([{'params':dynamics_parameters,'lr':1e-5},{'params':readout_parameters,'lr':3e-4}],weight_decay=1e-4,fused=True)
    scaler = torch.amp.GradScaler('cuda')
    prediction, _, reference_lai = evaluate(model,loaders['validation'],stats,device)
    best = rmse(arrays['validation']['target'],prediction)
    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    best_epoch, stale = 0,0
    rows = [dict(epoch=0,validation_rmse=best,validation_lai_rmse=reference_lai,eligible=True,loss=None,seconds=0.)]
    weight = {'frozen':0.,'joint_01':.1,'joint_1':1.,'joint_10':10.}[args.mode]
    start = time.monotonic()
    for epoch in range(1,args.epochs+1):
        tick = time.monotonic()
        joint = args.mode != 'frozen' and epoch > 3
        for parameter in dynamics_parameters:
            parameter.requires_grad_(joint)
        model.train()
        if not joint:
            model.dynamics.eval()
        loss_sum,count = 0.,0
        for raw in loaders['train']:
            b,labels = split_batch(raw,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output,lai = model(b)
                loss = nn.functional.mse_loss(output,labels['target_residual'])
                mask = labels['target_lai_valid']
                lai_loss = (((lai-labels['target_lai'])**2)*mask).sum()/mask.sum().clamp_min(1)
                if joint:
                    loss = loss + weight * lai_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint objective')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())*len(output)
            count += len(output)
        prediction,_,lai_error = evaluate(model,loaders['validation'],stats,device)
        error = rmse(arrays['validation']['target'],prediction)
        eligible = lai_error <= reference_lai*1.10
        rows.append(dict(epoch=epoch,validation_rmse=error,validation_lai_rmse=lai_error,eligible=eligible,loss=loss_sum/count,seconds=time.monotonic()-tick))
        print(f'[LATENT] {args.crop}/{args.mode}/{args.seed} epoch={epoch} yield={error:.6f} lai={lai_error:.6f} eligible={eligible} seconds={rows[-1]["seconds"]:.1f}',flush=True)
        if eligible and error < best-1e-6:
            best,best_epoch,stale=error,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience and epoch >= 6:
            break
    model.load_state_dict(best_state)
    torch.save(best_state,destination/'model_best.pt')
    write_csv(rows,destination/'training_history.csv')
    scores = {}
    for s in ('validation','test'):
        prediction,lai,lai_rmse=evaluate(model,loaders[s],stats,device)
        a=arrays[s]
        history=a['baseline']+a['history_base']*stats['residual_std']+stats['residual_mean']
        np.savez_compressed(destination/f'{s}_predictions.npz',prediction=prediction,predicted_lai=lai,history_prediction=history,**{k:a[k] for k in ('target','year','row','col','source_indices')})
        m=regression_metrics(a['target'],prediction)
        h=rmse(a['target'],history)
        scores[s]={**m,'history_rmse':h,'gain_over_history_percent':100*(h-m['rmse'])/h,'lai_rmse':lai_rmse}
    save_json(dict(crop=args.crop,seed=args.seed,mode=args.mode,best_epoch=best_epoch,reference_validation_lai_rmse=reference_lai,metrics=scores,elapsed_seconds=time.monotonic()-start,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20),destination/'test_metrics.json')
    print(json.dumps(scores),flush=True)


if __name__ == '__main__':
    main()
