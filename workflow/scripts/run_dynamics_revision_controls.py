"""Same-budget retraining controls for forcing and observable-state feedback."""
import argparse
import time
import numpy as np
import torch
from biid_world_model import CropWorldDynamics, WorldNormalizationStats, lai_metrics
from review_revision_data import CROPS, RESULT_ROOT, load_shared
from multimodal_baseline import save_json, write_csv, set_seed
from run_biid_world_model import make_loader, train_dynamics, predict_dynamics


class SelfModulationStack(torch.nn.Module):
    """Keep all projections and MLPs, but remove cross-branch relevance."""

    def __init__(self, stack):
        super().__init__()
        self.layers = stack.layers

    def forward(self, source, target):
        for layer in self.layers:
            source_bar = source + layer.target_to_source(source, source)
            target_bar = target + layer.source_to_target(target, target)
            source = source_bar + layer.source_mlp(layer.source_norm(source_bar))
            target = target_bar + layer.target_mlp(layer.target_norm(target_bar))
        return source, target


class ControlledDynamics(CropWorldDynamics):
    def __init__(self, control):
        super().__init__('biid_climate')
        self.control=control
        if control == 'self_modulation':
            self.climate_biid = SelfModulationStack(self.climate_biid)

    def forward(self,weather,*args,**kwargs):
        if self.control=='no_weather':weather=torch.zeros_like(weather)
        return super().forward(weather,*args,observation_feedback=self.control!='no_feedback',**kwargs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop',choices=CROPS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--control',choices=('full','no_weather','no_feedback','self_modulation'),required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);set_seed(a.seed)
    root=RESULT_ROOT/('dynamics_controls_smoke' if a.smoke else 'dynamics_controls')/a.crop/a.control/f'seed_{a.seed}'
    if (root/'test_metrics.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    arrays,meta=load_shared(a.crop,a.seed)
    if a.smoke:arrays={s:{k:v[:64].copy() for k,v in x.items()} for s,x in arrays.items()}
    stats=WorldNormalizationStats(**meta['normalization'])
    loaders={s:make_loader(x,1024,s=='train',True) for s,x in arrays.items()}
    device=torch.device('cuda');model=ControlledDynamics(a.control).to(device)
    save_json({**vars(a),'batch_size':1024,'learning_rate':3e-4,'weight_decay':1e-4,'initialization':'from scratch; same seed and parameter shapes in all conditions','registered_parameters':sum(p.numel() for p in model.parameters()),'feedback_control':'remove predicted-observation embedding, keep recurrent state and LayerNorm','weather_control':'all standardized weather channels set to zero; positions/context retained','self_modulation_control':'same projections and branch MLPs; each relevance field uses its own branch; downstream weather cross-attention and feedback retained','input_manifest':meta},root/'config.json')
    start=time.monotonic()
    state,history,epoch,score=train_dynamics(model,loaders['train'],loaders['validation'],arrays['validation'],stats,device,a.epochs,a.patience,3e-4,1e-4)
    model.load_state_dict(state);torch.save(state,root/'dynamics_best.pt');write_csv(history,root/'training_history.csv')
    metrics={}
    for split in ('validation','test'):
        x=arrays[split];pred=predict_dynamics(model,loaders[split],'biid_climate',device)
        metrics[split]=lai_metrics(x['target_lai'],pred,x['target_lai_valid'],x['relative_weight'],stats)
        np.savez_compressed(root/f'{split}_predictions.npz',prediction=pred,target=x['target_lai'],valid=x['target_lai_valid'],source_indices=x['source_indices'],year=x['year'],row=x['row'],col=x['col'])
    save_json(dict(crop=a.crop,seed=a.seed,control=a.control,metrics=metrics,best_epoch=epoch,validation_selection_rmse=score,elapsed_seconds=time.monotonic()-start),root/'test_metrics.json')


if __name__=='__main__':main()
