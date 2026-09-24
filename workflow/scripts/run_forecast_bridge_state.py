"""Inner-selected, full-window-refitted scalar state models; no test loader."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from forecast_bridge_data import ROOT, CROPS, PRODUCTS, load, root as data_root, fit_stats, identity_hash
from forecast_bridge_state import ForecastState, batch_arrays, masked_loss
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/forecast_state_bridge_v1/states'
CODE = ('forecast_bridge_data.py', 'forecast_bridge_state.py', 'run_forecast_bridge_state.py',
        'token_retention_state.py', 'dual_remote_state.py', 'biid_world_model.py')


def run_root(crop, product, architecture, cutoff, seed, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / product / architecture / f'cutoff_{cutoff}/seed_{seed}'


def tensors(b, device):
    return {k: torch.as_tensor(v, device=device) for k, v in b.items()}


@torch.no_grad()
def predict(model, a, take, stats, product, batch=256):
    model.eval()
    pieces = []
    device = next(model.parameters()).device
    for start in range(0, len(take), batch):
        selected = take[start:start+batch]
        b = tensors(batch_arrays(a, selected, stats, product), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p = model(b)
        pieces.append(p.float().cpu().numpy())
    return np.concatenate(pieces)*stats[product]['std']+stats[product]['mean']


def state_metrics(a, take, prediction, product):
    truth = a[f'observed_{product}'][take]
    valid = np.isfinite(truth) & (a['relative_valid'][take] > 0)
    if not np.isfinite(prediction[valid]).all():
        raise FloatingPointError('Nonfinite state forecast')
    errors = (prediction-truth)**2
    years = a['year'][take]
    return {str(int(year)): float(np.sqrt(errors[(years == year)[:, None] & valid].mean()))
            for year in np.unique(years)}


def train_epoch(model, a, take, stats, product, optimizer, rng, batch):
    model.train()
    shuffled = rng.permutation(take)
    total, count = 0., 0
    device = next(model.parameters()).device
    for start in range(0, len(shuffled), batch):
        indices = shuffled[start:start+batch]
        b = tensors(batch_arrays(a, indices, stats, product), device)
        truth = torch.as_tensor((a[f'observed_{product}'][indices]-stats[product]['mean'])/stats[product]['std'], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = model(b)
            loss = masked_loss(output, truth, b['relative_valid'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite state loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total += float(loss.detach())*len(indices)
        count += len(indices)
    return total/count


def sample_by_year(a, take, limit):
    return np.concatenate([take[a['year'][take] == y][:limit] for y in np.unique(a['year'][take])])


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for registered state training')
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    a, meta = load(args.crop)
    fit = np.flatnonzero(a['year'] <= args.cutoff-2)
    validation = np.flatnonzero((a['year'] > args.cutoff-2) & (a['year'] <= args.cutoff))
    full = np.flatnonzero(a['year'] <= args.cutoff)
    if not len(fit) or not len(validation) or args.cutoff > 2009:
        raise ValueError('Invalid forward state cutoff')
    if args.smoke:
        fit, validation, full = [sample_by_year(a, x, 16) for x in (fit, validation, full)]
    destination = run_root(args.crop, args.product, args.architecture, args.cutoff, args.seed, args.smoke)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(**vars(args), schema=1, data_sha256=sha256(data_root(args.crop) / 'manifest.json'),
            code_sha256={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            fit_identity=identity_hash(a['source_indices'][fit]), inner_validation_identity=identity_hash(a['source_indices'][validation]),
            full_identity=identity_hash(a['source_indices'][full]), selection_years=np.unique(a['year'][validation]).tolist(),
            target_observations_in_forward=False, target_quality_in_forward=False, evaluation_arrays_loaded=False,
            physical_product=args.product, architecture_detail='retain12, dim128, 2 BIID layers plus cross-attention/gate' if args.architecture == 'biid' else 'GRUCell128 recursive',
            optimizer=dict(lr=3e-4, weight_decay=1e-4), selection='Mean physical state RMSE over inner years; full refit for selected epochs',
            protocol=meta['protocol'])
        file = destination / 'config.json'
        if file.exists() and json.loads(file.read_text()) != config:
            raise ValueError('Frozen state specification changed')
        atomic_json(file, config)
        if (destination / 'complete.json').exists():
            previous = json.loads((destination / 'complete.json').read_text())
            for name, checksum in previous['files'].items():
                if sha256(destination / name) != checksum:
                    raise ValueError('Changed state model')
            return
        start = time.monotonic()
        stats = fit_stats(a, fit)
        set_seed(args.seed)
        model = ForecastState(args.architecture).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        best, epoch_best, stale, trace = float('inf'), 0, 0, []
        epochs = 1 if args.smoke else args.epochs
        for epoch in range(1, epochs+1):
            loss = train_epoch(model, a, fit, stats, args.product, optimizer, rng, args.batch)
            p = predict(model, a, validation, stats, args.product, args.batch)
            scores = state_metrics(a, validation, p, args.product)
            score = float(np.mean(list(scores.values())))
            trace.append(dict(stage='inner_selection', epoch=epoch, loss=loss, rmse=scores))
            if score < best:
                best, epoch_best, stale = score, epoch, 0
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, destination / 'inner_best.pt')
            else:
                stale += 1
            print(f'[STATE] {args.crop} {args.product} cutoff={args.cutoff} epoch={epoch} inner_rmse={score:.6f}', flush=True)
            atomic_json(destination / 'training_history.json', trace)
            if stale >= args.patience:
                break
        del optimizer, model
        torch.cuda.empty_cache()
        stats_full = fit_stats(a, full)
        set_seed(args.seed)
        model = ForecastState(args.architecture).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, epoch_best+1):
            loss = train_epoch(model, a, full, stats_full, args.product, optimizer, rng, args.batch)
            trace.append(dict(stage='full_refit', epoch=epoch, loss=loss))
            print(f'[STATE REFIT] {args.crop} cutoff={args.cutoff} {epoch}/{epoch_best} loss={loss:.6f}', flush=True)
            atomic_json(destination / 'training_history.json', trace)
        weight = destination / 'model.pt'
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weight)
        atomic_json(destination / 'normalization.json', stats_full)
        verify = full[:min(32, len(full))]
        p = predict(model, a, verify, stats_full, args.product, args.batch)
        restored = ForecastState(args.architecture).cuda()
        restored.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
        np.testing.assert_array_equal(p, predict(restored, a, verify, stats_full, args.product, args.batch))
        atomic_json(destination / 'complete.json', dict(smoke=args.smoke, selected_epochs=epoch_best,
            inner_state_rmse=best, full_training_rows=len(full), seconds=time.monotonic()-start,
            full_fit_cutoff=args.cutoff, maximum_replay_error=0.,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, evaluation_arrays_loaded=False,
            files={n: sha256(destination / n) for n in ('config.json', 'model.pt', 'inner_best.pt', 'normalization.json', 'training_history.json')}))
        print(f'[STATE COMPLETE] {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--product', choices=PRODUCTS, default='ndvi')
    parser.add_argument('--architecture', choices=('biid', 'gru'), default='biid')
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
