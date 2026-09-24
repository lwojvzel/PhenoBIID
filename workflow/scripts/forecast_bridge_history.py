"""Chronological history experts with inner-only selection and explicit lineage."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from forecast_bridge_data import ROOT, CROPS, load, root as data_root, history_features, identity_hash
from task_aligned_world import TaskAlignedWorld
from numeric_embedding_readout import NeuralReadout, training_bins
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from run_crop_signal_screen import year_weights, yearly_rmse

RESULT = ROOT / 'benchmark/results/forecast_state_bridge_v1/history'
CODE = ('forecast_bridge_history.py', 'forecast_bridge_data.py', 'task_aligned_world.py',
        'numeric_embedding_readout.py', 'run_history_multimodal_baselines.py')


def run_root(crop, kind, cutoff, seed, smoke=False):
    return RESULT / ('smoke' if smoke else 'pipelines') / crop / kind / f'cutoff_{cutoff}/seed_{seed}'


def make_model(kind, bins=None):
    if kind == 'mlp':
        return TaskAlignedWorld('history_mlp').yield_head
    return NeuralReadout('tabm', 20, bins)


@torch.no_grad()
def predict_component(model, x, batch=256):
    device = next(model.parameters()).device
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        data = torch.as_tensor(x[start:start+batch], dtype=torch.float32, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p = model(data)
        out.append(p.float().mean(-1).cpu().numpy())
    return np.concatenate(out).astype(float)


def saved_predict(a, take, crop, kind, cutoff, seed, smoke=False):
    directory = run_root(crop, kind, cutoff, seed, smoke)
    marker = json.loads((directory / 'complete.json').read_text())
    for name, checksum in marker['files'].items():
        if sha256(directory / name) != checksum:
            raise ValueError('Changed history expert asset')
    if np.any(a['year'][take] <= cutoff):
        raise ValueError('History expert prediction must be strictly forward')
    config = json.loads((directory / 'config.json').read_text())
    for name, checksum in config['code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != checksum:
            raise ValueError('Changed history expert implementation')
    norm = json.loads((directory / 'normalization.json').read_text())
    x, trend = history_features(a, take, norm['features'])
    if kind == 'tabm':
        x = (x-np.array(norm['x_mean'], np.float32))/np.array(norm['x_std'], np.float32)
        base, parents = saved_predict(a, take, crop, 'mlp', cutoff, seed, smoke)
        bins = torch.load(directory / 'bins.pt', map_location='cpu', weights_only=True)
    else:
        base, parents, bins = trend.astype(float), [], None
    model = make_model(kind, bins).cuda()
    model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
    value = predict_component(model, x)
    del model
    return base+norm['center']+norm['scale']*value, [*parents, str(directory / 'complete.json')]


def mlp_oof(a, take, crop, seed, smoke=False):
    years = a['year'][take]
    if np.any(years < 1987):
        raise ValueError('TabM residuals require five historical warmup years')
    out = np.full(len(take), np.nan)
    lineage = []
    for start in range(1987, int(years.max())+1, 3):
        selected = np.flatnonzero((years >= start) & (years <= start+2))
        if len(selected):
            out[selected], source = saved_predict(a, take[selected], crop, 'mlp', start-1, seed, smoke)
            lineage.extend(source)
    if not np.isfinite(out).all():
        raise ValueError('Incomplete chronological MLP predictions')
    return out, sorted(set(lineage))


def make_design(a, fit, take, kind):
    target = a['target'][fit]
    stats = dict(target_mean=float(target.mean()), target_std=max(float(target.std()), 1e-6),
                 fit_years=np.unique(a['year'][fit]).tolist(), fit_identity=identity_hash(a['source_indices'][fit]))
    xfit, bfit = history_features(a, fit, stats)
    x, base = history_features(a, take, stats)
    norm = dict(features=stats)
    if kind == 'tabm':
        mean, std = xfit.mean(0), np.maximum(xfit.std(0), 1e-6)
        norm.update(x_mean=mean.tolist(), x_std=std.tolist())
        xfit, x = (xfit-mean)/std, (x-mean)/std
        bins, _ = training_bins(xfit)
    else:
        bins = None
    return xfit.astype(np.float32), x.astype(np.float32), bfit.astype(float), base.astype(float), norm, bins


def train_epoch(model, optimizer, x, target, weights, rng):
    model.train()
    device = next(model.parameters()).device
    order = rng.permutation(len(x))
    summed = 0.
    for start in range(0, len(order), 1024):
        indices = order[start:start+1024]
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(indices), 256):
            take = indices[offset:offset+256]
            b = torch.as_tensor(x[take], device=device)
            y = torch.as_tensor(target[take, None], dtype=torch.float32, device=device)
            w = torch.as_tensor(weights[take, None], dtype=torch.float32, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                prediction = model(b)
                loss = ((prediction-y).square()*w).mean()*len(take)/len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite historical loss')
            loss.backward()
            summed += float(loss.detach())*len(indices)
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    return summed/len(x)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    a, _ = load(args.crop)
    begin = 1987 if args.kind == 'tabm' else 1982
    fit = np.flatnonzero((a['year'] >= begin) & (a['year'] <= args.cutoff-2))
    val = np.flatnonzero((a['year'] > args.cutoff-2) & (a['year'] <= args.cutoff))
    full = np.flatnonzero((a['year'] >= begin) & (a['year'] <= args.cutoff))
    if not len(fit) or not len(val) or args.cutoff > 2009 or (args.kind == 'tabm' and args.crop != 'soybean'):
        raise ValueError('Invalid chronological history experiment')
    if args.smoke:
        def small(take):
            return np.concatenate([take[a['year'][take] == y][:16] for y in np.unique(a['year'][take])])
        fit, val, full = small(fit), small(val), small(full)
    destination = run_root(args.crop, args.kind, args.cutoff, args.seed, args.smoke)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(**vars(args), fit_identity=identity_hash(a['source_indices'][fit]),
            inner_identity=identity_hash(a['source_indices'][val]), full_identity=identity_hash(a['source_indices'][full]),
            code_sha256={k: sha256(ROOT / 'scripts' / k) for k in CODE},
            input_manifest_sha256=sha256(data_root(args.crop) / 'manifest.json'),
            optimizer=dict(learning_rate=.001, weight_decay=.0001, batch=1024, microbatch=256),
            selection='Inner last two years annual mean RMSE, including initialization; full refit for fixed selected epochs',
            training_weight='Equal total weight per year', evaluation_arrays_loaded=False,
            remote_inputs=False, upstream_predictions='Chronological MLP OOF for TabM; no full-fit residual substitution')
        config = destination / 'config.json'
        if config.exists() and json.loads(config.read_text()) != spec:
            raise ValueError('Frozen history configuration changed')
        atomic_json(config, spec)
        if (destination / 'complete.json').exists():
            prior = json.loads((destination / 'complete.json').read_text())
            for name, checksum in prior['files'].items():
                if sha256(destination / name) != checksum:
                    raise ValueError('Changed history checkpoint')
            return
        started = time.monotonic()
        x, xv, base, bv, norm, bins = make_design(a, fit, val, args.kind)
        lineage = []
        if args.kind == 'tabm':
            base, l1 = mlp_oof(a, fit, args.crop, args.seed, args.smoke)
            bv, l2 = mlp_oof(a, val, args.crop, args.seed, args.smoke)
            lineage = sorted(set(l1+l2))
        residual = a['target'][fit].astype(float)-base
        norm.update(center=float(residual.mean()), scale=max(float(residual.std()), 1e-6))
        y = (residual-norm['center'])/norm['scale']
        set_seed(args.seed)
        model = make_model(args.kind, bins).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(args.seed)
        best, selected_epoch, stale, trace = float('inf'), 0, 0, []
        for epoch in range((1 if args.smoke else args.epochs)+1):
            loss = train_epoch(model, optimizer, x, y, year_weights(a['year'][fit]), rng) if epoch else None
            p = bv+norm['center']+norm['scale']*predict_component(model, xv)
            scores = yearly_rmse(a['target'][val], p, a['year'][val])
            score = float(np.mean(list(scores.values())))
            if score < best:
                best, selected_epoch, stale = score, epoch, 0
            else:
                stale += 1
            trace.append(dict(stage='inner', epoch=epoch, loss=loss, rmse=scores))
            atomic_json(destination / 'training_history.json', trace)
            print(f'[HISTORY] {args.crop} {args.kind} cutoff={args.cutoff} epoch={epoch} rmse={score:.6f}', flush=True)
            if epoch and stale >= args.patience:
                break
        del model, optimizer, x, xv
        torch.cuda.empty_cache()
        x, _, base, _, norm, bins = make_design(a, full, full[:1], args.kind)
        if args.kind == 'tabm':
            base, extra = mlp_oof(a, full, args.crop, args.seed, args.smoke)
            lineage = sorted(set(lineage+extra))
        residual = a['target'][full].astype(float)-base
        norm.update(center=float(residual.mean()), scale=max(float(residual.std()), 1e-6))
        y = (residual-norm['center'])/norm['scale']
        set_seed(args.seed)
        model = make_model(args.kind, bins).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, selected_epoch+1):
            loss = train_epoch(model, optimizer, x, y, year_weights(a['year'][full]), rng)
            trace.append(dict(stage='full_refit', epoch=epoch, loss=loss))
            atomic_json(destination / 'training_history.json', trace)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, destination / 'model.pt')
        atomic_json(destination / 'normalization.json', norm)
        if bins is not None:
            torch.save(bins, destination / 'bins.pt')
        restored = make_model(args.kind, bins).cuda()
        restored.load_state_dict(torch.load(destination / 'model.pt', map_location='cpu', weights_only=True))
        np.testing.assert_array_equal(predict_component(model, x[:32]), predict_component(restored, x[:32]))
        names = ['config.json', 'model.pt', 'normalization.json', 'training_history.json']
        if bins is not None:
            names.append('bins.pt')
        atomic_json(destination / 'complete.json', dict(smoke=args.smoke, cutoff=args.cutoff,
            selected_epochs=selected_epoch, inner_rmse=best, rows=len(full), seconds=time.monotonic()-started,
            upstream_sha256={p: sha256(Path(p)) for p in lineage}, maximum_replay_error=0.,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, evaluation_arrays_loaded=False,
            files={n: sha256(destination / n) for n in names}))
        print(f'[HISTORY COMPLETE] {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--kind', choices=('mlp', 'tabm'), default='mlp')
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 45, 48), default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
