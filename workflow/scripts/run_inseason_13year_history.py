"""Refit the full historical library before each thirteen-year evaluation block."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import time

import faiss
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from forecast_bridge_data import ROOT, history_features, identity_hash
from inseason_13year_data import CACHE, OUT, BLOCKS, ALPHAS, RECIPES, load, partition
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_inseason_direct_baselines import score
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from run_yield_only_classical_baselines import (ALL_METHODS, STATISTICAL_METHODS,
    build_model, _fit_model, _fit_predict_knn, _serializable_parameters)
from run_yield_only_neural_baselines import HistorySequenceRegressor, MODELS

CODE = ('run_inseason_13year_history.py', 'inseason_13year_data.py',
        'forecast_bridge_data.py', 'run_yield_only_classical_baselines.py',
        'run_yield_only_neural_baselines.py')


def root_for(crop, cutoff, kind, smoke=False):
    return OUT / ('smoke' if smoke else 'history') / crop / f'cutoff_{cutoff}/seed_42' / kind


def design(raw, fit, take):
    norm = dict(target_mean=float(np.mean(raw['target'][fit])),
                target_std=max(float(np.std(raw['target'][fit])), 1e-6))
    xf, bf = history_features(raw, fit, norm)
    xt, bt = history_features(raw, take, norm)
    rf = raw['target'][fit].astype(float)-bf
    center, scale = float(rf.mean()), max(float(rf.std()), 1e-6)
    norm.update(residual_center=center, residual_scale=scale)
    return dict(train=xf, other=xt, train_y=(rf-center)/scale,
        other_y=(raw['target'][take].astype(float)-bt-center)/scale,
        train_anchor=bf.astype(float), other_anchor=bt.astype(float), normalization=norm)


def physical(value, design):
    n = design['normalization']
    return design['other_anchor']+n['residual_center']+n['residual_scale']*value


def statistics(raw, crop, ix):
    inner, val, full, ev = (ix[k] for k in ('inner_fit', 'inner_validation', 'full_fit', 'evaluation'))
    mean = float(raw['target'][full].mean())
    h = raw['history'][ev]
    known = h[:, 14] > 0
    historical = np.where(known, h[:, 10], mean).astype(float)
    result = dict(global_mean=np.full(len(ev), mean), grid_climatology=historical,
        linear_trend=np.where(known, raw['baseline'][ev], mean).astype(float))
    for name, width in (('previous_year', 1), ('rolling_mean_3', 3), ('rolling_mean_5', 5)):
        valid = h[:, 5:5+width] > 0
        result[name] = np.divide(np.where(valid, h[:, :width], 0).sum(1), valid.sum(1),
            out=historical.copy(), where=valid.sum(1) > 0)
    candidates = {}
    for alpha in ALPHAS:
        values = np.load(CACHE / crop / f'smoothing_{alpha:g}.npy', mmap_mode='r')
        p = np.where(raw['history'][val, 14] > 0, values[val], raw['target'][inner].mean())
        candidates[alpha] = score(raw['target'][val], p, raw['year'][val])['mean_annual_rmse']
    alpha = min(candidates, key=candidates.get)
    values = np.load(CACHE / crop / f'smoothing_{alpha:g}.npy', mmap_mode='r')
    result['exponential_smoothing'] = np.where(known, values[ev], mean).astype(float)
    return result, dict(alpha=alpha, inner_mean_annual_rmse=candidates, global_mean=mean)


def classic(method, inner, full, dest, smoke):
    if method == 'approximate_knn':
        _, output, index, mean, std = _fit_predict_knn(
            full['train'].astype(np.float32), full['train_y'].astype(np.float32),
            full['other'].astype(np.float32), full['other'].astype(np.float32), 2)
        faiss.write_index(index, str(dest / 'index.faiss'))
        np.savez_compressed(dest / 'neighbors.npz', mean=mean, std=std, target=full['train_y'].astype(np.float32))
        restored = faiss.read_index(str(dest / 'index.faiss'))
        distances, neighbors = restored.search(np.ascontiguousarray((full['other']-mean)/std, dtype=np.float32), 25)
        weights = 1./np.maximum(distances, 1e-6)
        replay = (weights*full['train_y'].astype(np.float32)[neighbors]).sum(1)/weights.sum(1)
        np.testing.assert_array_equal(output, replay)
        return physical(output, full), dict(neighbors=25, ef_search=96, trained_rows=len(full['train']))
    model = build_model(method, 42, 2)
    selected = None
    if smoke and hasattr(model, 'n_estimators'):
        model.set_params(n_estimators=10)
    if method in ('lightgbm', 'xgboost'):
        _fit_model(model, method, inner['train'], inner['train_y'], inner['other'], inner['other_y'])
        selected = int(model.best_iteration_) if method == 'lightgbm' else int(model.best_iteration)+1
        model = build_model(method, 42, 2)
        model.set_params(n_estimators=max(selected, 1))
        if method == 'xgboost':
            model.set_params(early_stopping_rounds=None)
    # Double precision also avoids ill-conditioned float32 linear normal equations.
    model.fit(full['train'].astype(float), full['train_y'])
    output = model.predict(full['other'].astype(float)).astype(float)
    joblib.dump(model, dest / 'model.joblib')
    restored = joblib.load(dest / 'model.joblib')
    np.testing.assert_array_equal(output, restored.predict(full['other'].astype(float)))
    return physical(output, full), dict(selected_trees=selected, parameters=_serializable_parameters(model))


def sequence(x):
    return np.stack((x[:, :5][:, ::-1], x[:, 5:10][:, ::-1]), -1).copy(), x[:, 10:].copy()


@torch.no_grad()
def neural_predict(model, x, batch=2048):
    model.eval()
    seq, context = sequence(x)
    result = []
    for start in range(0, len(x), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model(torch.as_tensor(seq[start:start+batch], device='cuda'),
                      torch.as_tensor(context[start:start+batch], device='cuda'))
        result.append(y.float().cpu().numpy())
    return np.concatenate(result).astype(float)


def fit_neural(method, data, years, dest, smoke, epochs=None):
    set_seed(42)
    model = HistorySequenceRegressor(method, 10, 64, .2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(42)
    seq, static = sequence(data['train'])
    n = len(seq)
    batch = 256 if smoke else 4096
    maximum = epochs if epochs is not None else (2 if smoke else 40)
    best, selected, stale, traces = float('inf'), 0, 0, []
    for epoch in range(1, maximum+1):
        model.train()
        loss_sum = 0.
        order = rng.permutation(n)
        for start in range(0, n, batch):
            ix = order[start:start+batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(torch.as_tensor(seq[ix], device='cuda'), torch.as_tensor(static[ix], device='cuda'))
                loss = (p.float()-torch.as_tensor(data['train_y'][ix], dtype=torch.float32, device='cuda')).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite history training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            loss_sum += float(loss.detach())*len(ix)
        record = dict(epoch=epoch, loss=loss_sum/n)
        if epochs is None:
            value = score(data['other_y'], neural_predict(model, data['other']), years)['mean_annual_rmse']
            record['inner_mean_annual_standardized_rmse'] = value
            if value < best:
                best, selected, stale = value, epoch, 0
            else:
                stale += 1
        traces.append(record)
        atomic_json(dest / ('inner_training.json' if epochs is None else 'full_training.json'), traces)
        if epochs is None and stale >= 6:
            break
    if epochs is None:
        del model
        torch.cuda.empty_cache()
        return selected
    torch.save(model.state_dict(), dest / 'model.pt')
    pred = neural_predict(model, data['other'])
    restored = HistorySequenceRegressor(method, 10, 64, .2).cuda()
    restored.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location='cpu'))
    np.testing.assert_array_equal(pred, neural_predict(restored, data['other']))
    return physical(pred, data), dict(selected_epochs=epochs, optimizer_steps=epochs*((n+batch-1)//batch),
        parameter_count=sum(p.numel() for p in model.parameters()))


def run(crop, cutoff, kind, smoke=False):
    neural = kind != 'classical'
    if neural:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = root_for(crop, cutoff, kind, smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = {n: sha256(ROOT / 'scripts' / n) for n in CODE}
        if (root / 'complete.json').exists():
            done = json.loads((root / 'complete.json').read_text())
            check_files(root, done['files'])
            if done['code_sha256'] != hashes:
                raise ValueError('Changed frozen rolling history run')
            return
        start = time.monotonic()
        raw, _ = load(crop)
        ix = partition(raw, cutoff)
        if smoke:
            ix = {s: np.concatenate([v[raw['year'][v] == y][:20] for y in np.unique(raw['year'][v])]) for s, v in ix.items()}
        spec = dict(crop=crop, cutoff=cutoff, kind=kind, seed=42, smoke=smoke,
            code_sha256=hashes, cache_sha256=sha256(CACHE / crop / 'manifest.json'),
            partitions={s: dict(years=np.unique(raw['year'][v]).tolist(), rows=len(v),
                identity=identity_hash(raw['source_indices'][v])) for s, v in ix.items()},
            inner_selection_precedes_evaluation=True, refit_full_cutoff=True,
            history_input='Same causal lag/context features, source population precedes cohort filter',
            objective='Standardized trend residual MSE; uniform training-row weight',
            selection='Preceding two-year mean annual RMSE; all evaluation years excluded',
            neural=dict(hidden=64, dropout=.2, lr=.001, weight_decay=.0001,
                batch=256 if smoke else 4096, maximum_epochs=2 if smoke else 40, patience=6),
            evaluation_clipping=False)
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Changed rolling history registration')
        atomic_json(root / 'config.json', spec)
        inner = design(raw, ix['inner_fit'], ix['inner_validation'])
        full = design(raw, ix['full_fit'], ix['evaluation'])
        stats, stat_spec = statistics(raw, crop, ix) if not neural else ({}, {})
        methods = (kind,) if neural else ALL_METHODS
        if smoke and not neural:
            methods = (*STATISTICAL_METHODS, 'ridge', 'lightgbm', 'approximate_knn')
        for method in methods:
            dest = root / method
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'audit.json').exists():
                check_files(dest, json.loads((dest / 'audit.json').read_text())['files'])
                continue
            with threadpool_limits(limits=2):
                if method in STATISTICAL_METHODS:
                    pred, metadata = stats[method], stat_spec
                elif neural:
                    selected = fit_neural(method, inner, raw['year'][ix['inner_validation']], dest, smoke)
                    if selected < 1:
                        raise ValueError('Initialized neural candidate forbidden')
                    pred, metadata = fit_neural(method, full, None, dest, smoke, epochs=selected)
                else:
                    pred, metadata = classic(method, inner, full, dest, smoke)
            labels = {k: raw[k][ix['evaluation']] for k in ('target', 'year', 'row', 'col', 'source_indices')}
            measured = score(labels['target'], pred, labels['year'])
            np.savez_compressed(dest / 'evaluation_predictions.npz', prediction=pred, **labels)
            atomic_json(dest / 'metrics.json', dict(method=method, scores=measured,
                selection=metadata, inner_normalization=inner['normalization'], full_normalization=full['normalization']))
            files = {p.name: sha256(p) for p in dest.iterdir() if p.is_file()}
            atomic_json(dest / 'audit.json', dict(files=files,
                full_weight_replay=method not in STATISTICAL_METHODS,
                statistical_rule=method in STATISTICAL_METHODS,
                code_sha256=hashes, all_optimization_years_precede_evaluation=True))
            print(f'[13 YEAR HISTORY] {crop} {cutoff} {method} annual={measured["mean_annual_rmse"]:.6f}', flush=True)
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file() and p.name != 'run.lock'}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=hashes,
            seconds=time.monotonic()-start, methods=methods, smoke=smoke))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES)+('all',), required=True)
    p.add_argument('--cutoff', type=int, choices=tuple(BLOCKS))
    p.add_argument('--kind', choices=('classical', *MODELS), required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.crop == 'all':
        if args.kind != 'classical':
            raise ValueError('Neural processes require the shared GPU queue')
        jobs = [(c, t) for c in RECIPES for t in BLOCKS]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(run, c, t, args.kind, args.smoke) for c, t in jobs]:
                future.result()
    else:
        run(args.crop, args.cutoff, args.kind, args.smoke)
