"""Complete-quality seasonal LightGBM, GRU, and Transformer controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from forecast_bridge_data import ROOT, identity_hash
from inseason_complete_inputs import complete_inputs, structured_features, flat_features, RECIPES
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_inseason_direct_baselines import score
from run_input_matched_direct_yield import make_neural_model
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_complete_direct_v2'
CODE = ('inseason_complete_inputs.py', 'run_inseason_complete_direct.py',
        'inseason_direct_data.py', 'inseason_extension_data.py', 'inseason_ndvi_reuse.py',
        'forecast_bridge_data.py', 'run_input_matched_direct_yield.py',
        'ndvi_tail_replacement.py', 'observed_remote_benchmark.py')


def run_root(crop, model, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / model / 'fit_2009/seed_42'


@torch.no_grad()
def predict(model, inputs, batch=2048):
    model.eval()
    result = []
    for start in range(0, len(inputs['static']), batch):
        data = [torch.as_tensor(inputs[k][start:start+batch], device='cuda')
                for k in ('sequence', 'static', 'valid')]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(*data)
        result.append(output.float().cpu().numpy())
    return np.concatenate(result).astype(float)


def train_neural(name, x, residual, weights, val_years, dest, smoke):
    set_seed(42)
    model = make_neural_model(name, x['train']['sequence'].shape[-1], x['train']['static'].shape[-1], .1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch, epochs, patience = (256, 2, 2) if smoke else (2048, 40, 6)
    rng = np.random.default_rng(42)
    best, best_epoch, stale = float('inf'), 0, 0
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        total, count = 0., 0
        shuffled = rng.permutation(len(residual['train']))
        for start in range(0, len(shuffled), batch):
            ix = shuffled[start:start+batch]
            data = [torch.as_tensor(x['train'][k][ix], device='cuda') for k in ('sequence', 'static', 'valid')]
            target = torch.as_tensor(residual['train'][ix], dtype=torch.float32, device='cuda')
            weight = torch.as_tensor(weights[ix], dtype=torch.float32, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(*data)
                loss = ((output.float()-target).square()*weight).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite direct yield loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(ix)
            count += len(ix)
        val = predict(model, x['validation'], batch)
        metric = score(residual['validation'], val, val_years)['mean_annual_rmse']
        trace.append(dict(epoch=epoch, loss=total/count, validation_standardized_rmse=metric))
        if metric < best:
            best, best_epoch, stale = metric, epoch, 0
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, dest / 'model.pt')
        else:
            stale += 1
        atomic_json(dest / 'training_history.json', trace)
        print(f'[NEURAL] {name} {dest.parent.name} epoch={epoch} val={metric:.6f}', flush=True)
        if stale >= patience:
            break
    if best_epoch < 1:
        raise ValueError('No trained neural checkpoint')
    model.load_state_dict(torch.load(dest / 'model.pt', map_location='cpu', weights_only=True))
    outputs = {s: predict(model, x[s], batch) for s in ('validation', 'test')}
    restored = make_neural_model(name, x['train']['sequence'].shape[-1], x['train']['static'].shape[-1], .1).cuda()
    restored.load_state_dict(torch.load(dest / 'model.pt', map_location='cpu', weights_only=True))
    np.testing.assert_array_equal(outputs['test'], predict(restored, x['test'], batch))
    return outputs, dict(selected_epoch=best_epoch, epochs_run=len(trace),
                         parameter_count=sum(p.numel() for p in model.parameters()),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def train_tree(x, residual, weights, val_years, dest, smoke):
    features = {s: flat_features(v) for s, v in x.items()}
    best, candidates = None, []
    configs = [(15, 200, 10.), (31, 40, 1.)]
    for leaves, child, penalty in configs:
        params = dict(n_estimators=15 if smoke else 1200, learning_rate=.03,
            num_leaves=leaves, min_child_samples=child, reg_lambda=penalty,
            subsample=.9, subsample_freq=1, colsample_bytree=.9, n_jobs=2,
            random_state=42, verbosity=-1, deterministic=True, force_col_wise=True,
            objective='regression', metric='None')

        def metric(truth, pred):
            return 'annual_rmse', score(truth, pred, val_years)['mean_annual_rmse'], False

        model = lgb.LGBMRegressor(**params)
        trace = {}
        model.fit(features['train'], residual['train'], sample_weight=weights,
            eval_set=[(features['validation'], residual['validation'])], eval_metric=metric,
            callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.record_evaluation(trace)])
        value = metric(residual['validation'], model.booster_.predict(features['validation']))[1]
        candidates.append(dict(parameters=params, selected_trees=int(model.best_iteration_), score=value, trace=trace))
        if best is None or value < best[0]:
            best = (value, model, len(candidates)-1)
    model = best[1]
    joblib.dump(model, dest / 'model.joblib')
    restored = joblib.load(dest / 'model.joblib')
    outputs = {s: model.booster_.predict(features[s]) for s in ('validation', 'test')}
    np.testing.assert_array_equal(outputs['test'], restored.booster_.predict(features['test']))
    return outputs, dict(selected_candidate=best[2], candidates=candidates)


def run(crop, name, smoke=False):
    if name != 'lightgbm':
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(False)
        torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = run_root(crop, name, smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = {n: sha256(ROOT / 'scripts' / n) for n in CODE}
        if (root / 'complete.json').exists():
            marker = json.loads((root / 'complete.json').read_text())
            check_files(root, marker['files'])
            if code != marker['code_sha256']:
                raise ValueError('Changed completed direct control')
            return
        started = time.monotonic()
        raw, rows, metadata, support, stats, climate, provenance = complete_inputs(crop)
        if smoke:
            for split, ix in rows.items():
                selected = np.concatenate([np.flatnonzero(raw['year'][ix] == y)[:16] for y in np.unique(raw['year'][ix])])
                rows[split] = ix[selected]
                climate[split] = {p: a[selected] for p, a in climate[split].items()}
        ratios = (.1,) if smoke else (.1, .3, .5)
        spec = dict(crop=crop, model=name, recipe=RECIPES[crop], seed=42, smoke=smoke,
            ratios=ratios, code_sha256=code, normalization=stats,
            splits={s: dict(years=np.unique(raw['year'][ix]).tolist(), rows=len(ix),
                            identity=identity_hash(raw['source_indices'][ix])) for s, ix in rows.items()},
            selection='Mean annual validation RMSE; no test-driven selection',
            metadata='Exact six prefix-quality fields and hidden support from frozen seasonal interface',
            removed_unselected_previous_quality=True, target_state_supervision=False,
            weather='Supplied full-year actual weather, shared conditional setting',
            neural=dict(layers=2, dim=128, attention_heads=4, dropout=.1, lr=1e-3,
                        weight_decay=1e-4, max_epochs=2 if smoke else 40,
                        patience=2 if smoke else 6, batch=256 if smoke else 2048,
                        objective='Year-balanced standardized causal-trend residual MSE'),
            lightgbm_candidates=[dict(leaves=15, minimum_leaf=200, l2=10.),
                                 dict(leaves=31, minimum_leaf=40, l2=1.)],
            weight_reload_replay='Full test prediction replay required')
        spec = json.loads(json.dumps(spec))
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Changed registered direct protocol')
        atomic_json(root / 'config.json', spec)
        atomic_json(root / 'provenance.json', provenance)
        results = []
        for ratio in ratios:
            x = {s: structured_features(raw, ix, metadata, support, stats, climate[s], RECIPES[crop], ratio)
                 for s, ix in rows.items()}
            physical_residual = {s: raw['target'][ix].astype(float)-x[s]['anchor'] for s, ix in rows.items()}
            center = float(physical_residual['train'].mean())
            scale = max(float(physical_residual['train'].std()), 1e-6)
            residual = {s: (r-center)/scale for s, r in physical_residual.items()}
            dest = root / f'tail_{round(ratio*100):02d}'
            dest.mkdir(parents=True, exist_ok=True)
            weights = year_weights(raw['year'][rows['train']])
            with threadpool_limits(limits=2):
                outputs, selection = (train_tree(x, residual, weights, raw['year'][rows['validation']], dest, smoke)
                    if name == 'lightgbm' else train_neural(name, x, residual, weights,
                        raw['year'][rows['validation']], dest, smoke))
            metrics = {}
            for split, value in outputs.items():
                labels = {k: raw[k][rows[split]] for k in ('target', 'source_indices', 'year', 'row', 'col')}
                prediction = x[split]['anchor']+center+scale*value
                metrics[split] = score(labels['target'], prediction, labels['year'])
                np.savez_compressed(dest / f'{split}_predictions.npz', prediction=prediction, **labels)
            record = dict(crop=crop, model=name, ratio=ratio, metrics=metrics, selection=selection,
                residual_normalization=dict(center=center, scale=scale),
                sequence_shape=list(x['train']['sequence'].shape[1:]), static_width=x['train']['static'].shape[1],
                maximum_test_replay_error=0.)
            atomic_json(dest / 'metrics.json', record)
            results.append(record)
            print(f'[COMPLETE INPUTS] {crop} {name} {ratio:.1f} '
                  f'test={metrics["test"]["pooled_rmse"]:.6f}', flush=True)
            if name != 'lightgbm':
                torch.cuda.empty_cache()
        atomic_json(root / 'summary.json', results)
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
                 if p.is_file() and p.name not in ('run.lock', 'complete.json')}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=code,
                    seconds=time.monotonic()-started, smoke=smoke, test_used_for_selection=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES)+('all',), required=True)
    p.add_argument('--model', choices=('lightgbm', 'gru', 'transformer'), required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.crop == 'all':
        if args.model != 'lightgbm':
            raise ValueError('Neural jobs must use the shared GPU admission queue')
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(run, c, args.model, args.smoke) for c in RECIPES]:
                future.result()
    else:
        run(args.crop, args.model, args.smoke)
