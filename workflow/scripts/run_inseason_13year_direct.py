"""Seasonal direct controls with pre-evaluation selection and full-window refits."""
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

from forecast_bridge_data import ROOT, fit_stats, climatology
from inseason_13year_data import CACHE, OUT, BLOCKS, RECIPES, load, partition
from inseason_complete_inputs import structured_features, flat_features
from inseason_ndvi_reuse import historical_support
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_inseason_complete_direct import predict, train_tree
from run_inseason_direct_baselines import score
from run_input_matched_direct_yield import make_neural_model
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json

CODE = ('run_inseason_13year_direct.py', 'inseason_13year_data.py',
        'inseason_complete_inputs.py', 'run_inseason_complete_direct.py',
        'run_input_matched_direct_yield.py', 'inseason_ndvi_reuse.py', 'forecast_bridge_data.py')


def root_for(crop, cutoff, model, smoke=False):
    return OUT / ('direct_smoke' if smoke else 'direct') / crop / f'cutoff_{cutoff}/seed_42' / model


def make_design(raw, fit, other, recipe, ratio):
    stats = fit_stats(raw, fit)
    take = np.concatenate((fit, other))
    support = historical_support(raw, fit, np.arange(len(raw['year'])), raw['metadata'][fit])
    climates = {p: climatology(raw, fit, take, p) for p in recipe.split('_')}
    encoded = structured_features(raw, take, raw['metadata'], support, stats, climates, recipe, ratio)
    x = dict(train={k: v[:len(fit)] for k, v in encoded.items()},
             other={k: v[len(fit):] for k, v in encoded.items()})
    residual = raw['target'][fit].astype(float)-x['train']['anchor']
    center, scale = float(residual.mean()), max(float(residual.std()), 1e-6)
    return x, dict(train=(residual-center)/scale,
        other=(raw['target'][other].astype(float)-x['other']['anchor']-center)/scale), dict(
            center=center, scale=scale, features=stats)


def neural_fit(model_name, x, residual, weights, years, dest, smoke, epochs=None):
    set_seed(42)
    model = make_neural_model(model_name, x['train']['sequence'].shape[-1], x['train']['static'].shape[-1], .1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    batch = 256 if smoke else 2048
    maximum = epochs if epochs is not None else (2 if smoke else 40)
    rng = np.random.default_rng(42)
    best, chosen, stale, traces = float('inf'), 0, 0, []
    for epoch in range(1, maximum+1):
        model.train()
        summed = 0.
        order = rng.permutation(len(residual['train']))
        for start in range(0, len(order), batch):
            ix = order[start:start+batch]
            opt.zero_grad(set_to_none=True)
            args = [torch.as_tensor(x['train'][k][ix], device='cuda') for k in ('sequence', 'static', 'valid')]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(*args)
                target = torch.as_tensor(residual['train'][ix], dtype=torch.float32, device='cuda')
                weight = torch.as_tensor(weights[ix], dtype=torch.float32, device='cuda')
                loss = ((prediction.float()-target).square()*weight).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite rolling direct loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            summed += float(loss.detach())*len(ix)
        record = dict(epoch=epoch, loss=summed/len(order))
        if epochs is None:
            value = score(residual['other'], predict(model, x['other'], batch), years)['mean_annual_rmse']
            record['inner_mean_annual_standardized_rmse'] = value
            if value < best:
                best, chosen, stale = value, epoch, 0
            else:
                stale += 1
        traces.append(record)
        atomic_json(dest / ('inner_training.json' if epochs is None else 'full_training.json'), traces)
        if epochs is None and stale >= 6:
            break
    if epochs is None:
        del model
        torch.cuda.empty_cache()
        return chosen
    torch.save(model.state_dict(), dest / 'model.pt')
    pred = predict(model, x['other'], batch)
    restored = make_neural_model(model_name, x['train']['sequence'].shape[-1], x['train']['static'].shape[-1], .1).cuda()
    restored.load_state_dict(torch.load(dest / 'model.pt', map_location='cpu', weights_only=True))
    np.testing.assert_array_equal(pred, predict(restored, x['other'], batch))
    return pred


def run(crop, cutoff, model_name, smoke=False):
    if model_name != 'lightgbm':
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = root_for(crop, cutoff, model_name, smoke)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = {n: sha256(ROOT / 'scripts' / n) for n in CODE}
        if (root / 'complete.json').exists():
            marker = json.loads((root / 'complete.json').read_text())
            check_files(root, marker['files'])
            if marker['code_sha256'] != hashes:
                raise ValueError('Changed completed seasonal control')
            return
        raw, _ = load(crop)
        ix = partition(raw, cutoff)
        if smoke:
            ix = {s: np.concatenate([v[raw['year'][v] == y][:16] for y in np.unique(raw['year'][v])]) for s, v in ix.items()}
        ratios = (.1,) if smoke else (.1, .3, .5)
        spec = dict(crop=crop, cutoff=cutoff, model=model_name, seed=42, smoke=smoke,
            ratios=ratios, code_sha256=hashes, cache_sha256=sha256(CACHE / crop / 'manifest.json'),
            periods={s: np.unique(raw['year'][v]).tolist() for s,v in ix.items()},
            recipe=RECIPES[crop], inner_selection='Mean annual RMSE before all evaluation years',
            final_refit=True, direct_vegetation_target=False, input='Complete quality interface',
            objective='Year-balanced standardized causal-trend residual MSE',
            neural=dict(layers=2, width=128, dropout=.1, lr=.001, weight_decay=.0001,
                epochs=2 if smoke else 40, patience=6, batch=256 if smoke else 2048),
            lightgbm_candidates=[dict(leaves=15, minimum_leaf=200, l2=10.), dict(leaves=31, minimum_leaf=40, l2=1.)])
        spec = json.loads(json.dumps(spec))
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != spec:
            raise ValueError('Changed registered rolling seasonal protocol')
        atomic_json(root / 'config.json', spec)
        started = time.monotonic()
        for ratio in ratios:
            dest = root / f'tail_{round(100*ratio):02d}'
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'audit.json').exists():
                check_files(dest, json.loads((dest / 'audit.json').read_text())['files'])
                continue
            x, r, norm = make_design(raw, ix['inner_fit'], ix['inner_validation'], RECIPES[crop], ratio)
            atomic_json(dest / 'inner_normalization.json', norm)
            with threadpool_limits(limits=2):
                if model_name == 'lightgbm':
                    inner_dir = dest / 'inner'
                    inner_dir.mkdir(exist_ok=True)
                    _, selection = train_tree(dict(train=x['train'], validation=x['other'], test=x['other']),
                        dict(train=r['train'], validation=r['other'], test=r['other']),
                        year_weights(raw['year'][ix['inner_fit']]), raw['year'][ix['inner_validation']], inner_dir, smoke)
                else:
                    chosen = neural_fit(model_name, x, r, year_weights(raw['year'][ix['inner_fit']]),
                        raw['year'][ix['inner_validation']], dest, smoke)
                    if chosen < 1:
                        raise ValueError('Untrained seasonal candidate')
                    selection = dict(selected_epochs=chosen)
            del x, r
            x, r, norm = make_design(raw, ix['full_fit'], ix['evaluation'], RECIPES[crop], ratio)
            with threadpool_limits(limits=2):
                if model_name == 'lightgbm':
                    winner = selection['candidates'][selection['selected_candidate']]
                    parameters = dict(winner['parameters'], n_estimators=winner['selected_trees'])
                    model = lgb.LGBMRegressor(**parameters)
                    model.fit(flat_features(x['train']), r['train'], sample_weight=year_weights(raw['year'][ix['full_fit']]))
                    prediction = model.booster_.predict(flat_features(x['other']))
                    joblib.dump(model, dest / 'model.joblib')
                    restored = joblib.load(dest / 'model.joblib')
                    np.testing.assert_array_equal(prediction, restored.booster_.predict(flat_features(x['other'])))
                else:
                    prediction = neural_fit(model_name, x, r, year_weights(raw['year'][ix['full_fit']]),
                        None, dest, smoke, epochs=selection['selected_epochs'])
            prediction = x['other']['anchor']+norm['center']+norm['scale']*prediction
            labels = {k: raw[k][ix['evaluation']] for k in ('target', 'year', 'row', 'col', 'source_indices')}
            measured = score(labels['target'], prediction, labels['year'])
            np.savez_compressed(dest / 'evaluation_predictions.npz', prediction=prediction, **labels)
            atomic_json(dest / 'metrics.json', dict(scores=measured, selection=selection,
                full_normalization=norm, sequence_shape=list(x['train']['sequence'].shape[1:]),
                static_width=x['train']['static'].shape[1]))
            files = {str(p.relative_to(dest)): sha256(p) for p in dest.rglob('*') if p.is_file()}
            atomic_json(dest / 'audit.json', dict(files=files, weight_replay=True,
                future_values_and_quality_excluded=True, inner_selection_before_evaluation=True))
            print(f'[13 YEAR DIRECT] {crop} {cutoff} {model_name} suffix={ratio:.1f} annual={measured["mean_annual_rmse"]:.6f}', flush=True)
            del x, r
            if model_name != 'lightgbm':
                torch.cuda.empty_cache()
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file() and p.name != 'run.lock'}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=hashes, smoke=smoke,
                    seconds=time.monotonic()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--crop', choices=tuple(RECIPES)+('all',), required=True)
    p.add_argument('--cutoff', type=int, choices=tuple(BLOCKS))
    p.add_argument('--model', choices=('lightgbm', 'gru', 'transformer'), required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.crop == 'all':
        if args.model != 'lightgbm':
            raise ValueError('Use shared GPU queue')
        with ProcessPoolExecutor(max_workers=4) as pool:
            for f in [pool.submit(run,c,t,args.model,args.smoke) for c in RECIPES for t in BLOCKS]:
                f.result()
    else:
        run(args.crop, args.cutoff, args.model, args.smoke)
