"""Complete the registered Ridge, random forest, and CNN-RNN seasonal controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import time

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from inseason_13year_data import ROOT, CACHE, BLOCKS, RECIPES, load, partition
from inseason_complete_inputs import flat_features
from inseason_direct_extension_models import SeasonalCNNHistoryLSTM
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_inseason_13year_direct import make_design, CODE as BASE_CODE
from run_inseason_complete_direct import predict
from run_inseason_direct_baselines import score
from run_ndvi_signal_permutation import check_files
from run_review_revision_parallel import atomic_json
from run_yield_only_classical_baselines import build_model

OUT = ROOT / 'benchmark/results/inseason_direct_extension_v1'
MODELS = ('ridge', 'random_forest', 'cnn_rnn')
ALPHAS = (1., 10., 100., 1000., 10000.)
CODE = (*BASE_CODE, 'inseason_direct_extension_models.py',
        'run_inseason_direct_extension.py', 'run_yield_only_classical_baselines.py',
        'run_crop_signal_screen.py', 'run_inseason_direct_baselines.py')


def root_for(crop, cutoff, model, smoke=False):
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_42' / model


def verify(root):
    marker = json.loads((root / 'complete.json').read_text())
    check_files(root, marker['files'])
    if marker['code_sha256'] != {n: sha256(ROOT / 'scripts' / n) for n in CODE}:
        raise ValueError('Changed completed seasonal extension source')
    return marker


def ridge_fit(x, y, weight, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))
    model.fit(np.asarray(x, dtype=float), y, ridge__sample_weight=weight)
    return model


def select_ridge(x, residual, weights, years, dest, smoke):
    xx, xv = flat_features(x['train']).astype(float), flat_features(x['other']).astype(float)
    records, best = [], None
    for alpha in ALPHAS:
        model = ridge_fit(xx, residual['train'], weights, alpha)
        value = score(residual['other'], model.predict(xv), years)['mean_annual_rmse']
        records.append(dict(alpha=alpha, inner_mean_annual_standardized_rmse=value))
        if best is None or value < best[0]:
            best = value, alpha
    return dict(alpha=best[1], candidates=records)


def train_cnn(x, residual, weights, years, dest, smoke, epochs=None):
    set_seed(42)
    model = SeasonalCNNHistoryLSTM(x['train']['sequence'].shape[-1], x['train']['static'].shape[-1]).cuda()
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
            inputs = [torch.as_tensor(x['train'][k][ix], device='cuda') for k in ('sequence', 'static', 'valid')]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(*inputs)
                target = torch.as_tensor(residual['train'][ix], dtype=torch.float32, device='cuda')
                weight = torch.as_tensor(weights[ix], dtype=torch.float32, device='cuda')
                loss = ((output.float()-target).square()*weight).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite CNN-RNN training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            summed += float(loss.detach())*len(ix)
        row = dict(epoch=epoch, loss=summed/len(order))
        if epochs is None:
            value = score(residual['other'], predict(model, x['other'], batch), years)['mean_annual_rmse']
            row['inner_mean_annual_standardized_rmse'] = value
            if value < best:
                best, chosen, stale = value, epoch, 0
            else:
                stale += 1
        traces.append(row)
        atomic_json(dest / ('inner_training.json' if epochs is None else 'full_training.json'), traces)
        if epochs is None and stale >= 6:
            break
    if epochs is None:
        if chosen < 1:
            raise ValueError('An initialized model cannot be selected')
        del model
        torch.cuda.empty_cache()
        return dict(selected_epochs=chosen)
    torch.save(model.state_dict(), dest / 'model.pt')
    output = predict(model, x['other'], batch)
    restored = SeasonalCNNHistoryLSTM(x['train']['sequence'].shape[-1], x['train']['static'].shape[-1]).cuda()
    restored.load_state_dict(torch.load(dest / 'model.pt', map_location='cpu', weights_only=True))
    np.testing.assert_array_equal(output, predict(restored, x['other'], batch))
    atomic_json(dest / 'neural_audit.json', dict(selected_epochs=epochs,
        parameter_count=sum(p.numel() for p in model.parameters()),
        optimizer_steps=epochs*((len(residual['train'])+batch-1)//batch),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    return output


def run(crop, cutoff, name, smoke=False):
    if name == 'cnn_rnn':
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = root_for(crop, cutoff, name, smoke)
    root.mkdir(parents=True, exist_ok=True)
    code = {n: sha256(ROOT / 'scripts' / n) for n in CODE}
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify(root)
            return
        raw, _ = load(crop)
        groups = partition(raw, cutoff)
        if smoke:
            groups = {s: np.concatenate([ix[raw['year'][ix] == y][:16]
                      for y in np.unique(raw['year'][ix])]) for s, ix in groups.items()}
        spec = dict(crop=crop, cutoff=cutoff, model=name, seed=42, smoke=smoke,
            code_sha256=code, cache_sha256=sha256(CACHE / crop / 'manifest.json'),
            recipe=RECIPES[crop], ratios=[.1] if smoke else [.1, .3, .5],
            periods={s: np.unique(raw['year'][ix]).tolist() for s, ix in groups.items()},
            input_builder='Unchanged run_inseason_13year_direct.make_design',
            objective='Year-balanced standardized causal-trend residual MSE',
            ridge_alphas=ALPHAS, forest=build_model('random_forest', 42, 2).get_params(),
            cnn_rnn=dict(seasonal_conv=[64,128], kernel=3, masked_mean=True,
                history_lstm=dict(lags=5, channels=2, hidden=64, layers=1),
                head=[128,64,1], dropout=.1, lr=.001, weight_decay=.0001,
                maximum_epochs=2 if smoke else 40, patience=6, batch=256 if smoke else 2048,
                cudnn_benchmark=True, adaptation_not_original_paper_replication=True),
            selection='Preceding two-year annual RMSE for Ridge/CNN; fixed forest capacity',
            weight_replay=True, original_direct_weights_unchanged=True)
        spec = json.loads(json.dumps(spec))
        path = root / 'config.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Changed registered direct extension')
        atomic_json(path, spec)
        started = time.monotonic()
        for ratio in spec['ratios']:
            dest = root / f'tail_{round(100*ratio):02d}'
            dest.mkdir(exist_ok=True)
            if (dest / 'audit.json').exists():
                check_files(dest, json.loads((dest / 'audit.json').read_text())['files'])
                continue
            selection = dict(fixed_parameters=True)
            if name != 'random_forest':
                x, residual, norm = make_design(raw, groups['inner_fit'], groups['inner_validation'], RECIPES[crop], ratio)
                atomic_json(dest / 'inner_normalization.json', norm)
                weights = year_weights(raw['year'][groups['inner_fit']])
                with threadpool_limits(limits=2):
                    selection = (select_ridge if name == 'ridge' else train_cnn)(
                        x, residual, weights, raw['year'][groups['inner_validation']], dest, smoke)
                del x, residual
            x, residual, norm = make_design(raw, groups['full_fit'], groups['evaluation'], RECIPES[crop], ratio)
            weights = year_weights(raw['year'][groups['full_fit']])
            with threadpool_limits(limits=2):
                if name == 'cnn_rnn':
                    output = train_cnn(x, residual, weights, None, dest, smoke, selection['selected_epochs'])
                else:
                    xf, xv = flat_features(x['train']), flat_features(x['other'])
                    if name == 'ridge':
                        model = ridge_fit(xf, residual['train'], weights, selection['alpha'])
                        xv = xv.astype(float)
                    else:
                        model = build_model('random_forest', 42, 2)
                        if smoke:
                            model.set_params(n_estimators=8)
                        model.fit(xf, residual['train'], sample_weight=weights)
                        model.set_params(n_jobs=1)
                    output = model.predict(xv).astype(float)
                    joblib.dump(model, dest / 'model.joblib')
                    restored = joblib.load(dest / 'model.joblib')
                    np.testing.assert_array_equal(output, restored.predict(xv))
                    del model, restored, xf, xv
            prediction = x['other']['anchor']+norm['center']+norm['scale']*output
            labels = {k: raw[k][groups['evaluation']] for k in ('year', 'row', 'col', 'target', 'source_indices')}
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite direct extension prediction')
            np.savez_compressed(dest / 'evaluation_predictions.npz', prediction=prediction, **labels)
            metrics = score(labels['target'], prediction, labels['year'])
            atomic_json(dest / 'metrics.json', dict(scores=metrics, selection=selection,
                full_normalization=norm, sequence_shape=list(x['train']['sequence'].shape[1:]),
                static_width=x['train']['static'].shape[1]))
            files = {str(p.relative_to(dest)): sha256(p) for p in dest.rglob('*') if p.is_file()}
            atomic_json(dest / 'audit.json', dict(files=files, weight_replay=True,
                future_values_and_quality_excluded=True, all_fitting_before_evaluation=True))
            print(f'[DIRECT EXTENSION] {crop} {cutoff} {name} suffix={ratio} annual={metrics["mean_annual_rmse"]:.6f}', flush=True)
            del x, residual
            if name == 'cnn_rnn':
                torch.cuda.empty_cache()
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file() and p.name not in ('run.lock', 'complete.json')}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=code,
            seconds=time.monotonic()-started, smoke=smoke, final_models=len(spec['ratios'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=(*RECIPES, 'all'), required=True)
    parser.add_argument('--cutoff', type=int, choices=tuple(BLOCKS))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.crop == 'all':
        if args.model == 'cnn_rnn':
            parser.error('Use shared GPU admission for CNN-RNN')
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(run, crop, cutoff, args.model, args.smoke)
                           for crop in RECIPES for cutoff in BLOCKS]:
                future.result()
    elif args.cutoff is None:
        parser.error('--cutoff is required for a single crop')
    else:
        run(args.crop, args.cutoff, args.model, args.smoke)
