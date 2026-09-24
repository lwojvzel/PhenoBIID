"""Regional state adaptation with inner epoch selection and no numeric yield input."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from cybench_inseason_model_data import (ROOT, RESULT, RECIPES, SEEDS, DATA_CODE,
    BLOCKS, load_identity, state_population, load_seasons, state_arrays, fit_state_stats)
from forecast_bridge_data import identity_hash
from forecast_bridge_state import ForecastState, batch_arrays
from inseason_nested_common import register, finish, verify
from multimodal_baseline import set_seed
from review_revision_data import sha256
from run_forecast_bridge_state import train_epoch, predict, state_metrics
from run_review_revision_parallel import atomic_json

CODE = tuple(dict.fromkeys((*DATA_CODE, 'run_cybench_inseason_state.py',
    'forecast_bridge_state.py', 'run_forecast_bridge_state.py', 'token_retention_state.py',
    'dual_remote_state.py', 'biid_world_model.py', 'multimodal_baseline.py',
    'inseason_nested_common.py', 'run_ndvi_signal_permutation.py',
    'review_revision_data.py', 'run_review_revision_parallel.py')))
TRAINING = dict(batch=256, epochs=30, patience=5, learning_rate=3e-4,
    weight_decay=1e-4, gradient_clip=1., memory_limit_mib=3072)


def code_hashes():
    return {name: sha256(ROOT / 'scripts' / name) for name in CODE}


def root_for(crop, product, cutoff, seed, smoke=False):
    return RESULT / ('state_smoke' if smoke else 'states') / crop / product / f'cutoff_{cutoff}/seed_{seed}'


def validate_job(crop, product, cutoff, seed):
    if crop not in RECIPES or product not in RECIPES[crop] or cutoff not in BLOCKS or seed not in SEEDS:
        raise ValueError('Unregistered regional state job')


def preflight():
    path = RESULT / 'data_preflight.json'
    record = json.loads(path.read_text())
    if not record['passed'] or record['full_groups'] != 6 or record['new_fits'] != 0:
        raise ValueError('Verified six-group regional input preflight required')
    for name, digest in record['code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError(f'Changed preflight implementation: {name}')
    return path


def subset(arrays, take):
    return {key: value[take] for key, value in arrays.items()}


def temporal_indices(years, cutoff, smoke=False):
    years = np.asarray(years)
    if years.ndim != 1 or not len(years) or years.max() > cutoff:
        raise ValueError('Future or empty seasonal population')
    parts = dict(inner_fit=np.flatnonzero(years <= cutoff-2),
        inner_validation=np.flatnonzero((years >= cutoff-1) & (years <= cutoff)),
        full_fit=np.arange(len(years)))
    if set(years[parts['inner_validation']]) != {cutoff-1, cutoff} or not len(parts['inner_fit']):
        raise ValueError('Two preceding state validation years required')
    if smoke:
        parts = {key: np.sort(np.concatenate([ix[years[ix] == year][:16] for year in np.unique(years[ix])]))
                 for key, ix in parts.items()}
    np.testing.assert_array_equal(np.sort(np.concatenate((parts['inner_fit'], parts['inner_validation']))), parts['full_fit'])
    return parts


def source_guard(sources, cutoff):
    for filename in sources:
        path = Path(filename)
        if path.name in ('history.npz', 'targets.npz'):
            raise ValueError('Numeric yield source entered regional state training')
        if path.name.endswith(('_known.npz', '_state_targets.npz')):
            if int(path.name.split('_', 1)[0]) > cutoff:
                raise ValueError('Future seasonal source entered state training')


def verify_completed(crop, product, cutoff, seed, smoke=False):
    validate_job(crop, product, cutoff, seed)
    root = root_for(crop, product, cutoff, seed, smoke)
    record = verify(root, code_hashes())
    config = json.loads((root / 'config.json').read_text())
    expected = dict(crop=crop, product=product, cutoff=cutoff, seed=seed, smoke=smoke)
    if record['job'] != expected or config['job'] != expected:
        raise ValueError('Regional state identity mismatch')
    if not 1 <= record['selected_epochs'] <= (1 if smoke else TRAINING['epochs']):
        raise ValueError('Untrained or invalid state checkpoint')
    if record['changed_parameter_tensors'] < 1 or record['numeric_yield_arrays_loaded'] or record['evaluation_arrays_loaded']:
        raise ValueError('Invalid regional state training provenance')
    source_guard(config['sources'], cutoff)
    for filename, digest in config['sources'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f'Changed regional state source: {filename}')
    return record


def mean_metric(a, take, prediction, product):
    scores = state_metrics(a, take, prediction, product)
    score = float(np.mean(list(scores.values())))
    if not np.isfinite(score):
        raise FloatingPointError('No finite inner state selection score')
    return score, scores


def run(args):
    validate_job(args.crop, args.product, args.cutoff, args.seed)
    gate = preflight()
    job = dict(crop=args.crop, product=args.product, cutoff=args.cutoff, seed=args.seed, smoke=args.smoke)
    root = root_for(**job)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify_completed(**job)
            print(f'[REGIONAL STATE REUSE] {root}', flush=True)
            return
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('Registered regional state training requires CUDA')
        torch.cuda.set_per_process_memory_fraction(TRAINING['memory_limit_mib']*2**20/
            torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        ids, unused_history, label_parts, sources = load_identity(args.crop, args.cutoff, with_history=False)
        assert unused_history is None
        population = state_population(ids, label_parts, args.cutoff)
        full_indices = population['full_fit']
        known, targets = load_seasons(args.crop, ids, full_indices, sources, maximum_year=args.cutoff)
        source_guard(sources, args.cutoff)
        a = state_arrays(known, targets)
        parts = temporal_indices(a['year'], args.cutoff, args.smoke)
        fit, val, full = [parts[key] for key in ('inner_fit', 'inner_validation', 'full_fit')]
        sources[str(gate)] = sha256(gate)
        code = code_hashes()
        spec = dict(schema=1, job=job, training=TRAINING, sources=sources, code_sha256=code,
            architecture='Original ForecastState(biid): retain12, dim128, 2 BIID layers plus original cross-attention/gate',
            context_fields=['latitude/90', 'longitude/180', '(year-2000)/40', 'SOS/366', 'EOS/366'],
            inner_identity=identity_hash(full_indices[fit]), inner_validation_identity=identity_hash(full_indices[val]),
            full_identity=identity_hash(full_indices[full]), selection_years=[args.cutoff-1, args.cutoff],
            rows={key: len(ix) for key, ix in parts.items()}, maximum_loaded_season_year=int(a['year'].max()),
            numeric_yield_arrays_loaded=False, evaluation_arrays_loaded=False,
            current_state_targets_in_forward=False, target_quality_in_forward=False,
            region_population='Seen in inner yield fit; vegetation years may lack a numeric yield label',
            selection='Equal-year physical vegetation RMSE on the preceding two years, then fresh full refit',
            model_optimization=False)
        register(root, spec)
        np.savez_compressed(root / 'partition_indices.npz', **{key: full_indices[ix] for key, ix in parts.items()})
        ids.iloc[full_indices[full]].to_csv(root / 'state_identities.csv', index=False)
        start = time.monotonic()
        stats = fit_state_stats(subset(known, fit), subset(targets, fit), (args.product,), args.cutoff-2)
        atomic_json(root / 'inner_normalization.json', stats)
        set_seed(args.seed)
        model = ForecastState('biid').cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
        rng = np.random.default_rng(args.seed)
        best, epoch_best, stale, trace, best_prediction = float('inf'), 0, 0, [], None
        for epoch in range(1, (1 if args.smoke else TRAINING['epochs'])+1):
            loss = train_epoch(model, a, fit, stats, args.product, optimizer, rng, TRAINING['batch'])
            prediction = predict(model, a, val, stats, args.product, TRAINING['batch'])
            score, annual = mean_metric(a, val, prediction, args.product)
            trace.append(dict(stage='inner_selection', epoch=epoch, loss=loss, rmse=annual, score=score))
            if score < best:
                best, epoch_best, stale, best_prediction = score, epoch, 0, prediction.copy()
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, root / 'inner_best.pt')
            else:
                stale += 1
            atomic_json(root / 'training_history.json', trace)
            print(f'[REGIONAL STATE INNER] {job} epoch={epoch} rmse={score:.6f}', flush=True)
            if stale >= TRAINING['patience']:
                break
        if epoch_best < 1:
            raise ValueError('No trained checkpoint selected')
        model.load_state_dict(torch.load(root / 'inner_best.pt', map_location='cpu', weights_only=True))
        np.testing.assert_array_equal(best_prediction, predict(model, a, val, stats, args.product, TRAINING['batch']))
        np.savez_compressed(root / 'inner_validation_predictions.npz', sample_index=full_indices[val],
            region_index=known['region_index'][val], year=a['year'][val], prediction=best_prediction,
            target=a[f'observed_{args.product}'][val], active=a['relative_valid'][val])
        del optimizer, model
        torch.cuda.empty_cache()
        full_stats = fit_state_stats(subset(known, full), subset(targets, full), (args.product,), args.cutoff)
        atomic_json(root / 'normalization.json', full_stats)
        set_seed(args.seed)
        model = ForecastState('biid').cuda()
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, epoch_best+1):
            loss = train_epoch(model, a, full, full_stats, args.product, optimizer, rng, TRAINING['batch'])
            trace.append(dict(stage='full_refit', epoch=epoch, loss=loss))
            atomic_json(root / 'training_history.json', trace)
            print(f'[REGIONAL STATE REFIT] {job} epoch={epoch}/{epoch_best} loss={loss:.6f}', flush=True)
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        changed = sum(not torch.equal(value, initial[key]) for key, value in weights.items())
        if not changed:
            raise ValueError('Training left all state weights at initialization')
        torch.save(weights, root / 'model.pt')
        del optimizer, initial
        sample = full[np.linspace(0, len(full)-1, min(32, len(full)), dtype=int)]
        prediction = predict(model, a, sample, full_stats, args.product, TRAINING['batch'])
        restored = ForecastState('biid').cuda()
        restored.load_state_dict(torch.load(root / 'model.pt', map_location='cpu', weights_only=True))
        np.testing.assert_array_equal(prediction, predict(restored, a, sample, full_stats, args.product, TRAINING['batch']))
        inputs = batch_arrays(a, sample, full_stats, args.product)
        np.savez_compressed(root / 'replay_inputs.npz', sample_index=full_indices[sample], **inputs, prediction=prediction)
        for path, digest in sources.items():
            if sha256(Path(path)) != digest:
                raise ValueError('Regional source changed during fit')
        if code_hashes() != code:
            raise ValueError('Regional training code changed during fit')
        finish(root, code, job=job, selected_epochs=epoch_best, inner_state_rmse=best,
            full_training_rows=len(full), inner_fit_rows=len(fit), validation_rows=len(val),
            seconds=time.monotonic()-start, changed_parameter_tensors=changed, maximum_replay_error=0.,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            numeric_yield_arrays_loaded=False, evaluation_arrays_loaded=False,
            maximum_loaded_season_year=int(a['year'].max()), external_performance_evaluated=False)
        verify_completed(**job)
        print(f'[REGIONAL STATE COMPLETE] {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=RECIPES, required=True)
    parser.add_argument('--product', choices=('ndvi', 'gpp'), required=True)
    parser.add_argument('--cutoff', type=int, choices=BLOCKS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
