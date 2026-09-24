"""Forward-generated seasonal states with matched downstream training controls."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from forecast_bridge_data import (ROOT, fit_stats, climatology, remote_features,
                                  history_features)
from forecast_bridge_state import ForecastState
from inseason_direct_data import RECIPES, physical_inputs, seasonal_features, split_rows
from inseason_signal_matching import remap_product
from ndvi_tail_replacement import tail_mask, mix_trajectory
from review_revision_data import sha256
from run_crop_signal_screen import year_weights
from run_forecast_bridge_state import run_root as state_root
from run_inseason_direct_baselines import score
from run_ndvi_signal_permutation import check_files
from run_ndvi_tail_replacement import forecast_prefix
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'benchmark/results/inseason_matched_readout_v1'
RATIOS = (.1, .3, .5)
CODE = ('run_inseason_matched_readout.py', 'inseason_direct_data.py',
        'forecast_bridge_data.py', 'forecast_bridge_state.py',
        'run_ndvi_tail_replacement.py', 'ndvi_tail_replacement.py',
        'inseason_signal_matching.py', 'run_inseason_direct_baselines.py')


def upstream_cutoffs(years):
    years = np.asarray(years)
    if np.any(years < 1993):
        raise ValueError('No trained forward state before the first 1993 block')
    result = np.where(years <= 2009, 1992 + 3*((years-1993)//3), 2009)
    if np.any(result >= years):
        raise ValueError('Upstream state is not strictly earlier than the target year')
    return result


def verify_done(root):
    marker = json.loads((root / 'complete.json').read_text())
    check_files(root, marker['files'])
    for name, digest in marker['code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError('Changed readout implementation')


def export_states(crop, a, take, root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        verify_done(root)
        with np.load(root / 'identities.npz') as saved:
            np.testing.assert_array_equal(a['source_indices'][take], saved['source_indices'])
        return
    years = a['year'][take]
    cutoff = upstream_cutoffs(years)
    products = RECIPES[crop].split('_')
    outputs = {ratio: {p: np.full((len(take), 12), np.nan, np.float32)
                       for p in products} for ratio in RATIOS}
    climo = {p: np.full((len(take), 12), np.nan, np.float32) for p in products}
    sources, checks = {}, []
    for end in np.unique(cutoff):
        local = np.flatnonzero(cutoff == end)
        index = take[local]
        active = a['relative_valid'][index] > 0
        for product in products:
            state = state_root(crop, product, 'biid', int(end), 42)
            marker = json.loads((state / 'complete.json').read_text())
            if marker['smoke'] or marker['selected_epochs'] < 1 or marker['full_fit_cutoff'] != end:
                raise ValueError('Untrained or wrong-cutoff state checkpoint')
            check_files(state, marker['files'])
            cfg = json.loads((state / 'config.json').read_text())
            for name, digest in cfg['code_sha256'].items():
                if sha256(ROOT / 'scripts' / name) != digest:
                    raise ValueError('Changed upstream implementation')
            stats = json.loads((state / 'normalization.json').read_text())
            if max(stats['fit_years']) > end:
                raise ValueError('Future state normalization')
            model = ForecastState('biid').cuda().eval()
            model.load_state_dict(torch.load(state / 'model.pt', map_location='cpu', weights_only=True))
            mapped = remap_product(a, product)
            stats = dict(stats, ndvi=stats[product])
            observed = a[f'observed_{product}'][index]
            for ratio in RATIOS:
                tail = tail_mask(active, ratio)
                outputs[ratio][product][local] = forecast_prefix(
                    model, mapped, index, stats, observed, tail, active)
            fit = np.flatnonzero(a['year'] <= end)
            climo[product][local] = climatology(a, fit, index, product)
            checks.append(dict(product=product, cutoff=int(end),
                               predicted_years=np.unique(years[local]).tolist(),
                               trained_epochs=marker['selected_epochs'], rows=len(index)))
            for name in ('complete.json', 'model.pt', 'normalization.json'):
                sources[str(state / name)] = sha256(state / name)
            del model
            torch.cuda.empty_cache()
            print(f'[FORWARD PREFIX] {crop} {product} cutoff={end} rows={len(index)}', flush=True)
    for product in products:
        if not np.isfinite(climo[product]).all():
            raise ValueError('Missing forward climatology')
        for ratio in RATIOS:
            if not np.isfinite(outputs[ratio][product]).all():
                raise ValueError('Incomplete forward prediction')
    np.savez_compressed(root / 'identities.npz', raw_indices=take, cutoff=cutoff,
                        **{k: a[k][take] for k in ('source_indices', 'year', 'row', 'col')})
    np.savez_compressed(root / 'climatology.npz', **climo)
    for ratio in RATIOS:
        np.savez_compressed(root / f'prefix_{round(ratio*100):02d}.npz', **outputs[ratio])
    atomic_json(root / 'upstream.json', dict(sources=sources, checks=checks,
                all_state_epochs_positive=True, all_cutoffs_strictly_past=True))
    files = {p.name: sha256(p) for p in root.iterdir() if p.is_file()}
    atomic_json(root / 'complete.json', dict(files=files, code_sha256={
        n: sha256(ROOT / 'scripts' / n) for n in CODE}))


def completion_features(base, observed, forecast, tail, active, climo, scales, products):
    values = [base]
    for p in products:
        mixed = mix_trajectory(observed[p], forecast[p], tail)
        values.append(remote_features(mixed, active, climo[p], scales[p]))
    return np.concatenate(values, 1).astype(np.float32)


def run(crop):
    root = OUT / 'pipelines' / crop / 'fit_2009/seed_42'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists():
            verify_done(root)
            return
        start = time.monotonic()
        a, provenance = physical_inputs(crop)
        take = np.flatnonzero(a['year'] >= 1993)
        config = dict(crop=crop, seed=42, recipe=RECIPES[crop],
            fit_years=[1993, 2009], validation_years=[2010, 2011, 2012],
            test_years=[2013, 2014, 2015, 2016], ratios=RATIOS,
            conditions=['prefix_only', 'climatology', 'biid'],
            history='Identical causal-trend anchor in every condition; no neural anchor',
            selection='Validation annual RMSE for tree count; test never selects',
            purpose='Does training on forward-predicted states improve terminal utility?',
            old_head_changed=True, state_weights_changed=False,
            warmup='All three controls exclude pre-1993 rows; earliest state cutoff=1992',
            code_sha256={n: sha256(ROOT / 'scripts' / n) for n in CODE})
        config = json.loads(json.dumps(config))
        if (root / 'config.json').exists() and json.loads((root / 'config.json').read_text()) != config:
            raise ValueError('Changed registered paired-readout protocol')
        atomic_json(root / 'config.json', config)
        atomic_json(root / 'provenance.json', provenance)
        state_cache = root / 'forward_prefix'
        export_states(crop, a, take, state_cache)
        rows = split_rows(a, 2009)
        rows['train'] = rows['train'][a['year'][rows['train']] >= 1993]
        stats = fit_stats(a, rows['train'])
        atomic_json(root / 'normalization.json', stats)
        products = RECIPES[crop].split('_')
        climate = {s: {p: climatology(a, rows['train'], ix, p) for p in products}
                   for s, ix in rows.items()}
        anchors = {s: history_features(a, ix, stats)[1].astype(float) for s, ix in rows.items()}
        residual = {s: (a['target'][ix]-anchors[s])/stats['target_std'] for s, ix in rows.items()}
        cached_ix = {s: np.searchsorted(take, ix) for s, ix in rows.items()}
        for s, ix in rows.items():
            np.testing.assert_array_equal(take[cached_ix[s]], ix)
        with np.load(state_cache / 'climatology.npz') as f:
            forward_climo = {p: f[p] for p in products}
        params = dict(n_estimators=1200, learning_rate=.03, num_leaves=15,
                      min_child_samples=200, reg_lambda=10., subsample=.9,
                      subsample_freq=1, colsample_bytree=.9, n_jobs=2,
                      random_state=42, verbosity=-1, deterministic=True,
                      force_col_wise=True, objective='regression', metric='None')
        atomic_json(root / 'tree_parameters.json', params)
        weights = year_weights(a['year'][rows['train']])
        records = []
        for ratio in RATIOS:
            with np.load(state_cache / f'prefix_{round(ratio*100):02d}.npz') as f:
                forward_biid = {p: f[p] for p in products}
            base, tails = {}, {}
            for split, ix in rows.items():
                base[split], tails[split] = seasonal_features(
                    a, ix, stats, RECIPES[crop], ratio, climate[split])
            for condition in config['conditions']:
                x = {}
                for split, ix in rows.items():
                    if condition == 'prefix_only':
                        x[split] = base[split]
                    else:
                        source = forward_biid if condition == 'biid' else forward_climo
                        x[split] = completion_features(base[split],
                            {p: a[f'observed_{p}'][ix] for p in products},
                            {p: source[p][cached_ix[split]] for p in products},
                            tails[split], a['relative_valid'][ix]>0, climate[split], stats, products)

                def metric(truth, prediction):
                    return ('annual_rmse', score(truth, prediction,
                            a['year'][rows['validation']])['mean_annual_rmse'], False)

                trace = {}
                with threadpool_limits(limits=2):
                    model = lgb.LGBMRegressor(**params)
                    model.fit(x['train'], residual['train'], sample_weight=weights,
                        eval_set=[(x['validation'], residual['validation'])], eval_metric=metric,
                        callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False),
                                   lgb.record_evaluation(trace)])
                    dest = root / f'tail_{round(ratio*100):02d}' / condition
                    dest.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, dest / 'model.joblib')
                    restored = joblib.load(dest / 'model.joblib')
                    metrics = {}
                    for split in ('validation', 'test'):
                        ix = rows[split]
                        prediction = anchors[split]+stats['target_std']*model.booster_.predict(x[split])
                        replay = anchors[split]+stats['target_std']*restored.booster_.predict(x[split])
                        np.testing.assert_array_equal(prediction, replay)
                        labels = {k: a[k][ix] for k in ('target', 'year', 'row', 'col', 'source_indices')}
                        np.savez_compressed(dest / f'{split}_predictions.npz', prediction=prediction, **labels)
                        metrics[split] = score(labels['target'], prediction, labels['year'])
                record = dict(crop=crop, ratio=ratio, condition=condition,
                    selected_trees=int(model.best_iteration_), dimensions=x['train'].shape[1],
                    n_train=len(rows['train']), n_validation=len(rows['validation']),
                    metrics=metrics, model=str(dest / 'model.joblib'))
                atomic_json(dest / 'training_history.json', trace)
                atomic_json(dest / 'metrics.json', record)
                records.append(record)
                print(f'[MATCHED READOUT] {crop} {ratio:.1f} {condition} '
                      f'val={metrics["validation"]["pooled_rmse"]:.6f} '
                      f'test={metrics["test"]["pooled_rmse"]:.6f}', flush=True)
        atomic_json(root / 'summary.json', records)
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
                 if p.is_file() and p.name != 'run.lock' and p != root / 'complete.json'}
        atomic_json(root / 'complete.json', dict(files=files, code_sha256=config['code_sha256'],
                    seconds=time.monotonic()-start, trained_readouts=len(records),
                    state_fits=0, all_upstream_strictly_past=True, test_used_for_selection=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops', nargs='+', choices=tuple(RECIPES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    for crop in args.crops:
        run(crop)
