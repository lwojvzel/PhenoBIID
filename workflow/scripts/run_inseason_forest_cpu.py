"""Identical seasonal forests with a separate, eight-thread runtime registry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_inseason_direct_extension import (ROOT, CACHE, BLOCKS, RECIPES,
    CODE as BASE_CODE, load, partition, make_design, flat_features, build_model,
    year_weights, score, sha256, atomic_json, check_files)

OUT = ROOT / 'benchmark/results/inseason_direct_forest_cpu_v2'
CODE = (*BASE_CODE,'run_inseason_forest_cpu.py')


def root_for(crop,cutoff,model='random_forest',smoke=False):
    if model != 'random_forest':
        raise ValueError('This runtime adapter only fits Random Forest')
    return OUT / ('smoke' if smoke else 'pipelines') / crop / f'cutoff_{cutoff}/seed_42/random_forest'


def verify(root):
    marker = json.loads((root / 'complete.json').read_text())
    check_files(root,marker['files'])
    if marker['code_sha256'] != {name:sha256(ROOT / 'scripts' / name) for name in CODE}:
        raise ValueError('Changed eight-thread forest source')
    if marker['smoke']:
        raise ValueError('Smoke is not a final forest')
    return marker


def run(crop,cutoff,smoke=False):
    root = root_for(crop,cutoff,smoke=smoke)
    root.mkdir(parents=True,exist_ok=True)
    code = {name:sha256(ROOT / 'scripts' / name) for name in CODE}
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'complete.json').exists() and not smoke:
            verify(root)
            return
        raw,_ = load(crop)
        groups = partition(raw,cutoff)
        if smoke:
            groups = {name:np.concatenate([ix[raw['year'][ix]==year][:16]
                for year in np.unique(raw['year'][ix])]) for name,ix in groups.items()}
        ratios = [.1] if smoke else [.1,.3,.5]
        params = build_model('random_forest',42,8).get_params()
        if smoke:
            params['n_estimators'] = 8
        config = dict(crop=crop,cutoff=cutoff,model='random_forest',seed=42,smoke=smoke,
            code_sha256=code,cache_sha256=sha256(CACHE / crop / 'manifest.json'),
            recipe=RECIPES[crop],ratios=ratios,forest=params,
            periods={name:np.unique(raw['year'][ix]).tolist() for name,ix in groups.items()},
            input_builder='Unchanged run_inseason_13year_direct.make_design',
            runtime_change='Eight instead of two fit threads; identical trees, inputs, targets and seed',
            objective='Year-balanced standardized causal-trend residual MSE',
            original_two_thread_partials_retained=True,selection='Fixed forest capacity')
        config = json.loads(json.dumps(config))
        path = root / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Changed forest runtime registration')
        atomic_json(path,config)
        started = time.monotonic()
        for ratio in ratios:
            dest = root / f'tail_{round(100*ratio):02d}'
            dest.mkdir(exist_ok=True)
            if (dest / 'audit.json').exists():
                check_files(dest,json.loads((dest / 'audit.json').read_text())['files'])
                continue
            inputs,residual,norm = make_design(raw,groups['full_fit'],groups['evaluation'],RECIPES[crop],ratio)
            xf,xv = flat_features(inputs['train']),flat_features(inputs['other'])
            model = build_model('random_forest',42,8)
            model.set_params(**params)
            with threadpool_limits(limits=2):
                model.fit(xf,residual['train'],sample_weight=year_weights(raw['year'][groups['full_fit']]))
                model.set_params(n_jobs=1)
                output = model.predict(xv).astype(float)
                joblib.dump(model,dest / 'model.joblib')
                restored = joblib.load(dest / 'model.joblib')
                np.testing.assert_array_equal(output,restored.predict(xv))
            prediction = inputs['other']['anchor']+norm['center']+norm['scale']*output
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite forest prediction')
            labels = {key:raw[key][groups['evaluation']] for key in ('year','row','col','target','source_indices')}
            np.savez_compressed(dest / 'evaluation_predictions.npz',prediction=prediction,**labels)
            metrics = score(labels['target'],prediction,labels['year'])
            atomic_json(dest / 'metrics.json',dict(scores=metrics,selection=dict(fixed_parameters=True),
                full_normalization=norm,sequence_shape=list(inputs['train']['sequence'].shape[1:]),
                static_width=inputs['train']['static'].shape[1],fit_threads=8,inference_threads=1))
            files = {p.name:sha256(p) for p in dest.iterdir() if p.is_file()}
            atomic_json(dest / 'audit.json',dict(files=files,weight_replay=True,
                future_values_and_quality_excluded=True,all_fitting_before_evaluation=True))
            print(f'[FOREST CPU] {crop} {cutoff} {ratio} annual={metrics["mean_annual_rmse"]:.6f}',flush=True)
            del inputs,residual,xf,xv,model,restored
        files = {str(p.relative_to(root)):sha256(p) for p in root.rglob('*')
            if p.is_file() and p.name not in ('run.lock','complete.json')}
        atomic_json(root / 'complete.json',dict(files=files,code_sha256=code,smoke=smoke,
            final_models=len(ratios),seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop',choices=(*RECIPES,'all'),required=True)
    parser.add_argument('--cutoff',type=int,choices=tuple(BLOCKS))
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.crop == 'all':
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(run,crop,cutoff,args.smoke) for crop in RECIPES for cutoff in BLOCKS]:
                future.result()
    elif args.cutoff is None:
        parser.error('--cutoff is required for one crop')
    else:
        run(args.crop,args.cutoff,args.smoke)
