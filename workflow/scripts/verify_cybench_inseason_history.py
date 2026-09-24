"""Reconstruct historical inputs, replay all candidates, and freeze inner-only experts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cybench_inseason_model_data import ROOT, RESULT
from cybench_inseason_yield_models import MODELS, load_data
from run_cybench_inseason_history import root_for, verify_completed
from run_cybench_inseason_history_queue import jobs
from run_cybench_inseason_state_queue import process_identity
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json


def country_year_score(target, prediction, ids):
    records = []
    for country in sorted(ids.country.unique()):
        years = ids.loc[ids.country == country, 'year'].unique()
        for year in sorted(years):
            keep = ((ids.country == country) & (ids.year == year)).to_numpy()
            error = np.asarray(prediction, float)[keep]-np.asarray(target, float)[keep]
            records.append(dict(country=country, year=int(year), samples=int(keep.sum()),
                rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error)))))
    frame = pd.DataFrame(records)
    return float(frame.groupby('country').rmse.mean().mean()), records


def weights(ids):
    countries = ids.country.unique()
    output = np.empty(len(ids), float)
    for country in countries:
        country_mask = (ids.country == country).to_numpy()
        years = ids.loc[country_mask, 'year'].unique()
        for year in years:
            keep = country_mask & (ids.year.to_numpy() == year)
            output[keep] = 1/(len(countries)*len(years)*int(keep.sum()))
    return output/output.mean()


def verify_normalization(raw, target, anchor, ids, saved):
    mean, std = [], []
    for col in np.asarray(raw, np.float64).T:
        finite = col[np.isfinite(col)]
        mean.append(float(finite.mean()) if len(finite) else 0.)
        std.append(max(float(finite.std()), 1e-6) if len(finite) else 1e-6)
    np.testing.assert_allclose(saved['features']['history_mean'], mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(saved['features']['history_std'], std, rtol=1e-12, atol=1e-12)
    w = weights(ids)
    residual = target-anchor
    center = np.sum(w*residual)/w.sum()
    scale = max(float(np.sqrt(np.sum(w*(residual-center)**2)/w.sum())), 1e-6)
    np.testing.assert_allclose([saved['residual']['center'], saved['residual']['scale']], [center, scale], rtol=1e-12, atol=1e-12)


def matrix(raw, norm):
    scaled = (raw-np.asarray(norm['features']['history_mean']))/np.asarray(norm['features']['history_std'])
    return np.nan_to_num(scaled, nan=0., posinf=0., neginf=0.).astype(np.float32)


def replay(path, model_name, x, anchor, norm):
    model = joblib.load(path)
    if model_name == 'lightgbm':
        raw = model.booster_.predict(x, num_threads=2)
    else:
        if model_name == 'random_forest':
            assert model.n_jobs == 1 and len(model.estimators_) > 0
        raw = model.predict(x).astype(np.float64)
    p = np.maximum(anchor+norm['residual']['center']+norm['residual']['scale']*raw, 0)
    if not np.isfinite(p).all():
        raise FloatingPointError('Nonfinite historical replay')
    return p


def choose_expert(scores):
    if set(scores) != set(MODELS) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('All three finite inner scores are required')
    return min(MODELS, key=lambda name: scores[name])


def wait_for_models(smoke):
    stage = 'history_smoke' if smoke else 'history_full'
    marker = RESULT / f'{stage}_queue_complete.json'
    while not marker.exists():
        launcher = RESULT / 'history_launcher.json'
        if smoke or not launcher.exists():
            raise RuntimeError('Regional historical models have not completed')
        record = json.loads(launcher.read_text())
        current = process_identity(record['pid'])
        if current is None or current != record['process']:
            if marker.exists():
                break
            raise RuntimeError('Historical queue is no longer live; inspect its recorded failures')
        print(f'[WAIT VERIFIED HISTORY PROCESS] pid={record["pid"]}', flush=True)
        time.sleep(10)
    return marker


def run(smoke=False, wait=False):
    stage = 'history_smoke' if smoke else 'history_full'
    gate = wait_for_models(smoke) if wait else RESULT / f'{stage}_queue_complete.json'
    summary = json.loads(gate.read_text())
    planned = jobs(smoke)
    if not summary['passed'] or summary['jobs'] != planned:
        raise ValueError('Complete registered historical matrix required')
    sources = {str(gate): sha256(gate)}
    annual_rows, expert_candidates, rows_replayed = [], {}, 0
    for job in planned:
        record = verify_completed(**job)
        root = root_for(**job)
        data = load_data(job['crop'], job['cutoff'], smoke, vegetation=False)
        ids, y, anchor, parts = [data[k] for k in ('identities', 'target', 'trend', 'parts')]
        np.testing.assert_array_equal(anchor, np.maximum(data['history'][:, 13].astype(float), 0))
        sources.update(data['sources'])
        with np.load(root / 'partition_indices.npz') as saved:
            for key, ix in parts.items():
                np.testing.assert_array_equal(saved[key], data['sample_index'][ix])
        norms = [json.loads((root / name).read_text()) for name in ('inner_normalization.json', 'normalization.json')]
        for norm, part in zip(norms, ('inner_fit', 'full_fit')):
            ix = parts[part]
            verify_normalization(data['history'][ix], y[ix], anchor[ix], ids.iloc[ix], norm)
        val, ev = parts['inner_validation'], parts['evaluation']
        selection = json.loads((root / 'inner/selection.json').read_text())
        candidates = []
        for candidate in selection['candidates']:
            p = replay(root / 'inner' / candidate['weight'], job['model'], matrix(data['history'][val], norms[0]), anchor[val], norms[0])
            score, annual = country_year_score(y[val], p, ids.iloc[val])
            np.testing.assert_allclose(score, candidate['score'], rtol=0, atol=1e-12)
            candidates.append(score)
            if candidate['index'] == selection['selected_candidate']:
                with np.load(root / 'inner_validation_predictions.npz') as saved:
                    np.testing.assert_array_equal(saved['sample_index'], data['sample_index'][val])
                    np.testing.assert_array_equal(saved['target'], y[val])
                    np.testing.assert_array_equal(saved['prediction'], p)
            rows_replayed += len(val)
        assert int(np.argmin(candidates)) == selection['selected_candidate']
        np.testing.assert_allclose(min(candidates), selection['score'], rtol=0, atol=1e-12)
        p = replay(root / 'model.joblib', job['model'], matrix(data['history'][ev], norms[1]), anchor[ev], norms[1])
        with np.load(root / 'evaluation_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['sample_index'], data['sample_index'][ev])
            np.testing.assert_array_equal(saved['target'], y[ev])
            np.testing.assert_array_equal(saved['prediction'], p)
        csv = pd.read_csv(root / 'evaluation_identities.csv', dtype={'country': str, 'adm_id': str})
        pd.testing.assert_frame_equal(csv, ids.iloc[ev].reset_index(drop=True), check_dtype=False)
        score, annual = country_year_score(y[ev], p, ids.iloc[ev])
        metrics = json.loads((root / 'metrics.json').read_text())
        np.testing.assert_allclose(score, metrics['score'], rtol=0, atol=1e-12)
        measured = pd.DataFrame(annual).set_index(['country', 'year'])
        stored = pd.DataFrame(metrics['annual']).set_index(['country', 'year'])
        pd.testing.assert_frame_equal(measured, stored[measured.columns], check_exact=False, rtol=0, atol=1e-12)
        annual_rows.extend(dict(**{k: job[k] for k in ('crop', 'cutoff', 'seed', 'model')}, **row) for row in annual)
        key = (job['crop'], job['cutoff'], job['seed'])
        expert_candidates.setdefault(key, {})[job['model']] = selection['score']
        rows_replayed += len(ev)
        for name, digest in record['files'].items():
            sources[str(root / name)] = digest
        sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
        print(f'[REGIONAL HISTORY VERIFIED] {job} inner={selection["score"]:.6f}', flush=True)
    experts = []
    for (crop, cutoff, seed), scores in expert_candidates.items():
        winner = choose_expert(scores)
        root = root_for(crop, cutoff, seed, winner, smoke)
        config = json.loads((root / 'inner/selection.json').read_text())
        experts.append(dict(crop=crop, cutoff=cutoff, seed=seed, model=winner, inner_scores=scores,
            full_weight=str(root / 'model.joblib'), full_normalization=str(root / 'normalization.json'),
            inner_weight=config['selected_weight'], inner_normalization=str(root / 'inner_normalization.json'),
            completed_model=str(root / 'complete.json'), completed_model_sha256=sha256(root / 'complete.json'),
            selection='Only preceding two-year country-balanced RMSE; fixed family-order tie break'))
    destination = RESULT / ('history_smoke_summary' if smoke else 'history_summary')
    destination.mkdir(parents=True, exist_ok=True)
    annual = pd.DataFrame(annual_rows)
    expert_annual = []
    for expert in experts:
        selected = annual[(annual.crop == expert['crop']) & (annual.cutoff == expert['cutoff'])
            & (annual.seed == expert['seed']) & (annual.model == expert['model'])].copy()
        selected['model'] = 'validation_selected_expert'
        selected['selected_family'] = expert['model']
        expert_annual.append(selected)
    annual = pd.concat([annual, *expert_annual], ignore_index=True)
    annual.to_csv(destination / 'annual.csv', index=False)
    mean = annual.groupby(['crop', 'country', 'seed', 'model']).agg(
        rmse=('rmse', 'mean'), mae=('mae', 'mean'), years=('year', 'nunique'), samples=('samples', 'sum')).reset_index()
    mean.to_csv(destination / 'per_seed.csv', index=False)
    atomic_json(destination / 'experts.json', experts)
    for name in ('annual.csv', 'per_seed.csv', 'experts.json'):
        sources[str(destination / name)] = sha256(destination / name)
    for path, digest in sources.items():
        if sha256(Path(path)) != digest:
            raise ValueError('Historical source changed during independent verification')
    output = RESULT / ('history_smoke_verification.json' if smoke else 'history_verification.json')
    atomic_json(output, dict(passed=True, timestamp=datetime.now().astimezone().isoformat(), models=len(planned),
        trained_families=list(MODELS), experts=len(experts), annual_rows=len(annual),
        prediction_rows_replayed=rows_replayed, maximum_weight_replay_error=0.,
        independent_normalization_rebuilt=True, selection_uses_outer_scores=False,
        external_world_performance_evaluated=False, sources=sources, verifier_sha256=sha256(Path(__file__))))
    print(f'[REGIONAL HISTORICAL AUDIT COMPLETE] models={len(planned)} experts={len(experts)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.smoke, args.wait)
