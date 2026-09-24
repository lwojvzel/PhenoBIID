"""Replay regional readouts and audit prefix isolation before downstream use."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cybench_inseason_model_data import ROOT, RESULT, RECIPES
from cybench_inseason_yield_models import load_data
from cybench_seasonal_inputs import issue_masks
from run_cybench_inseason_readout import (root_for, verify_completed, design,
    history_expert, historical_prediction)
from run_cybench_inseason_readout_queue import jobs
from run_cybench_inseason_state_queue import process_identity
from verify_cybench_inseason_history import verify_normalization, country_year_score, replay
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

SCRIPT = Path(__file__).resolve()


def output_for(smoke=False, scope='all'):
    return RESULT / f'{"readout_smoke" if smoke else "readout"}_{scope}_verification.json'


def planned_jobs(smoke=False, scope='all'):
    if scope not in ('all', 'terminal'):
        raise ValueError('Unregistered readout audit scope')
    return [job for job in jobs(smoke) if scope == 'all' or job['route'] == 'terminal']


def wait_for_models(planned, wait):
    while not all((root_for(**job) / 'complete.json').exists() for job in planned):
        if not wait:
            raise RuntimeError('Registered readout matrix is incomplete')
        path = RESULT / 'readout_launcher.json'
        record = json.loads(path.read_text())
        current = process_identity(record['pid'])
        if current is None or current != record['process']:
            if all((root_for(**job) / 'complete.json').exists() for job in planned):
                break
            raise RuntimeError('Readout queue is not live; inspect its recorded failures')
        print(f'[WAIT VERIFIED READOUT PROCESS] pid={record["pid"]}', flush=True)
        time.sleep(10)


def verify_record(smoke=False, scope='all'):
    path = output_for(smoke, scope)
    record = json.loads(path.read_text())
    if not record['passed'] or record['jobs'] != planned_jobs(smoke, scope) or record['verifier_sha256'] != sha256(SCRIPT):
        raise ValueError('Readout verification scope or implementation changed')
    for filename, digest in record['sources'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Verified readout source changed')
    return record


def run(smoke=False, scope='all', wait=False):
    if output_for(smoke, scope).exists():
        verify_record(smoke, scope)
        print(f'[READOUT AUDIT REUSE] {output_for(smoke, scope)}', flush=True)
        return
    planned = planned_jobs(smoke, scope)
    wait_for_models(planned, wait)
    registration = RESULT / f'{"readout_smoke" if smoke else "readout_full"}_registration.json'
    registered = json.loads(registration.read_text())
    if registered['jobs'] != jobs(smoke) or registered['queue_sha256'] != sha256(ROOT / 'scripts/run_cybench_inseason_readout_queue.py'):
        raise ValueError('Readout training registration changed')
    sources = {str(registration): sha256(registration)}
    annual_rows, replayed_rows, poison_checks = [], 0, 0
    for job in planned:
        record = verify_completed(**job)
        root = root_for(**job)
        data = load_data(job['crop'], job['cutoff'], smoke, vegetation=True)
        ids, target, parts = [data[k] for k in ('identities', 'target', 'parts')]
        fit, val, full, ev = [parts[k] for k in ('inner_fit', 'inner_validation', 'full_fit', 'evaluation')]
        expert = history_expert(job['crop'], job['cutoff'], job['seed'], data['sources'], smoke) if job['route'] == 'terminal' else None
        inner_anchor = historical_prediction(data, expert, 'inner') if expert else data['trend']
        full_anchor = historical_prediction(data, expert, 'full') if expert else data['trend']
        with np.load(root / 'history_anchors.npz') as saved:
            np.testing.assert_array_equal(saved['sample_index'], data['sample_index'])
            np.testing.assert_array_equal(saved['inner'], inner_anchor)
            np.testing.assert_array_equal(saved['full'], full_anchor)
        with np.load(root / 'partition_indices.npz') as saved:
            for key, ix in parts.items():
                np.testing.assert_array_equal(saved[key], data['sample_index'][ix])
        selection = json.loads((root / 'inner/selection.json').read_text())
        norm_inner = json.loads((root / 'inner_normalization.json').read_text())
        x, rebuilt = design(data, fit, val, RECIPES[job['crop']], job['percent'], job['cutoff']-2)
        assert rebuilt == norm_inner['features']
        verify_normalization(data['history'][fit], target[fit], inner_anchor[fit], ids.iloc[fit], norm_inner)
        candidate_scores = []
        for candidate in selection['candidates']:
            p = replay(root / 'inner' / candidate['weight'], job['model'], x['validation'], inner_anchor[val], norm_inner)
            score, _ = country_year_score(target[val], p, ids.iloc[val])
            np.testing.assert_allclose(score, candidate['score'], rtol=0, atol=1e-12)
            candidate_scores.append(score)
            if candidate['index'] == selection['selected_candidate']:
                with np.load(root / 'inner_validation_predictions.npz') as saved:
                    np.testing.assert_array_equal(saved['sample_index'], data['sample_index'][val])
                    np.testing.assert_array_equal(saved['target'], target[val])
                    np.testing.assert_array_equal(saved['prediction'], p)
            replayed_rows += len(val)
        assert selection['selected_candidate'] == int(np.argmin(candidate_scores))
        np.testing.assert_allclose(selection['score'], min(candidate_scores), rtol=0, atol=1e-12)
        norm = json.loads((root / 'normalization.json').read_text())
        x, rebuilt = design(data, full, ev, RECIPES[job['crop']], job['percent'], job['cutoff'])
        assert rebuilt == norm['features']
        verify_normalization(data['history'][full], target[full], full_anchor[full], ids.iloc[full], norm)
        p = replay(root / 'model.joblib', job['model'], x['validation'], full_anchor[ev], norm)
        with np.load(root / 'evaluation_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['sample_index'], data['sample_index'][ev])
            np.testing.assert_array_equal(saved['target'], target[ev])
            np.testing.assert_array_equal(saved['prediction'], p)
        with np.load(root / 'replay_features.npz') as saved:
            np.testing.assert_array_equal(saved['features'], x['validation'][:32])
        score, annual = country_year_score(target[ev], p, ids.iloc[ev])
        metrics = json.loads((root / 'metrics.json').read_text())
        np.testing.assert_allclose(score, metrics['score'], rtol=0, atol=1e-12)
        measured = pd.DataFrame(annual).set_index(['country', 'year'])
        stored = pd.DataFrame(metrics['annual']).set_index(['country', 'year'])
        pd.testing.assert_frame_equal(measured, stored[measured.columns], check_exact=False, rtol=0, atol=1e-12)
        if job['route'] == 'direct':
            _, hidden = issue_masks(data['known']['active_mask'][ev], job['percent'])
            row, slot = np.nonzero(hidden)
            poisoned = {k: v.copy() for k, v in data['state_targets'].items()}
            poisoned['vegetation'][ev[row], slot, :] = 9876.
            poisoned['vegetation_support'][ev[row], slot, :] = .137
            altered, altered_norm = design(dict(data, state_targets=poisoned), full, ev,
                RECIPES[job['crop']], job['percent'], job['cutoff'])
            assert altered_norm == norm['features']
            np.testing.assert_array_equal(altered['fit'], x['fit'])
            np.testing.assert_array_equal(altered['validation'], x['validation'])
            poison_checks += 1
        annual_rows.extend(dict(**{k: job[k] for k in ('route', 'crop', 'cutoff', 'seed', 'model', 'percent')}, **row) for row in annual)
        replayed_rows += len(ev)
        sources.update(data['sources'])
        for name, digest in record['files'].items():
            sources[str(root / name)] = digest
        sources[str(root / 'complete.json')] = sha256(root / 'complete.json')
        print(f'[REGIONAL READOUT VERIFIED] {job}', flush=True)
    destination = RESULT / f'{"readout_smoke" if smoke else "readout"}_{scope}_summary'
    destination.mkdir(parents=True, exist_ok=True)
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(destination / 'annual.csv', index=False)
    mean = annual.groupby(['route', 'crop', 'country', 'seed', 'model', 'percent']).agg(
        rmse=('rmse', 'mean'), mae=('mae', 'mean'), years=('year', 'nunique'), samples=('samples', 'sum')).reset_index()
    mean.to_csv(destination / 'per_seed.csv', index=False)
    for name in ('annual.csv', 'per_seed.csv'):
        sources[str(destination / name)] = sha256(destination / name)
    for filename, digest in sources.items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Readout source changed during independent replay')
    atomic_json(output_for(smoke, scope), dict(passed=True, timestamp=datetime.now().astimezone().isoformat(),
        jobs=planned, models=len(planned), scope=scope, smoke=smoke,
        prediction_rows_replayed=replayed_rows, maximum_weight_replay_error=0.,
        hidden_value_and_support_poison_checks=poison_checks,
        feature_reconstruction_uses_registered_builder=True, train_only_normalization_rebuilt=True,
        historical_expert_weights_unchanged=True, predicted_state_yield_evaluated=False,
        sources=sources, verifier_sha256=sha256(SCRIPT)))
    print(f'[REGIONAL READOUT AUDIT COMPLETE] models={len(planned)} scope={scope}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--scope', choices=('all', 'terminal'), default='all')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.smoke, args.scope, args.wait)
