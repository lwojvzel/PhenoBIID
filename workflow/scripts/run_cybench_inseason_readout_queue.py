"""Bounded regional readout matrix: main direct fits, terminal heads, then leads."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cybench_inseason_model_data import ROOT, RESULT, RECIPES, BLOCKS, SEEDS, PERCENTS
from cybench_inseason_yield_models import hashes
from run_cybench_inseason_readout import root_for, verify_completed, CODE
from run_cybench_inseason_history_queue import free_memory_gib
from run_cybench_inseason_state_queue import process_identity
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

SCRIPT = Path(__file__).resolve()
LOGS = ROOT / 'benchmark/logs/cybench_inseason_models_v1'


def jobs(smoke=False):
    periods = (2001,) if smoke else BLOCKS
    seeds = (42,) if smoke else SEEDS
    def direct(percents):
        return [dict(route='direct', crop=crop, cutoff=cutoff, seed=seed, model=model, percent=percent, smoke=smoke)
            for percent in percents for crop in RECIPES for cutoff in periods for seed in seeds
            for model in ('lightgbm', 'random_forest')]
    terminal = [dict(route='terminal', crop=crop, cutoff=cutoff, seed=seed, model='lightgbm', percent=0, smoke=smoke)
        for crop in RECIPES for cutoff in periods for seed in seeds]
    return direct((10,))+terminal+direct((30, 50))


def tag(job):
    return f'{job["route"]}_{job["crop"]}_{job["cutoff"]}_{job["seed"]}_{job["model"]}_{job["percent"]:03d}'


def completed(job):
    if not (root_for(**job) / 'complete.json').exists():
        return False
    verify_completed(**job)
    return True


def verify_gate(path, expected):
    value = json.loads(path.read_text())
    if not value['passed'] or value['models'] != expected or value['selection_uses_outer_scores']:
        raise ValueError('Independent historical verification required')
    for filename, digest in value['sources'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Historical gate source changed')
    return value


def worker(smoke=False):
    prefix = 'readout_smoke' if smoke else 'readout_full'
    logs = LOGS / prefix
    logs.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    with (RESULT / f'{prefix}_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = RESULT / ('history_smoke_verification.json' if smoke else 'history_verification.json')
        verify_gate(gate, 6 if smoke else 54)
        if not smoke:
            for job in jobs(True):
                verify_completed(**job)
        planned = jobs(smoke)
        spec = dict(jobs=planned, code_sha256=hashes(CODE), queue_sha256=sha256(SCRIPT),
            queue_helpers_sha256={name: sha256(ROOT / 'scripts' / name) for name in
                ('run_cybench_inseason_history_queue.py', 'run_cybench_inseason_state_queue.py')},
            history_verification_sha256=sha256(gate), workers=2, fit_threads_each=2,
            max_load_average_1m=80, minimum_available_memory_gib=30,
            model_optimization=False, all_registered_seeds_retained=True)
        path = RESULT / f'{prefix}_registration.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Registered regional readout queue changed')
        atomic_json(path, spec)
        pending = [job for job in planned if not completed(job)]
        live, failures = {}, []
        done = len(planned)-len(pending)
        while pending or live:
            for name, (job, child, stream) in list(live.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                try:
                    ok, error = code == 0 and completed(job), None
                except Exception as exception:
                    ok, error = False, repr(exception)
                if ok:
                    done += 1
                    print(f'[REGIONAL READOUT QUEUE] {prefix} {done}/{len(planned)} {name}', flush=True)
                else:
                    failures.append(dict(job=job, exit_code=code, error=error, log=str(logs / f'{name}.log')))
                del live[name]
            if not failures and os.getloadavg()[0] < 80 and free_memory_gib() >= 30:
                while pending and len(live) < 2:
                    job = pending.pop(0)
                    args = [sys.executable, '-u', str(ROOT / 'scripts/run_cybench_inseason_readout.py')]
                    for key in ('route', 'crop', 'cutoff', 'seed', 'model', 'percent'):
                        args.extend(['--'+key, str(job[key])])
                    if smoke:
                        args.append('--smoke')
                    name = tag(job)
                    stream = (logs / f'{name}.log').open('a')
                    child = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
                    live[name] = job, child, stream
                    print(f'[REGIONAL READOUT START] {name} pid={child.pid}', flush=True)
            atomic_json(RESULT / f'{prefix}_queue_status.json', dict(timestamp=datetime.now().astimezone().isoformat(),
                completed=done, total=len(planned), pending=len(pending), failures=failures,
                live=[dict(job=job, process=process_identity(child.pid)) for job, child, _ in live.values()]))
            if failures and not live:
                raise RuntimeError(f'Regional readout failed; pending jobs retained: {failures}')
            if pending or live:
                time.sleep(3)
        for job in planned:
            verify_completed(**job)
        atomic_json(RESULT / f'{prefix}_queue_complete.json', dict(passed=True,
            timestamp=datetime.now().astimezone().isoformat(), jobs=planned,
            final_models=0 if smoke else len(planned), smoke_models=len(planned) if smoke else 0,
            direct_final_models=0 if smoke else 108, terminal_final_models=0 if smoke else 18,
            registration_sha256=sha256(path),
            completion_sha256={str(root_for(**job) / 'complete.json'): sha256(root_for(**job) / 'complete.json') for job in planned},
            independent_output_audit_completed=False, predicted_state_yield_evaluated=False))


def launch():
    RESULT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (RESULT / 'readout_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        final = RESULT / 'readout_full_queue_complete.json'
        if final.exists():
            record = json.loads(final.read_text())
            if not record['passed'] or record['final_models'] != 126 or record['jobs'] != jobs():
                raise ValueError('Invalid completed regional readout matrix')
            if sha256(RESULT / 'readout_full_registration.json') != record['registration_sha256']:
                raise ValueError('Completed readout registration changed')
            for job in jobs():
                verify_completed(**job)
                marker = root_for(**job) / 'complete.json'
                if sha256(marker) != record['completion_sha256'][str(marker)]:
                    raise ValueError('Readout completion changed')
            print(json.dumps(dict(already_complete=True, final_models=126, completion_preserved=True)))
            return
        path = RESULT / 'readout_launcher.json'
        if path.exists():
            previous = json.loads(path.read_text())
            current = process_identity(previous['pid'])
            if current is not None and current == previous['process'] and str(SCRIPT) in current['command'] and '--worker' in current['command']:
                print(json.dumps(dict(already_running=True, **previous), indent=2))
                return
        verify_gate(RESULT / 'history_verification.json', 54)
        for job in jobs(True):
            verify_completed(**job)
        with (LOGS / 'readout_queue.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--worker'], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError('Regional readout queue exited; inspect readout_queue.log')
        record = dict(timestamp=datetime.now().astimezone().isoformat(), pid=child.pid,
            process=process_identity(child.pid), log=str(LOGS / 'readout_queue.log'))
        atomic_json(path, record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--smoke', action='store_true')
    choice.add_argument('--launch', action='store_true')
    choice.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    launch() if args.launch else worker(args.smoke)
