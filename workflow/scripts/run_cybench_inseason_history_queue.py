"""Two-worker CPU queue for all fixed regional historical references."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cybench_inseason_model_data import ROOT, RESULT, RECIPES, BLOCKS, SEEDS
from cybench_inseason_yield_models import MODELS, hashes
from run_cybench_inseason_history import root_for, verify_completed, CODE
from run_cybench_inseason_state_queue import process_identity
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

SCRIPT = Path(__file__).resolve()
LOGS = ROOT / 'benchmark/logs/cybench_inseason_models_v1'


def jobs(smoke=False):
    return [dict(crop=crop, cutoff=cutoff, seed=seed, model=model, smoke=smoke)
        for crop in RECIPES for cutoff in ((2001,) if smoke else BLOCKS)
        for seed in ((42,) if smoke else SEEDS) for model in MODELS]


def tag(job):
    return f'{job["crop"]}_{job["cutoff"]}_{job["seed"]}_{job["model"]}'


def completed(job):
    if not (root_for(**job) / 'complete.json').exists():
        return False
    verify_completed(**job)
    return True


def free_memory_gib():
    lines = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(lines['MemAvailable'].split()[0])/1024**2


def worker(smoke=False):
    stage = 'history_smoke' if smoke else 'history_full'
    logs = LOGS / stage
    logs.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    with (RESULT / f'{stage}_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not smoke:
            for job in jobs(True):
                verify_completed(**job)
        planned = jobs(smoke)
        spec = dict(jobs=planned, code_sha256=hashes(CODE), queue_sha256=sha256(SCRIPT),
            process_probe_sha256=sha256(ROOT / 'scripts/run_cybench_inseason_state_queue.py'),
            workers=2, fit_threads_each=2, maximum_load_average_1m=80, minimum_available_gib=30,
            model_optimization=False, all_registered_seeds_retained=True)
        path = RESULT / f'{stage}_registration.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Regional historical registration changed')
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
                    print(f'[REGIONAL HISTORY QUEUE] {stage} {done}/{len(planned)} {name}', flush=True)
                else:
                    failures.append(dict(job=job, exit_code=code, error=error, log=str(logs / f'{name}.log')))
                del live[name]
            if not failures and os.getloadavg()[0] < 80 and free_memory_gib() >= 30:
                while pending and len(live) < 2:
                    job = pending.pop(0)
                    args = [sys.executable, '-u', str(ROOT / 'scripts/run_cybench_inseason_history.py')]
                    for key in ('crop', 'cutoff', 'seed', 'model'):
                        args.extend(['--'+key, str(job[key])])
                    if smoke:
                        args.append('--smoke')
                    name = tag(job)
                    stream = (logs / f'{name}.log').open('a')
                    child = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
                    live[name] = job, child, stream
                    print(f'[REGIONAL HISTORY START] {name} pid={child.pid}', flush=True)
            atomic_json(RESULT / f'{stage}_queue_status.json', dict(timestamp=datetime.now().astimezone().isoformat(),
                completed=done, total=len(planned), pending=len(pending), failures=failures,
                live=[dict(job=job, process=process_identity(child.pid)) for job, child, _ in live.values()]))
            if failures and not live:
                raise RuntimeError(f'Regional historical fits failed; pending work retained: {failures}')
            if pending or live:
                time.sleep(3)
        for job in planned:
            verify_completed(**job)
        atomic_json(RESULT / f'{stage}_queue_complete.json', dict(passed=True,
            timestamp=datetime.now().astimezone().isoformat(), jobs=planned,
            final_models=0 if smoke else len(planned), smoke_models=len(planned) if smoke else 0,
            registration_sha256=sha256(path),
            completion_sha256={str(root_for(**job) / 'complete.json'): sha256(root_for(**job) / 'complete.json') for job in planned},
            independent_output_audit_completed=False, external_world_performance_evaluated=False))


def launch():
    RESULT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (RESULT / 'history_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RESULT / 'history_launcher.json'
        if path.exists():
            previous = json.loads(path.read_text())
            current = process_identity(previous['pid'])
            if current is not None and current == previous['process'] and str(SCRIPT) in current['command'] and '--worker' in current['command']:
                print(json.dumps(dict(already_running=True, **previous), indent=2))
                return
        for job in jobs(True):
            verify_completed(**job)
        with (LOGS / 'history_queue.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--worker'], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError('Regional historical queue exited; inspect history_queue.log')
        record = dict(timestamp=datetime.now().astimezone().isoformat(), pid=child.pid,
            process=process_identity(child.pid), log=str(LOGS / 'history_queue.log'))
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
