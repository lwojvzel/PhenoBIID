"""Fixed regional state fits sharing the existing bounded GPU admission leases."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cybench_inseason_model_data import ROOT, RESULT, RECIPES, SEEDS, BLOCKS
from run_cybench_inseason_state import root_for, code_hashes, verify_completed, preflight
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from shared_gpu_queue_extended import run_extended_stage

SCRIPT = Path(__file__).resolve()
LOGS = ROOT / 'benchmark/logs/cybench_inseason_models_v1'
GPUS = ('4', '5', '6')


def jobs(smoke=False):
    return [dict(crop=crop, product=product, cutoff=cutoff, seed=seed, smoke=smoke)
        for cutoff in ((2001,) if smoke else BLOCKS)
        for crop, products in RECIPES.items() for product in products
        for seed in ((42,) if smoke else SEEDS)]


def tag(job):
    return (f'cybench_state_{"smoke" if job["smoke"] else "full"}_'
        f'{job["crop"]}_{job["product"]}_{job["cutoff"]}_{job["seed"]}')


def command(job):
    args = [sys.executable, '-u', str(ROOT / 'scripts/run_cybench_inseason_state.py')]
    for name in ('crop', 'product', 'cutoff', 'seed'):
        args.extend(['--'+name, str(job[name])])
    return args + (['--smoke'] if job['smoke'] else [])


def process_identity(pid):
    folder = Path('/proc') / str(pid)
    try:
        info = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        args = (folder / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return None
    if info[0] == 'Z':
        return None
    return dict(pid=pid, start_ticks=info[19], command=[x.decode() for x in args if x])


def run_stage(smoke):
    preflight()
    stage = 'state_smoke' if smoke else 'state_full'
    stage_logs = LOGS / stage
    RESULT.mkdir(parents=True, exist_ok=True)
    stage_logs.mkdir(parents=True, exist_ok=True)
    with (RESULT / f'{stage}_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not smoke:
            for job in jobs(True):
                verify_completed(**job)
        planned = jobs(smoke)
        records = []
        for job in planned:
            marker = root_for(**job) / 'complete.json'
            if marker.exists():
                verify_completed(**job)
            records.append(dict(key=tag(job), marker=str(marker), command=command(job)))
        spec = dict(jobs=planned, gpus=GPUS, code_sha256=code_hashes(),
            queue_code_sha256={name: sha256(ROOT / 'scripts' / name) for name in
                ('run_cybench_inseason_state_queue.py', 'shared_gpu_queue.py', 'shared_gpu_queue_extended.py')},
            data_preflight_sha256=sha256(RESULT / 'data_preflight.json'),
            max_jobs_per_card=3, admission_limit_mib=16000, per_process_limit_mib=3072,
            fixed_architecture=True, model_optimization=False, final_models=len(planned))
        spec = json.loads(json.dumps(spec))
        path = RESULT / f'{stage}_registration.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Regional state queue registration changed')
        atomic_json(path, spec)
        try:
            run_extended_stage(records, stage, GPUS, RESULT, stage_logs, len(planned))
            metrics = [verify_completed(**job) for job in planned]
            atomic_json(RESULT / f'{stage}_queue_complete.json', dict(passed=True,
                timestamp=datetime.now().astimezone().isoformat(), jobs=planned,
                final_models=0 if smoke else len(planned), smoke_models=len(planned) if smoke else 0,
                selected_epochs=[row['selected_epochs'] for row in metrics],
                maximum_peak_allocated_mib=max(row['peak_allocated_mib'] for row in metrics),
                registration_sha256=sha256(path), external_yield_performance_evaluated=False,
                completion_sha256={str(root_for(**job) / 'complete.json'): sha256(root_for(**job) / 'complete.json') for job in planned}))
        except Exception as error:
            atomic_json(RESULT / f'{stage}_queue_error.json', dict(timestamp=datetime.now().astimezone().isoformat(), error=repr(error)))
            raise


def launch():
    RESULT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (RESULT / 'state_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RESULT / 'state_launcher.json'
        if path.exists():
            previous = json.loads(path.read_text())
            current = process_identity(previous['pid'])
            if current is not None and current == previous['process'] and str(SCRIPT) in current['command'] and '--worker' in current['command']:
                print(json.dumps(dict(already_running=True, **previous), indent=2))
                return
        preflight()
        for job in jobs(True):
            verify_completed(**job)
        with (LOGS / 'state_queue.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--worker'], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError('Regional state queue exited; inspect state_queue.log')
        record = dict(timestamp=datetime.now().astimezone().isoformat(), pid=child.pid,
            process=process_identity(child.pid), log=str(LOGS / 'state_queue.log'))
        atomic_json(path, record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--smoke', action='store_true')
    choice.add_argument('--launch', action='store_true')
    choice.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        run_stage(args.smoke)
