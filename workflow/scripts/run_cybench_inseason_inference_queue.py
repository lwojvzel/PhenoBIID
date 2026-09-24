"""Bounded frozen regional inference, gated on independent component audits."""
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
from run_cybench_inseason_inference import root_for, verify_completed, code_hashes
from run_cybench_inseason_state_queue import process_identity, GPUS
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from shared_gpu_queue_extended import run_extended_stage
from verify_cybench_inseason_readouts import run as audit_heads, verify_record as verify_heads
from verify_cybench_inseason_states import verify_record as verify_states

SCRIPT = Path(__file__).resolve()
LOGS = ROOT / 'benchmark/logs/cybench_inseason_models_v1'


def jobs(smoke=False):
    return [dict(crop=crop, cutoff=cutoff, seed=seed, smoke=smoke)
        for cutoff in ((2001,) if smoke else BLOCKS) for crop in RECIPES
        for seed in ((42,) if smoke else SEEDS)]


def tag(job):
    return f'cybench_inference_{"smoke" if job["smoke"] else "full"}_{job["crop"]}_{job["cutoff"]}_{job["seed"]}'


def command(job):
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/run_cybench_inseason_inference.py')]
    for key in ('crop', 'cutoff', 'seed'):
        cmd.extend(['--'+key, str(job[key])])
    return cmd + (['--smoke'] if job['smoke'] else [])


def registration(smoke):
    spec = dict(jobs=jobs(smoke), code_sha256=code_hashes(),
        queue_code_sha256={name: sha256(ROOT / 'scripts' / name) for name in
            ('run_cybench_inseason_inference_queue.py', 'shared_gpu_queue.py', 'shared_gpu_queue_extended.py',
             'run_cybench_inseason_state_queue.py')},
        state_verification_sha256=sha256(RESULT / 'state_verification.json'),
        head_gate='readout_smoke_all_verification.json' if smoke else 'readout_terminal_verification.json',
        gpus=list(GPUS), max_jobs_per_card=3, admission_limit_mib=16000,
        process_limit_mib=3072, conditions_per_group=7, new_fits=0, model_optimization=False)
    if not smoke:
        gate = RESULT / 'inference_smoke_verification.json'
        record = json.loads(gate.read_text())
        if not record['passed'] or record['groups'] != 2 or record['conditions'] != 14:
            raise ValueError('Independent smoke inference replay is required')
        if record['verifier_sha256'] != sha256(ROOT / 'scripts/verify_cybench_inseason_inference.py'):
            raise ValueError('Smoke inference verifier changed')
        for filename, digest in record['sources'].items():
            if sha256(Path(filename)) != digest:
                raise ValueError('Verified smoke inference source changed')
        spec['smoke_verification_sha256'] = sha256(gate)
    return spec


def run_stage(smoke=False):
    stage = 'inference_smoke' if smoke else 'inference_full'
    LOGS.mkdir(parents=True, exist_ok=True)
    with (RESULT / f'{stage}_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = registration(smoke)
        path = RESULT / f'{stage}_registration.json'
        if path.exists():
            if json.loads(path.read_text()) != spec:
                raise ValueError('Registered inference implementation changed')
        else:
            atomic_json(path, spec)
        done = RESULT / f'{stage}_queue_complete.json'
        if done.exists():
            for job in jobs(smoke):
                verify_completed(**job)
            record = json.loads(done.read_text())
            if record['registration_sha256'] != sha256(path):
                raise ValueError('Completion does not match inference registration')
            print(f'[REGIONAL INFERENCE COMPLETE REUSE] {done}', flush=True)
            return
        verify_states()
        if smoke:
            verify_heads(True, 'all')
        else:
            # The existing CPU readout worker must be live while terminal weights are pending.
            audit_heads(False, 'terminal', wait=True)
            verify_heads(False, 'terminal')
        if registration(smoke) != spec:
            raise ValueError('Inference sources changed while waiting for terminal heads')
        records = []
        for job in jobs(smoke):
            marker = root_for(**job) / 'complete.json'
            if marker.exists():
                verify_completed(**job)
            records.append(dict(key=tag(job), marker=str(marker), command=command(job)))
        try:
            run_extended_stage(records, stage, GPUS, RESULT, LOGS / stage, len(records))
            metrics = [verify_completed(**job) for job in jobs(smoke)]
            atomic_json(done, dict(passed=True, timestamp=datetime.now().astimezone().isoformat(),
                jobs=jobs(smoke), groups=len(metrics), conditions=7*len(metrics), new_fits=0,
                maximum_peak_allocated_mib=max(row['peak_allocated_mib'] for row in metrics),
                registration_sha256=sha256(path),
                completion_sha256={str(root_for(**job) / 'complete.json'): sha256(root_for(**job) / 'complete.json')
                    for job in jobs(smoke)}))
        except Exception as error:
            atomic_json(RESULT / f'{stage}_queue_error.json', dict(error=repr(error),
                timestamp=datetime.now().astimezone().isoformat()))
            raise


def launch():
    LOGS.mkdir(parents=True, exist_ok=True)
    with (RESULT / 'inference_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RESULT / 'inference_launcher.json'
        if path.exists():
            old = json.loads(path.read_text())
            current = process_identity(old['pid'])
            if current == old['process'] and current is not None and str(SCRIPT) in current['command']:
                print(json.dumps(dict(already_running=True, **old), indent=2))
                return
        registration(False)
        if (RESULT / 'inference_full_queue_complete.json').exists():
            run_stage(False)
            return
        with (LOGS / 'inference_queue.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--worker'], cwd=ROOT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError('Regional inference worker exited; inspect inference_queue.log')
        record = dict(pid=child.pid, process=process_identity(child.pid),
            timestamp=datetime.now().astimezone().isoformat(), log=str(LOGS / 'inference_queue.log'))
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
