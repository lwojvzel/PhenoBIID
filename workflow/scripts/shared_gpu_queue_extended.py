"""Admission extension for newly released GPU7; preserve frozen queue/model code.

The original shared_gpu_queue.py is hash-pinned by running experiments. This
adapter shares its lease file and budget checks without changing those assets.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from shared_gpu_queue import COORD, live, legacy_jobs, inventory, room, process_start, RESERVE_MIB
from review_revision_data import ROOT
from run_review_revision_parallel import atomic_json


def run_extended_stage(queue, stage, gpus, result, logs, fits_expected):
    if not set(gpus).issubset({'0', '1', '4', '5', '6', '7'}):
        raise ValueError('GPU has not been approved for this extension')
    COORD.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    pending = [q for q in queue if not Path(q['marker']).exists()]
    existing = len(queue)-len(pending)
    running, completed, failed = [], [], []

    def public(job):
        return {k: v for k, v in job.items() if k not in ('process', 'stream')}

    try:
        while pending or running:
            for job in list(running):
                code = job['process'].poll()
                if code is None:
                    continue
                job['stream'].close()
                running.remove(job)
                ok = code == 0 and Path(job['marker']).exists()
                (completed if ok else failed).append(dict(**public(job), exit_code=code))
                print(f'[{"DONE" if ok else "FAILED"}] {job["key"]}', flush=True)
            with (COORD / 'admission.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = COORD / 'leases.json'
                leases = [j for j in json.loads(path.read_text()) if live(j)] if path.exists() else []
                legacy, legacy_pending = legacy_jobs()
                if pending and not legacy_pending:
                    cards, memory = inventory()
                    for gpu in gpus:
                        pending = [q for q in pending if not Path(q['marker']).exists()]
                        if not pending or not room(gpu, cards, memory, [*legacy, *leases]):
                            continue
                        active_keys = {j['key'] for j in leases}
                        ready = next((i for i, job in enumerate(pending) if job['key'] not in active_keys), None)
                        if ready is None:
                            continue
                        spec = pending.pop(ready)
                        stream = (logs / f'{spec["key"]}.log').open('ab')
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                        child = subprocess.Popen(spec['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        lease = dict(key=spec['key'], gpu=str(gpu), pid=child.pid, process_start=process_start(child.pid),
                            reserve_mib=RESERVE_MIB, logs=str(logs))
                        leases.append(lease)
                        running.append(dict(**spec, **{k: v for k, v in lease.items() if k != 'key'}, process=child, stream=stream))
                        print(f'[START] {spec["key"]} gpu={gpu} pid={child.pid}', flush=True)
                atomic_json(path, leases)
            atomic_json(logs / 'status.json', dict(stage=stage, existing=existing, completed=completed, failures=failed,
                running=[public(j) for j in running], pending=len(pending), fits_expected=fits_expected,
                admission_limit_mib=16000, max_jobs_per_card=3, reserved_mib_per_job=RESERVE_MIB, updated_unix=time.time()))
            if pending or running:
                time.sleep(5)
    finally:
        for job in running:
            job['process'].terminate()
        for job in running:
            try:
                job['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait()
            job['stream'].close()
    if failed:
        raise RuntimeError(f'{stage}: {len(failed)} registered jobs failed')
