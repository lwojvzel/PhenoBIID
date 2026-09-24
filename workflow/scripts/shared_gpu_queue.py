"""Short admission leases allow independent experiment batches to share free GPUs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from review_revision_data import ROOT
from run_review_revision_parallel import atomic_json

COORD = ROOT / 'benchmark/logs/shared_gpu_queue_v1'
LEGACY = ROOT / 'benchmark/logs/task_aligned_world_v1/status.json'
RESERVE_MIB = 4000


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(record):
    started = process_start(record['pid'])
    return started is not None and (not record.get('process_start') or started == record['process_start'])


def legacy_jobs():
    if not LEGACY.exists():
        return [], False
    state = json.loads(LEGACY.read_text())
    jobs = [r for r in state['running'] if live(r)]
    return jobs, bool(jobs and state['pending'])


def inventory():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    cards = {}
    for line in lines:
        index, uuid, used = (x.strip() for x in line.split(','))
        cards[index] = dict(uuid=uuid, used=int(used))
    memory = {}
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
    for line in lines:
        if not line.strip():
            continue
        uuid, pid, used = (x.strip() for x in line.split(','))
        if used.isdigit():
            memory[(uuid, int(pid))] = int(used)
    return cards, memory


def room(gpu, cards, memory, jobs):
    active = {j['pid']: j for j in jobs if str(j['gpu']) == str(gpu)}
    if len(active) >= 3:
        return False
    card = cards[str(gpu)]
    unfilled = sum(max(0, RESERVE_MIB - memory.get((card['uuid'], pid), 0)) for pid in active)
    return card['used'] + unfilled + RESERVE_MIB <= 16000


def run_shared_stage(queue, stage, gpus, result, logs, fits_expected):
    if not set(gpus).issubset({'0', '1', '4', '5', '6'}):
        raise ValueError('GPU not inspected for this experiment series')
    COORD.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    pending = [q for q in queue if not Path(q['marker']).exists()]
    existing = len(queue)-len(pending); running = []; completed = []; failed = []
    def public(j):
        return {k: v for k, v in j.items() if k not in ('process', 'stream')}
    try:
        while running or pending:
            for job in list(running):
                code = job['process'].poll()
                if code is None:
                    continue
                job['stream'].close(); running.remove(job)
                ok = code == 0 and Path(job['marker']).exists()
                (completed if ok else failed).append(dict(**public(job), exit_code=code))
                print(f'[{"DONE" if ok else "FAILED"}] {job["key"]}', flush=True)
            with (COORD / 'admission.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = COORD / 'leases.json'
                leases = json.loads(path.read_text()) if path.exists() else []
                leases = [j for j in leases if live(j)]
                legacy, old_queue_has_pending = legacy_jobs()
                if pending and not old_queue_has_pending:
                    cards, memory = inventory()
                    for gpu in gpus:
                        if not pending or not room(gpu, cards, memory, [*legacy, *leases]):
                            continue
                        spec = pending.pop(0); stream = (logs / f'{spec["key"]}.log').open('ab')
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                        child = subprocess.Popen(spec['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        lease = dict(key=spec['key'], gpu=str(gpu), pid=child.pid, process_start=process_start(child.pid),
                                     reserve_mib=RESERVE_MIB, logs=str(logs))
                        leases.append(lease); running.append(dict(**spec, **{k: v for k, v in lease.items() if k != 'key'}, process=child, stream=stream))
                        print(f'[START] {spec["key"]} gpu={gpu} pid={child.pid}', flush=True)
                atomic_json(path, leases)
            atomic_json(logs / 'status.json', dict(stage=stage, existing=existing, completed=completed, failures=failed,
                running=[public(j) for j in running], pending=len(pending), fits_expected=fits_expected,
                fits_completed=len(list((result / 'pipelines').rglob('metrics.json'))),
                admission_limit_mib=16000, max_jobs_per_card=3, reserved_mib_per_job=RESERVE_MIB, updated_unix=time.time()))
            if pending or running:
                time.sleep(5)
    finally:
        for j in running:
            j['process'].terminate()
        for j in running:
            try:
                j['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                j['process'].kill(); j['process'].wait()
            j['stream'].close()
    if failed:
        raise RuntimeError(f'{stage}: {len(failed)} experiment jobs failed')
