"""Bounded CPU queue for low-capacity state-to-yield diagnostics."""
import fcntl
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from linear_state_yield import ROOT, RESULT, CROPS
from run_review_revision_parallel import atomic_json


def main():
    logs = ROOT / 'benchmark/logs/linear_state_yield_v1'; logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = [(crop, origin) for crop in CROPS for origin in (2004, 2008, 2012)]
        atomic_json(logs / 'queue.json', dict(jobs=jobs, fits=480, cpu_workers=2, threads_per_worker=2))

        def run(job):
            crop, origin = job
            environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
            command = [sys.executable, str(ROOT / 'scripts/linear_state_yield.py'), '--crop', crop, '--origin', str(origin)]
            print(f'[START LINEAR YIELD] {crop} {origin}', flush=True)
            with (logs / f'{crop}_{origin}.log').open('a') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'[DONE LINEAR YIELD] {crop} {origin}', flush=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(run, jobs))
        atomic_json(RESULT / 'training_complete.json', dict(fits=480))


if __name__ == '__main__':
    main()
