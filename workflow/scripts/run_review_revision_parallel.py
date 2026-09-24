"""Resource-limited queue for isolated reviewer-response jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CROPS = ("maize", "rice", "soybean", "wheat")
SEEDS = (42, 45, 48)
RUN_ROOT = ROOT / "benchmark/results/review_revision_v2"


def configurations(stage):
    result = []
    for model in ("lightgbm", "mlp", "gru", "transformer"):
        result.append(dict(model=model))
    for climate in (False, True):
        for source in ("predicted", "previous", "climatology", "constant"):
            result.append(dict(model="fusion", state=source, climate=climate, moddrop="coverage"))
        if stage == "all":
            for drop in ("none", "uniform", "shuffled"):
                result.append(dict(model="fusion", climate=climate, moddrop=drop))
            for gate in ("no_bce", "fixed_01", "fixed_05"):
                result.append(dict(model="fusion", climate=climate, moddrop="coverage", gate=gate))
    if stage == "all":
        for model in ("multitask_gru", "trajectory_mlp", "trajectory_gru"):
            result.append(dict(model=model))
        result.append(dict(model="fusion", state="constant", moddrop="none"))
    return result


def experiment_id(spec):
    if spec["model"] != "fusion":
        return spec["model"]
    return "__".join(("fusion", spec.get("state", "predicted"), "climate" if spec.get("climate", False) else "state_only", spec.get("moddrop", "coverage"), spec.get("gate", "bce")))


def atomic_json(path, values):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(values, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("cache", "first", "all", "latent", "external", "dynamics", "dynamics_self", "interaction", "lai_inputs", "readout_only", "gpp"), required=True)
    p.add_argument("--gpus", default="0,1,4,5,6,7")
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--memory-limit", type=int, default=16000)
    p.add_argument("--reservation", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="42,45,48")
    args = p.parse_args()
    if not 1 <= args.per_gpu <= 3:
        raise ValueError("At most three of our tasks per GPU")
    gpus = args.gpus.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    logs = ROOT / "benchmark/logs" / ("review_revision_v3" if args.stage in ("interaction", "lai_inputs", "readout_only", "gpp") else "review_revision_v2") / args.stage
    logs.mkdir(parents=True, exist_ok=True)
    queue = []
    for crop in (("maize", "wheat") if args.stage == "external" else CROPS):
        for seed in seeds:
            specs = [dict(model=args.stage)] if args.stage in ("cache", "external") else (
                [dict(model=m) for m in ("frozen", "joint_01", "joint_1", "joint_10")]
                if args.stage == "latent" else configurations(args.stage))
            if args.stage == "dynamics":
                specs = [dict(model=m) for m in ("full", "no_weather", "no_feedback")]
            elif args.stage == "dynamics_self":
                specs = [dict(model="self_modulation")]
            elif args.stage == "interaction":
                specs = [dict(model=m) for m in ("original", "mean_normalized", "pairwise_relevance", "cross_value")]
            elif args.stage == "lai_inputs":
                specs = [dict(model=m) for m in ("no_lai", "previous_lai", "observed_lai")]
            elif args.stage == "gpp":
                specs = [dict(model=m) for m in ("gru", "fusion")]
            elif args.stage == "readout_only":
                specs = [dict(model="readout_only_"+m) for m in ("mean_normalized", "pairwise_relevance", "cross_value")]
            for spec in specs:
                key = "cache" if args.stage == "cache" else (spec["model"] if args.stage == "gpp" else experiment_id(spec))
                marker = ROOT / "benchmark/cache/review_revision_v2" / crop / f"seed_{seed}" / "manifest.json" if args.stage == "cache" else RUN_ROOT / crop / key / f"seed_{seed}" / "test_metrics.json"
                runner = "review_revision_data.py" if args.stage == "cache" else "run_review_revision.py"
                if args.stage == "latent":
                    marker = RUN_ROOT / "latent_candidates" / crop / key / f"seed_{seed}" / "test_metrics.json"
                    runner = "run_latent_world_revision.py"
                elif args.stage == "external":
                    marker = RUN_ROOT / "external_cybench" / crop / f"seed_{seed}" / "test_metrics.json"
                    runner = "run_external_world_revision.py"
                elif args.stage in ("dynamics", "dynamics_self"):
                    marker = RUN_ROOT / "dynamics_controls" / crop / key / f"seed_{seed}" / "test_metrics.json"
                    runner = "run_dynamics_revision_controls.py"
                elif args.stage in ("interaction", "lai_inputs", "readout_only"):
                    marker = ROOT / "benchmark/results/review_revision_v3/pipelines" / crop / key / f"seed_{seed}" / "test_metrics.json"
                    runner = "run_interaction_revision.py"
                elif args.stage == "gpp":
                    marker = ROOT / "benchmark/results/review_revision_v3/gpp" / crop / key / f"seed_{seed}" / "test_metrics.json"
                    runner = "run_gpp_readout_revision.py"
                command = [sys.executable, str(ROOT / "scripts" / runner), "--crop", crop, "--seed", str(seed)]
                if args.stage == "latent":
                    command += ["--mode", key, "--epochs", str(args.epochs), "--patience", str(args.patience)]
                elif args.stage in ("dynamics", "dynamics_self"):
                    command += ["--control", key, "--epochs", str(args.epochs), "--patience", str(args.patience)]
                elif args.stage in ("interaction", "lai_inputs", "readout_only"):
                    command += ["--kind", key, "--epochs", str(args.epochs), "--patience", str(args.patience)]
                elif args.stage == "gpp":
                    command += ["--model", key]
                elif args.stage not in ("cache", "external"):
                    command += ["--model", spec["model"], "--epochs", str(args.epochs), "--patience", str(args.patience)]
                    for option in ("state", "moddrop", "gate"):
                        if option in spec:
                            command += [f"--{option}", spec[option]]
                    if spec.get("climate"):
                        command += ["--climate"]
                queue.append(dict(crop=crop, seed=seed, key=key, command=command, marker=str(marker)))
    pending = [j for j in queue if not Path(j["marker"]).exists()]
    atomic_json(logs / "queue.json", queue)
    running, completed, failures = [], [], []
    def memory(gpu):
        r = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        return int(r.stdout.strip())
    baseline = {g: memory(g) for g in gpus}
    print(f"[QUEUE] {args.stage}: {len(pending)} pending, {len(queue)-len(pending)} existing", flush=True)
    try:
        while pending or running:
            for job in list(running):
                code = job["process"].poll()
                if code is not None:
                    job["log"].close()
                    running.remove(job)
                    record = {k: v for k, v in job.items() if k not in ("process", "log")}
                    record["exit_code"] = code
                    (completed if code == 0 and Path(job["marker"]).exists() else failures).append(record)
                    print(f"[{'DONE' if code == 0 else 'FAIL'}] {job['crop']}/{job['key']}/{job['seed']} gpu={job['gpu']}", flush=True)
            for gpu in gpus:
                current = [j for j in running if j["gpu"] == gpu]
                if not pending or len(current) >= args.per_gpu:
                    continue
                if max(memory(gpu), baseline[gpu] + len(current) * args.reservation) + args.reservation > args.memory_limit:
                    continue
                job = pending.pop(0)
                log_path = logs / f"{job['crop']}__{job['key']}__{job['seed']}.log"
                stream = log_path.open("ab")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                running.append({**job, "gpu": gpu, "process": process, "log": stream, "pid": process.pid, "log_path": str(log_path)})
                print(f"[START] {job['crop']}/{job['key']}/{job['seed']} gpu={gpu} pid={process.pid}", flush=True)
            atomic_json(logs / "status.json", {"pending": len(pending), "running": [{k: v for k, v in j.items() if k not in ("process", "log")} for j in running], "completed": completed, "failures": failures})
            if running or pending:
                time.sleep(5)
    finally:
        for job in running:
            job["process"].terminate()
        for job in running:
            try:
                job["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                job["process"].kill()
                job["process"].wait()
            job["log"].close()
    print(f"[FINISHED] {len(completed)} completed, {len(failures)} failed", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
