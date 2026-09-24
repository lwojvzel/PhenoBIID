"""Fill the fixed-seed/origin grid with all matched historical controls."""
import fcntl
import json
import sys

from review_revision_data import ROOT, CROPS
from run_gru_world_replication import RESULT, world_root
from task_aligned_data import CACHE
from stable_remote_models import ENGINES
from run_review_revision_parallel import atomic_json
from shared_gpu_queue import run_shared_stage

ORIGINS = (2004, 2008, 2012)
SEEDS = (42, 45, 48)


def main():
    logs = ROOT / 'benchmark/logs/gru_world_replication_v1'; logs.mkdir(parents=True, exist_ok=True)
    selected = json.loads((ROOT / 'benchmark/results/validated_world_anchor_v1/pilot_validation_selection.json').read_text())
    if selected['selected']['candidate'] != 'gru__joint_tenth':
        raise ValueError('Frozen validation candidate differs from the registered recipe')
    with (logs / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpus = ['0', '1', '4', '5', '6']
        prepare = [dict(key=f'prepare__{crop}__{origin}', marker=str(CACHE / crop / f'origin_{origin}/manifest.json'),
            command=[sys.executable, str(ROOT / 'scripts/task_aligned_data.py'), '--crop', crop, '--origin', str(origin)]) for crop in CROPS for origin in ORIGINS]
        run_shared_stage(prepare, 'prepare', gpus, result=RESULT, logs=logs, fits_expected=32)
        controls = []; worlds = []
        for crop in CROPS:
            for origin in ORIGINS:
                for seed in SEEDS:
                    common = ['--crop', crop, '--origin', str(origin), '--seed', str(seed)]
                    for engine in ENGINES:
                        path = ROOT / f'benchmark/results/stable_remote_v1/pipelines/{crop}/origin_{origin}/{engine}__w0/seed_{seed}/history/metrics.json'
                        controls.append(dict(key=f'history__{crop}__{origin}__{seed}__{engine}', marker=str(path),
                            command=[sys.executable, str(ROOT / 'scripts/run_stable_remote.py'), *common,
                                     '--window', '0', '--engine', engine, '--conditions', 'history']))
                    for variant in ('history_mlp', 'direct_gru'):
                        path = ROOT / f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_{origin}/{variant}/seed_{seed}/metrics.json'
                        controls.append(dict(key=f'neural__{crop}__{origin}__{seed}__{variant}', marker=str(path),
                            command=[sys.executable, str(ROOT / 'scripts/run_task_aligned_world.py'), *common, '--variant', variant]))
                    worlds.append(dict(key=f'world__{crop}__{origin}__{seed}', marker=str(world_root(crop, origin, seed) / 'metrics.json'),
                        command=[sys.executable, str(ROOT / 'scripts/run_gru_world_replication.py'), *common]))
        atomic_json(logs / 'queue.json', dict(origins=ORIGINS, seeds=SEEDS, controls=controls, worlds=worlds,
            total_control_fits=288, total_world_fits=36, reused_control_fits=80, reused_world_fits=4,
            new_control_fits=208, new_world_fits=32, state_recipe='joint_tenth',
            reason='Replicate validation-positive history increment; superiority over direct climate is not yet established',
            main_model_promoted=False))
        run_shared_stage(controls, 'controls', gpus, result=RESULT, logs=logs, fits_expected=32)
        smoke = [dict(key='smoke__replication', marker=str(RESULT / 'smoke/maize/origin_2004/seed_45/metrics.json'),
            command=[sys.executable, str(ROOT / 'scripts/run_gru_world_replication.py'), '--crop', 'maize', '--origin', '2004', '--seed', '45', '--smoke'])]
        run_shared_stage(smoke, 'smoke', gpus, result=RESULT, logs=logs, fits_expected=32)
        run_shared_stage(worlds, 'worlds', gpus, result=RESULT, logs=logs, fits_expected=32)
        atomic_json(RESULT / 'training_complete.json', dict(worlds=36, controls=288, newly_trained=240))


if __name__ == '__main__':
    main()
