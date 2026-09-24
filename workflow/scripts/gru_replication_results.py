"""Paths, fixed baseline identity, and paired spatial inference for replication."""
import json
from pathlib import Path

import numpy as np

from run_gru_replication_queue import ORIGINS, SEEDS
from run_gru_world_replication import RESULT, world_root, direct_root, PRETRAINED
from review_revision_data import ROOT, CROPS, sha256
from stable_remote_models import ENGINES
from run_review_revision_parallel import atomic_json
from multimodal_baseline import regression_metrics

HISTORY = (*ENGINES, 'history_mlp')


def history_root(crop, origin, seed, engine):
    if engine == 'history_mlp':
        return PRETRAINED / 'pipelines' / crop / f'origin_{origin}/history_mlp/seed_{seed}'
    return ROOT / 'benchmark/results/stable_remote_v1/pipelines' / crop / f'origin_{origin}/{engine}__w0/seed_{seed}/history'


def original_direct_root(crop, origin, seed):
    return PRETRAINED / 'pipelines' / crop / f'origin_{origin}/direct_gru/seed_{seed}'


def history_selection():
    rows = []; ranking = []
    for crop in CROPS:
        for origin in ORIGINS:
            scores = []
            for engine in HISTORY:
                path = history_root(crop, origin, 42, engine) / 'metrics.json'
                m = json.loads(path.read_text())
                scores.append(dict(crop=crop, origin=origin, engine=engine,
                                   validation_rmse=m['scores']['validation']['rmse'], metrics=str(path), sha256=sha256(path)))
            scores.sort(key=lambda a: (a['validation_rmse'], a['engine']))
            rows.append(scores[0]); ranking.extend(scores)
    selection = dict(selected=rows, all_validation_candidates=ranking,
        criterion='For each crop/origin choose the history recipe using seed 42 validation RMSE only, then use the same identity for all three seeds')
    path = RESULT / 'history_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Frozen history identity changed')
    atomic_json(path, selection)
    return {(r['crop'], r['origin']): r['engine'] for r in rows}


def read_prediction(root, split):
    with np.load(root / f'{split}_predictions.npz') as p:
        return {k: p[k] for k in p.files}


def assert_aligned(a, b):
    for key in ('target', 'source_indices', 'row', 'col', 'year'):
        np.testing.assert_array_equal(a[key], b[key])


def ensemble(arrays):
    if len(arrays) != 3:
        raise ValueError('All three fixed seeds required')
    for a in arrays:
        assert_aligned(arrays[0], a)
    output = {k: v for k, v in arrays[0].items()}
    output['prediction'] = np.mean([a['prediction'].astype(float) for a in arrays], axis=0)
    if 'prediction_lai' in arrays[0]:
        output['prediction_lai'] = np.mean([a['prediction_lai'].astype(float) for a in arrays], axis=0)
    return output


def cluster_interval(target, prediction, reference, row, col, repetitions=10000, seed=20260906):
    target, prediction, reference = (np.asarray(a, dtype=float) for a in (target, prediction, reference))
    if not (target.shape == prediction.shape == reference.shape == np.asarray(row).shape == np.asarray(col).shape):
        raise ValueError('Paired arrays have different shapes')
    block = (np.asarray(row, dtype=np.int64)//40)*18 + np.asarray(col, dtype=np.int64)//40
    unique, inverse = np.unique(block, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('At least two spatial clusters required')
    sse = np.bincount(inverse, weights=(target-prediction)**2)
    ref = np.bincount(inverse, weights=(target-reference)**2)
    if np.any(ref < 0) or ref.sum() <= 0:
        raise ValueError('Degenerate reference errors')
    rng = np.random.default_rng(seed); samples = []
    for start in range(0, repetitions, 500):
        indices = rng.integers(0, len(unique), size=(min(500, repetitions-start), len(unique)))
        numerator = sse[indices].sum(1); denominator = ref[indices].sum(1)
        if (denominator <= 0).any():
            raise ValueError('Degenerate bootstrap reference')
        samples.extend(100*(1-np.sqrt(numerator/denominator)))
    low, high = np.quantile(samples, [.025, .975])
    return dict(gain_percent=float(100*(1-np.sqrt(sse.sum()/ref.sum()))), ci_low=float(low), ci_high=float(high),
                clusters=len(unique), repetitions=repetitions, degrees=20)


if __name__ == '__main__':
    print(history_selection())
