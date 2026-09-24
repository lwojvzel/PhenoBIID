"""Replay every state prediction and export free rollouts for terminal fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yield_sensitive_state import (ROOT, RESULT, CACHE, MODES, STATE_INPUTS,
    run_root, load, YieldSensitiveState, predict, projection_rmse)
from review_revision_data import CROPS, sha256
from run_task_aligned_world import state_rmse
from run_review_revision_parallel import atomic_json


def export_root(crop, origin, mode):
    return RESULT / 'exports' / crop / f'origin_{origin}' / mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS, required=True)
    parser.add_argument('--origin', type=int, choices=(2004, 2008, 2012), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(2000*2**20/torch.cuda.get_device_properties(0).total_memory)
    arrays, meta = load(args.crop, args.origin)
    for split, digest in meta['files'].items():
        if sha256(CACHE / args.crop / f'origin_{args.origin}/{split}.npz') != digest:
            raise ValueError('State cache changed')
    source = meta['spec']
    if sha256(Path(source['teacher_weight'])) != source['teacher_sha256']:
        raise ValueError('Outcome teacher changed')
    for state in source['sources']['states'].values():
        if sha256(Path(state['weight'])) != state['sha256']:
            raise ValueError('State prior weight changed')
    data = {split: {k: torch.from_numpy(a[k]) for k in STATE_INPUTS} for split, a in arrays.items()}
    beta = np.asarray(source['beta'], dtype=np.float32)
    norm = source['input_manifest']['normalization']; records = []
    for mode in MODES:
        root = run_root(args.crop, args.origin, mode)
        metrics = json.loads((root / 'metrics.json').read_text())
        config = json.loads((root / 'config.json').read_text())
        if config['input_manifest'] != meta:
            raise ValueError('State training inputs changed')
        for name, digest in config['code_hashes'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise ValueError(f'State training source changed: {name}')
        if sha256(Path(metrics['weight'])) != metrics['weight_sha256']:
            raise ValueError('Trained state weight changed')
        output = export_root(args.crop, args.origin, mode)
        output.mkdir(parents=True, exist_ok=True)
        spec = dict(weight=metrics['weight'], weight_sha256=metrics['weight_sha256'],
            source_config=str(root / 'config.json'), source_config_sha256=sha256(root / 'config.json'),
            input_manifest=meta, exporter_sha256=sha256(Path(__file__)),
            inference='Full free rollout, BF16, batch 256, no target state or yield inputs.')
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if saved['spec'] != spec:
                raise ValueError('Export recipe changed')
            for split, digest in saved['files'].items():
                if sha256(output / f'{split}.npy') != digest:
                    raise ValueError('Exported trajectory changed')
            records.append(saved['audit'])
            continue
        model = YieldSensitiveState().cuda()
        model.load_state_dict(torch.load(metrics['weight'], map_location='cuda', weights_only=True))
        errors = {}
        for split, a in arrays.items():
            prediction = predict(model, data[split])
            if split != 'train':
                with np.load(root / f'{split}_predictions.npz') as saved:
                    for key in ('target_lai', 'target_lai_valid', 'relative_valid',
                                'prior_lai', 'source_indices', 'row', 'col', 'year'):
                        np.testing.assert_array_equal(saved[key], a[key])
                    errors[split] = float(np.max(np.abs(prediction-saved['prediction_lai'])))
                    np.testing.assert_array_equal(prediction, saved['prediction_lai'])
                scores = metrics['state_scores'][split]
                np.testing.assert_allclose(state_rmse(a, prediction, norm), scores['lai_rmse'], rtol=0, atol=1e-8)
                np.testing.assert_allclose(projection_rmse(a, prediction, beta), scores['projection_rmse'], rtol=0, atol=1e-8)
            np.save(output / f'{split}.npy', prediction)
        prior = metrics['state_scores']['validation']
        limit = min(.99*prior['persistence_lai_rmse'], 1.02*prior['prior_lai_rmse'])
        np.testing.assert_allclose(limit, metrics['state_limit'], rtol=0, atol=1e-8)
        if metrics['state_constraint_met'] != (prior['lai_rmse'] <= limit):
            raise ValueError('State constraint report differs')
        audit = dict(crop=args.crop, origin=args.origin, mode=mode, directory=str(root),
            replay_errors=errors, maximum_replay_error=max(errors.values()), full_array_replay=True,
            hashes_metrics_alignment_verified=True, selected_epoch=metrics['selected_epoch'])
        atomic_json(manifest_path, dict(spec=spec, audit=audit,
            files={split: sha256(output / f'{split}.npy') for split in arrays}))
        records.append(audit)
        print(f'[SENSITIVE EXPORT] {args.crop} {args.origin} {mode}: {errors}', flush=True)
        del model; torch.cuda.empty_cache()
    output = ROOT / 'visualize/paper_experiments' / RESULT.name / 'audit'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f'{args.crop}_{args.origin}.json', dict(fits=3, models=records,
        maximum_replay_error=max(r['maximum_replay_error'] for r in records), full_array_replay=True))


if __name__ == '__main__':
    main()
