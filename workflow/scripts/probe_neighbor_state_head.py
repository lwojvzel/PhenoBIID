"""Registered synthetic resource/replay tests for the official retrieval model."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from neighbor_state_head import ROOT, VENDOR, make, candidate_indices, encode_memory, from_memory, verify_source
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json
from shared_gpu_queue import run_shared_stage

RESULT = ROOT / 'benchmark/results/neighbor_state_probe_v1'
CASES = [(kind, n, d) for kind in ('linear', 'modern')
         for n, d in ((1024, 104), (400000, 104), (400000, 560))]


def one(kind, n, d):
    if (kind, n, d) not in CASES:
        raise ValueError('Unregistered synthetic probe')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(3000*2**20/torch.cuda.get_device_properties(0).total_memory)
    root = RESULT / f'{kind}_{n}_{d}'; root.mkdir(parents=True, exist_ok=True)
    source = verify_source()
    spec = dict(kind=kind, rows=n, features=d, seed=42, synthetic_only=True,
        crop_models_fitted=0, query_batch=256, encoding_batch=1024,
        official_source=source, code_hashes={f:sha256(ROOT / 'scripts' / f)
            for f in ('neighbor_state_head.py', 'probe_neighbor_state_head.py')},
        vendor_hashes={str(p.relative_to(VENDOR)):sha256(p) for p in VENDOR.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    marker = root / 'config.json'
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError('Synthetic probe recipe changed')
    atomic_json(marker, spec)
    start = time.monotonic()
    x = torch.randn(n, d, device='cuda'); y = x[:,0]+.3*x[:,1]*x[:,2]+.2*torch.randn(n, device='cuda')
    query = torch.randn(1024, d, device='cuda')
    train_take = torch.arange(512, device='cuda'); candidates = candidate_indices(n, train_take, 'cuda')
    sample_count = min(4096, len(candidates))
    model = make(d, kind, sample_count/len(candidates)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    prediction = model(x[train_take], y[train_take], x[candidates], y[candidates], True)
    loss = (prediction-y[train_take]).square().mean(); loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Invalid synthetic training gradient')
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    after = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    if all(torch.equal(before[k], after[k]) for k in before):
        raise ValueError('Synthetic optimizer did not update model')
    torch.save(after, root / 'synthetic_model.pt')
    del optimizer, loss, prediction, candidates, before, after
    gc.collect(); torch.cuda.empty_cache(); model.eval()
    training_seconds = time.monotonic()-start
    start = time.monotonic(); keys = encode_memory(model, x, 1024)
    torch.cuda.synchronize(); encoding_seconds = time.monotonic()-start

    def predict(m, memory, batch):
        return torch.cat([from_memory(m, part, memory, y).cpu() for part in query.split(batch)])

    start = time.monotonic(); result = predict(model, keys, 256)
    predict_seconds = time.monotonic()-start
    repeated = predict(model, keys, 256); chunked = predict(model, keys, 128)
    errors = dict(repeat=float((result-repeated).abs().max()), chunk=float((result-chunked).abs().max()))
    atomic_json(root / 'numerical_check.json', errors)
    torch.testing.assert_close(result, repeated, rtol=0, atol=0)
    torch.testing.assert_close(result, chunked, rtol=0, atol=1e-4)
    if n == 1024:
        with torch.no_grad():
            official = torch.cat([model(part, None, x, y, False).cpu() for part in query.split(256)])
        errors['official'] = float((result-official).abs().max())
        torch.testing.assert_close(result, official, rtol=0, atol=1e-4)
    del keys, model; gc.collect(); torch.cuda.empty_cache()
    restored = make(d, kind).cuda().eval()
    restored.load_state_dict(torch.load(root / 'synthetic_model.pt', weights_only=True, map_location='cuda'))
    keys = encode_memory(restored, x, 1024); replay = predict(restored, keys, 256)
    errors['checkpoint_replay'] = float((result-replay).abs().max())
    atomic_json(root / 'numerical_check.json', errors)
    torch.testing.assert_close(result, replay, rtol=0, atol=0)
    del keys; gc.collect(); torch.cuda.empty_cache()
    keys = encode_memory(restored, x, 1024); rebuilt = predict(restored, keys, 256)
    errors['memory_rebuild'] = float((result-rebuilt).abs().max())
    atomic_json(root / 'numerical_check.json', errors)
    torch.testing.assert_close(result, rebuilt, rtol=0, atol=0)
    peak = torch.cuda.max_memory_allocated()/2**20
    if peak > 3000:
        raise ValueError('Resource budget exceeded')
    np.save(root / 'prediction.npy', result.numpy())
    audit = dict(spec=spec, errors=errors, synthetic_optimizer_steps=1, crop_models_fitted=0,
        peak_allocated_mib=peak, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        training_seconds=training_seconds, encoding_seconds=encoding_seconds,
        prediction_1024_seconds=predict_seconds, parameters=sum(p.numel() for p in restored.parameters()),
        checkpoint_sha256=sha256(root / 'synthetic_model.pt'))
    atomic_json(root / 'audit.json', audit)
    print(json.dumps(audit), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--kind', choices=('linear','modern'))
    p.add_argument('--rows', type=int); p.add_argument('--features', type=int)
    args = p.parse_args()
    if args.kind:
        one(args.kind, args.rows, args.features); return
    jobs = [dict(key=f'neighbor_probe__{k}__{n}__{d}', marker=str(RESULT / f'{k}_{n}_{d}/audit.json'),
        command=[sys.executable, str(Path(__file__).resolve()), '--kind', k, '--rows', str(n), '--features', str(d)])
        for k,n,d in CASES]
    run_shared_stage(jobs, 'synthetic_resource', ['4','5','6'], RESULT,
        ROOT / 'benchmark/logs' / RESULT.name, 0)


if __name__ == '__main__':
    main()
