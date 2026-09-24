"""Fixed-epoch historical components, separate from development selection."""
import numpy as np
import torch

from multimodal_baseline import set_seed
from numeric_embedding_readout import NeuralReadout
from task_aligned_world import TaskAlignedWorld


def component_loss(value, target, kind, count, batch_count):
    # Preserve each archived trainer's arithmetic order for exact seed42 replay.
    if kind == 'mlp':
        return (value.float() - target).square().mean() * (count / batch_count)
    if kind == 'tabm':
        return ((value - target[:, None]) ** 2).mean() * count / batch_count
    raise ValueError('Unknown historical component')


def fit_fixed(features, target, kind, epochs, seed, bins=None, report=None):
    if kind not in ('mlp', 'tabm') or epochs < 0 or (kind == 'mlp' and epochs == 0):
        raise ValueError('Invalid fixed historical component recipe')
    if features.ndim != 2 or features.shape[1] != 20 or target.shape != (len(features),):
        raise ValueError('Expected twenty historical/context features and scalar residual')
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError('Historical fitting inputs must be finite')
    set_seed(seed)
    model = (TaskAlignedWorld('history_mlp') if kind == 'mlp'
             else NeuralReadout('tabm', 20, bins)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003 if kind == 'mlp' else .001,
                                 weight_decay=.0001, fused=True)
    x = torch.from_numpy(features)
    truth = torch.from_numpy(target)
    batch = 2048 if kind == 'mlp' else 1024
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        total = 0.
        for start in range(0, len(x), batch):
            selected = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(selected), 256):
                take = selected[offset:offset + 256]
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=kind == 'mlp'):
                    value = (model.yield_head(x[take].cuda()).squeeze(-1) if kind == 'mlp'
                             else model(x[take].cuda()))
                    y = truth[take].cuda()
                    loss = component_loss(value, y, kind, len(take), len(selected))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite fixed historical loss')
                loss.backward()
                total += float(loss.detach()) * len(selected)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1. if kind == 'mlp' else 5.)
            optimizer.step()
            steps += 1
        if report is not None:
            report(dict(epoch=epoch, optimizer_steps=steps, loss=total / len(x)))
    return model, steps


def residual_target(target, anchor, scale):
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Positive training residual scale required')
    return ((target.astype(float) - anchor.astype(float)) / scale).astype(np.float32)
