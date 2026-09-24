"""Replay all retained historical variants, including stronger older controls."""
import json
from pathlib import Path

import numpy as np

from inseason_13year_data import ROOT
from inseason_baseline_tables import history_name, aligned_prediction
from review_revision_data import sha256

PREFIX='extended:'
DISPLAY={
    'TabM residual (full window)':'TabM residual (full window)',
    'TabM residual (PLE)':'TabM residual (PLE)',
    'MLP residual (in-sample, 1993+)':'MLP residual (1993+, in-sample)',
    'Linear-NCA residual':'Linear-NCA residual',
}


def extended_predictions(crop,cutoff,labels,sources):
    origin=cutoff+3
    path=ROOT/f'benchmark/results/inseason_ndvi_reuse_v1/pipelines/{crop}/origin_{origin}/seed_42/history_reference.json'
    spec=json.loads(path.read_text())
    sources[str(path)]=sha256(path)
    audit_path=ROOT/'visualize/paper_experiments/inseason_paper_v1/baseline_audit.json'
    pinned=json.loads(audit_path.read_text())['current']['sources']
    sources[str(audit_path)]=sha256(audit_path)
    if sources[str(path)]!=pinned[str(path)]:
        raise ValueError('Archived historical inventory changed')
    outputs={}
    for candidate in spec['candidates']:
        name=PREFIX+history_name(candidate['path'])
        chunks=[]
        for split in (('validation','test') if cutoff==2009 else ('validation',)):
            mask=labels['year']<=origin if split=='validation' else labels['year']>origin
            part={k:v[mask] for k,v in labels.items()}
            file=Path(candidate['path'])/f'{split}_predictions.npz'
            digest=sha256(file)
            if digest!=pinned[str(file)]:
                raise ValueError('Archived historical prediction changed')
            sources[str(file)]=digest
            chunks.append(aligned_prediction(file,part,candidate['key']))
        outputs[name]=np.concatenate(chunks)
    if len(outputs)!=15:
        raise ValueError('Incomplete extended historical library')
    return outputs


def display(name):
    plain=name.removeprefix(PREFIX)
    return DISPLAY.get(plain,plain)
