"""Frozen crop-specific product heads on a common observable-prefix interface."""
import json
from pathlib import Path
import joblib
import numpy as np

from forecast_bridge_data import ROOT, load
from crop_signal_screen_data import cache_root, GPP
from crop_signal_history_reference import IDENTITY
from inseason_extension_data import prepare as extension_prepare
from inseason_ndvi_reuse import TrajectoryEncoder, historical_support
from run_inseason_ndvi_reuse import original_heads, run_root as screen_root
from run_inseason_extension import run_root as extension_root, cached_prediction
from run_crop_head_signal_match import references, expert_root, source_root
from run_ndvi_signal_permutation import check_files
from review_revision_data import sha256
from summarize_forecast_bridge import verify

RECIPES = ('ndvi', 'gpp', 'ndvi_gpp')


def physical_gpp(raw, previous=False):
    values = np.full(raw['source_month'].shape, np.nan, np.float32)
    quality = np.zeros_like(values)
    sources = {}
    for year in np.unique(raw['year']):
        source_year = int(year)-int(previous)
        if source_year < 1982:
            continue
        take = np.flatnonzero(raw['year'] == year)
        mm = np.minimum(raw['source_month'][take], 11)
        rr, cc = raw['row'][take, None], raw['col'][take, None]
        active = raw['relative_valid'][take] > 0
        for field, prefix in (('value','gpp_daily_rate'),('quality','valid_area_fraction')):
            file = GPP/'monthly_0p5'/f'{prefix}_{source_year}.npy'
            sources[str(file)] = sha256(file)
            selected = np.load(file,mmap_mode='r')[mm,rr,cc]
            if field == 'value':
                values[take] = np.where(active,selected,np.nan)
            else:
                quality[take] = np.where(active,selected,0)
    quality = np.where(np.isfinite(values),quality,0).astype(np.float32)
    return values, quality, sources


def remap_product(raw, product):
    if product not in ('ndvi','gpp'):
        raise ValueError(product)
    mapped = dict(raw)
    for prefix in ('observed','previous'):
        mapped[f'{prefix}_ndvi'] = raw[f'{prefix}_{product}']
    mapped['previous_ndvi_quality'] = raw[f'previous_{product}_quality']
    return mapped


def feature_matrix(common, encodings, tail, support, recipe):
    products = recipe.split('_')
    if recipe not in RECIPES or common.shape[1] != 465:
        raise ValueError('Unexpected product interface')
    base = common.copy()
    metadata = base[:,21:309].reshape(-1,12,24)
    metadata[...,5:11] = np.where(tail[...,None],support,metadata[...,5:11])
    return np.concatenate((base,*[encodings[p] for p in products]),1)


def load_group(crop,origin):
    if origin == 2012:
        raw, extra, indices, _, x, _, _, sources = extension_prepare(crop)
    else:
        raw, _ = load(crop)
        raw = dict(raw)
        indices = dict(validation=np.flatnonzero((raw['year']>origin-3)&(raw['year']<=origin)))
        x, extra, sources = {}, {}, {}
    raw['observed_gpp'], _, gsources = physical_gpp(raw)
    raw['previous_gpp'], raw['previous_gpp_quality'], psources = physical_gpp(raw,True)
    sources.update(gsources); sources.update(psources)
    fit = np.flatnonzero(raw['year']<=origin-3)
    cache = cache_root(crop,origin)
    cm = json.loads((cache/'manifest.json').read_text())
    sources[str(cache/'manifest.json')] = sha256(cache/'manifest.json')
    names = [f'{p}_validation.npy' for p in ('history','metadata','weather','ndvi','gpp')]
    names += ['metadata_train.npy','train_labels.npz']
    check_files(cache,{n:cm['files'][n] for n in names})
    with np.load(cache/'train_labels.npz') as f:
        for k in IDENTITY:
            np.testing.assert_array_equal(raw[k][fit],f[k])
    if 'validation' not in x:
        x['validation'] = np.concatenate([np.load(cache/f'{p}_validation.npy') for p in ('history','metadata','weather','ndvi')],1)
    scales = dict(ndvi=cm['spec']['upstream']['upstream']['ndvi_normalization'],
                  gpp=cm['spec']['upstream']['gpp_training_normalization'])
    groups = {}
    for split,take in indices.items():
        label = {k:raw[k][take] for k in IDENTITY}
        encoders = {p:TrajectoryEncoder(remap_product(raw,p),fit,take,scales[p]) for p in scales}
        encodings = {p:encoders[p].encode(raw[f'observed_{p}'][take]) for p in scales}
        if split == 'validation':
            for p in scales:
                np.testing.assert_array_equal(encodings[p],np.load(cache/f'{p}_validation.npy'))
        np.testing.assert_array_equal(encodings['ndvi'],x[split][:,-36:])
        support = historical_support(raw,fit,take,np.load(cache/'metadata_train.npy',mmap_mode='r'))
        prior = screen_root(crop,origin) if split == 'validation' else extension_root(crop)
        verify(prior)
        with np.load(prior/'labels.npz') as f:
            for k in IDENTITY:
                np.testing.assert_array_equal(label[k],f[k])
            label['strong'] = f['strong']
        sources[str(prior/'complete.json')] = sha256(prior/'complete.json')
        groups[split] = dict(take=take,labels=label,encoders=encoders,observed_encodings=encodings,
            common=x[split][:,:465],support=support,reference=prior)
    val_heads,epochs,hsources = original_heads(crop,origin,groups['validation']['labels'])
    sources.update(hsources)
    cfg = json.loads((expert_root(crop,origin)/'config.json').read_text())
    if 'test' in groups:
        if crop == 'soybean':
            base_source,key = Path(cfg['source_directory']),'component_prediction'
        else:
            base_source = ROOT/f'benchmark/results/task_aligned_world_v1/pipelines/{crop}/origin_{origin}/history_mlp/seed_42'
            key = 'prediction'
        base = cached_prediction(base_source/'test_predictions.npz',groups['test']['labels'],key,sources)
        test_heads = [(m,c,extra['test']['baseline'].astype(float) if branch=='trend' else base)
            for (m,c,_),(branch,_) in zip(val_heads,references(crop,origin))]
    else:
        test_heads = None
    for split,g in groups.items():
        original = val_heads if split=='validation' else test_heads
        g['heads'] = dict(ndvi=original)
        for recipe, family in (('gpp','crop_head_signal_match_v1'),('ndvi_gpp','crop_head_signal_pair_v1')):
            parent = ROOT/f'benchmark/results/{family}/pipelines/{crop}/origin_{origin}/seed_42/{recipe}'
            check_files(parent,json.loads((parent/'audit.json').read_text())['files'])
            recipe_cfg = json.loads((parent/'config.json').read_text())
            if len(recipe_cfg['branches']) != len(original):
                raise ValueError('Crop branch count changed')
            heads = []
            for branch,(old_model,old_cfg,base) in zip(recipe_cfg['branches'],original):
                weight = parent/f'{branch["name"]}.joblib'
                norm = old_cfg['normalization']
                np.testing.assert_allclose([norm.get('center',norm.get('residual_mean')),norm.get('scale',norm.get('residual_std'))],
                    [branch['normalization'].get('center',branch['normalization'].get('residual_mean')),
                     branch['normalization'].get('scale',branch['normalization'].get('residual_std'))],rtol=0,atol=0)
                model = joblib.load(weight)
                heads.append((model,old_cfg,base))
                sources[str(weight)] = sha256(weight)
            g['heads'][recipe] = heads
            sources[str(parent/'audit.json')] = sha256(parent/'audit.json')
            g.setdefault('observed_sources',{})[recipe] = parent/'validation_predictions.npz'
        g['observed_sources']['ndvi'] = source_root(crop,origin)/'validation_predictions.npz'
    return raw,groups,scales,epochs,sources
