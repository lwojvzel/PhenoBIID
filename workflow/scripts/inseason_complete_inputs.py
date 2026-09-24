"""Direct seasonal inputs with the reference pipeline's complete quality fields."""
import json

import numpy as np

from crop_signal_screen_data import cache_root, active_slots
from forecast_bridge_data import (ROOT, fit_stats, history_features, climatology,
                                  remote_features)
from inseason_direct_data import physical_inputs, split_rows, RECIPES
from inseason_extension_data import prepare as extension_prepare
from inseason_ndvi_reuse import historical_support
from ndvi_tail_replacement import tail_mask
from review_revision_data import sha256


def complete_inputs(crop):
    raw, provenance = physical_inputs(crop)
    extension, extra, indices, _, features, test_support, _, sources = extension_prepare(crop)
    for key in ('source_indices', 'year', 'row', 'col', 'target'):
        np.testing.assert_array_equal(raw[key], extension[key])
    rows = split_rows(raw, 2009)
    folder = cache_root(crop, 2012)
    manifest = json.loads((folder / 'manifest.json').read_text())
    metadata_parts = []
    for split in ('train', 'validation'):
        path = folder / f'metadata_{split}.npy'
        if sha256(path) != manifest['files'][path.name]:
            raise ValueError('Changed reference metadata')
        sources[str(path)] = sha256(path)
        with np.load(folder / f'{split}_labels.npz') as labels:
            for key in ('source_indices', 'year', 'row', 'col', 'target'):
                np.testing.assert_array_equal(labels[key], raw[key][rows[split]])
        metadata_parts.append(np.load(path))
    np.testing.assert_array_equal(rows['test'], indices['test'])
    metadata_parts.append(extra['test']['metadata'])
    metadata = np.concatenate(metadata_parts)
    if metadata.shape != (len(raw['year']), 289):
        raise ValueError('Wrong complete metadata shape')
    for split in ('validation', 'test'):
        np.testing.assert_array_equal(metadata[rows[split]], features[split][:, 20:309])
    support = historical_support(raw, rows['train'], np.arange(len(raw['year'])), metadata[rows['train']])
    np.testing.assert_array_equal(support[rows['test']], test_support)
    stats = fit_stats(raw, rows['train'])
    products = RECIPES[crop].split('_')
    climate = {split: {p: climatology(raw, rows['train'], ix, p) for p in products}
               for split, ix in rows.items()}
    provenance['complete_quality_sources'] = sources
    provenance['reference_metadata_replay'] = dict(validation=True, test=True, hidden_test_support=True)
    return raw, rows, metadata, support, stats, climate, provenance


def mask_metadata(metadata, tail, support):
    if metadata.shape != (len(tail), 289) or support.shape != (*tail.shape, 6):
        raise ValueError('Unexpected reference metadata layout')
    values = np.array(metadata[:, 1:].reshape(-1, 12, 24), copy=True)
    values[..., 5:11] = np.where(tail[..., None], support, values[..., 5:11])
    return values


def structured_features(raw, take, metadata, support, stats, climo, recipe, ratio):
    a = {k: raw[k][take] for k in ('relative_valid', 'source_month')}
    active, _ = active_slots(a)
    tail = tail_mask(active, ratio)
    slots = mask_metadata(metadata[take], tail, support[take])
    weather = np.nan_to_num((raw['weather'][take]-np.asarray(stats['weather_mean'])) /
                            np.asarray(stats['weather_std']), nan=0., posinf=0., neginf=0.)
    sequence = [slots, weather, tail[..., None]]
    history, anchor = history_features(raw, take, stats)
    static = [history, raw['crop_coverage'][take, None]]
    for product in recipe.split('_'):
        previous = raw[f'previous_{product}'][take]
        prefix = np.where(active & ~tail, raw[f'observed_{product}'][take], np.nan)
        for name, physical in (('previous', previous), ('prefix', prefix)):
            valid = active & np.isfinite(physical)
            encoded = remote_features(physical, active, climo[product], stats[product])
            sequence.extend((encoded[:, :12, None], encoded[:, 18:30, None], valid[..., None]))
            static.extend((encoded[:, 12:18], encoded[:, 30:36]))
            if name == 'previous':
                quality = np.where(valid, raw[f'previous_{product}_quality'][take], 0)
                sequence.append(quality[..., None])
    seq = np.where(active[..., None], np.concatenate(sequence, -1), 0).astype(np.float32)
    context = np.concatenate(static, 1).astype(np.float32)
    nproducts = len(recipe.split('_'))
    if seq.shape != (len(take), 12, 38+7*nproducts) or context.shape != (len(take), 21+24*nproducts):
        raise ValueError('Unexpected seasonal model dimensions')
    if not np.isfinite(seq).all() or not np.isfinite(context).all():
        raise ValueError('Nonfinite model input')
    return dict(sequence=seq, static=context, valid=active.astype(np.float32),
                anchor=anchor.astype(float), tail=tail)


def flat_features(inputs):
    return np.concatenate((inputs['sequence'].reshape(len(inputs['static']), -1), inputs['static']), 1)
