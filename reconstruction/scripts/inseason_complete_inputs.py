from __future__ import annotations
import json
import numpy as np
from crop_signal_screen_data import cache_root, active_slots
from forecast_bridge_data import ROOT, fit_stats, history_features, climatology, remote_features
from inseason_direct_data import physical_inputs, split_rows, RECIPES
from inseason_extension_data import prepare as extension_prepare
from inseason_ndvi_reuse import historical_support
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
