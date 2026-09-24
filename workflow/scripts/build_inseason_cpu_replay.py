"""Export a local, independently runnable sample of the actual frozen pipeline."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import lightgbm as lgb
import numpy as np

from inseason_signal_matching import ROOT, load_group, feature_matrix
from inseason_ndvi_reuse import tail_mask
from forecast_bridge_state import batch_arrays
from run_forecast_bridge_state import run_root as state_root
from run_inseason_signal_match import run_root as signal_root
from run_ndvi_tail_replacement import yield_prediction
from run_crop_head_signal_match import expert_root
from run_inseason_pipeline_seed_history import load_mlp
from run_ndvi_signal_permutation import check_files
from review_revision_data import sha256
from summarize_forecast_bridge import verify

RECIPES = dict(maize='gpp', rice='ndvi', soybean='ndvi', wheat='ndvi_gpp')
RATIOS = (.1, .3, .5, .7)
SOURCE = ROOT / 'Paper/iclr2027/reproducibility/inseason_demo'
DEST = ROOT / 'Paper/release/inseason_cpu_replay_v1'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def select_rows(labels, per_year):
    if per_year < 1:
        raise ValueError('At least one sample per year is required')
    out = []
    for year in np.unique(labels['year']):
        rows = np.flatnonzero(labels['year'] == year)
        order = np.lexsort((labels['col'][rows], labels['row'][rows]))
        positions = np.linspace(0, len(rows)-1, min(len(rows), per_year), dtype=int)
        out.extend(rows[order[positions]])
    return np.asarray(out, dtype=int)


def selected_climatology(encoder, rows):
    train = encoder.train
    valid = (train['relative_valid'] > 0) & (train['observed_ndvi_valid'] > 0)
    key = (train['row'].astype(np.int64) * 720 + train['col'])[:, None] * 12 + np.minimum(train['source_month'], 11)
    sums = np.bincount(key[valid], weights=train['observed_ndvi'][valid], minlength=360*720*12)
    counts = np.bincount(key[valid], minlength=len(sums))
    month_sums = np.bincount(train['source_month'][valid], weights=train['observed_ndvi'][valid], minlength=12)
    month_counts = np.bincount(train['source_month'][valid], minlength=12)
    fields = encoder.fields
    months = np.minimum(fields['source_month'][rows], 11)
    keys = (fields['row'][rows].astype(np.int64) * 720 + fields['col'][rows])[:, None] * 12 + months
    fallback = np.divide(month_sums[months], month_counts[months], out=np.zeros_like(month_sums[months]), where=month_counts[months] > 0)
    return np.divide(sums[keys], counts[keys], out=fallback, where=counts[keys] > 0)


def export_symbols(destination, name, names, imports):
    source = ROOT / 'scripts' / name
    text = source.read_text()
    nodes = ast.parse(text).body
    chunks, found = [], set()
    for node in nodes:
        label = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            label = node.targets[0].id
        if label not in names:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        chunks.append('\n'.join(text.splitlines()[start-1:node.end_lineno]))
        found.add(label)
    if found != set(names):
        raise ValueError(f'Missing source symbols: {set(names)-found}')
    extracted = '\n\n'.join(chunks) + '\n'
    output = destination / 'core' / name
    output.write_text('"""Unmodified definitions extracted from the registered source snapshot."""\n' + imports + '\n\n' + extracted)
    archive = destination / 'source_snapshot' / name
    shutil.copy2(source, archive)
    return dict(file=name, full_source_sha256=sha256(source), symbols=sorted(names),
                extracted_definitions_sha256=hashlib.sha256(extracted.encode()).hexdigest())


def export_code(destination):
    for folder in ('core', 'source_snapshot', 'assets'):
        (destination / folder).mkdir(exist_ok=True)
    records = []
    biid = ('FeedForward', 'RelevanceModulation', 'BIIDLayer', 'BIIDStack', 'HistoryEncoder',
            'PreviousLAIStateEncoder', 'WeatherTokenizer', 'CropWorldDynamics')
    records.append(export_symbols(destination, 'biid_world_model.py', biid,
        'from __future__ import annotations\nimport math\nimport torch\nfrom torch import nn'))
    for name in ('dual_remote_state.py', 'token_retention_state.py', 'forecast_bridge_state.py',
                 'ndvi_tail_replacement.py', 'task_aligned_world.py'):
        file = ROOT / 'scripts' / name
        shutil.copy2(file, destination / 'core' / name)
        shutil.copy2(file, destination / 'source_snapshot' / name)
        records.append(dict(file=name, full_source_sha256=sha256(file), copied_whole=True))
    records.append(export_symbols(destination, 'numeric_embedding_readout.py', ('NeuralReadout',),
        'import warnings\nimport torch\nfrom torch import nn\nfrom tabm import TabM\nfrom rtdl_num_embeddings import PiecewiseLinearEmbeddings'))
    records.append(export_symbols(destination, 'observed_remote_benchmark.py', ('trajectory_features',), 'import numpy as np'))
    records.append(export_symbols(destination, 'run_ndvi_signal_permutation.py', ('compose',), ''))
    for name in ('replay.py', 'test_replay.py', 'README.md', 'requirements.txt'):
        shutil.copy2(SOURCE / name, destination / name)
    return records


def export_crop(destination, crop, origin, per_year):
    folder = destination / 'assets' / crop
    folder.mkdir()
    products = RECIPES[crop].split('_')
    raw, groups, scales, epochs, sources = load_group(crop, origin)
    group = groups['validation']
    labels = group['labels']
    chosen = select_rows(labels, per_year)
    take = group['take'][chosen]
    current = signal_root(crop, origin)
    verify(current)
    with np.load(current / 'validation_labels.npz') as saved:
        for key in ('target', 'row', 'col', 'year', 'source_indices'):
            np.testing.assert_array_equal(labels[key], saved[key])
    mlp_x, mlp_labels, norm, mlp_epoch, mlp_root, mlp_sources = load_mlp(crop, origin-3)
    sources.update(mlp_sources)
    for key in ('target', 'row', 'col', 'year', 'source_indices'):
        np.testing.assert_array_equal(labels[key], mlp_labels['validation'][key])
    shutil.copy2(mlp_root / 'model_best.pt', folder / 'history.pt')
    meta = dict(crop=crop, origin=origin, fitted_through=origin-3, products=products,
        history_normalization=norm, history_selected_epoch=mlp_epoch,
        encoder_scales={p: scales[p] for p in products}, state_scales={}, heads=[])
    base_inputs = dict(history_x=mlp_x['validation'][chosen], trend=mlp_labels['validation']['baseline'][chosen])
    if crop == 'soybean':
        econfig = json.loads((expert_root(crop, origin) / 'config.json').read_text())
        tabm_root = Path(econfig['source_directory'])
        tabm_cache = ROOT / f'benchmark/cache/neural_process_readout_v1/soybean/origin_{origin}'
        tabm_meta = json.loads((tabm_cache / 'manifest.json').read_text())
        for name in ('history_validation.npy', 'validation_labels.npz'):
            if sha256(tabm_cache / name) != tabm_meta['files'][name]:
                raise ValueError('Soybean historical input changed')
            sources[str(tabm_cache / name)] = sha256(tabm_cache / name)
        with np.load(tabm_cache / 'validation_labels.npz') as saved:
            for key in ('target', 'row', 'col', 'year', 'source_indices'):
                np.testing.assert_array_equal(labels[key], saved[key])
        base_inputs['tabm_x'] = np.load(tabm_cache / 'history_validation.npy', mmap_mode='r')[chosen]
        meta['tabm_scale'] = tabm_meta['normalization']['residual_std']
        meta['tabm_selected_epoch'] = json.loads((tabm_root / 'metrics.json').read_text())['selected_epoch']
        shutil.copy2(tabm_root / 'model_best.pt', folder / 'tabm.pt')
        bin_root = ROOT / f'benchmark/cache/numeric_embedding_readout_v1/soybean/origin_{origin}'
        bins_meta = json.loads((bin_root / 'manifest.json').read_text())
        check_files(bin_root, {'history_bins.pt': bins_meta['files']['history_bins.pt']})
        shutil.copy2(bin_root / 'history_bins.pt', folder / 'history_bins.pt')
        for file in (tabm_root / 'model_best.pt', bin_root / 'history_bins.pt'):
            sources[str(file)] = sha256(file)
    stats = {}
    for product in products:
        root = state_root(crop, product, 'biid', origin-3, 42)
        marker = json.loads((root / 'complete.json').read_text())
        check_files(root, marker['files'])
        if marker['smoke'] or marker['full_fit_cutoff'] != origin-3 or marker['selected_epochs'] < 1:
            raise ValueError('Unverified original state model')
        config = json.loads((root / 'config.json').read_text())
        for name, expected in config['code_sha256'].items():
            if sha256(ROOT / 'scripts' / name) != expected:
                raise ValueError('State source changed')
        stats[product] = json.loads((root / 'normalization.json').read_text())
        meta['state_scales'][product] = stats[product][product]
        shutil.copy2(root / 'model.pt', folder / f'{product}.pt')
        for file in (root / 'model.pt', root / 'normalization.json', root / 'complete.json'):
            sources[str(file)] = sha256(file)
        base_inputs.update({f'{product}_{k}': v for k, v in batch_arrays(raw, take, stats[product], product).items()})
        base_inputs[f'{product}_climatology'] = selected_climatology(group['encoders'][product], chosen)
    heads = group['heads'][RECIPES[crop]]
    for index, (model, config, _) in enumerate(heads):
        iterations = int(model.best_iteration_ or model.booster_.current_iteration())
        model.booster_.save_model(str(folder / f'head_{index}.txt'), num_iteration=iterations)
        meta['heads'].append(dict(head=config['head'], normalization=config['normalization'], iterations=iterations))
    write_json(folder / 'metadata.json', meta)
    source_records = []
    for ratio in RATIOS:
        percent = round(ratio*100)
        active = raw['relative_valid'][group['take']] > 0
        tail = tail_mask(active, ratio)
        trajectory_file = current / f'validation_trajectories_{percent:03d}.npz'
        prediction_file = current / f'validation_{RECIPES[crop]}_biid_{percent:03d}.npz'
        with np.load(trajectory_file) as saved:
            mixed = {p: saved[p] for p in products}
        encodings = {p: group['encoders'][p].encode(mixed[p], tail, True) for p in products}
        features = feature_matrix(group['common'], encodings, tail, group['support'], RECIPES[crop])
        with np.load(prediction_file) as saved:
            prediction = saved['prediction']
        np.testing.assert_array_equal(yield_prediction(heads, features, crop), prediction)
        inputs = dict(base_inputs, common=features[chosen, :465], active=active[chosen], tail=tail[chosen])
        reference = dict(prediction=prediction[chosen], features=features[chosen],
            **{key: labels[key][chosen] for key in ('target', 'row', 'col', 'year', 'source_indices')},
            **{f'anchor_{k}': base[chosen] for k, (_, _, base) in enumerate(heads)})
        for product in products:
            values = raw[f'observed_{product}'][take]
            inputs[f'{product}_prefix'] = np.where(active[chosen] & ~tail[chosen], values, np.nan).astype(np.float32)
            reference[f'{product}_mixed'] = mixed[product][chosen]
        name = f'cutoff_{percent:03d}'
        np.savez_compressed(folder / f'{name}_inputs.npz', **inputs)
        np.savez_compressed(folder / f'{name}_reference.npz', **reference)
        for file in (trajectory_file, prediction_file):
            sources[str(file)] = sha256(file)
        source_records.append(dict(crop=crop, ratio=ratio, samples=len(chosen)))
    lineage = dict(crop=crop, original_selected_epochs=epochs, selected_rows=chosen.tolist(),
                   years=np.unique(labels['year'][chosen]).tolist(), sources=sources)
    print(f'[EXPORT] {crop}: {len(chosen)} samples, {len(products)} state models', flush=True)
    return source_records, lineage


def main(destination, origin, per_year):
    if destination.exists():
        raise ValueError('Use a new destination; existing exports are never overwritten')
    destination.mkdir(parents=True)
    code = export_code(destination)
    cases, lineage = [], []
    for crop in RECIPES:
        rows, sources = export_crop(destination, crop, origin, per_year)
        cases.extend(rows)
        lineage.append(sources)
    packages = {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'lightgbm', 'scipy', 'tabm', 'rtdl-num-embeddings')}
    write_json(destination / 'source_provenance.json', dict(code=code, builder_sha256=sha256(Path(__file__)), runtime=packages))
    files = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(destination / 'manifest.json', dict(files=files, cases=cases, origin=origin,
        per_year=per_year, sample_selection='Equally spaced grid-sorted rows within every year; no target/prediction filtering',
        public_release=False, raw_products_redistribution_authorized=False,
        training_reproduced=False, full_thirteen_year_reproduction=False))
    private = ROOT / 'benchmark/results/inseason_reproducibility_v1'
    private.mkdir(parents=True, exist_ok=True)
    write_json(private / f'{destination.name}_lineage.json', dict(bundle=str(destination), crops=lineage))
    print(json.dumps(dict(bundle=str(destination), files=len(files), cases=len(cases))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, default=DEST)
    p.add_argument('--origin', type=int, choices=(2004, 2008, 2012), default=2008)
    p.add_argument('--per-year', type=int, default=8)
    args = p.parse_args()
    main(args.destination.resolve(), args.origin, args.per_year)
