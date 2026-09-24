"""Package full evaluation cohorts and the independently rebuilt components."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from audit_inseason_component_handoffs import LABELS, matched_identity
from build_inseason_cpu_replay import export_symbols, selected_climatology
from export_inseason_state_retrain import export_code, write_json
from forecast_bridge_state import batch_arrays
from inseason_13year_data import RECIPES, BLOCKS
from inseason_signal_matching import load_group
from ndvi_tail_replacement import tail_mask
from review_revision_data import ROOT, sha256

EVIDENCE = ROOT / 'benchmark/results/inseason_reproducibility_v1'


def require_complete_matrix(jobs):
    expected = {(f'{crop}/cutoff_{cutoff}', seed) for crop in RECIPES
                for cutoff in BLOCKS for seed in (42, 45, 48)}
    actual = [(j['case'], j['seed']) for j in jobs]
    if len(actual) != 36 or set(actual) != expected:
        raise ValueError('Require all four crops, three windows, and three seeds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError('Do not overwrite an existing reconstruction package')
    sources, bundles, completed = {}, {}, {}

    def read_json(path):
        sources[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def copy(source, destination):
        sources[str(source)] = sha256(source)
        shutil.copy2(source, destination)
        if sha256(destination) != sources[str(source)]:
            raise ValueError('Copied reconstruction asset changed')

    handoff = EVIDENCE / 'component_handoffs_20260909.json'
    if not read_json(handoff)['passed']:
        raise ValueError('Full-row training and historical handoffs have not passed')
    for stage in ('history', 'state', 'terminal'):
        folder = EVIDENCE / f'{stage}_retrain_20260909'
        complete, audit = read_json(folder/'complete.json'), read_json(folder/'original_weight_audit.json')
        if not complete['passed'] or not audit['passed']:
            raise ValueError('A required reconstruction stage is incomplete')
        bundle = Path(complete['relocated']).resolve()
        if bundle.is_relative_to(ROOT):
            raise ValueError('Components must originate from independent relocation')
        bundles[stage], completed[stage] = bundle, complete
    destination = args.destination.resolve()
    destination.mkdir(parents=True)
    code = export_code(destination)
    additions = (
        ('ndvi_tail_replacement.py', ('tail_mask',), 'import numpy as np'),
        ('inseason_signal_matching.py', ('RECIPES', 'feature_matrix'), 'import numpy as np'),
        ('observed_remote_benchmark.py', ('trajectory_features',), 'import numpy as np'),
        ('run_ndvi_signal_permutation.py', ('compose',), ''),
    )
    for entry in additions:
        code.append(export_symbols(destination, *entry))
    (destination/'core/ndvi_tail_replacement.py').rename(destination/'core/tail_math.py')
    for part in ('core', 'source_snapshot'):
        copy(ROOT/'scripts/ndvi_tail_replacement.py', destination/part/'ndvi_tail_replacement.py')
    (destination/'core/chain_math.py').write_text(
        'from tail_math import tail_mask\nfrom inseason_signal_matching import feature_matrix\n'
        'from run_ndvi_signal_permutation import compose\n')
    copy(ROOT/'Paper/iclr2027/reproducibility/reconstructed_chain/run.py', destination/'run.py')
    copy(ROOT/'Paper/iclr2027/reproducibility/reconstructed_chain/README.md', destination/'README.md')
    terminal_outputs = {(r['job']['case'], r['job']['seed']):Path(r['results'])
                        for r in completed['terminal']['cases']}
    jobs = []
    for crop, recipe_name in RECIPES.items():
        products = recipe_name.split('_')
        for cutoff in BLOCKS:
            name = f'{crop}/cutoff_{cutoff}'
            case = destination/'cases'/name
            case.mkdir(parents=True)
            raw, groups, scales, _, original_sources = load_group(crop, cutoff+3)
            sources.update(original_sources)
            state_case = read_json(bundles['state']/f'cases/{name}/case.json')
            stats = state_case['statistics']
            split_sizes = {}
            for split, group in groups.items():
                folder = case/split
                folder.mkdir()
                take = group['take']
                split_sizes[split] = len(take)
                active = raw['relative_valid'][take] > 0
                history_labels_path = bundles['history']/f'cases/{name}/mlp/reference/{split}_labels.npz'
                sources[str(history_labels_path)] = sha256(history_labels_path)
                with np.load(history_labels_path) as saved:
                    history_labels = {k:saved[k] for k in saved.files}
                matched_identity(group['labels'], history_labels)
                if crop == 'maize':
                    np.testing.assert_array_equal(group['heads'][recipe_name][0][2],
                                                  history_labels['baseline'].astype(float))
                for key, array in dict(active=active, common=group['common'], support=group['support'],
                                       trend=history_labels['baseline'].astype(float)).items():
                    np.save(folder/f'{key}.npy', array)
                for product in products:
                    observed = raw[f'observed_{product}'][take]
                    if np.isfinite(observed[~active]).any():
                        raise ValueError('Inactive observations require explicit original handling')
                    np.save(folder/f'observed_{product}.npy', observed)
                    for key, array in batch_arrays(raw, take, stats, product).items():
                        np.save(folder/f'{product}_{key}.npy', array)
                    climate = selected_climatology(group['encoders'][product], np.arange(len(take)))
                    np.save(folder/f'climatology_{product}.npy', climate)
                    for percent in (10, 30, 50):
                        tail = tail_mask(active, percent/100)
                        prefix = np.where(active & ~tail, observed, np.nan).astype(np.float32)
                        np.save(folder/f'prefix_{product}_{percent:03d}.npy', prefix)
            years = np.concatenate([g['labels']['year'] for g in groups.values()])
            if tuple(np.unique(years)) != BLOCKS[cutoff]:
                raise ValueError('Evaluation cohort does not cover the complete registered window')
            write_json(case/'case.json', dict(crop=crop, cutoff=cutoff, recipe=recipe_name, products=products,
                splits=split_sizes, evaluation_years=np.unique(years).tolist(), feature_width=465+36*len(products),
                encoder_scales={p:scales[p] for p in products}, state_scales={p:stats[p] for p in products}))
            for seed in (42, 45, 48):
                target = case/f'seed_{seed}'
                reference = target/'reference'
                reference.mkdir(parents=True)
                original = ROOT/f'benchmark/results/inseason_pipeline_seeds_v1/inference/{name}/seed_{seed}'
                original_marker = read_json(original/'complete.json')
                for file, checksum in original_marker['files'].items():
                    if sha256(original/file) != checksum:
                        raise ValueError('Original full-pipeline reference changed')
                    sources[str(original/file)] = checksum
                original_labels = np.load(original/'labels.npz')
                offset = 0
                for split, group in groups.items():
                    size = split_sizes[split]
                    matched_identity(group['labels'], {k:original_labels[k][offset:offset+size] for k in LABELS})
                    component = 'tabm' if crop == 'soybean' else 'mlp'
                    history = bundles['history']/f'reconstructed/{name}/seed_{seed}/{component}/checked_{split}_prediction.npy'
                    copy(history, target/f'history_{split}.npy')
                    np.testing.assert_array_equal(np.load(history), original_labels['history_anchor'][offset:offset+size])
                    for percent in (10, 30, 50):
                        with np.load(original/f'trajectories_{percent:03d}.npz') as trajectories:
                            for product in products:
                                np.save(reference/f'{split}_{product}_{percent:03d}.npy', trajectories[product][offset:offset+size])
                    offset += size
                if offset != len(original_labels['year']):
                    raise ValueError('Incomplete split concatenation')
                original_labels.close()
                for product in products:
                    copy(bundles['state']/f'reconstructed/{name}/{product}/seed_{seed}/model.pt', target/f'state_{product}.pt')
                recipe_path = bundles['terminal']/f'cases/{name}/seed_{seed}/recipe.json'
                recipe = read_json(recipe_path)
                copy(recipe_path, target/'recipe.json')
                for i in range(len(recipe['branches'])):
                    copy(terminal_outputs[name, seed]/f'branch_{i}.txt', target/f'head_{i}.txt')
                for percent, mode in [(0, 'observed')] + [(p, m) for p in (10,30,50) for m in ('biid','climatology')]:
                    with np.load(original/f'tail_{percent:03d}_{mode}.npz') as values:
                        np.save(reference/f'tail_{percent:03d}_{mode}.npy', values['prediction'])
                jobs.append(dict(case=name, seed=seed))
            print(f'[COMPLETE CHAIN EXPORTED] {name} {split_sizes}', flush=True)
    require_complete_matrix(jobs)
    for record in code:
        sources[str(ROOT/'scripts'/record['file'])] = record['full_source_sha256']
    for path in (Path(__file__), ROOT/'scripts/build_inseason_cpu_replay.py', ROOT/'scripts/export_inseason_state_retrain.py'):
        sources[str(path.resolve())] = sha256(path)
    for name, checksum in sources.items():
        if sha256(Path(name)) != checksum:
            raise ValueError('Source changed during complete-cohort export: '+name)
    write_json(destination/'provenance.json', dict(sources=sources, code=code,
        original_group_builder_used_only_during_export=True, original_reference_outputs_excluded_from_forward=True,
        state_readout_and_history_components_reconstructed=True, raw_preprocessing_reproduced=False))
    files = {str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(destination/'manifest.json', dict(files=files, jobs=jobs, systems=36, state_components=45,
        independent_recipe_confirmation=False, raw_processing_reproduced=False, public_release_ready=False,
        original_scientific_results_replaced=False, scope='Complete evaluation using reconstructed components and original frozen physical encodings'))
    print(f'[CHAIN PACKAGE READY] {len(files)} files, {len(jobs)} complete systems', flush=True)


if __name__ == '__main__':
    main()
