"""Export registered, non-selected-by-performance state-to-yield examples."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from inseason_13year_data import ROOT, CACHE
from inseason_nested_common import verify, hashes
from inseason_state_yield_cases import (YEAR, CUTOFF, SEED, PERCENTS, PRIMARY,
    MIN_ACTIVE, PRODUCTS, FIELDS, select_case, locate, suffix, state_scores)
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'visualize/paper_experiments/inseason_state_yield_cases_v1'
PIPELINES = ROOT / 'benchmark/results/inseason_pipeline_seeds_v1'
AUDIT = PIPELINES / 'inference_all_verification.json'
METHODS = ('biid', 'climatology', 'observed_suffix')


def selection_inputs(crop):
    folder = CACHE / crop
    manifest = json.loads((folder / 'manifest.json').read_text())
    paths = [folder / f'{k}.npy' for k in FIELDS]
    for path in paths:
        if sha256(path) != manifest['files'][path.name]:
            raise ValueError(f'Changed selection source: {path}')
    arrays = {p.stem: np.load(p, mmap_mode='r') for p in paths}
    return arrays, {str(p): sha256(p) for p in [folder / 'manifest.json', *paths]}


def register_selection():
    cases, sources = [], {}
    for crop in PRODUCTS:
        inputs, files = selection_inputs(crop)
        cases.append(select_case(crop, inputs))
        sources.update(files)
    spec = dict(year=YEAR, cutoff=CUTOFF, seed=SEED, primary_percent=PRIMARY,
        percentages=list(PERCENTS), minimum_active_slots=MIN_ACTIVE,
        products={k: list(v) for k, v in PRODUCTS.items()}, cases=cases, sources=sources,
        selector_sha256=sha256(ROOT / 'scripts/inseason_state_yield_cases.py'),
        fields_read=list(FIELDS), outcomes_used_for_selection=False,
        rule='Minimum SHA256 of state-yield-case-v1|CROP|YEAR|ROW|COL among eligible identities')
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / 'selection.json'
    if file.exists():
        old = json.loads(file.read_text())
        if {k: v for k, v in old.items() if k != 'registered_at'} != spec:
            raise ValueError('Case selection already registered differently')
        return old
    spec['registered_at'] = datetime.now().astimezone().isoformat()
    atomic_json(file, spec)
    print(json.dumps(spec['cases'], indent=2), flush=True)
    return spec


def export():
    # Require the separate selection-only invocation before loading any outcomes.
    if not (OUT / 'selection.json').exists():
        raise ValueError('Run --select first; never select from individual outcomes')
    selection = register_selection()
    audit = json.loads(AUDIT.read_text())
    if (not audit['passed'] or audit['groups'] != 36 or
            not audit['readouts_replayed_from_saved_trajectories']):
        raise ValueError('Complete source pipeline audit required')
    for name, digest in audit['pipeline_code_sha256'].items():
        if sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError('Changed source pipeline implementation')
    from inseason_signal_matching import load_group, remap_product, feature_matrix
    from inseason_ndvi_reuse import TrajectoryEncoder
    from run_ndvi_signal_permutation import compose
    from run_inseason_pipeline_seed_inference import CODE
    from run_inseason_pipeline_seed_terminal import CODE as TERMINAL_CODE
    trajectories, yields, states, case_records, replays = [], [], [], [], []
    sources = dict(selection['sources'])
    sources.update({str(p): sha256(p) for p in (AUDIT, OUT / 'selection.json')})
    for case in selection['cases']:
        crop, products = case['crop'], PRODUCTS[case['crop']]
        recipe = '_'.join(products)
        folder = PIPELINES / f'inference/{crop}/cutoff_{CUTOFF}/seed_{SEED}'
        marker = verify(folder, hashes(CODE))
        if audit['sources'][str(folder / 'complete.json')] != sha256(folder / 'complete.json'):
            raise ValueError('Pipeline no longer matches independent audit')
        config = json.loads((folder / 'config.json').read_text())
        terminal = Path(config['terminal'])
        verify(terminal, hashes(TERMINAL_CODE))
        tconfig = json.loads((terminal / 'config.json').read_text())
        with np.load(folder / 'labels.npz') as f:
            labels = {k: f[k] for k in f.files}
        global_i = locate(labels, case)
        raw, groups, scales, _, old_sources = load_group(crop, CUTOFF + 3)
        sources.update(old_sources)
        raw_i = locate(raw, case)
        matched = [(name, g, int(np.flatnonzero(g['take'] == raw_i)[0]))
                   for name, g in groups.items() if np.any(g['take'] == raw_i)]
        if len(matched) != 1:
            raise ValueError('Case not uniquely in original inference groups')
        split, group, i = matched[0]
        for key in ('year', 'row', 'col', 'target', 'source_indices'):
            np.testing.assert_array_equal(raw[key][raw_i], labels[key][global_i])
        active = raw['relative_valid'][raw_i] > 0
        np.testing.assert_array_equal(active, labels['active'][global_i])
        fit = np.flatnonzero(raw['year'] <= CUTOFF)
        encoders = {p: TrajectoryEncoder(remap_product(raw, p), fit, np.array([raw_i]), scales[p])
                    for p in products}
        truth = {p: raw[f'observed_{p}'][[raw_i]] for p in products}
        climates = {p: -encoders[p].encode(np.full_like(truth[p], scales[p]['mean']), active[None], True)
                    [:, -18:-6] * scales[p]['std'] + scales[p]['mean'] for p in products}
        models = [joblib.load(terminal / b['name'] / 'model.joblib') for b in tconfig['branches']]
        bases = [np.array([group['heads'][recipe][j][2][i]]) if b['name'] == 'trend'
                 else labels['history_anchor'][[global_i]] for j, b in enumerate(tconfig['branches'])]
        case_records.append(dict(case, split=split, pipeline=str(folder), terminal=str(terminal),
            latitude=-89.75 + .5*case['row'], longitude=-179.75 + .5*case['col'],
            target_yield=float(labels['target'][global_i]),
            historical_reference=config['reference']['baseline_label']))
        payload = dict(active=active, source_month=raw['source_month'][raw_i],
            common=group['common'][[i]], support=group['support'][[i]],
            anchor=np.concatenate(bases), target=labels['target'][[global_i]],
            strong=labels['strong'][[global_i]])
        for percent in PERCENTS:
            tail = suffix(active, percent)
            with np.load(folder / f'trajectories_{percent:03d}.npz') as f:
                np.testing.assert_array_equal(tail, f['tail'][global_i])
                biid = {p: f[p][[global_i]] for p in products}
            values = dict(biid=biid,
                climatology={p: np.where(tail[None], climates[p], truth[p]) for p in products},
                observed_suffix=truth)
            for mode in METHODS:
                encoded = {p: encoders[p].encode(values[mode][p], tail[None], True) for p in products}
                x = feature_matrix(group['common'][[i]], encoded, tail[None], group['support'][[i]], recipe)
                if x.shape != (1, 465 + 36*len(products)):
                    raise ValueError('Changed terminal dimensions')
                parts = [compose(b['config'], base, model.booster_.predict(x, num_threads=1))
                         for b, base, model in zip(tconfig['branches'], bases, models)]
                prediction = float((.5*parts[0]+.5*parts[1] if crop == 'maize' else parts[0])[0])
                if not np.isfinite(prediction):
                    raise ValueError('Nonfinite case yield')
                payload[f'features_{percent}_{mode}'] = x
                if mode != 'observed_suffix':
                    path = folder / f'tail_{percent:03d}_{mode}.npz'
                    with np.load(path) as f:
                        old = float(f['prediction'][global_i])
                    np.testing.assert_allclose(prediction, old, rtol=0, atol=1e-12)
                    replays.append(dict(crop=crop, percent=percent, mode=mode,
                        difference=abs(prediction-old)))
                target = float(labels['target'][global_i])
                yields.append(dict(crop=crop, percent=percent, mode=mode, prediction=prediction,
                    target=target, absolute_error=abs(prediction-target),
                    active_slots=int(active.sum()), hidden_slots=int(tail.sum())))
            yields.append(dict(crop=crop, percent=percent, mode='historical_reference',
                prediction=float(labels['strong'][global_i]), target=target,
                absolute_error=abs(float(labels['strong'][global_i])-target),
                active_slots=int(active.sum()), hidden_slots=int(tail.sum())))
            for p in products:
                payload[f'truth_{p}'] = truth[p][0]
                for mode in ('biid', 'climatology'):
                    payload[f'{p}_{percent}_{mode}'] = values[mode][p][0]
                    states.append(dict(crop=crop, product=p, percent=percent, mode=mode,
                        **state_scores(truth[p][0], values[mode][p][0], active, tail)))
                for slot in np.flatnonzero(active):
                    trajectories.append(dict(crop=crop, product=p, percent=percent, slot=int(slot+1),
                        source_month=int(raw['source_month'][raw_i, slot]+1),
                        hidden=bool(tail[slot]), truth=float(truth[p][0, slot]),
                        biid=float(biid[p][0, slot]), climatology=float(values['climatology'][p][0, slot])))
            np.testing.assert_array_equal(payload[f'features_{percent}_biid'][:, :465],
                payload[f'features_{percent}_observed_suffix'][:, :465])
            np.testing.assert_array_equal(payload[f'features_{percent}_biid'][:, :465],
                payload[f'features_{percent}_climatology'][:, :465])
        np.savez_compressed(OUT / f'{crop}_case.npz', **payload)
        for file in [folder / 'complete.json', *[folder / n for n in marker['files']],
                     terminal / 'complete.json', terminal / 'config.json',
                     *[terminal / b['name'] / 'model.joblib' for b in tconfig['branches']]]:
            sources[str(file)] = sha256(file)
        sources.update(config['sources'])
        print(f'[CASE EXPORTED] {crop} identity={case["source_indices"]}; same-head replay passed', flush=True)
    for name, digest in sources.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Source changed during case export: {name}')
    pd.DataFrame(trajectories).to_csv(OUT / 'trajectories.csv', index=False)
    pd.DataFrame(yields).to_csv(OUT / 'yield_predictions.csv', index=False)
    pd.DataFrame(states).to_csv(OUT / 'state_scores.csv', index=False)
    atomic_json(OUT / 'cases.json', case_records)
    files = [OUT / n for n in ('trajectories.csv', 'yield_predictions.csv', 'state_scores.csv', 'cases.json')]
    files += [OUT / f'{c}_case.npz' for c in PRODUCTS]
    atomic_json(OUT / 'export_audit.json', dict(passed=True, timestamp=datetime.now().astimezone().isoformat(),
        sources=sources, files={p.name: sha256(p) for p in files},
        code_sha256={n: sha256(ROOT / 'scripts' / n) for n in
                     ('inseason_state_yield_cases.py', Path(__file__).name)},
        replays=replays, cases=4, state_rows=len(states), yield_rows=len(yields),
        maximum_replay_error=max(r['difference'] for r in replays), new_fits=0,
        observed_suffix_keeps_cutoff_metadata=True, outcome_blind_case_selection=True,
        statistical_representativeness_claimed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--select', action='store_true')
    group.add_argument('--export', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        register_selection() if args.select else export()
