"""Freeze established validation-chosen history controls for the signal screen."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_world_anchor import leave_year_out, coefficient
from crop_signal_screen_data import ROOT, CROPS, ORIGINS, CONDITIONS, run_root
from stable_remote_data import cache_root as original_cache_root
from run_crop_signal_screen import yearly_rmse
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'visualize/paper_experiments/crop_signal_screen_v1/strong_history_reference'
CHOICES = ROOT / 'visualize/paper_experiments/neighbor_state_readout_v1/control_choices.csv'
IDENTITY = ('target', 'source_indices', 'year', 'row', 'col')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CHOICES, usecols=['crop', 'origin', 'history_directory'])
    if len(frame) != 12 or set(zip(frame.crop, frame.origin)) != {(c, o) for c in CROPS for o in ORIGINS}:
        raise ValueError('Incomplete established controls')
    records, annual = [], []
    for row in frame.itertuples():
        root = Path(row.history_directory)
        calibration_path = root / 'calibration.json'
        cal = json.loads(calibration_path.read_text())
        weight = Path(cal['weight'])
        if (root / 'audit.json').exists():
            audit_path = root / 'audit.json'
            audit = json.loads(audit_path.read_text())
            if audit['smoke'] or not audit['full_array_replay'] or audit['maximum_replay_error'] != 0:
                raise ValueError('Invalid existing history audit')
            expected = audit['weight_sha256']
        else:
            if 'linear_state_yield_v1' not in root.parts:
                raise ValueError('Unsupported history audit layout')
            audit_path = root.parents[1] / 'complete.json'
            audit = json.loads(audit_path.read_text())
            if audit['maximum_replay_error'] != 0:
                raise ValueError('Invalid linear history replay')
            expected = audit['weights_sha256'][str(weight)]
        if sha256(weight) != expected:
            raise ValueError('History weight differs from original verified weight')
        original = original_cache_root(row.crop, row.origin, 0)
        meta = json.loads((original / 'manifest.json').read_text())
        original_path = original / 'validation.npz'
        if sha256(original_path) != meta['splits']['validation']['file_sha256']:
            raise ValueError('Original validation cohort changed')
        path = root / 'validation_predictions.npz'
        with np.load(path) as source, np.load(original_path) as labels:
            for k in IDENTITY:
                np.testing.assert_array_equal(source[k], labels[k])
            y, h, p, years = (source[k] for k in ('target', 'history_prediction', 'component_prediction', 'year'))
            if coefficient(y, h, p, years) != cal['coefficient']:
                raise ValueError('Frozen full-year calibration does not replay')
            np.testing.assert_array_equal(source['prediction'], h+cal['coefficient']*(p-h))
            held, coefficients = leave_year_out(y, h, p, years)
            if coefficients != cal['leave_year_out_coefficients']:
                raise ValueError('Frozen leave-year-out calibration does not replay')
            values = dict(prediction=held, raw_prediction=p, anchor_prediction=h,
                          **{k: source[k] for k in IDENTITY})
        destination = OUT / f'{row.crop}_{row.origin}_validation.npz'
        if destination.exists():
            with np.load(destination) as old:
                for k, v in values.items():
                    np.testing.assert_array_equal(old[k], v)
        else:
            np.savez_compressed(destination, **values)
        for year, rmse in yearly_rmse(values['target'], held, values['year']).items():
            annual.append(dict(crop=row.crop, origin=row.origin, year=int(year), rmse=rmse))
        records.append(dict(crop=row.crop, origin=row.origin, directory=str(root), weight=str(weight),
            weight_sha256=expected, original_audit=str(audit_path), original_audit_sha256=sha256(audit_path),
            source_prediction_sha256=sha256(path), calibration_sha256=sha256(calibration_path),
            original_validation_sha256=sha256(original_path), output=str(destination), output_sha256=sha256(destination)))
    specification = dict(records=records, selection_csv_sha256=sha256(CHOICES),
        rule='Reuse previously validation-chosen per-origin history head; reconstruct its saved leave-calibration-year-out predictions',
        evaluation_split_loaded=False, new_fits=0, original_weight_hashes_verified=True,
        limitation='Source model selection used these validation years; leave-year-out calibration is not fully nested validation',
        code_sha256=sha256(Path(__file__)), calibration_code_sha256=sha256(ROOT / 'scripts/calibrate_world_anchor.py'))
    marker = OUT / 'manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != specification:
        raise ValueError('Frozen history reference changed')
    atomic_json(marker, specification)
    pd.DataFrame(annual).to_csv(OUT / 'annual_rmse.csv', index=False)
    print('[STRONG HISTORY READY] 12 weights verified; validation identities and LOO calibration replayed; no new fit.', flush=True)


def compare():
    meta = json.loads((OUT / 'manifest.json').read_text())
    paired = []
    for record in meta['records']:
        path = Path(record['output'])
        if sha256(path) != record['output_sha256']:
            raise ValueError('History reference prediction changed')
        with np.load(path) as reference:
            for condition in CONDITIONS:
                root = run_root(record['crop'], record['origin'], condition)
                audit = json.loads((root / 'audit.json').read_text())
                if not audit['full_validation_replay'] or audit['smoke']:
                    raise ValueError('Incomplete formal signal fit')
                file = root / 'validation_predictions.npz'
                if sha256(file) != audit['files'][file.name]:
                    raise ValueError('Signal validation predictions changed')
                with np.load(file) as current:
                    for k in IDENTITY:
                        np.testing.assert_array_equal(reference[k], current[k])
                    numer = yearly_rmse(current['target'], current['prediction'], current['year'])
                    denom = yearly_rmse(reference['target'], reference['prediction'], reference['year'])
                    for year in numer:
                        paired.append(dict(crop=record['crop'], origin=record['origin'], year=int(year),
                            condition=condition, rmse=numer[year], strong_history_rmse=denom[year],
                            gain_percent=100*(1-numer[year]/denom[year])))
    if len(paired) != 216:
        raise ValueError('Incomplete 72-fit strong-history comparison')
    frame = pd.DataFrame(paired)
    frame.to_csv(OUT / 'paired_gains.csv', index=False)
    summary = frame.groupby(['crop', 'condition']).agg(mean_gain_percent=('gain_percent', 'mean'),
        positive_years=('gain_percent', lambda v: int((v > 0).sum())), pairs=('year', 'count')).reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    atomic_json(OUT / 'comparison_audit.json', dict(fits_compared=72, validation_year_pairs=216,
        exact_target_and_identity_matches=True, evaluation_split_loaded=False,
        reference_manifest_sha256=sha256(OUT / 'manifest.json')))
    report = ['# 固定头信号筛选：与已有强历史参考比较', '',
        '采用已按验证选定的12个历史模型，核验原权重并重新计算原登记的留一年校准预测；没有根据本轮结果换对手。新信号模型无额外后验校准。', '',
        summary.to_markdown(index=False), '',
        '正数为相对强历史的逐年RMSE降低，九个验证年等权。不是评价集收益，也不是多种子确认。'
        '同头遥感增量仍需同时看固定头报告中的weather对照，不能用跨模型比较代替模态消融。', '',
        '旧历史头本身也使用过这些验证年选型；留一年校准不等同于完全嵌套验证。这张表用于避免弱基线误导，不能冒充独立确认。', '',
        f'来源、权重与配对：`{OUT}`。']
    (ROOT / 'Paper/task/固定头信号筛选_强历史参考比较_20260907.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('[STRONG HISTORY COMPARISON COMPLETE] 72 fits, 216 validation pairs.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    compare() if args.compare else prepare()
