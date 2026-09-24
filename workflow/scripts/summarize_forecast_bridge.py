"""Replay every completed readout and report paired annual gains without selection."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from forecast_bridge_data import ROOT, CROPS, ORIGINS, load, identity_hash
from export_forecast_bridge import run_root as export_root
from run_forecast_bridge_readout import CONDITIONS, run_root, features, compose, inner_split
from crop_signal_history_reference import OUT as STRONG, IDENTITY
from run_crop_signal_screen import yearly_rmse
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json


def verify(directory, marker='complete.json'):
    file = directory / marker
    audit = json.loads(file.read_text())
    if audit.get('evaluation_arrays_loaded', False):
        raise ValueError('Unexpected held-out evaluation access')
    for n, digest in audit['files'].items():
        if sha256(directory / n) != digest:
            raise ValueError(f'Changed result asset: {directory / n}')
    return sha256(file)


def summarize(frame):
    if frame.duplicated(['crop', 'condition', 'seed', 'year']).any():
        raise ValueError('Duplicate paired year')
    annual = frame.groupby(['crop', 'condition', 'year'], sort=False).agg(
        rmse=('rmse', 'mean'), **{f'gain_{k}': (f'gain_{k}', 'mean') for k in ('strong', 'weather', 'previous')}).reset_index()
    records = []
    rng = np.random.default_rng(20260907)
    draws = rng.integers(0, 9, (10000, 9))
    for (crop, condition), rows in annual.groupby(['crop', 'condition'], sort=False):
        if len(rows) != 9:
            raise ValueError('Expected exactly nine complete years')
        record = dict(crop=crop, condition=condition, rmse=float(rows.rmse.mean()))
        for ref in ('strong', 'weather', 'previous'):
            values = rows.sort_values('year')[f'gain_{ref}'].to_numpy()
            lo, hi = np.quantile(values[draws].mean(1), [.025, .975])
            record.update({f'gain_{ref}': float(values.mean()), f'positive_{ref}_years': int((values > 0).sum()),
                f'pass_{ref}': bool(values.mean() > 0 and (values > 0).sum() >= 5),
                f'ci_low_{ref}': float(lo), f'ci_high_{ref}': float(hi)})
        records.append(record)
    return annual, pd.DataFrame(records)


def main(args):
    seeds = [int(s) for s in args.seeds.split(',')]
    if seeds not in ([42], [42, 45, 48]):
        raise ValueError('Only registered first-seed or full-three-seed reports are permitted')
    tag = f'{args.architecture}_seeds_'+'_'.join(map(str, seeds))
    out = ROOT / 'visualize/paper_experiments/forecast_state_bridge_v1' / tag
    figdir = ROOT / 'visualize/forecast_state_bridge_v1' / tag
    rows, weights, sources, inventories = [], [], {}, []
    reference_manifest = STRONG / 'manifest.json'
    references = {(r['crop'], r['origin']): r for r in json.loads(reference_manifest.read_text())['records']}
    sources[str(reference_manifest)] = sha256(reference_manifest)
    checked_parents = set()

    def check_parent(file, expected):
        if sha256(file) != expected:
            raise ValueError(f'Changed upstream parent: {file}')
        if str(file) in checked_parents:
            return
        sources[str(file)] = verify(file.parent)
        checked_parents.add(str(file))
        record = json.loads(file.read_text())
        for parent, digest in record.get('upstream_sha256', {}).items():
            check_parent(Path(parent), digest)

    for crop in CROPS:
        raw, _ = load(crop, verify=True)
        for origin in ORIGINS:
            strong_file = STRONG / f'{crop}_{origin}_validation.npz'
            reference = references[crop, origin]
            for file, digest in ((strong_file, reference['output_sha256']),
                    (Path(reference['weight']), reference['weight_sha256']),
                    (Path(reference['original_audit']), reference['original_audit_sha256'])):
                if sha256(file) != digest:
                    raise ValueError('Frozen strong historical reference changed')
            with np.load(strong_file) as f:
                labels = {k: f[k] for k in IDENTITY}
                strong = yearly_rmse(f['target'], f['prediction'], f['year'])
            sources[str(strong_file)] = sha256(strong_file)
            for seed in seeds:
                src = export_root(crop, origin, seed, 'ndvi', args.architecture)
                sources[str(src / 'complete.json')] = verify(src)
                export_config = json.loads((src / 'config.json').read_text())
                for parent, digest in export_config['parents'].items():
                    check_parent(Path(parent), digest)
                with np.load(src / 'predictions.npz') as f:
                    cache = {k: f[k] for k in f.files}
                control_src = export_root(crop, origin, seed, 'ndvi', 'biid')
                if args.architecture == 'gru':
                    sources[str(control_src / 'complete.json')] = verify(control_src)
                    with np.load(control_src / 'predictions.npz') as f:
                        if set(f.files) != set(cache):
                            raise ValueError('Unmatched architecture export schema')
                        for k in f.files:
                            if k != 'predicted_state':
                                np.testing.assert_array_equal(f[k], cache[k])
                inner, val, full, outer = inner_split(cache['year'], origin)
                if np.any(cache['upstream_cutoff'] >= cache['year']):
                    raise ValueError('Upstream model sees target year')
                for k in IDENTITY:
                    np.testing.assert_array_equal(cache[k][outer], labels[k])
                    np.testing.assert_array_equal(cache[k], raw[k][cache['raw_indices']])
                scores = {}
                for condition in CONDITIONS:
                    parent_arch = args.architecture if condition == 'predicted' else 'biid'
                    dest = run_root(crop, origin, seed, 'ndvi', parent_arch, condition)
                    sources[str(dest / 'complete.json')] = verify(dest)
                    cfg = json.loads((dest / 'config.json').read_text())
                    if cfg['full_normalization']['identity'] != identity_hash(cache['source_indices'][full]):
                        raise ValueError('Normalization uses a different training window')
                    if cfg['inner_years'] != np.unique(cache['year'][val]).tolist():
                        raise ValueError('Outer years used for early stopping')
                    for name, digest in cfg['code_sha256'].items():
                        if sha256(ROOT / 'scripts' / name) != digest:
                            raise ValueError('Changed readout implementation')
                    feature_source = src if condition == 'predicted' else control_src
                    if cfg['export_sha256'] != sha256(feature_source / 'complete.json'):
                        raise ValueError('Changed export parent')
                    x = features(raw, cache, outer, cfg['full_normalization'], 'ndvi', condition)
                    details = json.loads((dest / 'metrics.json').read_text())['branches']
                    pieces = []
                    for b in details:
                        weight = dest / f'{b["branch"]}.joblib'
                        model = joblib.load(weight)
                        pieces.append(cache[f'history_{b["branch"]}'][outer]+b['center']+b['scale']*model.predict(x))
                        weights.append(str(weight))
                    prediction = compose(pieces, crop)
                    with np.load(dest / 'validation_predictions.npz') as f:
                        for k in IDENTITY:
                            np.testing.assert_array_equal(f[k], labels[k])
                        np.testing.assert_array_equal(f['prediction'], prediction)
                    scores[condition] = yearly_rmse(labels['target'], prediction, labels['year'])
                    inventories.append(dict(crop=crop, origin=origin, seed=seed, condition=condition, config=str(dest / 'config.json')))
                for condition in CONDITIONS:
                    for year, error in scores[condition].items():
                        refs = dict(strong=strong[year], weather=scores['weather'][year], previous=scores['previous'][year])
                        rows.append(dict(crop=crop, origin=origin, seed=seed, condition=condition, year=int(year), rmse=error,
                            **{f'{k}_rmse': v for k, v in refs.items()}, **{f'gain_{k}': 100*(1-error/v) for k, v in refs.items()}))
    frame = pd.DataFrame(rows)
    if len(frame) != 180*len(seeds) or len(set(weights)) != 75*len(seeds):
        raise ValueError('Incomplete bridge experiment grid')
    annual, summary = summarize(frame)
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'seed_year.csv', index=False)
    annual.to_csv(out / 'annual.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(inventories).to_csv(out / 'inventory.csv', index=False)
    atomic_json(out / 'audit.json', dict(seeds=seeds, readouts=60*len(seeds), terminal_refits=len(set(weights)),
        new_terminal_refits=(15 if args.architecture == 'gru' else 75)*len(seeds),
        reused_control_refits=(60 if args.architecture == 'gru' else 0)*len(seeds),
        non_state_export_fields_equal_for_control_reuse=True if args.architecture == 'gru' else None,
        all_terminal_weights_replayed=True, evaluation_arrays_loaded=False, sources=sources, weights=weights,
        uncertainty='Year bootstrap after within-year seed averaging, 10000 draws; descriptive development uncertainty, not selection-corrected independent inference'))
    target = annual[annual.condition == 'predicted']
    bound = max(1., float(np.max(np.abs(target[['gain_strong', 'gain_weather', 'gain_previous']].to_numpy()))))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    for ax, ref, title in zip(axes, ('strong', 'weather', 'previous'),
            ('Forecast-state readout vs frozen strong history', 'Forecast-state readout vs matched weather',
             'Forecast-state readout vs weather + previous remote sensing')):
        matrix = target.pivot(index='crop', columns='year', values=f'gain_{ref}').reindex(CROPS)
        im = ax.imshow(matrix, cmap='RdBu', norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect='auto')
        ax.set_yticks(range(4), [c.title() for c in CROPS])
        ax.set_xticks(range(9), matrix.columns)
        ax.set_title(title, fontsize=11)
        for i in range(4):
            for j in range(9):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{value:+.1f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(value) > .65*bound else 'black')
    fig.colorbar(im, ax=axes, label='Annual RMSE reduction (%)', shrink=.8)
    fig.suptitle('Conditional annual NDVI forecast bridge; seeds '+', '.join(map(str, seeds)), fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(figdir / f'annual_gain.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, ref in zip(axes, ('strong', 'weather')):
        for condition, label, color, shift in (('previous', 'Previous NDVI', '#9568a6', -.13),
                ('predicted', 'Predicted NDVI', '#287fa3', 0), ('observed', 'Observed NDVI', '#329568', .13)):
            values = summary[summary.condition == condition].set_index('crop').reindex(CROPS)[f'gain_{ref}']
            ax.scatter(np.arange(4)+shift, values, label=label, color=color, s=48)
        ax.axhline(0, color='#444444', lw=.8)
        ax.set_xticks(range(4), [c.title() for c in CROPS])
        ax.set_ylabel('Mean annual RMSE reduction (%)')
        ax.set_title('Relative to '+('strong history' if ref == 'strong' else 'matched weather'), fontsize=11)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(figdir / f'observed_forecast_bridge.{ext}', dpi=180)
    selected = summary[summary.condition == 'predicted']
    columns = ['crop', 'gain_strong', 'positive_strong_years', 'pass_strong', 'gain_weather',
        'positive_weather_years', 'pass_weather', 'gain_previous', 'positive_previous_years', 'pass_previous']
    report = ['# NDVI预测状态桥接结果', '',
        f'架构：{args.architecture}；固定种子：{seeds}。'+('这是首种子结果，不能当作三种子验收。' if len(seeds) == 1 else '每年先平均三个种子的配对相对收益，再对九年等权平均。'), '',
        '实验为给定目标年实际气象与回溯日历的年度有效槽条件预测，不是业务提前天气预报。状态与历史专家按前向块生成训练预测，终端树只用内层年份选轮数。开发年份不被重新称为独立测试。', '',
        '## 三层比较', '', selected[columns].round(3).to_markdown(index=False), '',
        'gain为RMSE下降百分比，正数越大越好；positive是九年中的正年数；pass要求均值正且至少5/9年正。对强历史通过说明整体方案有用；对气象通过说明状态分支有增量；对气象+过去遥感通过才支持显式演化优于这一直接读出。', '',
        f'![年度收益]({figdir / "annual_gain.png"})', '',
        '每格是一种作物的一个完整年份。蓝色为改善、红色为退步，三面板使用相同色标、不同参照，不能互换分母。负年份没有删除。', '',
        f'![观测预测桥接]({figdir / "observed_forecast_bridge.png"})', '',
        '每种作物的三个点分别用上一年、预测目标年、真实目标年NDVI进行产量读出。所有条件共享历史结构与基础信息。真实观测只作诊断，不是保证最优的数学上界。', '',
        '## 全部条件', '', summary.round(4).to_markdown(index=False), '',
        '区间为以年份为单位的描述性bootstrap，不是选择偏差校正后的独立确认；不要求区间下界为正。冻结强历史来自已有较长训练窗口，保留作为有竞争力的外部参考；history行则是同前向窗口的新历史读出，两者不混淆。', '',
        ('GRU只新增12个预测轨迹读出逻辑方案、15个终端树模型。历史/气象/过去/真实轨迹对照在所有非预测状态字段逐值相同后，复用BIID流程的48个条件、60个树模型，不重复训练或重复计数。' if args.architecture == 'gru' else '本架构每个种子有60个读出条件、75个终端树模型，内层选轮数拟合另计。'), '',
        '树模型数指LightGBM回归器数，不是内部决策树的棵数；具体提升轮数保存在各回归器的指标和权重中。', '',
        f'全量权重重放记录、配置索引与年度表：`{out}`。']
    (ROOT / 'Paper/task' / f'NDVI预测状态桥接_{tag}_结果.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(selected[columns].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--architecture', choices=('biid', 'gru'), default='biid')
    with threadpool_limits(limits=4):
        main(parser.parse_args())
