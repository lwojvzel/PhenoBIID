"""Training/validation-only linear LAI-change diagnosis with paired-year weather."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from multimodal_baseline import ERA5_ROOT, ERA5_VARIABLES, CROPS, PROJECT_ROOT as ROOT
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

RESULT = ROOT / 'benchmark/results/weather_innovation_diagnostic_v1'
PENALTIES = np.array([.01, .1, 1., 10., 100.])
KEYS = ('weather', 'weather_anomaly', 'previous_lai', 'previous_lai_valid',
        'lai_climo', 'relative_valid', 'source_month', 'context', 'target_lai',
        'target_lai_valid', 'year', 'row', 'col')
GROUPS = {'state': np.arange(33), 'current': np.arange(59),
          'innovation': np.r_[np.arange(33), np.arange(59, 72)],
          'combined': np.arange(72)}


def read_split(root, split):
    if split not in ('train', 'validation'):
        raise ValueError('This diagnostic must not access evaluation data')
    with np.load(root / f'{split}.npz') as f:
        return {key: f[key] for key in KEYS}


def paired_weather(a, normalization, sources):
    previous = np.zeros_like(a['weather'])
    mean = np.asarray(normalization['weather_mean'], dtype=np.float32)
    std = np.asarray(normalization['weather_std'], dtype=np.float32)
    maximum = 0.
    for year in np.unique(a['year']):
        ix = np.flatnonzero(a['year'] == year)
        valid = a['relative_valid'][ix] > 0
        month = np.minimum(a['source_month'][ix], 11)
        row, col = a['row'][ix, None], a['col'][ix, None]
        for channel, variable in enumerate(ERA5_VARIABLES):
            for offset in (0, -1):
                path = ERA5_ROOT / variable / f'{variable}_{int(year)+offset}.npy'
                if str(path) not in sources:
                    sources[str(path)] = sha256(path)
                raw = np.load(path, mmap_mode='r')[month, row, col]
                values = np.nan_to_num((raw-mean[channel])/std[channel], nan=0., posinf=0., neginf=0.)
                if offset == -1:
                    previous[ix, :, channel] = np.where(valid, values, 0)
                elif valid.any():
                    difference = float(np.max(np.abs(values[valid]-a['weather'][ix, :, channel][valid])))
                    maximum = max(maximum, difference)
    if maximum > 1e-6:
        raise ValueError(f'Natural-month weather does not match current cache: {maximum}')
    return previous, maximum


def features(a, previous_weather):
    sample, slot = np.where((a['relative_valid'] > 0) & (a['target_lai_valid'] > 0))
    phase = np.minimum(a['source_month'][sample, slot], 11)*np.float64(2*np.pi/12)
    state = np.column_stack((a['previous_lai'][sample], a['previous_lai_valid'][sample],
        a['previous_lai'][sample, slot], a['lai_climo'][sample, slot],
        np.sin(phase), np.cos(phase), a['context'][sample]))
    x = np.column_stack((state, a['weather'][sample, slot], a['weather_anomaly'][sample, slot],
                         a['weather'][sample, slot]-previous_weather[sample, slot])).astype(np.float64)
    if x.shape[1] != 72 or not np.isfinite(x).all():
        raise ValueError('Invalid feature layout')
    target = (a['target_lai'][sample, slot]-a['previous_lai'][sample, slot]).astype(np.float64)
    return x, target, a['year'][sample]


def fit(crop, origin):
    destination = RESULT / crop / f'origin_{origin}'
    destination.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / f'benchmark/cache/task_aligned_world_v1/{crop}/origin_{origin}'
    config = dict(crop=crop, origin=origin, penalties=PENALTIES.tolist(), windows=[0, 12],
        groups={key: value.tolist() for key, value in GROUPS.items()},
        code_sha256=sha256(Path(__file__)), input_sha256={s: sha256(source_root / f'{s}.npz')
            for s in ('train', 'validation')}, metadata_sha256=sha256(source_root / 'manifest.json'),
        scope='Only train/validation LAI labels. No evaluation arrays or yield labels accessed.',
        window_scope='Regression/scaler fitting window; local climatology retains full-training cache.')
    config_path = destination / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing configuration changed')
    atomic_json(config_path, config)
    if (destination / 'complete.json').exists():
        print(f'[SKIP] {crop} {origin}', flush=True)
        return
    meta = json.loads((source_root / 'manifest.json').read_text())
    sources, data, weather_errors = {}, {}, {}
    for split in ('train', 'validation'):
        arrays = read_split(source_root, split)
        previous, error = paired_weather(arrays, meta['normalization'], sources)
        data[split] = features(arrays, previous)
        weather_errors[split] = error
        del arrays, previous
    x, y, year = data['train']; xv, yv, yearv = data['validation']
    rows, yearly, models, maximum_replay = [], [], {}, 0.
    lai_std = meta['normalization']['lai_std']
    for window in (0, 12):
        keep = year >= (year.min() if window == 0 else year.max()-window+1)
        for group, columns in GROUPS.items():
            scaler = StandardScaler()
            xt = scaler.fit_transform(x[keep][:, columns])
            model = Ridge(alpha=len(xt)*PENALTIES, solver='cholesky')
            model.fit(xt, np.repeat(y[keep, None], len(PENALTIES), axis=1))
            validation = scaler.transform(xv[:, columns])
            prediction = model.predict(validation)
            key = f'window_{window}__{group}'
            weight = destination / f'{key}.joblib'
            joblib.dump(dict(scaler=scaler, model=model, columns=columns), weight)
            restored = joblib.load(weight)
            replay = restored['model'].predict(restored['scaler'].transform(xv[:, restored['columns']]))
            difference = float(np.max(np.abs(replay-prediction)))
            np.testing.assert_array_equal(replay, prediction)
            maximum_replay = max(maximum_replay, difference)
            models[str(weight)] = sha256(weight)
            for j, penalty in enumerate(PENALTIES):
                ratios = []
                for val_year in np.unique(yearv):
                    select = yearv == val_year
                    mse = np.mean((prediction[select, j]-yv[select])**2)
                    persistence = np.mean(yv[select]**2)
                    ratio = mse/persistence
                    ratios.append(ratio)
                    yearly.append(dict(crop=crop, origin=origin, window=window, group=group,
                        penalty=float(penalty), year=int(val_year), mse_ratio=float(ratio),
                        rmse=float(np.sqrt(mse)*lai_std), persistence_rmse=float(np.sqrt(persistence)*lai_std)))
                rmse = np.sqrt(np.mean((prediction[:, j]-yv)**2))*lai_std
                persistence = np.sqrt(np.mean(yv**2))*lai_std
                rows.append(dict(crop=crop, origin=origin, window=window, group=group,
                    penalty=float(penalty), mean_year_mse_ratio=float(np.mean(ratios)),
                    validation_rmse=float(rmse), persistence_rmse=float(persistence),
                    gain=float(100*(1-rmse/persistence)), train_slots=int(keep.sum()),
                    validation_slots=len(yv), input_dimensions=len(columns), weight=str(weight), output_column=j))
            print(f'[INNOVATION FIT] {crop} {origin} window={window} {group}', flush=True)
    pd.DataFrame(rows).to_csv(destination / 'validation_metrics.csv', index=False)
    pd.DataFrame(yearly).to_csv(destination / 'validation_years.csv', index=False)
    atomic_json(destination / 'complete.json', dict(fits=len(rows), test_data_accessed=False,
        weather_source_sha256=sources, current_weather_max_errors=weather_errors,
        weights_sha256=models, maximum_validation_replay_error=maximum_replay))


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frames = []
    for crop in CROPS:
        for origin in (2004, 2008, 2012):
            path = RESULT / crop / f'origin_{origin}'
            audit = json.loads((path / 'complete.json').read_text())
            if audit['fits'] != 40 or audit['test_data_accessed'] or audit['maximum_validation_replay_error']:
                raise ValueError('Incomplete or inconsistent diagnosis')
            frames.append(pd.read_csv(path / 'validation_metrics.csv'))
    frame = pd.concat(frames, ignore_index=True)
    chosen = frame.sort_values(['mean_year_mse_ratio', 'penalty']).groupby(
        ['crop', 'origin', 'window', 'group'], as_index=False).first()
    output = ROOT / 'visualize/paper_experiments/weather_innovation_diagnostic_v1'
    figure = ROOT / 'visualize/weather_innovation_diagnostic_v1'
    output.mkdir(parents=True, exist_ok=True); figure.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'all_validation_metrics.csv', index=False)
    chosen.to_csv(output / 'validation_selected_penalties.csv', index=False)
    labels = ['Previous state', 'State + current weather', 'State + weather change', 'State + both']
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), layout='constrained', sharey=True)
    index = pd.MultiIndex.from_product([CROPS, (2004, 2008, 2012)])
    matrices = [chosen[chosen.window == window].pivot(index=['crop', 'origin'], columns='group', values='gain').reindex(
        index=index, columns=GROUPS) for window in (0, 12)]
    limit = max(1., max(float(np.abs(m.to_numpy()).max()) for m in matrices))
    for ax, matrix, title in zip(axes, matrices, ('Full training history', 'Most recent 12 training years')):
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(4), labels, rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(12), [f'{c.title()}: validation {o-2}-{o}' for c, o in index], fontsize=9)
        for i in range(12):
            for j in range(4):
                v = matrix.iloc[i, j]
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > .7*limit else 'black')
    fig.colorbar(im, ax=axes, location='bottom', label='Validation LAI RMSE reduction vs. persistence (%)', shrink=.8)
    for ext in ('png', 'pdf'):
        fig.savefig(figure / f'validation_state_information.{ext}', dpi=200)
    plt.close(fig)
    text = ['# 前后年气象变化是否更有助于 LAI 预测', '',
        '共 480 个确定性岭回归拟合，仅使用训练和验证期。每个格子在五个预设正则中按三个验证年的等权误差比选择。这里没有读取评价期或产量标签，不能作为产量收益结论。', '',
        f'![状态信息诊断]({figure / "validation_state_information.png"})', '',
        '一行是一个作物与验证时间段。左图完整训练历史，右图最近 12 个训练年；两图的局地常态都保留完整训练期估计。四列从左到右依次增加当期气象、前后年气象差分、两者。正数表示比沿用去年 LAI 更准。比较第二和第四列才是在已有当期气象下增加前季气象的效果；不能只看第四列是否为正。', '',
        '该线性诊断直接预测每槽相对去年同槽的变化，没有多步自回归反馈；它用于判断输入和窗口是否值得进入新的转移模型，不是新的主模型。气象自然月抽取已与现有当期缓存核对；持久性、有效槽与所有特征组使用相同行。', '',
        chosen[['crop', 'origin', 'window', 'group', 'penalty', 'gain', 'mean_year_mse_ratio']].to_markdown(index=False, floatfmt='.5f'), '',
        f'全部配置 CSV：{output / "all_validation_metrics.csv"}', '',
        f'每个拟合的权重、气象原文件散列、验证重放：{RESULT}']
    (ROOT / 'Paper/task/前后季气象与训练窗口_480组状态诊断_20260906.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(chosen[['crop', 'origin', 'window', 'group', 'gain']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', choices=CROPS)
    parser.add_argument('--origin', type=int, choices=(2004, 2008, 2012))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.summarize:
            summarize()
        elif args.crop and args.origin:
            fit(args.crop, args.origin)
        else:
            parser.error('Specify crop and origin, or summarize')
