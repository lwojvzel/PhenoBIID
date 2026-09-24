"""Global paired yield maps using every registered evaluation year."""
import json

import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from inseason_13year_data import ROOT, CACHE, BLOCKS, YEARS, RECIPES, load, partition
from inseason_13year_uncertainty import historical
from inseason_nested_common import LABELS
from review_revision_data import sha256
from run_review_revision_parallel import atomic_json

OUT = ROOT / 'visualize/paper_experiments/inseason_13year_maps_v1'


def spatial_mean(rows, cols, values):
    index = rows.astype(np.int64)*720+cols.astype(np.int64)
    counts = np.bincount(index,minlength=360*720)
    total = np.bincount(index,weights=values,minlength=360*720)
    means = np.divide(total,counts,out=np.full(360*720,np.nan),where=counts>0)
    return means.reshape(360,720),counts.reshape(360,720)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference_path = ROOT / 'visualize/paper_experiments/inseason_13year_v1/comparisons.csv'
    references = pd.read_csv(reference_path).set_index('crop')
    sources = {str(reference_path):sha256(reference_path)}
    summaries, report = [], ['# 十三年主实验：全球产量空间图','',
        '所有图使用同样13个评价年份的原冻结世界模型，10%名义未观测后缀，seed42。没有选择某个效果特别好的年份；每个格点仅对该格点实际参与评价的年份取平均。','',
        '上排：真实产量、具名历史参照预测、世界模型预测。下排：历史预测减真实值、世界模型预测减真实值、历史绝对误差减世界模型绝对误差的时间平均。最后一幅大于零表示世界模型误差更小。偏差的时间平均可能正负抵消，不能替代逐年RMSE；最后一幅先逐样本取绝对值再平均。','']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    lon,lat = np.linspace(-180,180,721),np.linspace(-90,90,361)
    for crop,recipe in RECIPES.items():
        raw,_ = load(crop)
        pieces,indices = [],[]
        sources[str(CACHE / crop / 'manifest.json')] = sha256(CACHE / crop / 'manifest.json')
        world_parts = []
        for cutoff in BLOCKS:
            ix = partition(raw,cutoff)['evaluation']
            labels = {key:raw[key][ix] for key in LABELS}
            pieces.append(historical(crop,cutoff,references.loc[crop,'baseline'],labels,sources))
            path = CACHE / crop / f'world_{cutoff}_{recipe}_biid_010.npy'
            sources[str(path)] = sha256(path)
            world_parts.append(np.load(path))
            indices.append(ix)
        ix = np.concatenate(indices)
        if tuple(np.unique(raw['year'][ix])) != YEARS:
            raise ValueError('Incomplete map years')
        truth,hist,world = raw['target'][ix].astype(float),np.concatenate(pieces),np.concatenate(world_parts)
        if truth.shape != hist.shape or hist.shape != world.shape or not np.isfinite(world).all():
            raise ValueError('Invalid spatial map predictions')
        hist_error,world_error = hist-truth,world-truth
        values = [truth,hist,world,hist_error,world_error,np.abs(hist_error)-np.abs(world_error)]
        maps = [spatial_mean(raw['row'][ix],raw['col'][ix],v)[0] for v in values]
        counts = spatial_mean(raw['row'][ix],raw['col'][ix],truth)[1]
        if counts.max()>13:
            raise ValueError('Duplicated grid-year in spatial map')
        vmax = max(1.,float(np.ceil(np.nanpercentile(maps[0],99))))
        error_limit = max(.1,float(np.nanpercentile(np.abs(np.stack(maps[3:5])),95)))
        gain_limit = max(.05,float(np.nanpercentile(np.abs(maps[5]),95)))
        fig,axes = plt.subplots(2,3,figsize=(12.8,7.0),subplot_kw=dict(projection=ccrs.Robinson()))
        fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.16,wspace=.045,hspace=.08)
        titles = ('Ground truth','Historical prediction','World-model prediction',
                  'History: prediction - truth','World model: prediction - truth','MAE reduction: history - world')
        images = []
        for index,(ax,grid,title) in enumerate(zip(axes.flat,maps,titles)):
            ax.set_global()
            ax.coastlines(resolution='110m',linewidth=.35,color='.28')
            kwargs = (dict(cmap='YlGn',vmin=0,vmax=vmax) if index<3 else
                      dict(cmap='RdBu_r',vmin=-error_limit,vmax=error_limit) if index<5 else
                      dict(cmap='RdYlGn',vmin=-gain_limit,vmax=gain_limit))
            images.append(ax.pcolormesh(lon,lat,np.ma.masked_invalid(grid),transform=ccrs.PlateCarree(),
                          shading='flat',rasterized=True,**kwargs))
            ax.set_title(title,fontsize=10,pad=5)
        fig.suptitle(crop.title()+' | '+recipe.upper().replace('_',' + ')+
            '\n13-year eligible-record means; nominal 10% unobserved suffix',fontsize=12,y=.99)
        for rectangle,image,label in [([.035,.085,.29,.023],images[0],'Yield (t/ha)'),
                                      ([.355,.085,.29,.023],images[3],'Mean signed error (t/ha)'),
                                      ([.675,.085,.29,.023],images[5],'Mean absolute-error reduction (t/ha)')]:
            bar = fig.colorbar(image,cax=fig.add_axes(rectangle),orientation='horizontal',extend='both')
            bar.set_label(label,fontsize=9)
            bar.locator = MaxNLocator(nbins=4)
            bar.update_ticks()
        fig.text(.5,.91,'Historical reference: '+references.loc[crop,'baseline_label'],ha='center',fontsize=9)
        for ext in ('png','pdf'):
            fig.savefig(OUT / f'{crop}_yield_maps.{ext}',dpi=230,bbox_inches='tight')
        plt.close(fig)
        np.savez_compressed(OUT / f'{crop}_map_values.npz',truth=maps[0],history=maps[1],world=maps[2],
            history_bias=maps[3],world_bias=maps[4],mae_reduction=maps[5],evaluation_year_count=counts)
        summaries.append(dict(crop=crop,records=len(ix),cells=int((counts>0).sum()),
            positive_mae_cells=int((maps[5]>0).sum()),yield_display_max=vmax,
            signed_error_display_limit=error_limit,mae_display_limit=gain_limit,
            historical_reference=references.loc[crop,'baseline_label']))
        report += [f'## {crop.title()}', '',f'![{crop}全球产量对比]({OUT}/{crop}_yield_maps.png)','',
            f'参照为{references.loc[crop,"baseline_label"]}。共{len(ix)}条格点-年份、{int((counts>0).sum())}个格点；'
            f'其中{int((maps[5]>0).sum())}个格点的平均绝对误差下降。','']
    report += ['## 色标与解释','',
        '同一作物的三张产量图使用同一色标；两个偏差图使用同一对称色标。显示范围由真实产量99分位及两模型偏差绝对值95分位确定，超范围只在颜色上饱和，色条三角形标明延伸；评分和保存数组未截断。', '',
        '这组图显示空间分布和局部改进/退化，不证明每个地区都更好，也不直接证明BIID相对常态的机制优势。空白区域没有参与本次共同样本评价，不是零产量。', '',
        '原始格点图数组、评价年份数量及来源哈希一并保存，不能将不同模型的有效格点分别过滤后再比较。']
    pd.DataFrame(summaries).to_csv(OUT / 'summary.csv',index=False)
    (OUT / '读图说明.md').write_text('\n'.join(report)+'\n')
    atomic_json(OUT / 'audit.json',dict(sources=sources,years=YEARS,seed=42,
        selected_years=False,all_registered_years=True,common_grid_years=True,
        primary_predictions_changed=False,generator_sha256=sha256(ROOT / 'scripts/visualize_inseason_13year_maps.py')))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
