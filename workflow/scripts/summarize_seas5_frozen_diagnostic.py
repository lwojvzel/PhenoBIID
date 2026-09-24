"""Summarize the frozen SEAS5 connection diagnostic without selection."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from review_revision_data import ROOT

SOURCE = ROOT / "benchmark/results/seas5_weather_reliability_v1/frozen_diagnostic"
OUT = ROOT / "visualize/paper_experiments/seas5_weather_reliability_v1/frozen_diagnostic"
CROPS = ("maize", "rice", "soybean", "wheat")


def load():
    paths = sorted(SOURCE.glob("*/origin_*/suffix_*/annual.csv"))
    frames=[]
    for path in paths:
        cfg=json.loads((path.parent/"config.json").read_text())
        if cfg.get("schema") != 2:
            continue
        frames.append(pd.read_csv(path))
    if len(frames) != 36:
        raise RuntimeError(f"Expected 36 complete conditions, found {len(frames)}")
    return pd.concat(frames,ignore_index=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True); data=load()
    keys=["crop","ratio","condition","observed_feedback"]
    values=[c for c in data if c.endswith("rmse") or c in ("coverage","samples","covered_samples")]
    aggregate=data.groupby(keys,as_index=False)[values].mean()
    aggregate.to_csv(OUT/"condition_summary.csv",index=False)
    rows=[]
    for crop in CROPS:
        for ratio in (.1,.3,.5):
            d=aggregate[(aggregate.crop==crop)&(aggregate.ratio==ratio)].set_index(["condition","observed_feedback"])
            for forecast in ("seas5_raw","seas5_bias_corrected"):
                record=dict(crop=crop,ratio=ratio,forecast=forecast,
                    actual_feedback=d.loc[("actual_3var",True),"rmse"],
                    forecast_feedback=d.loc[(forecast,True),"rmse"],
                    climatology_feedback=d.loc[("climatology_3var",True),"rmse"])
                record["weather_penalty_feedback"]=record["forecast_feedback"]-record["actual_feedback"]
                no=d.loc[(forecast,False),"rmse"]-d.loc[("actual_3var",False),"rmse"]
                record["weather_penalty_no_feedback"]=no
                record["feedback_buffer"]=no-record["weather_penalty_feedback"]
                record["forecast_minus_climatology"]=record["forecast_feedback"]-record["climatology_feedback"]
                record["coverage"]=d.loc[(forecast,True),"coverage"]
                rows.append(record)
    effects=pd.DataFrame(rows);effects.to_csv(OUT/"weather_feedback_effects.csv",index=False)
    chosen=effects[effects.forecast.eq("seas5_bias_corrected")]
    colors={.1:"#29966f",.3:"#2878b5",.5:"#c36a32"}
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.7),constrained_layout=True)
    x=np.arange(4); width=.22
    for i,ratio in enumerate((.1,.3,.5)):
        q=chosen[chosen.ratio.eq(ratio)].set_index("crop").loc[list(CROPS)]
        axes[0].bar(x+(i-1)*width,100*q.weather_penalty_feedback,width,color=colors[ratio],label=f"{int(ratio*100)}% hidden")
        axes[1].bar(x+(i-1)*width,100*q.feedback_buffer,width,color=colors[ratio])
        axes[2].bar(x+(i-1)*width,100*q.forecast_minus_climatology,width,color=colors[ratio])
    for ax,title,ylabel in zip(axes,
        ("Forecast-weather penalty","Buffering from observed feedback","SEAS5 versus weather climatology"),
        ("Change in yield RMSE (x100 t/ha)","Reduction in weather penalty (x100 t/ha)","Change in yield RMSE (x100 t/ha)")):
        ax.axhline(0,color="#4f5b60",lw=.8);ax.set_title(title,fontweight="bold")
        ax.set_xticks(x,list(map(str.title,CROPS)),rotation=20);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(1.65,1.22))
    fig.suptitle("Frozen-model SEAS5 diagnostic (13 evaluation years)",fontweight="bold",y=1.07)
    fig.savefig(OUT/"frozen_weather_diagnostic.png",dpi=240,bbox_inches="tight")
    fig.savefig(OUT/"frozen_weather_diagnostic.pdf",bbox_inches="tight")
    print(effects.to_string(index=False))


if __name__=="__main__":
    main()
