"""Frozen-switch diagnostic for SEAS5-conditioned in-season trajectories.

This deliberately keeps the existing state/readout weights fixed. Hidden
weather channels unavailable in SEAS5 are set to their training means. The
script is a connection and distribution-shift diagnostic, not the final
three-variable refit comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from forecast_bridge_data import CROPS
from forecast_bridge_state import ForecastState, batch_arrays
from inseason_signal_matching import load_group, remap_product, feature_matrix
from inseason_ndvi_reuse import tail_mask, mix_trajectory
from ndvi_tail_replacement import prefix_values
from run_forecast_bridge_state import run_root as state_root, tensors
from run_ndvi_tail_replacement import yield_prediction
from run_crop_signal_screen import yearly_rmse
from crop_signal_screen_data import cache_root
from seas5_crop_weather import extract
from seas5_reliability_protocol import WEATHER_INDICES
from run_review_revision_parallel import atomic_json
from review_revision_data import ROOT, sha256

RESULT = ROOT / "benchmark/results/seas5_weather_reliability_v1/frozen_diagnostic"
RECIPE = dict(maize="gpp", rice="ndvi", soybean="ndvi", wheat="ndvi_gpp")
PRODUCTS = dict(maize=("gpp",), rice=("ndvi",), soybean=("ndvi",), wheat=("ndvi", "gpp"))


def training_climatology(raw, fit, take):
    output = np.zeros((len(take), 12, 3), np.float32)
    train_grid = raw["row"][fit].astype(np.int64) * 720 + raw["col"][fit]
    test_grid = raw["row"][take].astype(np.int64) * 720 + raw["col"][take]
    train_month = np.minimum(raw["source_month"][fit], 11).astype(np.int64)
    test_month = np.minimum(raw["source_month"][take], 11).astype(np.int64)
    active = raw["relative_valid"][fit] > 0
    keys = train_grid[:, None] * 12 + train_month
    query = test_grid[:, None] * 12 + test_month
    for q, j in enumerate(WEATHER_INDICES):
        value = raw["weather"][fit, :, j]
        valid = active & np.isfinite(value)
        sums = np.bincount(keys[valid], weights=value[valid], minlength=360*720*12)
        counts = np.bincount(keys[valid], minlength=len(sums))
        ms = np.bincount(train_month[valid], weights=value[valid], minlength=12)
        mc = np.bincount(train_month[valid], minlength=12)
        fallback = np.divide(ms, mc, out=np.full(12, value[valid].mean()), where=mc > 0)
        output[..., q] = np.divide(sums[query], counts[query], out=fallback[test_month].copy(), where=counts[query] > 0)
    return output


def bias_table(raw, fit, ratio):
    active = raw["relative_valid"][fit] > 0
    forecast, plan = extract(raw["year"][fit], raw["source_month"][fit], active,
                             raw["row"][fit], raw["col"][fit], ratio)
    actual = raw["weather"][fit][..., WEATHER_INDICES]
    table = np.zeros((12, 6, 3), np.float64)
    count = np.zeros((12, 6, 3), np.int64)
    for month in range(1, 13):
        for lead in range(1, 7):
            mask = plan["hidden"] & (plan["initialization_month"][:, None] == month) & (plan["leadtime_month"] == lead)
            valid = mask[..., None] & np.isfinite(forecast) & np.isfinite(actual)
            total = np.where(valid, actual-forecast, 0.).sum((0, 1))
            n = valid.sum((0, 1))
            table[month-1, lead-1] = np.divide(total, n, out=np.zeros(3), where=n > 0)
            count[month-1, lead-1] = n
    return table.astype(np.float32), count


def apply_bias(forecast, plan, table):
    out = forecast.copy()
    for month in range(1, 13):
        for lead in range(1, 7):
            mask = plan["hidden"] & (plan["initialization_month"][:, None] == month) & (plan["leadtime_month"] == lead)
            out[mask] += table[month-1, lead-1]
    out[..., 1] = np.maximum(out[..., 1], 0.)
    out[..., 2] = np.maximum(out[..., 2], 0.)
    return out


def physical_condition(actual, hidden, shared, training_means):
    """Past uses all observations; hidden slots expose only three shared fields."""
    result = actual.copy()
    result[hidden] = np.asarray(training_means)
    for q, j in enumerate(WEATHER_INDICES):
        result[..., j] = np.where(hidden, shared[..., q], result[..., j])
    return result.astype(np.float32)


@torch.no_grad()
def rollout(model, mapped, take, stats, observed, tail, active, weather, observed_feedback):
    prefix, known = prefix_values(observed, active, tail, stats["ndvi"]["mean"], stats["ndvi"]["std"])
    pieces = []
    for start in range(0, len(take), 256):
        end = min(start + 256, len(take)); selected = take[start:end]
        b = batch_arrays(mapped, selected, stats, "ndvi")
        b["weather"] = np.nan_to_num((weather[start:end]-np.asarray(stats["weather_mean"])) /
                                      np.asarray(stats["weather_std"])).astype(np.float32)
        b = tensors(b, "cuda")
        values = torch.as_tensor(prefix[start:end], device="cuda")
        visible = torch.as_tensor(known[start:end], device="cuda")
        dynamics = model.dynamics; states = dynamics.initial_states(dict(
            weather=b["weather"], previous_lai=torch.zeros_like(b["previous"]),
            previous_lai_valid=torch.zeros_like(b["previous"]), previous_ndvi=b["previous"],
            previous_ndvi_valid=b["previous_valid"], previous_ndvi_quality=b["previous_quality"],
            relative_valid=b["relative_valid"], context=b["context"]))
        weather_tokens = dynamics.weather(b["weather"], b["context"]); outputs=[]
        for k in range(12):
            on = b["relative_valid"][:, k, None, None].bool()
            states = {p: torch.where(on, dynamics.transitions[p](s, weather_tokens[:, k]), s) for p,s in states.items()}
            previous = torch.where(b["previous_valid"][:, k].bool(), b["previous"][:, k], torch.zeros_like(b["previous"][:, k]))
            predicted = previous + dynamics.heads["ndvi"](states["ndvi"].mean(1)).squeeze(-1)
            outputs.append(predicted)
            feedback = torch.where(visible[:, k], values[:, k], predicted) if observed_feedback else predicted
            message = dynamics.feedback["ndvi"](feedback[:, None])[:, None]
            states["ndvi"] = torch.where(on, dynamics.norms["ndvi"](states["ndvi"] + message), states["ndvi"])
        pieces.append(torch.stack(outputs, 1).float().cpu().numpy())
    return np.concatenate(pieces) * stats["ndvi"]["std"] + stats["ndvi"]["mean"]


def run(crop, origin, ratio):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(3500*2**20/torch.cuda.get_device_properties(0).total_memory)
    percent = round(100*ratio); destination = RESULT/crop/f"origin_{origin}"/f"suffix_{percent:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"complete.json").exists():
        previous = json.loads((destination/"config.json").read_text())
        if previous.get("schema") == 2:
            return
    started=time.monotonic(); raw,groups,scales,_,sources=load_group(crop,origin)
    group=groups["validation"]; take=group["take"]; labels=group["labels"]
    fit=np.flatnonzero(raw["year"] <= origin-3); active=raw["relative_valid"][take]>0
    tail=tail_mask(active,ratio); actual=raw["weather"][take]
    seas,plan=extract(labels["year"],raw["source_month"][take],active,raw["row"][take],raw["col"][take],ratio)
    climo=training_climatology(raw,fit,take); table,count=bias_table(raw,fit,ratio); corrected=apply_bias(seas,plan,table)
    variants={"actual_3var":actual[...,WEATHER_INDICES],"climatology_3var":climo,
              "seas5_raw":np.where(np.isfinite(seas),seas,climo),
              "seas5_bias_corrected":np.where(np.isfinite(corrected),corrected,climo)}
    models={}; stats={}
    for product in PRODUCTS[crop]:
        state=state_root(crop,product,"biid",origin-3,42)
        models[product]=ForecastState("biid").cuda().eval()
        models[product].load_state_dict(torch.load(state/"model.pt",map_location="cpu",weights_only=True))
        stats[product]=json.loads((state/"normalization.json").read_text())
    terminal=json.loads((cache_root(crop,origin)/"manifest.json").read_text())["spec"]["upstream"]["upstream"]["normalization"]
    tm,ts=np.asarray(terminal["weather_mean"]),np.asarray(terminal["weather_std"])
    rows=[]; recipe=RECIPE[crop]; observed={p:raw[f"observed_{p}"][take] for p in PRODUCTS[crop]}
    for condition,shared in variants.items():
        physical=physical_condition(actual,tail,shared,tm)
        common=group["common"].copy(); normalized=(physical-tm)/ts
        common[:,309:465]=np.nan_to_num(normalized).reshape(len(take),-1)
        for feedback in (True,False):
            complete={}; forecast_by_product={}
            for product in PRODUCTS[crop]:
                mapped=remap_product(raw,product); st=dict(stats[product],ndvi=stats[product][product])
                forecast=rollout(models[product],mapped,take,st,observed[product],tail,active,physical,feedback)
                forecast_by_product[product]=forecast
                complete[product]=mix_trajectory(observed[product],forecast,tail)
            enc={p:group["encoders"][p].encode(v,tail,True) for p,v in complete.items()}
            x=feature_matrix(common,enc,tail,group["support"],recipe)
            prediction=yield_prediction(group["heads"][recipe],x,crop)
            scores=yearly_rmse(labels["target"],prediction,labels["year"])
            for year,rmse in scores.items():
                yy=labels["year"]==int(year); covered=plan["supported"]&yy
                record=dict(crop=crop,origin=origin,year=int(year),ratio=ratio,condition=condition,
                    observed_feedback=feedback,rmse=rmse,samples=int(yy.sum()),covered_samples=int(covered.sum()),
                    coverage=float(plan["supported"][yy].mean()),
                    supported_rmse=float(np.sqrt(np.mean((prediction[covered]-labels["target"][covered])**2))) if covered.any() else np.nan)
                weather_valid=yy[:,None]&tail
                for q,name in enumerate(("t2m","tp","ssrd")):
                    delta=(shared[...,q]-actual[...,WEATHER_INDICES[q]])[weather_valid]
                    record[f"weather_{name}_rmse"]=float(np.sqrt(np.mean(delta**2)))
                for product,forecast in forecast_by_product.items():
                    valid=weather_valid&np.isfinite(observed[product])
                    delta=(forecast-observed[product])[valid]
                    record[f"state_{product}_rmse"]=float(np.sqrt(np.mean(delta**2))) if len(delta) else np.nan
                rows.append(record)
            np.savez_compressed(destination/f"{condition}_{'feedback' if feedback else 'no_feedback'}.npz",
                                prediction=prediction,supported=plan["supported"],year=labels["year"])
    pd.DataFrame(rows).to_csv(destination/"annual.csv",index=False)
    np.savez_compressed(destination/"bias.npz",bias=table,count=count)
    atomic_json(destination/"config.json",dict(schema=2,crop=crop,origin=origin,ratio=ratio,recipe=recipe,
        role="frozen-switch diagnostic; not final three-variable refit",raw_seas5_fallback="training climatology beyond lead 6",
        observed_prefix_weather="actual 13-variable weather",hidden_weather="three shared variables; other ten at training means",
        products=PRODUCTS[crop],source_files=sources))
    atomic_json(destination/"complete.json",dict(seconds=time.monotonic()-started,
        files={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and p.name!="complete.json"}))
    print(f"[SEAS5 DIAGNOSTIC COMPLETE] {crop} {origin} {percent}%",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--crop",choices=CROPS,required=True)
    p.add_argument("--origin",type=int,choices=(2004,2008,2012),default=2004)
    p.add_argument("--ratio",type=float,choices=(.1,.3,.5),default=.1)
    a=p.parse_args();run(a.crop,a.origin,a.ratio)
