"""Train-only local seasonal anomalies, appended to complete observed trajectories."""
import numpy as np

from observed_remote_benchmark import INPUTS, make_features, trajectory_features


def seasonal_anomalies(arrays, product):
    allowed=('row','col','source_month','relative_valid',f'observed_{product}',f'observed_{product}_valid')
    data={s:{k:a[k] for k in allowed} for s,a in arrays.items()}
    train=data['train']
    def keys(a):
        grid=a['row'].astype(np.int64)*720+a['col']
        return grid[:,None]*12+np.minimum(a['source_month'],11)
    def valid(a):
        return (a['relative_valid']>0)&(a[f'observed_{product}_valid']>0)
    mask=valid(train)
    values=train[f'observed_{product}']
    index=keys(train)
    sums=np.bincount(index[mask],weights=values[mask],minlength=360*720*12)
    counts=np.bincount(index[mask],minlength=len(sums))
    months=train['source_month'][mask]
    monthly_sums=np.bincount(months,weights=values[mask],minlength=12)
    monthly_counts=np.bincount(months,minlength=12)
    result={}
    for s,a in data.items():
        index=keys(a); mask=valid(a); values=a[f'observed_{product}']
        month=np.minimum(a['source_month'],11)
        numerator=sums[index].copy(); denominator=counts[index].copy()
        fallback_sum=monthly_sums[month].copy(); fallback_count=monthly_counts[month].copy()
        if s=='train':
            # One crop/grid/year row and no duplicate source months: excluding
            # this slot also excludes its complete same-year grid/month record.
            numerator-=np.where(mask,values,0); denominator-=mask
            fallback_sum-=np.where(mask,values,0); fallback_count-=mask
        fallback=np.divide(fallback_sum,fallback_count,out=np.zeros_like(fallback_sum),where=fallback_count>0)
        climatology=np.divide(numerator,denominator,out=fallback,where=denominator>0)
        result[s]=np.where(mask,values-climatology,0).astype(np.float32)
    return result


def build_features(arrays, variant):
    products=('lai','ndvi') if variant=='observed_both' else (variant.removeprefix('observed_'),)
    if not set(products).issubset({'lai','ndvi'}):
        raise ValueError('Only observed remote conditions have anomaly extensions')
    anomalies={p:seasonal_anomalies(arrays,p) for p in products}
    result={}
    for s,a in arrays.items():
        x,_,names=make_features({k:a[k] for k in INPUTS},variant)
        chunks=[x]
        for p in products:
            mask=(a['relative_valid']>0)&(a[f'observed_{p}_valid']>0)
            chunks.append(trajectory_features(anomalies[p][s],mask))
            names += [f'local_{p}_anomaly_slot_{i}' for i in range(12)]
            names += [f'local_{p}_anomaly_{k}' for k in ('mean','std','max','min','sum','peak_slot')]
        result[s]=np.concatenate(chunks,1).astype(np.float32)
    return result,names
