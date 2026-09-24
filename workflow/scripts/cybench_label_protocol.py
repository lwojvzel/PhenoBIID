"""Past-only regional yield features and fixed forward evaluation partitions."""
import numpy as np
import pandas as pd

BLOCKS = {2001:(2002,2003,2004), 2005:(2006,2007,2008), 2009:tuple(range(2010,2017))}
SUMMARY_NAMES = ('latest', 'mean', 'std', 'linear_trend', 'slope_per_year', 'count', 'latest_age')
BASELINES = ('latest_available', 'mean_last_three', 'mean_last_five', 'past_mean', 'past_linear_trend')


def past_features(labels, identities):
    """Feature values depend only on the same region and strictly earlier years."""
    keys = ['country_code', 'adm_id', 'harvest_year']
    if labels.duplicated(keys).any():
        raise ValueError('Duplicate source label identity')
    if identities.duplicated(['country', 'adm_id', 'year']).any():
        raise ValueError('Duplicate prediction identity')
    valid = np.isfinite(labels['yield']) & labels['yield'].ge(0)
    source = labels.loc[valid, keys+['yield']].sort_values(keys)
    groups = {k:g for k,g in source.groupby(['country_code', 'adm_id'])}
    n = len(identities)
    lags = np.full((n,5), np.nan, np.float32)
    lag_years = identities.year.to_numpy(np.int32)[:,None]-np.arange(1,6,dtype=np.int32)[None,:]
    summary = np.full((n,len(SUMMARY_NAMES)), np.nan, np.float32)
    summary[:,5] = 0
    latest_source = np.zeros(n,dtype=np.int32)
    for i,row in enumerate(identities.itertuples()):
        records = groups.get((row.country, row.adm_id))
        if records is None:
            continue
        past = records[records.harvest_year < row.year]
        if not len(past):
            continue
        years = past.harvest_year.to_numpy(np.int32)
        values = past['yield'].to_numpy(np.float64)
        lookup = dict(zip(years, values))
        lags[i] = [lookup.get(y,np.nan) for y in lag_years[i]]
        centered = years.astype(np.float64)-years.mean()
        slope = float(centered @ (values-values.mean()) / (centered @ centered)) if len(past)>1 else 0.
        level = values.mean()+slope*(row.year-years.mean())
        summary[i] = [values[-1],values.mean(),values.std(),level,slope,len(values),row.year-years[-1]]
        latest_source[i] = years[-1]
    return dict(lag_yield=lags,lag_valid=np.isfinite(lags),lag_source_year=lag_years,
                history_summary=summary,latest_source_year=latest_source)


def partition(identities, target, history, cutoff):
    if cutoff not in BLOCKS:
        raise ValueError('Unknown temporal block')
    years = identities.year.to_numpy()
    available = np.isfinite(target) & (target>=0) & (history['history_summary'][:,5]>0)
    inner = available & (years<=cutoff-2)
    seen = set(map(tuple,identities.loc[inner,['country','adm_id']].to_numpy()))
    seen_mask = np.array([(r.country,r.adm_id) in seen for r in identities.itertuples()])
    eligible = available & seen_mask
    return dict(inner_fit=np.flatnonzero(eligible & (years<=cutoff-2)),
        inner_validation=np.flatnonzero(eligible & (years>cutoff-2) & (years<=cutoff)),
        full_fit=np.flatnonzero(eligible & (years<=cutoff)),
        evaluation=np.flatnonzero(eligible & np.isin(years,BLOCKS[cutoff])))


def simple_predictions(history):
    summary = history['history_summary'].astype(np.float64)
    result = dict(latest_available=summary[:,0],past_mean=summary[:,1],past_linear_trend=summary[:,3])
    for k,name in ((3,'mean_last_three'),(5,'mean_last_five')):
        lags = history['lag_yield'][:,:k].astype(np.float64)
        count = np.isfinite(lags).sum(1)
        result[name] = np.divide(np.nansum(lags,axis=1),count,
            out=summary[:,1].copy(),where=count>0)
    return {name:np.maximum(result[name],0).astype(np.float32) for name in BASELINES}


def annual_scores(identities, target, predictions, indices):
    rows = []
    chosen = identities.iloc[indices].copy()
    chosen['array_index'] = indices
    for (country,year),frame in chosen.groupby(['country','year']):
        ix = frame.array_index.to_numpy()
        truth = target[ix].astype(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError('Missing evaluation target')
        for name,pred in predictions.items():
            value = pred[ix].astype(np.float64)
            if not np.isfinite(value).all():
                raise ValueError('Missing baseline on paired cohort')
            error = value-truth
            rows.append(dict(country=country,year=int(year),method=name,samples=len(ix),
                             rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(np.abs(error)))))
    return rows
