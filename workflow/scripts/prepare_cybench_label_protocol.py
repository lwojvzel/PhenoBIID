"""Join statistical labels, freeze forward identities, and score simple history rules."""
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from cybench_label_protocol import BLOCKS,SUMMARY_NAMES,BASELINES,past_features,partition,simple_predictions,annual_scores
from prepare_cybench_regional_monthly import sha256

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ROOT/'Data/processed/cybench_seasonal_v1'
PROVIDER = ROOT/'Data/external/CYBench/full_v1_10'
AUDIT = ROOT/'benchmark/results/cybench_inseason_inputs_v1'
OUT = ROOT/'benchmark/cache/cybench_inseason_labels_v1'
SCORES = ROOT/'benchmark/results/cybench_inseason_labels_v1'
LABEL_COLUMNS = ['country_code','adm_id','harvest_year','yield','crop_name']


def atomic_json(path,data):
    tmp=path.with_suffix('.part')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)


def source_tables():
    audit=json.loads((AUDIT/'input_audit.json').read_text())
    sources={str(AUDIT/'input_audit.json'):sha256(AUDIT/'input_audit.json')}
    tables={}
    for crop in ('maize','wheat'):
        parts=[]
        for country in ('DE','FR','PL'):
            path=PROVIDER/crop/country/f'yield_{crop}_{country}.csv'
            if sha256(path)!=audit['sources'][str(path)]:
                raise ValueError('Statistical labels changed since source audit')
            marker_path=path.with_suffix('.csv.source.json')
            marker=json.loads(marker_path.read_text())
            if zlib.crc32(path.read_bytes())!=marker['zip_crc32_verified']:
                raise ValueError('Statistical label CRC mismatch')
            frame=pd.read_csv(path,usecols=LABEL_COLUMNS)
            if not frame.country_code.eq(country).all() or frame.duplicated(['adm_id','harvest_year']).any():
                raise ValueError('Provider country or label identity mismatch')
            if not np.isfinite(frame.harvest_year).all() or not (frame.harvest_year==frame.harvest_year.astype(int)).all():
                raise ValueError('Invalid harvest year')
            # Later provider labels are outside this registered study, including lag construction.
            parts.append(frame[frame.harvest_year<=2016].copy())
            for p in (path,marker_path):
                sources[str(p)]=sha256(p)
        tables[crop]=pd.concat(parts,ignore_index=True)
    for kind in ('DE','EU'):
        path=ROOT/f'Data/external/CYBench/code/data_preparation/crop_statistics_{kind}/README.md'
        sources[str(path)]=sha256(path)
    return tables,sources


def main():
    seasonal=json.loads((SEASONS/'manifest.json').read_text())
    verify_path=AUDIT/'seasonal_verification.json'
    verified=json.loads(verify_path.read_text())
    if not verified['passed'] or verified['manifest_sha256']!=sha256(SEASONS/'manifest.json'):
        raise ValueError('Verified seasonal inputs are required')
    if not seasonal['complete'] or seasonal['labels_loaded']:
        raise ValueError('Expected a complete separate input layer')
    for path,digest in seasonal['outputs'].items():
        if sha256(SEASONS/path)!=digest:
            raise ValueError('Seasonal input changed')
    tables,sources=source_tables()
    sources[str(SEASONS/'manifest.json')]=sha256(SEASONS/'manifest.json')
    sources[str(verify_path)]=sha256(verify_path)
    config=dict(blocks=BLOCKS,years=list(range(1983,2017)),history_lag_years=5,
        history_summary_names=SUMMARY_NAMES,baseline_methods=BASELINES,target_unit='t/ha',
        source_sha256=sources,code_sha256={n:sha256(ROOT/'scripts'/n) for n in ('cybench_label_protocol.py','prepare_cybench_label_protocol.py')},
        source_columns=LABEL_COLUMNS,maximum_source_label_year=2016,
        partition='Fixed expanding blocks; region must have a history-supported target in inner fitting years',
        eligibility='Finite nonnegative target and at least one strictly earlier valid regional yield',
        history_information='All strictly earlier harvest-year statistics assumed available; no publication dates are supplied',
        evaluation_history='Earlier evaluation-year labels may become later-year lags; weights stay fixed within block',
        first_ever_label='Retained as a later historical source, not a history-supported prediction target',
        missing_lag='Exact calendar-year lag remains NaN with validity mask; summary uses all available earlier records',
        mean_fallback='If none of the previous 3/5 calendar years has a label, use the all-past regional mean',
        prediction_clipping='Nonnegative final predictions for every registered simple method',
        source_category={'maize_DE':'grain_maize','maize_FR_PL':'grain_maize_from_EU_source',
                         'wheat_DE':'winter_wheat','wheat_FR_PL':'soft_wheat_from_EU_source'},
        region_weighting='Equal regions per country and year, then equal years; do not merge countries with unequal year coverage',
        fitted_models=0,normalization_fitted=False,hyperparameter_search=False,
        scope='Label/identity preparation and deterministic historical reference scores; no world-model skill result')
    OUT.mkdir(parents=True,exist_ok=True)
    SCORES.mkdir(parents=True,exist_ok=True)
    registration=OUT/'registration.json'
    if registration.exists() and json.loads(registration.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Registered label/history protocol changed')
    atomic_json(registration,config)
    files,counts,cohorts,score_rows={},[],[],[]
    for crop,labels in tables.items():
        regions=pd.read_csv(SEASONS/crop/'regions.csv')
        records=[]
        for year in config['years']:
            frame=regions[['country','adm_id','region_index']].copy()
            frame['year']=year
            frame['season_row']=np.arange(len(frame),dtype=np.int32)
            records.append(frame)
        identities=pd.concat(records,ignore_index=True)
        identities.insert(0,'sample_index',np.arange(len(identities),dtype=np.int32))
        match=identities.merge(labels.rename(columns={'country_code':'country','harvest_year':'year'}),
            on=['country','adm_id','year'],how='left',validate='one_to_one',indicator=True)
        target=match['yield'].to_numpy(np.float32)
        history=past_features(labels,identities)
        valid=np.isfinite(target)&(target>=0)
        match['label_present']=match['_merge'].eq('both')
        match['target_valid']=valid
        match['has_history']=history['history_summary'][:,5]>0
        match['eligible_before_block']=valid&match.has_history
        match['exclusion_reason']=np.where(~match.label_present,'no_label',np.where(~valid,'invalid_target',np.where(~match.has_history,'no_past_label','eligible')))
        folder=OUT/crop
        folder.mkdir(exist_ok=True)
        for name,table in (('identities',identities),('label_join',match.drop(columns=['yield','_merge']))):
            path=folder/f'{name}.csv'
            table.to_csv(path,index=False)
            files[str(path.relative_to(OUT))]=sha256(path)
        for name,arrays in (('history',history),('targets',{'yield':target})):
            path=folder/f'{name}.npz'
            with path.with_suffix('.part').open('wb') as stream:
                np.savez_compressed(stream,**arrays)
            path.with_suffix('.part').replace(path)
            files[str(path.relative_to(OUT))]=sha256(path)
        predictions=simple_predictions(history)
        prediction_path=folder/'simple_history_predictions.npz'
        with prediction_path.with_suffix('.part').open('wb') as stream:
            np.savez_compressed(stream,**predictions)
        prediction_path.with_suffix('.part').replace(prediction_path)
        files[str(prediction_path.relative_to(OUT))]=sha256(prediction_path)
        for country,frame in match.groupby('country'):
            counts.append(dict(crop=crop,country=country,input_rows=len(frame),label_rows=int(frame.label_present.sum()),
                valid_targets=int(frame.target_valid.sum()),history_supported=int(frame.eligible_before_block.sum())))
        for cutoff in BLOCKS:
            indices=partition(identities,target,history,cutoff)
            path=folder/f'block_{cutoff}.npz'
            with path.with_suffix('.part').open('wb') as stream:
                np.savez_compressed(stream,**indices)
            path.with_suffix('.part').replace(path)
            files[str(path.relative_to(OUT))]=sha256(path)
            for split,ix in indices.items():
                for country in ('DE','FR','PL'):
                    group=identities.iloc[ix]
                    group=group[group.country.eq(country)]
                    cohorts.append(dict(crop=crop,cutoff=cutoff,split=split,country=country,
                        samples=len(group),regions=group.adm_id.nunique(),years=';'.join(map(str,sorted(group.year.unique())))))
            for row in annual_scores(identities,target,predictions,indices['evaluation']):
                score_rows.append(dict(crop=crop,cutoff=cutoff,**row))
        print(f'[LABELS] {crop}: {int(valid.sum())} valid; {int(match.eligible_before_block.sum())} with history',flush=True)
    for name,rows in (('label_readiness',counts),('partition_counts',cohorts)):
        path=OUT/f'{name}.csv'
        pd.DataFrame(rows).to_csv(path,index=False)
        files[path.name]=sha256(path)
    annual=pd.DataFrame(score_rows)
    if annual.duplicated(['crop','country','year','method']).any():
        raise ValueError('Overlapping evaluation years')
    annual_path=SCORES/'simple_history_annual.csv'
    annual.to_csv(annual_path,index=False)
    summary=annual.groupby(['crop','country','method'],sort=False).agg(
        mean_annual_rmse=('rmse','mean'),mean_annual_mae=('mae','mean'),years=('year','nunique'),samples=('samples','sum')).reset_index()
    summary_path=SCORES/'simple_history_scores.csv'
    summary.to_csv(summary_path,index=False)
    atomic_json(OUT/'manifest.json',dict(complete=True,registration_sha256=sha256(registration),files=files,
        rows=int(sum(r['input_rows'] for r in counts)),valid_label_rows=int(sum(r['valid_targets'] for r in counts)),
        feature_targets_separate=True,fitted_models=0,world_model_evaluated=False,
        simple_score_files={str(p):sha256(p) for p in (annual_path,summary_path)}))
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
