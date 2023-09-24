import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm
from typing import List, Dict, Tuple

import awswrangler as wr

def plot_time_distr(tot, title, n_days=2, delta_hours=6):
    omega_hours = 24 // delta_hours
    m0 = tot[tot.time>=0]
    f, ax = plt.subplots()
    b = np.linspace(0,n_days*24*60,100)
    _ = m0.groupby('a_patientid').time.min().hist(bins=b, ax=ax)
    _ = ax.set_title(title)
    _ = ax.set_xticks([i*60*delta_hours for i in range(n_days*omega_hours)])
    _ = ax.set_xticklabels([i*delta_hours for i in range(n_days*omega_hours)])
    _ = ax.set_xlabel('Time [hours]')
    return f, ax

def get_item_id(
    item_ids: List, 
    time_window: int,
    agg_fnc: List|None = [np.min, np.max]):
    """Simple function which returns the observations for the 'item_ids' 
    found in the s3://icusics-db/labresults_numeric tables. 
    Functions specified in the agg_fnc are then applied to the observations 
    in order to have a single observation per patient

    Args:
        item_ids (List): List of lab result items
        time_window (int): Number of minutes that quantify the observation window taken in consideration
        agg_fnc (List): List of aggregation functions to apply
    """
    
    labs = []
    # tot = []
    for i in tqdm(range(1,7)):
        labresults_numeric = wr.s3.read_parquet(f's3://icusics-db/labresults_numeric/labresults_numeric_h{i}.parquet')
        ith = labresults_numeric[labresults_numeric.a_variableid.isin(item_ids)]
        # tot.append(ith)
        ith = ith[(ith.time>=0) & (ith.time<=time_window)]
        if agg_fnc is not None:
            ith = ith.groupby('a_patientid').agg({'value': agg_fnc})
        labs.append(ith)
        
    labs = pd.concat(labs)
    if agg_fnc is not None:
        labs.columns = ['value_' + fnc.__name__ for fnc in agg_fnc]
        labs = labs.reset_index()
    labs['hospital_id'] = labs.a_patientid.apply(lambda x: int(str(x)[0]))
    return labs
    
def normalize_albumine(
    albumine: pd.DataFrame) -> pd.DataFrame:
    """This function normalizes the albumine values of hospital 1 in order 
    to have the same unit of measure used in the other hospitals

    Args:
        albumine (pd.DataFrame): queried albumine dataframe
        agg_fnc (List): aggregation functions used 

    Returns:
        pd.DataFrame: normalized albumine dataframe
    """
    
    albumine_h1 = albumine.query('hospital_id==1')
    for col in albumine_h1.columns:
        if col.startswith('value_'):
            albumine_h1[col] = albumine_h1[col] / 10
    
    albumine_hx = albumine.query('hospital_id!=1')
    albumine = pd.concat([albumine_h1, albumine_hx])
    return albumine

def urea_to_bun(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.startswith('value_'):
            df[col] = df[col] * 0.357
    return df

# q05 = partial(np.quantile, q=0.05)
# q95 = partial(np.quantile, q=0.95)

def process_ith(
    df_ith: pd.DataFrame, 
    time_window: int, 
    var_idx: List[int], 
    agg_fnc: List, 
    value_min: float|None=None, value_max: float|None=None, 
    col_labels: List[str]|None=None) -> pd.DataFrame:
    
    ith = df_ith[df_ith.a_variableid.isin(var_idx)]
    ith = ith[(ith.time>=0) & (ith.time<=time_window)]
    if value_min is not None:
        ith = ith[ith.value >= value_min]
    if value_max is not None:
        ith = ith[ith.value <= value_max]
    ith = ith.groupby('a_patientid').agg({'value':agg_fnc})
    if col_labels:
        ith.columns = col_labels
    return ith

def get_tables(s3):
    my_bucket = s3.Bucket('icusics-db')

    tables = []
    for my_bucket_object in my_bucket.objects.all():
        tables.append(my_bucket_object.key)
    tables = pd.Series(tables)
    return tables

def finalize_vitals(list_df: List[pd.DataFrame], var_name: str) -> pd.DataFrame:
    df = pd.concat(list_df)
    # df.columns = [var_name + '_' + col for col in df.columns]
    df = df.reset_index()
    df['hospital_id'] = df.a_patientid.apply(lambda x: int(str(x)[0]))
    df = df.set_index(['hospital_id','a_patientid'])
    return df

def finalize_outcome(flux: List[pd.DataFrame]) -> pd.DataFrame:
    flux = pd.concat(flux)
    asd = flux.reset_index()
    asd['hospital_id'] = asd.a_patientid.apply(lambda x: int(str(x)[0]))
    asd.time = asd.time.astype(float)
    # asd.a_variableid = asd.a_variableid.apply(lambda x: str(x)[2:])
    asd = asd[asd.a_variableid.astype(str).str.endswith('1380')]
    outcome = asd.drop(columns=['a_variableid'])
    outcome = outcome.rename(columns={'time':'crrt_time'})
    outcome = outcome.set_index(['hospital_id','a_patientid'])
    return outcome

def from_monitored_numeric(s3, flux_idx: List, config: Dict, time_window: int, agg_fnc: List) -> Tuple[pd.DataFrame]:
    tables = get_tables(s3)
    
    vitals_list = {k: [] for k in config.keys()}
    flux = []
    for tbl_name in tqdm(tables.str.extract('(.*monitored_numeric.*)').dropna()[0].values):
        ith_monitored = wr.s3.read_parquet(f's3://icusics-db/{tbl_name}')
        flux_ith = ith_monitored[ith_monitored.a_variableid.isin(flux_idx)]
        flux_ith_grp = flux_ith.groupby(['a_patientid','a_variableid']).time.min()
        flux.append(flux_ith_grp)
        for k,v in config.items():
            processed_ith = process_ith(
                df_ith=ith_monitored, 
                time_window=time_window, 
                var_idx=v['codes'], 
                agg_fnc=agg_fnc, 
                value_min=v['vmin'], value_max=v['vmax'])
            # print(processed_ith.columns)
            processed_ith.columns = [k + '_' + col[1] for col in processed_ith.columns]
            vitals_list[k].append(processed_ith)
    
    outcome = finalize_outcome(flux)
    vitals_df = dict.fromkeys(list(config.keys()), None)
    for k,v in vitals_list.items():
        vitals_df[k] = finalize_vitals(v, k)
        
    vitals_df = pd.concat(list(vitals_df.values()), join='outer', axis=1)
    return vitals_df, outcome

def get_comorbidity(
    icd_codes: pd.DataFrame, 
    comorbidity_codes: List[str], 
    name: str, 
    ids_to_exclude: List[str]|None=None) -> pd.DataFrame:
    comorbidity = icd_codes[icd_codes["referencecode"].str.startswith(comorbidity_codes)]
    if ids_to_exclude is not None:
        comorbidity = comorbidity[~comorbidity.referencecode.isin(ids_to_exclude)]
    hf = comorbidity.groupby("a_patientid").referencecode.count().reset_index().rename(columns={"referencecode": name})
    hf[name] = 1
    return hf

def get_icd_codes() -> pd.DataFrame:
    
    icd_codes=[]
    for h in range(1,7):
        diagnoses = wr.s3.read_parquet(path="s3://icusics-db/diagnoses/diagnoses_h%s.parquet"%h)
        icd_codes.append(diagnoses)

    icd_codes= pd.concat(icd_codes, ignore_index=True)
    return icd_codes

def extract_comorbidities(config: Dict) -> pd.DataFrame:
    icd_codes = get_icd_codes()
    
    data = []
    for key,obj in config.items():
        df = get_comorbidity(icd_codes, obj['icd'], key, obj['to_exclude'])
        df['hospital_id'] = df.a_patientid.apply(lambda x: int(str(x)[0]))
        df = df.set_index(['hospital_id','a_patientid'])
        data.append(df)
    data = pd.concat(data, join='outer', axis=1)
    return data

def get_demography() -> pd.DataFrame:
    patients = wr.s3.read_parquet(
        path="s3://icusics-db/patients/patients.parquet")
    patients['hospital_id'] = patients.a_patientid.apply(lambda x: int(str(x)[0]))
    patients = patients.set_index(['hospital_id','a_patientid'])
    cols = ['patientsex','age','height','weight','bmi']
    return patients.loc[:, cols]