import boto3

import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from typing import Dict

from configs.get_data import lab_config, vitals_config, \
    comorbidities_config, crrt_idx, \
    TIME_WINDOW_START, TIME_WINDOW_STOP, AGG_FNC
from libs.data_extraction import get_item_id, from_monitored_numeric, \
    extract_comorbidities, get_demography

DATA_PATH = Path('./data')
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

def login():
    load_dotenv()
    # Log in
    key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
    s3 = boto3.resource('s3',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key)
    return s3

def main(filename, aki_filename):
    s3 = login()
    
    # Get lab results
    print('-'*50)
    print('Quering Lab results\n\n')
    labs = []
    for lab_name,obj in lab_config.items():
        print(lab_name)
        df = get_item_id(obj['idx'], TIME_WINDOW_START, AGG_FNC)
        if obj['post_fnc'] is not None:
            df = obj['post_fnc'](df)
        df.columns = [col.replace('value', lab_name) for col in df.columns]
        df = df.set_index(['hospital_id', 'a_patientid'])
        labs.append(df)
    labs = pd.concat(labs, join='outer', axis=1)
    
    # Get vitals
    print('-'*50)
    print('Quering vitals\n\n')
    vitals, outcome = from_monitored_numeric(
        s3, crrt_idx, vitals_config, TIME_WINDOW_START, AGG_FNC)
    
    # Get comorbidities
    print('-'*50)
    print('Quering comorbidities\n\n')
    comorbidities = extract_comorbidities(comorbidities_config)
    
    # Get AKI score
    aki_score = pd.read_csv(aki_filename)
    aki_score['hospital_id'] = aki_score.a_patientid.apply(lambda x: int(str(x)[0]))
    aki_score = aki_score.set_index(['hospital_id','a_patientid'])
    
    # Get demography
    print('-'*50)
    print('Quering demograpy\n\n')
    demography = get_demography()
    
    dataset = pd.concat(
        [labs, vitals, comorbidities, aki_score, outcome], 
        join='outer', axis=1)
    
    dataset = demography.merge(
        dataset, how='right', 
        left_index=True, right_index=True)
    
    ### Exclusion criteria
    print('-'*50)
    print('Applying exclusion criteria\n\n')
    # crrt in the first 24 hours
    dataset['exclude_per_crrt'] = np.where(dataset.crrt_time <= TIME_WINDOW_START, 1, 0)
    # crrt after 5 days
    dataset['exclude_per_crrt'] = np.where(dataset.crrt_time > TIME_WINDOW_STOP, 1, dataset['exclude_per_crrt'])
    # AKI < 1
    dataset['exclude_per_stage'] = np.where(dataset.aki_stage < 1, 1,0)
    
    final_ds = dataset.query(
        "exclude_per_crrt == 0 and \
        exclude_per_stage == 0 and \
        rrt != 1"
    )
    
    print('-'*50)
    print('Finalizing..\n\n')
    final_ds.crrt_time = final_ds.crrt_time.fillna(-1)
    final_ds['crrt_24_134_h'] = np.where(
        (final_ds.crrt_time> TIME_WINDOW_START) & (final_ds.crrt_time<= TIME_WINDOW_STOP), 1, 0)
    final_ds = final_ds.drop(
        columns=['exclude_per_stage','exclude_per_crrt','rrt','crrt_time'])
    final_ds.reset_index().to_csv(filename, index=False)
    print('Done!')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--filename",type=str,default="./data/dataset.csv")
    args.add_argument("--aki_filename",type=str,default="max_aki_24h.csv")
    args = args.parse_args()
    main(args.filename, args.aki_filename)