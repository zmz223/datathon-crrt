import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt

from libs.analytics import DataProcessing, CollinearityRemover, FeatureSelector, Optimizer
from libs.evaluation import cv_performance, oddsRatio, plot_rocs, probability_inspection
from configs.predictions import episode_config

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


RESULTS_PATH = Path('./results')
if not os.path.isdir(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
EPISODE = datetime.now().strftime('%Y%m%d_%H%M$S')
os.mkdir(RESULTS_PATH / EPISODE)

def main():
    # Load data
    data_path = episode_config['dataset_path']
    print(f'Reading dataset from: {data_path}')
    df_raw = pd.read_csv(data_path)

    # Data processing
    df_raw.patientsex = df_raw.patientsex.replace({'M':0, 'F':1})
    processor = DataProcessing(df_raw)
    df = processor.transform()
    ans = processor.post(df, 'aki_stage', to_drop=['hospital_id','a_patientid','crrt_24_134_h'])
    df_X = ans['df']

    # Remove collinearity
    cr = CollinearityRemover(df_X)
    vif = cr.run()
    cols = vif.Variable
    binary_cols = list(set(cols).difference(ans['continous']))
    continous_cols = list(set(cols).difference(ans['binary']))
    df_X = df_X.loc[:, binary_cols+continous_cols]

    # Features selection
    mdl = LGBMClassifier(
        max_depth=3,
        num_leaves=12,
        n_estimators=200,
        learning_rate=1e-2,
        class_weight='balanced',
        random_state=17,
        n_jobs=-1,
    )
    fs = FeatureSelector(mdl, continous_cols, binary_cols)
    features = fs.transform(df_X, df.crrt_24_134_h, k=5)
    selected_features = features[features>=5].index.tolist()
    binary_cols = list(set(selected_features).difference(ans['continous']))
    continous_cols = list(set(selected_features).difference(ans['binary']))

    # Model optimization
    mdl = LogisticRegression(
        penalty='elasticnet',
        random_state=17,
        solver='saga',
        max_iter=100,
        l1_ratio=0.5
    )
    params = {
        'mdl__C': [1e-3, 1e-1, 1, 1e1, 1e3], 
        'mdl__class_weight': [None,'balanced'],
        'mdl__l1_ratio': [1e-5, 0.25, 0.5, 0.75, 1] 
    }
    opt = Optimizer(mdl, continous_cols, binary_cols)
    clf = opt.run(df_X.loc[:, selected_features], df.crrt_24_134_h, params)
    joblib.dump(clf, RESULTS_PATH / EPISODE / 'best_model.sav')
    df_X.loc[:, selected_features].to_csv(RESULTS_PATH / EPISODE / 'features.csv')
    df.crrt_24_134_h.to_csv(RESULTS_PATH / EPISODE / 'target.csv')
    
    # Cross-validation
    res = cv_performance(
        clf, 
        df_X.loc[:, selected_features],#.values, # 
        df.crrt_24_134_h.values, 
        k=5) 
    stats = res[0].stack().groupby(level=1).median()
    stats.to_excel(RESULTS_PATH / EPISODE / 'stats.xlsx')
    
    f, ax = plt.subplots()
    _ = plot_rocs(res, ax=ax)
    f.savefig(RESULTS_PATH / EPISODE / 'roc_curves.png')
    
    f, ax = plt.subplots()
    _ = probability_inspection(res, ax=ax)
    f.savefig(RESULTS_PATH / EPISODE / 'probability_inspection.png')
    
    # Odds Ratio
    df_coef = oddsRatio(
        df_X.loc[:, selected_features], 
        df.crrt_24_134_h, 
        columnnames=selected_features)
    f, ax = plt.subplots()
    _ = ax.boxplot(df_coef, whis=[5, 95], showfliers=False, vert=False)
    _ = ax.set_yticklabels(df_coef.columns)
    _ = ax.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    f.savefig(RESULTS_PATH / EPISODE / 'odds_ratio.png')
    
    print('\n\nDone!')


if __name__ == '__main__':
    main()