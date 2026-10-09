# -*- coding: utf-8 -*-
"""补跑方案A所需的配对检验：PAO vs SVM/Importance/Uniform（决定能否写"显著优于"）。"""
import numpy as np
import pandas as pd
import config as cfg
from data_utils import load_features, make_subset
from experiments import METHOD_PREDICTORS, pao_predictor, baseline_predictor
from stats_tests import dietterich_5x2

X_all, y_all, g_all = load_features()
rows = []
for ratio in cfg.RATIOS:
    X, y, g = make_subset(X_all, y_all, g_all, ratio)
    for other in ['SVM', 'Importance', 'Uniform', 'Stratified']:
        res = dietterich_5x2(METHOD_PREDICTORS['Kernel-PAO'],
                             baseline_predictor(other), X, y, g, metric='G-mean')
        rows.append({'IR': f'1:{ratio}', 'Comparison': f'Kernel-PAO vs {other}',
                     't': round(res['t'], 4), 'p': round(res['p'], 4),
                     'mean_diff': round(res['mean_diff'], 4)})
        print(rows[-1])
pd.DataFrame(rows).to_csv(f'{cfg.RESULTS_DIR}/ttests_extra.csv', index=False)