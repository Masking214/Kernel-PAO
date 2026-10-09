# -*- coding: utf-8 -*-
"""评估指标：G-mean / F-value / AUC / PR-AUC / 少数类召回与精度。"""
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)

METRIC_NAMES = ['G-mean', 'F-value', 'AUC', 'PR-AUC', 'Recall_min', 'Precision_min']


def all_metrics(y_true, y_pred, y_score):
    """y_score: 正类(少数类=故障)的决策值或概率，用于 AUC/PR-AUC。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = float(np.sqrt(recall_min * recall_maj))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f_value = (2 * precision * recall_min / (precision + recall_min)
               if (precision + recall_min) > 0 else 0.0)
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        auc, pr_auc = 0.5, 0.0
    return {
        'G-mean': g_mean,
        'F-value': f_value,
        'AUC': auc,
        'PR-AUC': pr_auc,
        'Recall_min': float(recall_min),
        'Precision_min': float(precision),
    }
