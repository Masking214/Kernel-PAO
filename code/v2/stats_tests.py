# -*- coding: utf-8 -*-
"""Dietterich 5×2 交叉验证配对 t 检验（R1-8）。

普通 5 折 CV 各折训练集重叠 75%，折间指标不独立，直接配对 t 检验会低估方差、
虚低 p 值。5×2cv 每轮用不相交的两半数据做两次训练/测试，共 5 轮，满足检验前提。

统计量（Dietterich, 1998）：
    t = p̄_1 / sqrt( (1/5) * Σ_r s_r² ),  自由度 = 5
其中 p_r1,p_r2 为第 r 轮两半上方法 A 相对 B 的指标差，
p̄_r=(p_r1+p_r2)/2，s_r²=(p_r1-p̄_r)²+(p_r2-p̄_r)²。
"""
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

from data_utils import fold_train_test


def _fold_score(predict_fn, Xtr, ytr, gtr, Xte, yte, metric='G-mean'):
    from metrics_utils import all_metrics
    y_pred, y_score = predict_fn(Xtr, ytr, Xte, gtr)
    return all_metrics(yte, y_pred, y_score)[metric]


def dietterich_5x2(predict_a, predict_b, X, y, groups,
                   metric='G-mean', n_rounds=5, base_seed=100):
    diffs = []          # 每轮 (p1, p2)
    for r in range(n_rounds):
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True,
                                    random_state=base_seed + r)
        halved = []
        for tr_idx, te_idx in sgkf.split(np.zeros((len(y), 1)), y, groups):
            Xtr, ytr, Xte, yte, gtr, _ = fold_train_test(X, y, groups, tr_idx, te_idx)
            sa = _fold_score(predict_a, Xtr, ytr, gtr, Xte, yte, metric)
            sb = _fold_score(predict_b, Xtr, ytr, gtr, Xte, yte, metric)
            halved.append(sa - sb)
        diffs.append(halved)
    diffs = np.asarray(diffs)                     # (5,2)
    pbar = diffs.mean(axis=1)                     # 每轮均值
    s2 = ((diffs - pbar[:, None]) ** 2).sum(axis=1)
    # Dietterich(1998) 分子为第一轮第一个差值 p_1^(1)，非首轮均值
    t_stat = diffs[0, 0] / np.sqrt(s2.sum() / n_rounds + 1e-12)
    p_val = 2 * stats.t.sf(abs(t_stat), df=n_rounds)
    return {'t': float(t_stat), 'p': float(p_val),
            'diffs': diffs.tolist(), 'mean_diff': float(diffs.mean())}
