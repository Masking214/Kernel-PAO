# -*- coding: utf-8 -*-
"""数据加载、不平衡子集构造、按文件分组交叉验证。

修订点：
- R1-1：用 StratifiedGroupKFold 按 source_file（原始振动文件）分组，
        同一文件的窗口绝不同时出现在训练/测试折，杜绝数据泄漏。
- R1-6：健康为多数类(y=0)固定，对故障类(y=1)随机欠采样构造 IR 子集。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import config as cfg


def load_features():
    """返回 X, y, groups(文件名), file_labels。"""
    df = pd.read_csv(cfg.FEATURE_CSV)
    X = df[cfg.FEATURE_COLS].values.astype(float)
    y = df['label'].values.astype(int)
    groups = df['source_file'].values
    return X, y, groups


def make_subset(X, y, groups, ratio, seed=cfg.RANDOM_STATE):
    """固定多数类(健康 y=0)，对少数类(故障 y=1)欠采样至 n_maj/ratio。

    与论文一致使用固定随机种子，保证可复现。
    """
    maj_idx = np.where(y == 0)[0]
    min_idx = np.where(y == 1)[0]
    n_min_target = int(round(len(maj_idx) / ratio))
    rng = np.random.RandomState(seed)
    min_keep = rng.choice(min_idx, size=min(n_min_target, len(min_idx)), replace=False)
    keep = np.concatenate([maj_idx, min_keep])
    rng.shuffle(keep)
    return X[keep], y[keep], groups[keep]


def grouped_folds(y, groups, n_splits=5, seed=cfg.RANDOM_STATE):
    """按文件分组、同时尽量保持各折类比的折索引生成器。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # split 需要 X 占位（仅用 y/groups）
    folds = list(sgkf.split(np.zeros((len(y), 1)), y, groups))
    # 断言检查：每个折的训练/测试原始振动文件集合必须零交集（防止同一文件跨折泄漏）
    for tr_idx, te_idx in folds:
        assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), \
            "data leakage: train/test folds share source files"
    return folds


def fold_train_test(X, y, groups, tr_idx, te_idx):
    """折内 Z-score 标准化：均值/标准差只由训练折估计（论文 4.4）。"""
    scaler = StandardScaler().fit(X[tr_idx])
    Xtr = scaler.transform(X[tr_idx])
    Xte = scaler.transform(X[te_idx])
    return Xtr, y[tr_idx], Xte, y[te_idx], groups[tr_idx], groups[te_idx]
