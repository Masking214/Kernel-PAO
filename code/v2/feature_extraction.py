# -*- coding: utf-8 -*-
"""从 UORED 原始振动 CSV 提取无重叠滑窗时域特征。

对应论文 4.2 节；修订点：
- R1-1：step=window_size（无重叠），从源头消除相邻窗口共享数据点。
- R1-6：故障=1（少数类），健康=0（多数类）。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sstats

import config as cfg


def extract_features(signal):
    """论文 Table 2 的 10 个时域统计特征（与原代码一致）。"""
    sig = np.asarray(signal, dtype=float)
    rms = np.sqrt(np.mean(sig ** 2))
    abs_mean = np.mean(np.abs(sig))
    return {
        'mean': np.mean(sig),
        'std': np.std(sig),
        'rms': rms,
        'max': np.max(sig),
        'min': np.min(sig),
        'peak_to_peak': np.max(sig) - np.min(sig),
        'kurtosis': sstats.kurtosis(sig),
        'skew': sstats.skew(sig),
        'crest_factor': np.max(np.abs(sig)) / (rms + 1e-8),
        'energy': np.sum(sig ** 2),
    }


def main():
    rows = []
    for folder, label in cfg.FOLDER_LABEL.items():
        folder_path = os.path.join(cfg.RAW_ROOT, folder)
        csv_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith('.csv'))
        print(f'{folder}: {len(csv_files)} csv, label={label}')
        for fname in csv_files:
            df = pd.read_csv(os.path.join(folder_path, fname), usecols=[cfg.SIGNAL_COL])
            sig = df[cfg.SIGNAL_COL].dropna().values
            n_pts = len(sig)
            for start in range(0, n_pts - cfg.WINDOW_SIZE + 1, cfg.STEP):
                win = sig[start:start + cfg.WINDOW_SIZE]
                feats = extract_features(win)
                feats['label'] = label
                feats['source_file'] = f'{folder}/{fname}'
                feats['window_start'] = start
                rows.append(feats)

    out = pd.DataFrame(rows)
    out.to_csv(cfg.FEATURE_CSV, index=False)
    print(f'\n总窗口数: {len(out)}')
    print(out['label'].value_counts().sort_index())
    print('文件数:', out['source_file'].nunique())
    print('每文件窗口数分布:',
          out.groupby('source_file').size().value_counts().to_dict())
    print('已保存:', cfg.FEATURE_CSV)


if __name__ == '__main__':
    main()
