# -*- coding: utf-8 -*-
"""V7 多种子稳健性实验。

在 3 个随机种子（42 / 0 / 2024）下重复主实验（8 方法 × 4 IR × 分组 5 折），
报告跨种子的 mean ± std，回应"单随机种子"的稳健性质疑。

每个种子同时改变三处随机来源：
1) make_subset 对故障类的欠采样；
2) StratifiedGroupKFold 的折划分（shuffle 种子）；
3) 各过采样方法内部的 RNG（统一通过 cfg.RANDOM_STATE 传递）。

注：Kernel-PAO 内层选 σ 的 3 折划分保持固定设计（不随后续种子变化），
模型选择流程在种子间一致，种子差异只来自数据子集/外折/采样过程。

产出：
- results_v2/multiseed_per_seed.csv  每个种子的折均值（6 指标）
- results_v2/multiseed_summary.csv   跨 3 种子的 mean ± std（6 指标）

用法：python multiseed.py
"""
import numpy as np
import pandas as pd

import config as cfg
import experiments as E
from data_utils import load_features, make_subset
from metrics_utils import METRIC_NAMES

SEEDS = [42, 0, 2024]


def main():
    X_all, y_all, g_all = load_features()
    per_seed_rows = []

    for seed in SEEDS:
        cfg.RANDOM_STATE = seed          # 统一驱动基线/过采样的内部 RNG
        for ratio in cfg.RATIOS:
            X, y, g = make_subset(X_all, y_all, g_all, ratio, seed=seed)
            for method in cfg.METHOD_ORDER:
                df = E.cv_loop(E.METHOD_PREDICTORS[method], X, y, g, seed=seed)
                row = {'Seed': seed, 'IR': f'1:{ratio}', 'Method': method}
                for mt in METRIC_NAMES:
                    row[mt] = df[mt].mean()      # 该种子下 5 折均值
                per_seed_rows.append(row)
                print(f'seed={seed} 1:{ratio} {method:18s} '
                      f'G-mean={row["G-mean"]:.4f}', flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(f'{cfg.RESULTS_DIR}/multiseed_per_seed.csv', index=False)

    # 跨种子汇总
    summary_rows = []
    for (ir, method), sub in per_seed.groupby(['IR', 'Method'], sort=False):
        row = {'IR': ir, 'Method': method}
        for mt in METRIC_NAMES:
            vals = sub[mt].values
            row[mt] = f'{vals.mean():.4f} ± {vals.std(ddof=1):.4f}'
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f'{cfg.RESULTS_DIR}/multiseed_summary.csv', index=False)
    print('\n===== 跨种子汇总 (G-mean / F-value) =====')
    print(summary[['IR', 'Method', 'G-mean', 'F-value']].to_string(index=False))


if __name__ == '__main__':
    main()