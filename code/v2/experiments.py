# -*- coding: utf-8 -*-
"""全部重跑实验（按文件分组 CV，无数据泄漏；超参内层选择，R1-5）。

产出（results_v2/）：
- table5_6_main.csv / main_per_fold.csv  表5、表6（8 方法 × 4 IR，6 指标）
- table7_kernels.csv                     表7（核函数，嵌套 CV 选择）
- table8_ablation.csv                    表8（M0–M4；M4 钟形权重 σ 内层选择）
- table9_N.csv                           表9（N=1..5，合成总数恒定，R1-7/10）
- table10_noise.csv                      表10（噪声鲁棒性）
- ttests_5x2.csv                         Dietterich 5×2 配对检验（R1-8）
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg
from data_utils import load_features, make_subset, grouped_folds, fold_train_test
from metrics_utils import METRIC_NAMES, all_metrics
from baselines import run_method
from kernel_pao import KernelPAO

warnings.filterwarnings('ignore')

SIGMA_CANDS = [0.15, 0.30, 0.50]      # 钟形权重带宽候选（内层选择）


def add_noise(X, snr_db, seed=0):
    """逐特征按目标 SNR(dB) 加高斯白噪声（仅测试集，论文 5.5）。"""
    if np.isinf(snr_db):
        return X.copy()
    rng = np.random.RandomState(seed)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        p_sig = np.var(X[:, j])
        p_noise = max(p_sig / (10 ** (snr_db / 10.0)), 1e-12)
        out[:, j] = X[:, j] + rng.normal(0, np.sqrt(p_noise), size=len(X))
    return out


# ---------------- 预测器（统一签名 (Xtr,ytr,Xte,gtr)） ----------------
def select_sigma(Xtr, ytr, gtr, kernel='rbf', N=1, candidates=SIGMA_CANDS):
    """内层 3 折分组 CV 选钟形带宽 σ；只使用外层训练数据（R1-5）。"""
    best_sig, best_val = candidates[0], -1.0
    for sig in candidates:
        vals = []
        for tri, vai in grouped_folds(ytr, gtr, n_splits=3):
            sc = StandardScaler().fit(Xtr[tri])
            Xa, Xb = sc.transform(Xtr[tri]), sc.transform(Xtr[vai])
            m = KernelPAO(kernel=kernel, gamma=cfg.GAMMA, C=cfg.C,
                          sigma_w=sig, N=N, k=cfg.K_NEIGHBORS, mode='pao',
                          random_state=cfg.RANDOM_STATE).fit(Xa, ytr[tri])
            vals.append(all_metrics(ytr[vai], m.predict(Xb),
                                    m.decision_function(Xb))['G-mean'])
        v = float(np.mean(vals))
        if v > best_val:
            best_sig, best_val = sig, v
    return best_sig


def baseline_predictor(method):
    def fn(Xtr, ytr, Xte, gtr=None):
        return run_method(method, Xtr, ytr, Xte)
    return fn


def pao_predictor(kernel='rbf', N=1, mode='pao', nested_sigma=False,
                  sigma=cfg.SIGMA_W, degree=3):
    """Kernel-PAO 预测器；nested_sigma=True 时 σ 由内层 CV 选择。"""
    def fn(Xtr, ytr, Xte, gtr=None):
        sig = select_sigma(Xtr, ytr, gtr, kernel, N) if nested_sigma else sigma
        m = KernelPAO(kernel=kernel, gamma=cfg.GAMMA, C=cfg.C, degree=degree,
                      sigma_w=sig, N=N, k=cfg.K_NEIGHBORS, mode=mode,
                      random_state=cfg.RANDOM_STATE).fit(Xtr, ytr)
        return m.predict(Xte), m.decision_function(Xte)
    return fn


METHOD_PREDICTORS = {m: baseline_predictor(m) for m in cfg.METHOD_ORDER}
METHOD_PREDICTORS['Kernel-PAO'] = pao_predictor(mode='pao', nested_sigma=True)


def cv_loop(predictor, X, y, groups, n_splits=5, noise_snr=None, timing=False,
            seed=None):
    if seed is None:
        seed = cfg.RANDOM_STATE
    rows, fit_times = [], []
    for fi, (tr_idx, te_idx) in enumerate(grouped_folds(y, groups, n_splits, seed)):
        Xtr, ytr, Xte, yte, gtr, _ = fold_train_test(X, y, groups, tr_idx, te_idx)
        if noise_snr is not None:
            Xte = add_noise(Xte, noise_snr, seed=seed + fi)
        t0 = time.time()
        y_pred, y_score = predictor(Xtr, ytr, Xte, gtr)
        fit_times.append(time.time() - t0)
        m = all_metrics(yte, y_pred, y_score)
        m['Fold'] = fi + 1
        rows.append(m)
    df = pd.DataFrame(rows)
    return (df, float(np.mean(fit_times))) if timing else df


# ---------------- 表5 / 表6 ----------------
def experiment_main(methods=None, ratios=None):
    methods = methods or cfg.METHOD_ORDER
    ratios = ratios or cfg.RATIOS
    X_all, y_all, g_all = load_features()
    per_fold, summary = [], []
    for ratio in ratios:
        X, y, g = make_subset(X_all, y_all, g_all, ratio)
        for method in methods:
            df = cv_loop(METHOD_PREDICTORS[method], X, y, g)
            df.insert(0, 'Method', method)
            df.insert(0, 'IR', f'1:{ratio}')
            per_fold.append(df)
            row = {'IR': f'1:{ratio}', 'Method': method}
            for mt in METRIC_NAMES:
                row[mt] = f'{df[mt].mean():.4f} ± {df[mt].std(ddof=1):.4f}'
            summary.append(row)
            print(f'1:{ratio} {method:18s} G-mean={df["G-mean"].mean():.4f} '
                  f'F={df["F-value"].mean():.4f}')
    pd.concat(per_fold, ignore_index=True).to_csv(
        f'{cfg.RESULTS_DIR}/main_per_fold.csv', index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(f'{cfg.RESULTS_DIR}/table5_6_main.csv', index=False)
    return sm


# ---------------- 表7：核函数（嵌套选择，R1-5） ----------------
KERNEL_CANDS = [('RBF', 'rbf', 3), ('Poly-d2', 'poly', 2), ('Poly-d3', 'poly', 3),
                ('Linear', 'linear', 3), ('Sigmoid', 'sigmoid', 3)]


def experiment_kernels(ratio=16):
    X_all, y_all, g_all = load_features()
    X, y, g = make_subset(X_all, y_all, g_all, ratio)
    rec = {name: {'g': [], 'f': [], 't': [], 'warn': set()}
           for name, _, _ in KERNEL_CANDS}
    nested_g, nested_f, chosen = [], [], []

    for tr_idx, te_idx in grouped_folds(y, g, 5):
        sc = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = sc.transform(X[tr_idx]), sc.transform(X[te_idx])
        ytr, yte, gtr = y[tr_idx], y[te_idx], g[tr_idx]

        # 内层只比较核函数（σ 固定 0.30 以隔离核的效应）
        best_name, best_val = None, -1
        for name, ker, deg in KERNEL_CANDS:
            vals = []
            try:
                for tri, vai in grouped_folds(ytr, gtr, 3):
                    sc2 = StandardScaler().fit(Xtr[tri])
                    m = KernelPAO(kernel=ker, degree=deg, sigma_w=0.30,
                                  random_state=cfg.RANDOM_STATE).fit(
                        sc2.transform(Xtr[tri]), ytr[tri])
                    vals.append(all_metrics(
                        ytr[vai], m.predict(sc2.transform(Xtr[vai])),
                        m.decision_function(sc2.transform(Xtr[vai])))['G-mean'])
                v = float(np.mean(vals))
            except Exception:
                v = -1.0
            if v > best_val:
                best_name, best_val = name, v
        chosen.append(best_name)

        for name, ker, deg in KERNEL_CANDS:
            t0 = time.time()
            try:
                with warnings.catch_warnings(record=True) as wn:
                    warnings.simplefilter('always')
                    m = KernelPAO(kernel=ker, degree=deg, sigma_w=0.30,
                                  random_state=cfg.RANDOM_STATE).fit(Xtr, ytr)
                    pred, score = m.predict(Xte), m.decision_function(Xte)
                rec[name]['warn'].update(
                    {str(w.message) for w in wn if 'overflow' in str(w.message).lower()})
            except Exception as e:
                pred = np.zeros_like(yte)
                score = np.zeros_like(yte, float)
                rec[name]['warn'].add(f'ERROR:{e}')
            mt = all_metrics(yte, pred, score)
            rec[name]['g'].append(mt['G-mean'])
            rec[name]['f'].append(mt['F-value'])
            rec[name]['t'].append(time.time() - t0)
            if name == best_name:
                nested_g.append(mt['G-mean'])
                nested_f.append(mt['F-value'])

    rows = []
    for name, _, _ in KERNEL_CANDS:
        r = rec[name]
        rows.append({'Kernel': name,
                     'G-mean': f"{np.mean(r['g']):.4f} ± {np.std(r['g'], ddof=1):.4f}",
                     'F-value': f"{np.mean(r['f']):.4f} ± {np.std(r['f'], ddof=1):.4f}",
                     'Train_time_s': f"{np.mean(r['t']):.2f}",
                     'note': ';'.join(sorted(r['warn']))[:80]})
    rows.append({'Kernel': 'NestedSelected(outer)',
                 'G-mean': f'{np.mean(nested_g):.4f} ± {np.std(nested_g, ddof=1):.4f}',
                 'F-value': f'{np.mean(nested_f):.4f} ± {np.std(nested_f, ddof=1):.4f}',
                 'Train_time_s': '-', 'note': 'inner-chosen: ' + ','.join(chosen)})
    df = pd.DataFrame(rows)
    df.to_csv(f'{cfg.RESULTS_DIR}/table7_kernels.csv', index=False)
    print(df.to_string(index=False))
    return df


# ---------------- 表8：消融 ----------------
ABLATION = [('M0', None), ('M1', 'random'), ('M2', 'posterior_uniform'),
            ('M3', 'proportional'), ('M4', 'pao')]


def experiment_ablation(ratio=16):
    X_all, y_all, g_all = load_features()
    X, y, g = make_subset(X_all, y_all, g_all, ratio)
    rows = []
    for name, mode in ABLATION:
        if name == 'M0':
            pred = baseline_predictor('SVM')
        elif mode == 'pao':
            pred = pao_predictor(mode='pao', nested_sigma=True)   # M4 内层选 σ
        else:
            pred = pao_predictor(mode=mode, nested_sigma=False, sigma=0.30)
        df = cv_loop(pred, X, y, g)
        rows.append({'Model': name,
                     'G-mean': f"{df['G-mean'].mean():.4f} ± {df['G-mean'].std(ddof=1):.4f}",
                     'F-value': f"{df['F-value'].mean():.4f} ± {df['F-value'].std(ddof=1):.4f}",
                     'Gmean_mean': round(df['G-mean'].mean(), 4),
                     'Fvalue_mean': round(df['F-value'].mean(), 4)})
        print(name, rows[-1]['G-mean'], rows[-1]['F-value'])
    out = pd.DataFrame(rows)
    out.to_csv(f'{cfg.RESULTS_DIR}/table8_ablation.csv', index=False)
    return out


# ---------------- 表9：倍数 N（合成总数恒定，R1-7/10） ----------------
def experiment_N(ratio=16):
    X_all, y_all, g_all = load_features()
    X, y, g = make_subset(X_all, y_all, g_all, ratio)
    rows = []
    per_N_g = {}
    for N in cfg.N_GRID:
        df, tmean = cv_loop(pao_predictor(mode='pao', N=N, nested_sigma=False,
                                          sigma=0.30), X, y, g, timing=True)
        per_N_g[N] = df['G-mean'].values
        rows.append({'N': N, 'n_synthetic_total': 'N0-N1 (constant)',
                     'G-mean': f"{df['G-mean'].mean():.4f} ± {df['G-mean'].std(ddof=1):.4f}",
                     'F-value': f"{df['F-value'].mean():.4f} ± {df['F-value'].std(ddof=1):.4f}",
                     'Train_time_s': f'{tmean:.2f}'})
        print('N=', N, rows[-1]['G-mean'], rows[-1]['F-value'], f'{tmean:.1f}s')

    # 内层选择 N（只用训练折）
    nested_g = []
    chosen_n = []
    for tr_idx, te_idx in grouped_folds(y, g, 5):
        Xtr_raw, ytr, gtr = X[tr_idx], y[tr_idx], g[tr_idx]
        best_n, best_v = 1, -1
        for N in cfg.N_GRID:
            vals = []
            for tri, vai in grouped_folds(ytr, gtr, 3):
                sc = StandardScaler().fit(Xtr_raw[tri])
                m = KernelPAO(sigma_w=0.30, N=N, random_state=cfg.RANDOM_STATE).fit(
                    sc.transform(Xtr_raw[tri]), ytr[tri])
                vals.append(all_metrics(
                    ytr[vai], m.predict(sc.transform(Xtr_raw[vai])),
                    m.decision_function(sc.transform(Xtr_raw[vai])))['G-mean'])
            if np.mean(vals) > best_v:
                best_n, best_v = N, np.mean(vals)
        chosen_n.append(best_n)
        sc = StandardScaler().fit(Xtr_raw)
        m = KernelPAO(sigma_w=0.30, N=best_n, random_state=cfg.RANDOM_STATE).fit(
            sc.transform(Xtr_raw), ytr)
        Xte = sc.transform(X[te_idx])
        nested_g.append(all_metrics(y[te_idx], m.predict(Xte),
                                    m.decision_function(Xte))['G-mean'])
    rows.append({'N': 'NestedSelected', 'n_synthetic_total': 'constant',
                 'G-mean': f'{np.mean(nested_g):.4f} ± {np.std(nested_g, ddof=1):.4f}',
                 'F-value': '-', 'Train_time_s': '-',
                 'note': 'inner-chosen N: ' + ','.join(map(str, chosen_n))})
    out = pd.DataFrame(rows)
    out.to_csv(f'{cfg.RESULTS_DIR}/table9_N.csv', index=False)
    return out


# ---------------- 表10：噪声鲁棒性 ----------------
def experiment_noise(ratio=16):
    X_all, y_all, g_all = load_features()
    X, y, g = make_subset(X_all, y_all, g_all, ratio)
    noise_preds = {'SVM': baseline_predictor('SVM'),
                   'SMOTE': baseline_predictor('SMOTE'),
                   'Kernel-PAO': pao_predictor(mode='pao', nested_sigma=True)}
    rows = []
    for snr in cfg.SNR_GRID:
        for method in ['SVM', 'SMOTE', 'Kernel-PAO']:
            df = cv_loop(noise_preds[method], X, y, g, noise_snr=snr)
            rows.append({'SNR_dB': 'inf' if np.isinf(snr) else snr, 'Method': method,
                         'G-mean': f"{df['G-mean'].mean():.4f} ± {df['G-mean'].std(ddof=1):.4f}",
                         'F-value': f"{df['F-value'].mean():.4f} ± {df['F-value'].std(ddof=1):.4f}",
                         'Gmean_mean': round(df['G-mean'].mean(), 4)})
            print('SNR', rows[-1]['SNR_dB'], method, rows[-1]['G-mean'])
    out = pd.DataFrame(rows)
    out.to_csv(f'{cfg.RESULTS_DIR}/table10_noise.csv', index=False)
    return out


# ---------------- 5×2 显著性检验（R1-8） ----------------
def _ablation_predictor(model):
    if model == 'M0':
        return baseline_predictor('SVM')
    if model == 'M4':
        return pao_predictor(mode='pao', nested_sigma=True)
    mode = {'M1': 'random', 'M2': 'posterior_uniform', 'M3': 'proportional'}[model]
    return pao_predictor(mode=mode, sigma=0.30)


def experiment_ttests(ratios=None):
    from stats_tests import dietterich_5x2
    ratios = ratios or cfg.RATIOS
    X_all, y_all, g_all = load_features()
    results = []
    for ratio in ratios:
        X, y, g = make_subset(X_all, y_all, g_all, ratio)
        res = dietterich_5x2(METHOD_PREDICTORS['Kernel-PAO'],
                             METHOD_PREDICTORS['SMOTE'], X, y, g, metric='G-mean')
        results.append({'Comparison': 'Kernel-PAO vs SMOTE', 'IR': f'1:{ratio}',
                        'metric': 'G-mean', 't': round(res['t'], 4),
                        'p': round(res['p'], 4), 'mean_diff': round(res['mean_diff'], 4)})
        print(results[-1])
    X, y, g = make_subset(X_all, y_all, g_all, 16)
    for a, b in [('M1', 'M0'), ('M2', 'M1'), ('M3', 'M2'), ('M4', 'M3')]:
        res = dietterich_5x2(_ablation_predictor(a), _ablation_predictor(b),
                             X, y, g, metric='G-mean')
        results.append({'Comparison': f'{a} vs {b}', 'IR': '1:16',
                        'metric': 'G-mean', 't': round(res['t'], 4),
                        'p': round(res['p'], 4), 'mean_diff': round(res['mean_diff'], 4)})
        print(results[-1])
    out = pd.DataFrame(results)
    out.to_csv(f'{cfg.RESULTS_DIR}/ttests_5x2.csv', index=False)
    return out
