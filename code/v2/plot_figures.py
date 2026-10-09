# -*- coding: utf-8 -*-
"""基于 results_v2/ 的严格重跑数据，重绘正文 Figure 1–8（figures_v2/）。

对应审稿意见：
- R1-9/11：图编号统一、方法名 Kernel-CCP -> Kernel-PAO、图例术语一致。
- 数据全部来自无重叠窗口 + 分组 CV 的严格版结果。
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 150})

METHOD_ORDER = cfg.METHOD_ORDER
IR_LABELS = {'1:2': '1:2', '1:4': '1:4', '1:8': '1:8', '1:16': '1:16'}
COLORS = plt.cm.tab10(np.linspace(0, 1, len(METHOD_ORDER)))


def mean_std(s):
    m = re.match(r'\s*([0-9.]+)\s*[±~]\s*([0-9.]+)', str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


# ---------------- 数据加载 ----------------
main = pd.read_csv(f'{cfg.RESULTS_DIR}/table5_6_main.csv')
per_fold = pd.read_csv(f'{cfg.RESULTS_DIR}/main_per_fold.csv')
kernels = pd.read_csv(f'{cfg.RESULTS_DIR}/table7_kernels.csv')
ablation = pd.read_csv(f'{cfg.RESULTS_DIR}/table8_ablation.csv')
Ntab = pd.read_csv(f'{cfg.RESULTS_DIR}/table9_N.csv')
noise = pd.read_csv(f'{cfg.RESULTS_DIR}/table10_noise.csv')

OUT = cfg.FIG_DIR

# ============ Figure 1-3: 箱线图（IR=1:16，逐折分布） ============
for metric, fname, title in [
    ('G-mean', 'Figure1_gmean_boxplot', 'G-mean'),
    ('F-value', 'Figure2_fvalue_boxplot', 'F-value'),
    ('AUC', 'Figure3_auc_boxplot', 'AUC'),
]:
    sub = per_fold[per_fold['IR'] == '1:16']
    data = [sub[sub['Method'] == m][metric].values for m in METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, labels=METHOD_ORDER, patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.tick_params(axis='x', rotation=40)
    ax.set_ylabel(metric)
    ax.set_title(f'{title} distribution across the 5 folds at IR = 1:16')
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/{fname}.pdf', bbox_inches='tight')
    plt.close(fig)

# ============ Figure 4: G-mean 柱状图（4 个 IR，误差棒） ============
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(METHOD_ORDER))
width = 0.2
for j, (ir, lab) in enumerate(IR_LABELS.items()):
    sub = main[main['IR'] == ir]
    means, stds = [], []
    for m in METHOD_ORDER:
        row = sub[sub['Method'] == m]
        if len(row):
            mu, sd = mean_std(row['G-mean'].iloc[0])
            means.append(mu); stds.append(sd)
        else:
            means.append(np.nan); stds.append(np.nan)
    ax.bar(x + (j - 1.5) * width, means, width, yerr=stds, capsize=2,
           label=f'IR={lab}')
ax.set_xticks(x)
ax.set_xticklabels(METHOD_ORDER, rotation=40)
ax.set_ylabel('G-mean')
ax.set_title('G-mean across methods under different imbalance ratios')
ax.legend(title='Imbalance ratio')
fig.tight_layout()
fig.savefig(f'{OUT}/Figure4_gmean_bar.png', bbox_inches='tight')
fig.savefig(f'{OUT}/Figure4_gmean_bar.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Figure 5: G-mean 折线（随 IR 变化） ============
fig, ax = plt.subplots(figsize=(8, 4.8))
xs = [2, 4, 8, 16]
for m, c in zip(METHOD_ORDER, COLORS):
    ys = []
    for ir, lab in IR_LABELS.items():
        row = main[(main['IR'] == ir) & (main['Method'] == m)]
        ys.append(mean_std(row['G-mean'].iloc[0])[0] if len(row) else np.nan)
    ax.plot(xs, ys, marker='o', label=m, color=c)
ax.set_xscale('log', base=2)
ax.set_xticks(xs)
ax.set_xticklabels([f'1:{v}' for v in xs])
ax.set_xlabel('Imbalance ratio (fault : health)')
ax.set_ylabel('G-mean')
ax.set_title('G-mean as a function of imbalance ratio')
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/Figure5_gmean_line.png', bbox_inches='tight')
fig.savefig(f'{OUT}/Figure5_gmean_line.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Figure 6: 核函数柱状图 ============
kn = kernels[kernels['Kernel'] != 'NestedSelected(outer)']
kg = [mean_std(v)[0] for v in kn['G-mean']]
kf = [mean_std(v)[0] for v in kn['F-value']]
klab = kn['Kernel'].tolist()
xx = np.arange(len(klab))
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(xx - 0.2, kg, 0.4, label='G-mean')
ax.bar(xx + 0.2, kf, 0.4, label='F-value')
ax.set_xticks(xx)
ax.set_xticklabels(klab)
ax.set_ylabel('Score')
ax.set_title('Kernel-PAO under different kernel functions (IR=1:16)')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/Figure6_kernel_bar.png', bbox_inches='tight')
fig.savefig(f'{OUT}/Figure6_kernel_bar.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Figure 7: 消融堆叠贡献 ============
ab = ablation.copy()
ab_g = ab['Gmean_mean'].values
ab_f = ab['Fvalue_mean'].values
labels = ab['Model'].tolist()
xx = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(xx, ab_g, 0.6, label='G-mean')
ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylabel('G-mean')
ax.set_xlabel('Ablation model (M0–M4)')
ax.set_title('Ablation study (IR=1:16)')
for i, v in enumerate(ab_g):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/Figure7_ablation.png', bbox_inches='tight')
fig.savefig(f'{OUT}/Figure7_ablation.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Figure 8: 噪声鲁棒性 ============
fig, ax1 = plt.subplots(figsize=(8, 4.8))
# SNR_dB 列含字符串 'inf'，统一转字符串处理，按 SNR 从高到低排序
snr_levels = sorted(noise['SNR_dB'].astype(str).unique(),
                    key=lambda s: 1e9 if s == 'inf' else float(s), reverse=True)
for method in ['SVM', 'SMOTE', 'Kernel-PAO']:
    sub = noise[noise['Method'] == method]
    ys = [sub[sub['SNR_dB'].astype(str) == s]['Gmean_mean'].iloc[0] for s in snr_levels]
    xidx = np.arange(len(snr_levels))
    ax1.plot(xidx, ys, marker='o', label=method)
ax1.set_xticks(np.arange(len(snr_levels)))
ax1.set_xticklabels(['No noise' if s == 'inf' else f'{s} dB' for s in snr_levels])
ax1.set_xlabel('Noise level (SNR)')
ax1.set_ylabel('G-mean')
ax1.set_title('G-mean degradation with noise')
ax1.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/Figure8_noise.png', bbox_inches='tight')
fig.savefig(f'{OUT}/Figure8_noise.pdf', bbox_inches='tight')
plt.close(fig)

print('已生成 8 张图到', OUT)