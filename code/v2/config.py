# -*- coding: utf-8 -*-
"""全局配置：路径与论文固定超参数。

修订要点（对应审稿意见）
- R1-1：特征提取窗口改为无重叠（step = window_size），交叉验证按原始文件分组。
- R1-6：类方向反转——故障样本为少数类（label=1），健康样本为多数类（label=0）。
"""
import os

# ---------------- 路径 ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))   # Kernel-PAO/（v2 上两级）
DATA_DIR = os.path.join(PKG_ROOT, 'data')
RESULTS_DIR = os.path.join(PKG_ROOT, 'results_v2')
FIG_DIR = os.path.join(PKG_ROOT, 'figures_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# UORED 原始振动 CSV（60 个文件：20 健康 + 40 故障）
RAW_ROOT = r"D:\1_CSV_Raw_Data_Files (.csv)"

# v2 无重叠特征数据集
FEATURE_CSV = os.path.join(DATA_DIR, 'UORED_features_nonoverlap.csv')

# ---------------- 数据参数（论文 4.2 节） ----------------
WINDOW_SIZE = 5000     # 约 0.12 s @42kHz
STEP = WINDOW_SIZE     # R1-1：无重叠（原 step=2500，50% 重叠）
SIGNAL_COL = 'Accelerometer'

# 文件夹 -> 标签。R1-6：故障=少数类 1，健康=多数类 0
FOLDER_LABEL = {
    '1_Healthy': 0,
    '2_Inner_Race_Faults': 1,
    '3_Outer_Race_Faults': 1,
    '4_Ball_Faults': 1,
    '5_Cage_Faults': 1,
}

FEATURE_COLS = ['mean', 'std', 'rms', 'max', 'min', 'peak_to_peak',
                'kurtosis', 'skew', 'crest_factor', 'energy']

# ---------------- 超参数（论文 4.6 节，先验固定） ----------------
GAMMA = 0.1
C = 1.0
A = 10.0          # 钟形权重的灵敏度参数（沿用论文 a=10）
SIGMA_W = 0.15    # 钟形权重带宽：p=0.5 取峰值，p≈0/1 趋零（新公式 3.8）
K_NEIGHBORS = 5
RANDOM_STATE = 42

RATIOS = [2, 4, 8, 16]          # IR = N_maj / N_min
N_GRID = [1, 2, 3, 4, 5]        # 过采样倍数（修订后总合成数恒定）
SNR_GRID = [float('inf'), 30, 20, 15, 10, 5]

METHOD_ORDER = ['SVM', 'Uniform', 'Stratified', 'Importance', 'SMOTE',
                'RandomUnder', 'OriginalAdaptive', 'Kernel-PAO']

METHOD_LABEL_CN = {
    'SVM': '标准SVM',
    'Uniform': '均匀过采样',
    'Stratified': '分层过采样',
    'Importance': '重要性过采样',
    'SMOTE': 'SMOTE',
    'RandomUnder': '随机欠采样',
    'OriginalAdaptive': 'ASVB-SMOTE',
    'Kernel-PAO': 'Kernel-PAO',
}
