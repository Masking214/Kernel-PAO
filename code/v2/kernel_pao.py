# -*- coding: utf-8 -*-
"""Kernel-PAO：核空间后验概率自适应过采样（修订版）。

对应审稿意见的三处方法学修正：
- R1-3：核空间平方距离采用正确形式 d²(Φ_i,Φ_j)=κ(i,i)-2κ(i,j)+κ(j,j)，
        并对任意核通用（不再假设 κ(x,x)=1）。
- R1-4：采样权重改为在 p=P(y=1|x)=0.5 处取峰、向两侧衰减的钟形函数
        w(p)=exp(-(p-0.5)²/(2σ²))，真正把采样集中于边界；
        旧稿单调递增 sigmoid 会令深层内部样本权重反而最高。
- R1-7/10：合成样本总数恒为 m=N0-N1（恰好平衡）。倍数 N 表示每个种子生成
        N 个插值样本，种子数取 ceil(m/N)，最终截断到 m；N>1 不再造成反向不平衡。
"""
import numpy as np
from sklearn.svm import SVC

_EPS = 1e-8


def kernel_matrix(X1, X2, kernel='rbf', gamma=0.1, degree=3, coef0=0.0):
    """支持 rbf / linear / poly / sigmoid 的 Gram 矩阵。"""
    dot = X1 @ X2.T
    if kernel == 'rbf':
        sq1 = np.sum(X1 ** 2, axis=1)[:, None]
        sq2 = np.sum(X2 ** 2, axis=1)[None, :]
        dist2 = np.maximum(sq1 + sq2 - 2 * dot, 0.0)
        return np.exp(-gamma * dist2)
    if kernel == 'linear':
        return dot
    if kernel == 'poly':
        return (gamma * dot + coef0) ** degree
    if kernel == 'sigmoid':
        return np.tanh(gamma * dot + coef0)
    raise ValueError(f'未知核函数: {kernel}')


def boundary_weights(p, mode, sigma_w=0.15):
    """根据后验概率返回未归一化权重。

    mode:
      random             : 不用后验，等权（消融 M1）
      posterior_uniform : 计算后验但等权采样（消融 M2）
      proportional       : 直接以 p 为权重（消融 M3）
      pao                : 钟形边界聚焦权重（完整方法 M4，修订后公式 3.8）
    """
    p = np.clip(p, _EPS, 1 - _EPS)
    if mode in ('random', 'posterior_uniform'):
        return np.ones_like(p)
    if mode == 'proportional':
        return p
    if mode == 'pao':
        return np.exp(-((p - 0.5) ** 2) / (2.0 * sigma_w ** 2))
    raise ValueError(f'未知权重模式: {mode}')


class KernelPAO:
    """统一封装 Kernel-PAO 及其消融变体。

    标签约定：y=1 少数类（故障），y=0 多数类（健康）。
    """

    def __init__(self, kernel='rbf', gamma=0.1, C=1.0, degree=3, coef0=0.0,
                 sigma_w=0.15, N=1, k=5, mode='pao', random_state=42):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.sigma_w = sigma_w
        self.N = N
        self.k = k
        self.mode = mode
        self.random_state = random_state

    # ---------- 后验概率（公式 3.5–3.7） ----------
    def _posterior(self, K, y):
        min_idx = np.where(y == 1)[0]
        maj_idx = np.where(y == 0)[0]
        rho_min = K[:, min_idx].sum(axis=1)
        rho_maj = K[:, maj_idx].sum(axis=1)
        p_all = rho_min / (rho_min + rho_maj + _EPS)
        return p_all, min_idx

    def _gram(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return kernel_matrix(X1, X2, kernel=self.kernel, gamma=self.gamma,
                             degree=self.degree, coef0=self.coef0)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        rng = np.random.RandomState(self.random_state)
        self.X_train_ = X

        n = len(X)
        min_idx = np.where(y == 1)[0]
        maj_idx = np.where(y == 0)[0]
        n_min, n_maj = len(min_idx), len(maj_idx)
        m = n_maj - n_min                       # 总合成数：恰好平衡（R1-7/10）

        K = self._gram(X)

        if m <= 0 or n_min == 0:                # 已平衡，直接训练
            self.clf_ = SVC(kernel='precomputed', C=self.C).fit(K, y)
            self.generators_ = None
            return self

        # ---- 选择种子 ----
        if self.mode == 'random':
            seed_weights = np.ones(n_min)
        else:
            p_all, _ = self._posterior(K, y)
            p_min = p_all[min_idx]
            seed_weights = boundary_weights(p_min, self.mode, self.sigma_w)
        seed_weights = seed_weights / seed_weights.sum()

        n_seeds = int(np.ceil(m / self.N))
        seeds = rng.choice(min_idx, size=n_seeds, p=seed_weights, replace=True)

        # ---- 每个种子在少数类 k 近邻中随机取 N 个，核空间线性插值（公式 3.10） ----
        # 通用核距离 d²=κ(i,i)-2κ(i,j)+κ(j,j)（R1-3）
        kii = np.diag(K)
        K_min = K[np.ix_(min_idx, min_idx)]
        d2_min = (kii[min_idx][:, None] - 2 * K_min + kii[min_idx][None, :])
        d2_min = np.maximum(d2_min, 0.0)

        gen_seed, gen_neigh, gen_delta = [], [], []
        pos_of = {orig: q for q, orig in enumerate(min_idx)}
        for sd in seeds:
            q = pos_of[sd]
            order = np.argsort(d2_min[q])
            cand = [min_idx[t] for t in order if min_idx[t] != sd][:self.k]
            if not cand:
                continue
            take = min(self.N, len(cand))
            for nb in rng.choice(cand, size=take, replace=False):
                gen_seed.append(sd)
                gen_neigh.append(int(nb))
                gen_delta.append(rng.uniform(0.0, 1.0))
            if len(gen_seed) >= m:
                break
        gen_seed = np.asarray(gen_seed[:m])
        gen_neigh = np.asarray(gen_neigh[:m])
        gen_delta = np.asarray(gen_delta[:m])
        self.generators_ = (gen_seed, gen_neigh, gen_delta)

        # ---- 增广 Gram 矩阵（公式 3.11–3.13，向量化双线性展开） ----
        m_actual = len(gen_seed)
        nodes = np.concatenate([gen_seed, gen_neigh])          # 2m'
        coeff = np.zeros((2 * m_actual, m_actual))
        coeff[np.arange(m_actual), np.arange(m_actual)] = 1.0 - gen_delta
        coeff[m_actual + np.arange(m_actual), np.arange(m_actual)] = gen_delta
        B = K[np.ix_(nodes, nodes)]
        G_new_new = coeff.T @ B @ coeff
        G_new_orig = coeff.T @ K[nodes, :]

        K_aug = np.block([[K, G_new_orig.T],
                          [G_new_orig, G_new_new]])
        y_aug = np.concatenate([y, np.ones(m_actual)])
        self.clf_ = SVC(kernel='precomputed', C=self.C).fit(K_aug, y_aug)
        return self

    def _test_kernel(self, X_test):
        K_to = self._gram(X_test, self.X_train_)
        if self.generators_ is None:
            return K_to
        gen_seed, gen_neigh, gen_delta = self.generators_
        m_actual = len(gen_seed)
        nodes = np.concatenate([gen_seed, gen_neigh])
        coeff = np.zeros((2 * m_actual, m_actual))
        coeff[np.arange(m_actual), np.arange(m_actual)] = 1.0 - gen_delta
        coeff[m_actual + np.arange(m_actual), np.arange(m_actual)] = gen_delta
        K_new = self._gram(X_test, self.X_train_[nodes]) @ coeff
        return np.hstack([K_to, K_new])

    def predict(self, X_test):
        return self.clf_.predict(self._test_kernel(np.asarray(X_test, float)))

    def decision_function(self, X_test):
        return self.clf_.decision_function(self._test_kernel(np.asarray(X_test, float)))
