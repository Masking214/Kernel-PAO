# -*- coding: utf-8 -*-
"""8 个对比方法（论文 4.5 节），统一接口 run_method(name, Xtr, ytr, Xte)。

返回 (y_pred, y_score)，y_score 为正类决策值（decision_function），
各方法口径一致，用于 G-mean/F-value/AUC/PR-AUC。

注：ASVB-SMOTE（OriginalAdaptive）与 ADWI 为忠实移植自原实验代码的核空间方法；
复制式过采样与 SMOTE 等在输入空间进行。
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

import config as cfg

_EPS = 1e-8


def _rbf(X1, X2, gamma):
    return np.exp(-gamma * cdist(X1, X2, 'sqeuclidean'))


def _svm_fit_predict(Xtr, ytr, Xte, gamma, C):
    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte), clf.decision_function(Xte)


# ---------------- 输入空间复制式过采样 ----------------
def _duplicate_oversample(X, y, weights=None, seed=None):
    # 种子在调用时从 cfg 读取，便于多种子实验统一改 cfg.RANDOM_STATE
    if seed is None:
        seed = cfg.RANDOM_STATE
    rng = np.random.RandomState(seed)
    maj = np.where(y == 0)[0]
    minor = np.where(y == 1)[0]
    n_syn = len(maj) - len(minor)
    if n_syn <= 0:
        return X, y
    if weights is None:
        chosen = rng.choice(minor, size=n_syn, replace=True)
    else:
        chosen = rng.choice(minor, size=n_syn, p=weights, replace=True)
    return np.vstack([X, X[chosen]]), np.hstack([y, np.ones(n_syn)])


def _uniform(X, y):
    return _duplicate_oversample(X, y)


def _stratified(X, y, feature_idx=6, n_bins=5):
    """按峭度五分层、按层占比复制（论文 4.5）。"""
    minor = np.where(y == 1)[0]
    feat = X[:, feature_idx][minor]
    bins = np.linspace(feat.min(), feat.max(), n_bins + 1)
    lab = np.clip(np.digitize(feat, bins) - 1, 0, n_bins - 1)
    uniq, counts = np.unique(lab, return_counts=True)
    n_syn = int(np.sum(y == 0)) - len(minor)
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    syn_idx = []
    alloc = (counts / len(minor) * n_syn).astype(int)
    alloc[-1] += n_syn - alloc.sum()
    for b, n_gen in zip(uniq, alloc):
        pool = minor[lab == b]
        if n_gen > 0 and len(pool):
            syn_idx.append(rng.choice(pool, size=n_gen, replace=True))
    if not syn_idx:
        return X, y
    chosen = np.concatenate(syn_idx)
    return np.vstack([X, X[chosen]]), np.hstack([y, np.ones(len(chosen))])


def _importance(X, y, gamma, C):
    """临时 SVM 决策值加权：|decision| 越小（越近边界）权重越高（论文表4）。"""
    temp = SVC(kernel='rbf', C=C, gamma=gamma).fit(X, y)
    dec = temp.decision_function(X)
    minor = np.where(y == 1)[0]
    w = np.exp(-np.abs(dec[minor]))
    w = w / w.sum()
    return _duplicate_oversample(X, y, weights=w)


def _random_under(X, y):
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    maj = np.where(y == 0)[0]
    minor = np.where(y == 1)[0]
    keep_maj = rng.choice(maj, size=len(minor), replace=False)
    keep = np.concatenate([keep_maj, minor])
    rng.shuffle(keep)
    return X[keep], y[keep]


# ---------------- 核空间插值公共过程（ASVB / ADWI） ----------------
def _kernel_space_predict(Xtr, ytr, Xte, generators, K, gamma, C):
    n_orig = len(Xtr)
    gen = np.asarray(generators)
    seeds, neigh, deltas = gen[:, 0].astype(int), gen[:, 1].astype(int), gen[:, 2].astype(float)
    G2 = (1 - deltas)[:, None] * K[seeds, :] + deltas[:, None] * K[neigh, :]
    m = len(seeds)
    ss, sn, nn = K[np.ix_(seeds, seeds)], K[np.ix_(seeds, neigh)], K[np.ix_(neigh, neigh)]
    G3 = ((1 - deltas)[:, None] * (1 - deltas)[None, :] * ss +
          (1 - deltas)[:, None] * deltas[None, :] * sn +
          deltas[:, None] * (1 - deltas)[None, :] * sn.T +
          deltas[:, None] * deltas[None, :] * nn)
    K_aug = np.block([[K, G2.T], [G2, G3]])
    y_aug = np.concatenate([ytr, np.ones(m)])
    clf = SVC(kernel='precomputed', C=C).fit(K_aug, y_aug)
    Kt = _rbf(Xte, Xtr, gamma)
    Kt_new = ((1 - deltas)[None, :] * Kt[:, seeds] + deltas[None, :] * Kt[:, neigh])
    Kt_aug = np.hstack([Kt, Kt_new])
    return clf.predict(Kt_aug), clf.decision_function(Kt_aug)


def _asvb(Xtr, ytr, Xte, k=5, N=1, gamma=cfg.GAMMA, C=cfg.C):
    """Guo et al. 的 ASVB-SMOTE：临时 SVM 取少数类支持向量，
    依近邻多数类数决定内插/外插方向。忠实移植原实现。"""
    temp = SVC(kernel='rbf', C=C, gamma=gamma).fit(Xtr, ytr)
    sv_idx = temp.support_
    sv_pos = sv_idx[ytr[sv_idx] == 1]
    if len(sv_pos) == 0:
        Xr, yr = SMOTE(random_state=cfg.RANDOM_STATE).fit_resample(Xtr, ytr)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    K = _rbf(Xtr, Xtr, gamma)
    minor = np.where(ytr == 1)[0]
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    gens = []
    for idx in sv_pos:
        neigh_all = np.argsort(2 - 2 * K[idx])[1:k + 1]
        n_maj_nb = int(np.sum(ytr[neigh_all] == 0))
        if n_maj_nb == k:
            continue
        delta_range = (-1, 0) if n_maj_nb <= k / 2 else (0, 1)
        if len(minor) <= 1:
            continue
        Km = K[np.ix_(minor, minor)]
        pos = np.where(minor == idx)[0][0]
        top = minor[np.argsort(2 - 2 * Km[pos])[1:N + 1]]
        for j in top:
            gens.append((idx, j, rng.uniform(delta_range[0], delta_range[1])))
    if not gens:
        return _svm_fit_predict(Xtr, ytr, Xte, gamma, C)
    return _kernel_space_predict(Xtr, ytr, Xte, gens, K, gamma, C)


def _adwi(Xtr, ytr, Xte, k=5, N=1, gamma=cfg.GAMMA, C=cfg.C):
    """ADWI：基于支持向量决策值自适应插值强度。忠实移植原实现。"""
    temp = SVC(kernel='rbf', C=C, gamma=gamma).fit(Xtr, ytr)
    sv_idx = temp.support_
    sv_pos = sv_idx[ytr[sv_idx] == 1]
    if len(sv_pos) == 0:
        Xr, yr = SMOTE(random_state=cfg.RANDOM_STATE).fit_resample(Xtr, ytr)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    dual = temp.dual_coef_[0]
    sv = temp.support_vectors_
    intercept = temp.intercept_[0]
    fvals = np.abs(dual.dot(_rbf(sv, sv, gamma)) + intercept)
    sv_pos_in = {o: i for i, o in enumerate(sv_idx)}
    w = {o: fvals[sv_pos_in[o]] for o in sv_pos}
    wmax = max(w.values()) if w else 1.0
    w = {o: v / wmax for o, v in w.items()}
    K = _rbf(Xtr, Xtr, gamma)
    minor = np.where(ytr == 1)[0]
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    gens = []
    for idx in sv_pos:
        neigh_all = np.argsort(2 - 2 * K[idx])[1:k + 1]
        n_maj_nb = int(np.sum(ytr[neigh_all] == 0))
        if n_maj_nb == k or len(minor) <= 1:
            continue
        wi = w.get(idx, 0.5)
        if n_maj_nb > k / 2:
            delta = 0.5 * wi * rng.uniform(0, 1)
        else:
            delta = -0.5 * (1 + wi) * rng.uniform(0, 1)
        Km = K[np.ix_(minor, minor)]
        pos = np.where(minor == idx)[0][0]
        top = minor[np.argsort(2 - 2 * Km[pos])[1:N + 1]]
        for j in top:
            gens.append((idx, j, delta))
    if not gens:
        return _svm_fit_predict(Xtr, ytr, Xte, gamma, C)
    return _kernel_space_predict(Xtr, ytr, Xte, gens, K, gamma, C)


# ---------------- 统一入口 ----------------
def run_method(name, Xtr, ytr, Xte, gamma=cfg.GAMMA, C=cfg.C,
               pao_kwargs=None):
    if name == 'SVM':
        return _svm_fit_predict(Xtr, ytr, Xte, gamma, C)
    if name == 'Uniform':
        Xr, yr = _uniform(Xtr, ytr)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    if name == 'Stratified':
        Xr, yr = _stratified(Xtr, ytr)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    if name == 'Importance':
        Xr, yr = _importance(Xtr, ytr, gamma, C)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    if name == 'SMOTE':
        Xr, yr = SMOTE(random_state=cfg.RANDOM_STATE).fit_resample(Xtr, ytr)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    if name == 'RandomUnder':
        Xr, yr = _random_under(Xtr, ytr)
        return _svm_fit_predict(Xr, yr, Xte, gamma, C)
    if name == 'OriginalAdaptive':
        return _asvb(Xtr, ytr, Xte, gamma=gamma, C=C)
    if name == 'ADWI':
        return _adwi(Xtr, ytr, Xte, gamma=gamma, C=C)
    if name == 'Kernel-PAO':
        from kernel_pao import KernelPAO
        kw = dict(kernel='rbf', gamma=gamma, C=C, sigma_w=cfg.SIGMA_W,
                  N=1, k=cfg.K_NEIGHBORS, mode='pao',
                  random_state=cfg.RANDOM_STATE)
        kw.update(pao_kwargs or {})
        model = KernelPAO(**kw).fit(Xtr, ytr)
        return model.predict(Xte), model.decision_function(Xte)
    raise ValueError(f'未知方法: {name}')
