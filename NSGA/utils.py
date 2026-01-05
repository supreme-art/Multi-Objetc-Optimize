# @Author : CY
# @Time : 2026/1/5 15:47
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# ================================
# 【新增】Pareto/HV/SP 工具函数
# ================================
def nondominated_mask(min_objs: np.ndarray) -> np.ndarray:
    """
    min_objs: (M, d) 全部目标均为“越小越好”
    返回: (M,) True 表示非支配解
    """
    M = min_objs.shape[0]
    mask = np.ones(M, dtype=bool)
    for i in range(M):
        if not mask[i]:
            continue
        dominates_i = np.all(min_objs <= min_objs[i], axis=1) & np.any(min_objs < min_objs[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            mask[i] = False
    return mask


def normalize_to_unit_box(X: np.ndarray, eps: float = 1e-12):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    denom = np.maximum(x_max - x_min, eps)
    Xn = (X - x_min) / denom
    return Xn, x_min, x_max


def spacing_metric(front_norm: np.ndarray) -> float:
    K = front_norm.shape[0]
    if K < 2:
        return 0.0
    diffs = front_norm[:, None, :] - front_norm[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dist, np.inf)
    d_i = np.min(dist, axis=1)
    d_mean = np.mean(d_i)
    return float(np.sqrt(np.sum((d_i - d_mean) ** 2) / (K - 1)))


def hypervolume_mc(front_norm_min: np.ndarray, ref: np.ndarray, n_samples: int = 20000, seed: int = 42) -> float:
    """
    Monte Carlo 估计 HV（越大越好）
    front_norm_min: 已归一化，且全最小化（越小越好）
    ref: 参考点（应比 front 更差），例如 [1.1, 1.1, 1.1]
    """
    rng = np.random.default_rng(seed)
    d = front_norm_min.shape[1]
    samples = rng.random((n_samples, d)) * ref
    dominated = np.any(np.all(front_norm_min[None, :, :] <= samples[:, None, :], axis=2), axis=1)
    return float(dominated.mean() * np.prod(ref))
