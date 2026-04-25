from __future__ import annotations

import numpy as np
from algoquantengine.opt.mean_variance import efficient_frontier


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def mean_variance_best_sharpe(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    frontier = efficient_frontier(Sigma, mu, n_points=25)
    best = max(frontier, key=lambda p: p["sharpe"])
    return best["weights"]


def min_variance(Sigma: np.ndarray) -> np.ndarray:
    n = Sigma.shape[0]
    ones = np.ones(n)

    try:
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma)

    w = inv @ ones
    w = w / (ones @ inv @ ones)

    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n