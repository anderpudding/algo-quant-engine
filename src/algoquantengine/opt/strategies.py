from __future__ import annotations

import numpy as np

from algoquantengine.graph.algorithms import spectral_clusters_from_corr
from algoquantengine.graph.constraints import cluster_weight_caps
from algoquantengine.opt.mean_variance import efficient_frontier


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def mean_variance_best_sharpe(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    frontier = efficient_frontier(Sigma, mu, n_points=25)
    best = max(frontier, key=lambda p: p["sharpe"])
    return best["weights"]


def hybrid_graph_constrained(
    Sigma: np.ndarray,
    mu: np.ndarray,
    corr: np.ndarray,
    n_clusters: int = 4,
    max_per_cluster: float = 0.40,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[tuple[list[int], float]]]:
    """Select the best-Sharpe frontier portfolio under spectral cluster caps."""
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    corr = np.asarray(corr, dtype=float)

    if mu.ndim != 1 or mu.size < 2:
        raise ValueError("mu must be 1D with at least 2 assets")
    n = mu.size
    if Sigma.shape != (n, n) or corr.shape != (n, n):
        raise ValueError("Sigma and corr must have shape (N,N) matching mu")
    if not all(np.isfinite(a).all() for a in (Sigma, mu, corr)):
        raise ValueError("Non-finite Sigma/mu/corr")
    if not np.allclose(Sigma, Sigma.T) or not np.allclose(corr, corr.T):
        raise ValueError("Sigma and corr must be symmetric")
    if not np.allclose(np.diag(corr), 1.0) or np.any(np.abs(corr) > 1.0 + 1e-12):
        raise ValueError("corr must have unit diagonal and values in [-1, 1]")
    if not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not (0.0 < max_per_cluster <= 1.0):
        raise ValueError("max_per_cluster must be in (0, 1]")

    k = min(max(n_clusters, 2), n)
    if k * max_per_cluster < 1.0:
        raise ValueError("Infeasible cluster caps: total cluster capacity must be >= 1")

    labels = spectral_clusters_from_corr(corr, n_clusters=k, seed=seed)
    caps = cluster_weight_caps(labels, max_per_cluster=max_per_cluster)
    # Discretization can yield fewer nonempty clusters than requested.
    if len(caps) * max_per_cluster < 1.0:
        raise ValueError("Infeasible cluster caps for the generated clusters")
    frontier = efficient_frontier(Sigma, mu, n_points=25, extra_caps=caps)
    best = max(frontier, key=lambda p: p["sharpe"])
    weights = best["weights"]
    if any(weights[indices].sum() > cap + 1e-6 for indices, cap in caps):
        raise RuntimeError("Cluster cap projection did not converge within tolerance")
    return weights, labels, caps


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
