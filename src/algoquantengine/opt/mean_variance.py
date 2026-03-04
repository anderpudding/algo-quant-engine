from __future__ import annotations

import numpy as np

from algoquantengine.opt.solvers import projected_gradient_descent


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> tuple[float, float]:
    ret = float(mu @ w)
    vol = float(np.sqrt(max(0.0, w @ (Sigma @ w))))
    return ret, vol


def efficient_frontier(
    Sigma: np.ndarray,
    mu: np.ndarray,
    n_points: int = 25,
    extra_caps: list[tuple[list[int], float]] | None = None,
    max_iter: int = 8000,
    lr: float = 0.05,
) -> list[dict]:
    mu = np.asarray(mu, dtype=float)
    n = mu.size
    if n_points < 5:
        raise ValueError("n_points should be >= 5")

    # feasible target returns: from min achievable to max achievable (long-only)
    r_min = float(mu.min())
    r_max = float(mu.max())
    targets = np.linspace(r_min, r_max, n_points)

    points: list[dict] = []
    for tr in targets:
        w = projected_gradient_descent(
            Sigma=Sigma,
            mu=mu,
            target_return=float(tr),
            extra_caps=extra_caps,
            max_iter=max_iter,
            lr=lr,
        )
        ret, vol = _portfolio_stats(w, mu, Sigma)
        sharpe = ret / vol if vol > 0 else 0.0
        points.append(
            {
                "target_return": float(tr),
                "return": float(ret),
                "vol": float(vol),
                "sharpe": float(sharpe),
                "weights": w,
            }
        )
    return points