from __future__ import annotations

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project v onto simplex: {w >= 0, sum(w)=1}.
    Sorting-based projection, O(N log N).
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be 1D")

    n = v.size
    if n == 0:
        raise ValueError("v must be non-empty")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        # fallback: uniform
        return np.ones(n) / n
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    # numerical cleanup
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s


def _apply_group_caps(
    w: np.ndarray,
    caps: list[tuple[list[int], float]] | None,
    *,
    eps: float = 1e-12,
    max_rounds: int = 50,
) -> np.ndarray:
    """
    Iteratively enforce group caps and simplex constraint.

    This is not a perfect constrained projection, but it enforces caps tightly
    in practice by alternating:
      (1) scale down violating groups
      (2) project back to simplex
    """
    if not caps:
        return project_to_simplex(w)

    w2 = project_to_simplex(w)

    for _ in range(max_rounds):
        violated = False

        # scale down any group that exceeds its cap
        for idxs, cap in caps:
            idxs = list(idxs)
            if not idxs:
                continue

            gsum = float(w2[idxs].sum())
            if gsum > cap + eps and gsum > 0:
                violated = True
                # scale slightly under cap to avoid numerical rebound
                scale = (cap - eps) / gsum if cap > eps else 0.0
                scale = max(scale, 0.0)
                w2[idxs] *= scale

        # if no violations, we're done
        if not violated:
            break

        # restore simplex constraint
        w2 = project_to_simplex(w2)

    return w2


def projected_gradient_descent(
    Sigma: np.ndarray,
    mu: np.ndarray,
    target_return: float,
    max_iter: int = 5000,
    lr: float = 0.05,
    tol: float = 1e-9,
    extra_caps: list[tuple[list[int], float]] | None = None,
    penalty: float = 50.0,
) -> np.ndarray:
    """
    Solve: minimize w^T Sigma w
      s.t. mu^T w >= target_return, sum w = 1, w >= 0,
      plus optional group caps.
    Uses PGD on simplex with a soft penalty for return constraint.

    Complexity per iter: O(N^2) due to Sigma @ w.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)

    n = mu.size
    if Sigma.shape != (n, n):
        raise ValueError("Sigma shape must be (N,N)")
    if n < 2:
        raise ValueError("Need at least 2 assets")
    if not np.all(np.isfinite(Sigma)) or not np.all(np.isfinite(mu)):
        raise ValueError("Non-finite Sigma/mu")

    # init: uniform
    w = np.ones(n) / n
    w = _apply_group_caps(w, extra_caps)

    prev_obj = None
    for _ in range(max_iter):
        # objective and gradient: 2 Sigma w
        Sw = Sigma @ w
        grad = 2.0 * Sw

        # return constraint penalty if violated
        ret = float(mu @ w)
        if ret < target_return:
            # penalty term: penalty*(target_return - mu^T w)^2
            # gradient: -2*penalty*(target_return - mu^T w)*mu
            grad -= 2.0 * penalty * (target_return - ret) * mu

        w_new = project_to_simplex(w - lr * grad)
        w_new = _apply_group_caps(w_new, extra_caps)

        # convergence check
        obj = float(w_new @ (Sigma @ w_new))
        if prev_obj is not None and abs(prev_obj - obj) < tol:
            w = w_new
            break
        prev_obj = obj
        w = w_new

    return w