from __future__ import annotations

import numpy as np


def portfolio_pnl_from_scenarios(scenario_returns: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    scenario_returns:
      - shape (paths, N) for one-step returns, OR
      - shape (paths, horizon, N) for multi-step returns
    Returns portfolio PnL/return per path (simple return).
    """
    R = np.asarray(scenario_returns, dtype=float)
    w = np.asarray(w, dtype=float)

    if R.ndim == 2:
        # (paths, N)
        if R.shape[1] != w.size:
            raise ValueError("Scenario N must match weights size")
        pnl = R @ w
        return pnl
    elif R.ndim == 3:
        # (paths, horizon, N) -> compound per path
        if R.shape[2] != w.size:
            raise ValueError("Scenario N must match weights size")
        path_port = (R @ w)  # (paths, horizon)
        # compound: Π(1+r_t) - 1
        pnl = np.prod(1.0 + path_port, axis=1) - 1.0
        return pnl
    else:
        raise ValueError("scenario_returns must be 2D or 3D")


def var_cvar(pnl: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """
    VaR/CVaR on losses. Returns positive numbers representing loss levels.
    pnl: portfolio return distribution (higher is better).
    """
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1 or pnl.size < 10:
        raise ValueError("pnl must be 1D with at least 10 samples")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    losses = -pnl
    q = float(np.quantile(losses, alpha))
    cvar = float(losses[losses >= q].mean())
    return q, cvar


def max_drawdown(equity: np.ndarray) -> float:
    """
    equity: array of portfolio value over time (must be positive).
    Returns max drawdown as positive fraction (e.g., 0.25 means -25% peak-to-trough).
    """
    eq = np.asarray(equity, dtype=float)
    if eq.ndim != 1 or eq.size < 2:
        raise ValueError("equity must be 1D with length >= 2")
    if (eq <= 0).any():
        raise ValueError("equity must be positive")

    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)
    return float(np.max(dd))