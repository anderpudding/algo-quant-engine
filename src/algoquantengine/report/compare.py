from __future__ import annotations

import numpy as np
import pandas as pd

from algoquantengine.opt.risk import var_cvar, max_drawdown, portfolio_pnl_from_scenarios
from algoquantengine.sim.scenarios import bootstrap_return_scenarios


def evaluate_strategy(
    name: str,
    weights: np.ndarray,
    rets: pd.DataFrame,
    prices: pd.DataFrame,
    paths: int,
    horizon: int,
    alpha: float,
    backtest_fn,
):
    mu = rets.mean().to_numpy() * 252
    cov = rets.cov().to_numpy() * 252

    ret = float(mu @ weights)
    vol = float(np.sqrt(weights @ (cov @ weights)))
    sharpe = ret / vol if vol > 0 else 0.0

    # risk
    scen = bootstrap_return_scenarios(rets, horizon=horizon, n_paths=paths, seed=42)
    pnl = portfolio_pnl_from_scenarios(scen, weights)
    var, cvar = var_cvar(pnl, alpha=alpha)

    # backtest
    equity = backtest_fn(prices, weights)
    mdd = max_drawdown(equity.to_numpy())

    return {
        "strategy": name,
        "return": ret,
        "volatility": vol,
        "sharpe": sharpe,
        "VaR": var,
        "CVaR": cvar,
        "max_drawdown": mdd,
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values(by="sharpe", ascending=False)

def static_backtest(prices, weights):
    rets = prices.pct_change().dropna()
    eq = 1.0
    equity = []

    for i in range(len(rets)):
        r = float(rets.iloc[i].to_numpy() @ weights)
        eq *= (1.0 + r)
        equity.append(eq)

    return pd.Series(equity, index=rets.index)

def compute_turnover(prev_w: np.ndarray, new_w: np.ndarray) -> float:
    return float(np.abs(new_w - prev_w).sum())

def apply_transaction_costs(
    equity: np.ndarray,
    turnover_series: np.ndarray,
    cost_rate: float
) -> np.ndarray:
    """
    Applies proportional transaction cost to equity curve.
    cost_rate: e.g. 0.001 = 10bps per unit turnover
    """
    eq = equity.copy()

    for i in range(1, len(eq)):
        cost = turnover_series[i] * cost_rate
        eq[i] = eq[i] * (1.0 - cost)

    return eq