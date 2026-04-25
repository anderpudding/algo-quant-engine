from __future__ import annotations

import numpy as np
import pandas as pd

from algoquantengine.opt.risk import (
    var_cvar,
    max_drawdown,
    portfolio_pnl_from_scenarios,
)
from algoquantengine.sim.scenarios import bootstrap_return_scenarios


def compute_turnover(prev_w: np.ndarray, new_w: np.ndarray) -> float:
    return float(np.abs(new_w - prev_w).sum())


def apply_transaction_costs(
    equity: np.ndarray,
    turnover_series: np.ndarray,
    cost_rate: float,
) -> np.ndarray:
    """
    Applies proportional transaction cost to an equity curve.

    cost_rate = 0.001 means 10 bps per unit turnover.
    """
    eq = np.asarray(equity, dtype=float).copy()
    turnover_series = np.asarray(turnover_series, dtype=float)

    if eq.shape[0] != turnover_series.shape[0]:
        raise ValueError("equity and turnover_series must have the same length")

    for i in range(len(eq)):
        cost = turnover_series[i] * cost_rate
        eq[i] *= 1.0 - cost

    return eq


def evaluate_strategy(
    name: str,
    weights: np.ndarray,
    rets: pd.DataFrame,
    prices: pd.DataFrame,
    paths: int,
    horizon: int,
    alpha: float,
    backtest_fn,
    cost_rate: float = 0.001,
) -> dict:
    weights = np.asarray(weights, dtype=float)

    # Annualized ex-ante statistics
    mu = rets.mean().to_numpy() * 252
    cov = rets.cov().to_numpy() * 252

    gross_return = float(mu @ weights)
    gross_volatility = float(np.sqrt(max(0.0, weights @ (cov @ weights))))
    gross_sharpe = gross_return / gross_volatility if gross_volatility > 0 else 0.0

    # Scenario risk
    scen = bootstrap_return_scenarios(
        rets,
        horizon=horizon,
        n_paths=paths,
        seed=42,
    )
    pnl = portfolio_pnl_from_scenarios(scen, weights)
    var, cvar = var_cvar(pnl, alpha=alpha)

    # Backtest + turnover
    backtest_result = backtest_fn(prices, weights)

    if isinstance(backtest_result, tuple):
        equity, turnover_series = backtest_result
    else:
        equity = backtest_result
        turnover_series = np.zeros(len(equity), dtype=float)

    equity_values = equity.to_numpy()
    turnover_series = np.asarray(turnover_series, dtype=float)

    if len(equity_values) != len(turnover_series):
        raise ValueError("equity and turnover_series length mismatch")

    mdd = max_drawdown(equity_values)
    avg_turnover = float(np.mean(turnover_series))
    total_turnover = float(np.sum(turnover_series))

    # Cost-adjusted equity
    net_equity = apply_transaction_costs(
        equity=equity_values,
        turnover_series=turnover_series,
        cost_rate=cost_rate,
    )

    gross_backtest_return = float(equity_values[-1] - 1.0)
    net_backtest_return = float(net_equity[-1] - 1.0)

    # Simple realized volatility from equity returns
    gross_equity_returns = pd.Series(equity_values).pct_change().dropna().to_numpy()
    net_equity_returns = pd.Series(net_equity).pct_change().dropna().to_numpy()

    gross_realized_vol = float(np.std(gross_equity_returns) * np.sqrt(252)) if len(gross_equity_returns) > 1 else 0.0
    net_realized_vol = float(np.std(net_equity_returns) * np.sqrt(252)) if len(net_equity_returns) > 1 else 0.0

    net_sharpe = net_backtest_return / net_realized_vol if net_realized_vol > 0 else 0.0

    return {
        "strategy": name,
        "expected_return": gross_return,
        "expected_volatility": gross_volatility,
        "expected_sharpe": gross_sharpe,
        "backtest_return": gross_backtest_return,
        "net_backtest_return": net_backtest_return,
        "net_sharpe": net_sharpe,
        "VaR": float(var),
        "CVaR": float(cvar),
        "max_drawdown": float(mdd),
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "cost_rate": float(cost_rate),
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values(by="net_sharpe", ascending=False)