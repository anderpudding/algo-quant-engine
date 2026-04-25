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