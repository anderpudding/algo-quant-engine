from __future__ import annotations

import numpy as np
import pandas as pd

from algoquantengine.data.features import estimate_mu_cov
from algoquantengine.opt.strategies import (
    equal_weight,
    mean_variance_best_sharpe,
    min_variance,
)
from algoquantengine.opt.risk import max_drawdown


def _annualized_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    returns = equity.pct_change().dropna()

    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    ann_return = float((1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(equity.to_numpy())

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def _strategy_weights(
    strategy: str,
    window_returns: pd.DataFrame,
) -> np.ndarray:
    n = window_returns.shape[1]

    if strategy == "Equal Weight":
        return equal_weight(n)

    mu, cov = estimate_mu_cov(window_returns)

    if strategy == "Mean-Variance":
        return mean_variance_best_sharpe(cov, mu)

    if strategy == "Min Variance":
        return min_variance(cov)

    raise ValueError(f"Unknown strategy: {strategy}")


def run_rolling_strategy_comparison(
    prices: pd.DataFrame,
    strategies: list[str] | None = None,
    lookback: int = 60,
    rebalance: int = 21,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling walk-forward comparison.

    transaction_cost:
      0.001 = 10 bps per unit turnover.

    Returns:
      metrics_df, equity_df
    """
    if strategies is None:
        strategies = ["Equal Weight", "Mean-Variance", "Min Variance"]

    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if rebalance < 1:
        raise ValueError("rebalance must be >= 1")

    rets = prices.pct_change().dropna(how="any")
    if len(rets) <= lookback:
        raise ValueError("Not enough data for rolling comparison.")

    equity = {s: 1.0 for s in strategies}
    prev_w: dict[str, np.ndarray | None] = {s: None for s in strategies}
    equity_rows: list[dict] = []
    turnover_sum = {s: 0.0 for s in strategies}
    rebalance_count = {s: 0 for s in strategies}

    for t in range(lookback, len(rets)):
        date = rets.index[t]
        row = {"Date": date}

        for strategy in strategies:
            should_rebalance = (t - lookback) % rebalance == 0 or prev_w[strategy] is None

            if should_rebalance:
                window = rets.iloc[t - lookback : t]
                new_w = _strategy_weights(strategy, window)

                if prev_w[strategy] is None:
                    turnover = 0.0
                else:
                    turnover = float(np.abs(new_w - prev_w[strategy]).sum())

                cost = turnover * transaction_cost
                equity[strategy] *= 1.0 - cost

                turnover_sum[strategy] += turnover
                rebalance_count[strategy] += 1
                prev_w[strategy] = new_w

            r_t = float(rets.iloc[t].to_numpy() @ prev_w[strategy])
            equity[strategy] *= 1.0 + r_t
            row[strategy] = equity[strategy]

        equity_rows.append(row)

    equity_df = pd.DataFrame(equity_rows).set_index("Date")

    metrics = []
    for strategy in strategies:
        m = _annualized_metrics(equity_df[strategy])
        avg_turnover = turnover_sum[strategy] / max(rebalance_count[strategy], 1)
        m["strategy"] = strategy
        m["avg_turnover"] = float(avg_turnover)
        m["transaction_cost"] = float(transaction_cost)
        metrics.append(m)

    metrics_df = pd.DataFrame(metrics).sort_values("sharpe", ascending=False)

    return metrics_df, equity_df