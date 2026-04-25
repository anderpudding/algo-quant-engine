from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_rebalance(
    prices: pd.DataFrame,
    rebalance_every: int,
    lookback: int,
    make_weights_fn,
    return_turnover: bool = False,
):
    """
    Walk-forward backtest using simple returns.

    At each rebalance date:
    - estimate weights using past lookback returns
    - hold until next rebalance
    - track turnover when weights change

    If return_turnover=False:
        returns equity curve only.

    If return_turnover=True:
        returns (equity curve, turnover series).
    """
    if rebalance_every < 1 or lookback < 2:
        raise ValueError("rebalance_every must be >= 1 and lookback >= 2")

    rets = prices.pct_change().dropna(how="any")
    if len(rets) <= lookback:
        raise ValueError("Not enough data for backtest lookback")

    equity: list[float] = []
    eq_dates: list[pd.Timestamp] = []
    turnover_values: list[float] = []

    current_w = None
    prev_w = None
    eq = 1.0

    for t in range(lookback, len(rets)):
        is_rebalance = (t - lookback) % rebalance_every == 0 or current_w is None

        if is_rebalance:
            window = rets.iloc[t - lookback : t]
            new_w = np.asarray(make_weights_fn(window), dtype=float)

            if new_w.ndim != 1 or new_w.size != rets.shape[1]:
                raise ValueError("make_weights_fn returned invalid weight vector")

            if prev_w is None:
                turnover = 0.0
            else:
                turnover = float(np.abs(new_w - prev_w).sum())

            current_w = new_w
            prev_w = new_w.copy()
        else:
            turnover = 0.0

        r_t = float(rets.iloc[t].to_numpy() @ current_w)
        eq *= 1.0 + r_t

        equity.append(eq)
        eq_dates.append(rets.index[t])
        turnover_values.append(turnover)

    equity_series = pd.Series(equity, index=eq_dates, name="equity")
    turnover_array = np.asarray(turnover_values, dtype=float)

    if return_turnover:
        return equity_series, turnover_array

    return equity_series