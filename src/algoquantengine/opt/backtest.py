from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_rebalance(
    prices: pd.DataFrame,
    rebalance_every: int,
    lookback: int,
    make_weights_fn,
) -> pd.Series:
    """
    Walk-forward backtest using simple returns.
    - At each rebalance date, compute weights from past lookback returns.
    - Hold until next rebalance.
    Returns equity curve (starting at 1.0).
    """
    if rebalance_every < 1 or lookback < 2:
        raise ValueError("rebalance_every must be >=1 and lookback >=2")

    rets = prices.pct_change().dropna(how="any")
    if len(rets) <= lookback:
        raise ValueError("Not enough data for backtest lookback")

    equity = [1.0]
    eq_dates = [rets.index[lookback]]

    current_w = None

    for t in range(lookback, len(rets)):
        if (t - lookback) % rebalance_every == 0 or current_w is None:
            window = rets.iloc[t - lookback : t]
            current_w = np.asarray(make_weights_fn(window), dtype=float)
            if current_w.ndim != 1 or current_w.size != rets.shape[1]:
                raise ValueError("make_weights_fn returned invalid weight vector")

        r_t = float(rets.iloc[t].to_numpy() @ current_w)
        equity.append(equity[-1] * (1.0 + r_t))
        eq_dates.append(rets.index[t])

    return pd.Series(equity[1:], index=eq_dates)