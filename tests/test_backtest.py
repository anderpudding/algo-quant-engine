import numpy as np
import pandas as pd

from algoquantengine.opt.backtest import backtest_rebalance


def test_backtest_returns_equity_and_turnover():
    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104, 105, 106],
            "B": [100, 99, 101, 102, 103, 104, 105],
        },
        index=pd.date_range("2024-01-01", periods=7),
    )

    def make_weights_fn(window_returns):
        return np.array([0.5, 0.5])

    equity, turnover = backtest_rebalance(
        prices=prices,
        rebalance_every=1,
        lookback=2,
        make_weights_fn=make_weights_fn,
        return_turnover=True,
    )

    assert len(equity) == len(turnover)
    assert len(equity) > 0
    assert np.isfinite(equity.to_numpy()).all()
    assert np.isfinite(turnover).all()
    assert turnover[0] == 0.0