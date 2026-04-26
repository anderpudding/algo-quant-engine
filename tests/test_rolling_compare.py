import numpy as np
import pandas as pd

from algoquantengine.report.rolling_compare import run_rolling_strategy_comparison


def test_run_rolling_strategy_comparison_outputs_shapes():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")

    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 30),
            "B": np.linspace(80, 90, 30),
            "C": np.linspace(50, 55, 30),
        },
        index=dates,
    )

    metrics_df, equity_df = run_rolling_strategy_comparison(
        prices=prices,
        strategies=["Equal Weight", "Min Variance"],
        lookback=5,
        rebalance=2,
        transaction_cost=0.001,
    )

    assert not metrics_df.empty
    assert not equity_df.empty
    assert "strategy" in metrics_df.columns
    assert "avg_turnover" in metrics_df.columns
    assert set(equity_df.columns) == {"Equal Weight", "Min Variance"}