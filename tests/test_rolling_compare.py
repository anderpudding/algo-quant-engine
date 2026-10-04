import numpy as np
import pandas as pd
import pytest

from algoquantengine.data.features import estimate_mu_cov, corr_matrix
from algoquantengine.report import rolling_compare
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


def _two_asset_prices():
    returns = np.array([
        [0.01, -0.02], [0.02, 0.01], [-0.01, 0.03],
        [0.03, -0.01], [0.01, 0.02], [-0.02, 0.01],
        [0.02, -0.03], [0.01, 0.02], [-0.01, 0.01],
    ])
    values = np.vstack([np.ones(2), np.cumprod(1.0 + returns, axis=0)]) * 100
    return pd.DataFrame(
        values, columns=["A", "B"], index=pd.date_range("2024-01-01", periods=10)
    )


def test_default_rolling_comparison_includes_hybrid():
    metrics, equity = run_rolling_strategy_comparison(
        _two_asset_prices(), lookback=3, rebalance=2, hybrid_clusters=2, hybrid_cap=0.9
    )
    expected = {"Equal Weight", "Mean-Variance", "Min Variance", "Hybrid Graph-Constrained"}
    assert set(equity.columns) == expected
    assert set(metrics["strategy"]) == expected
    assert np.isfinite(equity.to_numpy()).all()
    assert np.isfinite(metrics.select_dtypes(include="number").to_numpy()).all()


def test_rolling_hybrid_uses_each_historical_window_and_transaction_costs(monkeypatch):
    prices = _two_asset_prices()
    returns = prices.pct_change().dropna()
    calls = []
    weights = [np.array([0.8, 0.2]), np.array([0.3, 0.7]), np.array([0.6, 0.4])]

    def spy_hybrid(Sigma, mu, corr, n_clusters, max_per_cluster, seed):
        calls.append((Sigma.copy(), mu.copy(), corr.copy(), n_clusters, max_per_cluster, seed))
        w = weights[len(calls) - 1]
        return w, np.array([0, 1]), [([0], max_per_cluster), ([1], max_per_cluster)]

    monkeypatch.setattr(rolling_compare, "hybrid_graph_constrained", spy_hybrid)
    metrics, equity = run_rolling_strategy_comparison(
        prices,
        strategies=["Hybrid Graph-Constrained"],
        lookback=3,
        rebalance=2,
        transaction_cost=0.01,
        hybrid_clusters=2,
        hybrid_cap=0.9,
        seed=17,
    )

    assert len(calls) == 3
    for call, t in zip(calls, range(3, len(returns), 2)):
        historical = returns.iloc[t - 3 : t]
        mu, cov = estimate_mu_cov(historical)
        np.testing.assert_allclose(call[0], cov)
        np.testing.assert_allclose(call[1], mu)
        np.testing.assert_allclose(call[2], corr_matrix(historical))
        assert call[3:] == (2, 0.9, 17)

    # Independently reconstruct equity and costs, including holding periods.
    value = 1.0
    expected_equity = []
    total_turnover = 0.0
    for t in range(3, len(returns)):
        j = (t - 3) // 2
        if (t - 3) % 2 == 0 and j > 0:
            turnover = float(np.abs(weights[j] - weights[j - 1]).sum())
            total_turnover += turnover
            value *= 1.0 - turnover * 0.01
        value *= 1.0 + float(returns.iloc[t].to_numpy() @ weights[j])
        expected_equity.append(value)

    np.testing.assert_allclose(equity["Hybrid Graph-Constrained"], expected_equity)
    assert metrics.iloc[0]["avg_turnover"] == pytest.approx(total_turnover / 3)
    assert metrics.iloc[0]["transaction_cost"] == 0.01


def test_rolling_hybrid_future_changes_do_not_affect_prior_equity():
    prices = _two_asset_prices()
    changed = prices.copy()
    changed.iloc[7:, 0] *= 1.01
    kwargs = dict(
        strategies=["Hybrid Graph-Constrained"], lookback=3, rebalance=1,
        hybrid_clusters=2, hybrid_cap=0.9, seed=42,
    )
    _, equity = run_rolling_strategy_comparison(prices, **kwargs)
    _, changed_equity = run_rolling_strategy_comparison(changed, **kwargs)
    pd.testing.assert_frame_equal(equity.loc[:prices.index[6]], changed_equity.loc[:prices.index[6]])
