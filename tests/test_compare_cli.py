import numpy as np
import pandas as pd
import pytest

from algoquantengine import cli
from algoquantengine.data.features import estimate_mu_cov, corr_matrix
from algoquantengine.data.preprocess import compute_returns


@pytest.mark.parametrize("command", ["compare", "rolling-compare"])
def test_comparison_cli_hybrid_defaults(command):
    args = cli.build_parser().parse_args([command, "--data", "prices.csv"])
    assert (args.clusters, args.cap, args.seed) == (4, 0.40, 42)


def test_static_comparison_hybrid_uses_same_returns_and_reporting(tmp_path, monkeypatch):
    prices = pd.DataFrame(
        {"A": [100, 101, 102, 101, 103], "B": [100, 99, 101, 102, 101]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    calls = []
    evaluated = []
    dashboards = []
    weights = np.array([0.6, 0.4])

    def spy_hybrid(Sigma, mu, corr, n_clusters, max_per_cluster, seed):
        calls.append((Sigma, mu, corr, n_clusters, max_per_cluster, seed))
        return weights, np.array([0, 1]), [([0], 0.9), ([1], 0.9)]

    def spy_evaluate(**kwargs):
        evaluated.append(kwargs)
        return {"strategy": kwargs["name"], "net_sharpe": 0.0}

    monkeypatch.setattr(cli, "load_clean_validate_prices", lambda args: prices)
    monkeypatch.setattr(cli, "hybrid_graph_constrained", spy_hybrid)
    monkeypatch.setattr(cli, "evaluate_strategy", spy_evaluate)
    monkeypatch.setattr(cli, "export_strategy_dashboard", lambda df, out_dir: dashboards.append(df))
    args = cli.build_parser().parse_args([
        "compare", "--data", "prices.csv", "--clusters", "2", "--cap", "0.9",
        "--seed", "17", "--cost-rate", "0.002", "--out-dir", str(tmp_path),
    ])
    args.func(args)

    expected_returns = compute_returns(prices, method=args.returns)
    mu, cov = estimate_mu_cov(expected_returns, annualize=args.annualize)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], cov)
    np.testing.assert_allclose(calls[0][1], mu)
    np.testing.assert_allclose(calls[0][2], corr_matrix(expected_returns))
    assert calls[0][3:] == (2, 0.9, 17)
    assert [r["name"] for r in evaluated] == [
        "Equal Weight", "Mean-Variance", "Min Variance", "Hybrid Graph-Constrained"
    ]
    for result in evaluated:
        pd.testing.assert_frame_equal(result["rets"], expected_returns)
        assert result["cost_rate"] == 0.002
    np.testing.assert_array_equal(evaluated[-1]["weights"], weights)
    saved = pd.read_csv(tmp_path / "strategy_comparison.csv")
    assert len(saved) == 4
    pd.testing.assert_frame_equal(saved, dashboards[0].reset_index(drop=True))
