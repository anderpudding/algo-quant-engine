import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from algoquantengine import cli
from algoquantengine.data.loaders import load_benchmark_prices_csv
from algoquantengine.report.factor_analysis import (
    build_factor_exposure_table,
    build_rolling_factor_tables,
    compute_factor_exposure,
    compute_rolling_factor_exposure,
)
from algoquantengine.report.factor_plots import (
    plot_rolling_beta,
    plot_rolling_correlation,
    plot_strategy_vs_benchmark,
)


@pytest.fixture
def benchmark():
    return pd.Series(
        [0.01, -0.02, 0.015, 0.003, -0.008, 0.02, -0.004, 0.012],
        index=pd.date_range("2024-01-01", periods=8), name="Market",
    )


def test_known_beta_and_annualized_alpha(benchmark):
    exposure = compute_factor_exposure(0.0002 + 1.5 * benchmark, benchmark)
    assert exposure["beta"] == pytest.approx(1.5)
    assert exposure["alpha"] == pytest.approx(0.0002 * 252)
    assert exposure["correlation"] == pytest.approx(1)
    assert exposure["r_squared"] == pytest.approx(1)
    assert exposure["residual_volatility"] == pytest.approx(0, abs=1e-12)


def test_effective_annual_risk_free_conversion(benchmark):
    annualization, risk_free_rate = 12, 0.06
    periodic_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    strategy = periodic_rf + 0.001 + 1.5 * (benchmark - periodic_rf)
    exposure = compute_factor_exposure(strategy, benchmark, risk_free_rate, annualization)
    assert exposure["alpha"] == pytest.approx(0.001 * annualization)
    assert exposure["beta"] == pytest.approx(1.5)


def test_identical_strategy_zero_tracking_error(benchmark):
    exposure = compute_factor_exposure(benchmark, benchmark, risk_free_rate=0.04)
    for key in ("beta", "correlation", "r_squared"):
        assert exposure[key] == pytest.approx(1)
    for key in ("alpha", "active_return", "tracking_error", "residual_volatility"):
        assert exposure[key] == pytest.approx(0)
    assert np.isnan(exposure["information_ratio"])


def test_noisy_regression_and_active_metrics(benchmark):
    strategy = 0.0002 + 1.5 * benchmark + np.array([0.002, 0, -0.003, 0.001, 0, 0.004, -0.001, 0])
    exposure = compute_factor_exposure(strategy, benchmark, annualization=12)
    intercept, slope = np.linalg.lstsq(
        np.column_stack([np.ones(len(benchmark)), benchmark]), strategy, rcond=None
    )[0]
    residuals = strategy - (intercept + slope * benchmark)
    active = strategy - benchmark
    tracking_error = np.std(active, ddof=1) * np.sqrt(12)
    assert exposure["alpha"] == pytest.approx(intercept * 12)
    assert exposure["beta"] == pytest.approx(slope)
    assert exposure["correlation"] == pytest.approx(np.corrcoef(strategy, benchmark)[0, 1])
    assert exposure["r_squared"] == pytest.approx(1 - sum(residuals**2) / sum((strategy - strategy.mean())**2))
    assert exposure["active_return"] == pytest.approx(active.mean() * 12)
    assert exposure["tracking_error"] == pytest.approx(tracking_error)
    assert exposure["information_ratio"] == pytest.approx(active.mean() * 12 / tracking_error)
    assert exposure["residual_volatility"] == pytest.approx(residuals.std(ddof=1) * np.sqrt(12))


def test_date_alignment_drops_nan_and_uses_only_common_dates(benchmark):
    strategy = (0.0002 + 1.5 * benchmark).iloc[2:].copy()
    strategy.iloc[1] = np.nan
    strategy.loc[pd.Timestamp("2024-01-09")] = 100
    market = benchmark.iloc[:-1].copy()
    market.iloc[4] = np.nan
    expected_dates = [benchmark.index[i] for i in (2, 5, 6)]
    result = compute_factor_exposure(strategy.iloc[::-1], market.iloc[::-1])
    expected = compute_factor_exposure(strategy.loc[expected_dates], market.loc[expected_dates])
    for metric in result:
        assert result[metric] == pytest.approx(expected[metric])


@pytest.mark.parametrize("value", [0.0, 0.01])
def test_constant_benchmark_raises(benchmark, value):
    with pytest.raises(ValueError, match="Benchmark return variance"):
        compute_factor_exposure(benchmark, pd.Series(value, index=benchmark.index))


def test_constant_strategy_undefined_correlation_and_r_squared(benchmark):
    result = compute_factor_exposure(pd.Series(0.01, index=benchmark.index), benchmark)
    assert result["beta"] == pytest.approx(0)
    assert np.isnan(result["correlation"])
    assert np.isnan(result["r_squared"])
    assert result["residual_volatility"] == pytest.approx(0)


@pytest.mark.parametrize("annualization", [0, -1, 2.5, True])
def test_invalid_annualization(benchmark, annualization):
    with pytest.raises(ValueError, match="annualization"):
        compute_factor_exposure(benchmark, benchmark, annualization=annualization)


@pytest.mark.parametrize("risk_free_rate", [-1, -2, np.nan, np.inf])
def test_invalid_risk_free_rate(benchmark, risk_free_rate):
    with pytest.raises(ValueError, match="risk_free_rate"):
        compute_factor_exposure(benchmark, benchmark, risk_free_rate=risk_free_rate)


def test_insufficient_nonfinite_and_duplicate_samples(benchmark):
    for sample in (benchmark.iloc[:0], benchmark.iloc[:1], benchmark.shift(20)):
        with pytest.raises(ValueError, match="at least 2 common"):
            compute_factor_exposure(sample, benchmark)
    invalid = benchmark.copy()
    invalid.iloc[1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        compute_factor_exposure(invalid, benchmark)
    with pytest.raises(ValueError, match="unique"):
        compute_factor_exposure(pd.concat([benchmark, benchmark.iloc[:1]]), benchmark)


def test_rolling_full_window_alignment_and_known_beta(benchmark):
    strategy = 0.0002 + 1.5 * benchmark
    strategy.iloc[1] = np.nan
    market = benchmark.iloc[1:]
    rolling = compute_rolling_factor_exposure(strategy, market, window=3)
    expected_index = benchmark.index[2:].rename("Date")
    pd.testing.assert_index_equal(rolling.index, expected_index)
    assert list(rolling.columns) == ["beta", "correlation"]
    assert rolling.iloc[:2].isna().all().all()
    np.testing.assert_allclose(rolling["beta"].iloc[2:], 1.5)
    np.testing.assert_allclose(rolling["correlation"].iloc[2:], 1)


def test_rolling_matches_summary_for_noisy_windows(benchmark):
    strategy = benchmark + np.array([0.002, 0, -0.003, 0.001, 0, 0.004, -0.001, 0])
    rolling = compute_rolling_factor_exposure(strategy, benchmark, window=3)
    for end in range(2, len(benchmark)):
        expected = compute_factor_exposure(strategy.iloc[end-2:end+1], benchmark.iloc[end-2:end+1])
        assert rolling.iloc[end]["beta"] == pytest.approx(expected["beta"])
        assert rolling.iloc[end]["correlation"] == pytest.approx(expected["correlation"])


def test_rolling_short_and_constant_windows(benchmark):
    assert compute_rolling_factor_exposure(benchmark, benchmark, window=60).isna().all().all()
    market = benchmark.copy()
    market.iloc[:3] = 0.01
    result = compute_rolling_factor_exposure(benchmark, market, window=3)
    assert result.iloc[2].isna().all()
    assert result.iloc[3:].notna().all().all()
    constant = compute_rolling_factor_exposure(benchmark * 0, benchmark, window=3)
    assert constant["correlation"].isna().all()
    np.testing.assert_allclose(constant["beta"].iloc[2:], 0)


@pytest.mark.parametrize("window", [0, 1, -1, 2.5, True])
def test_invalid_rolling_window(benchmark, window):
    with pytest.raises(ValueError, match="window"):
        compute_rolling_factor_exposure(benchmark, benchmark, window=window)


def _equity_from_returns(returns):
    index = pd.DatetimeIndex([returns.index[0] - pd.Timedelta(days=1), *returns.index], name="Date")
    return pd.Series(np.r_[1.0, (1 + returns).cumprod()], index=index)


def test_multi_strategy_table_uses_same_sample(benchmark):
    equity = pd.DataFrame({
        "Custom A": _equity_from_returns(benchmark),
        "Custom B": _equity_from_returns(0.0002 + 1.5 * benchmark),
    })
    equity.loc[benchmark.index[3], "Custom B"] = np.nan
    table = build_factor_exposure_table(equity, benchmark)
    assert list(table["strategy"]) == list(equity.columns)
    assert list(table.columns) == [
        "strategy", "alpha", "beta", "correlation", "r_squared", "active_return",
        "tracking_error", "information_ratio", "residual_volatility",
    ]
    valid_dates = benchmark.index.delete([3, 4])
    for i, column in enumerate(equity.columns):
        expected = compute_factor_exposure(equity[column].pct_change(fill_method=None).loc[valid_dates], benchmark)
        for metric in expected:
            np.testing.assert_allclose(table.iloc[i][metric], expected[metric], atol=1e-12)
    beta, correlation = build_rolling_factor_tables(equity, benchmark, window=3)
    assert list(beta.columns) == list(correlation.columns) == list(equity.columns)
    pd.testing.assert_index_equal(beta.index, valid_dates.rename("Date"))
    np.testing.assert_allclose(beta.iloc[2:], np.tile([1, 1.5], (len(beta)-2, 1)))


def test_equity_returns_are_computed_before_alignment(benchmark):
    equity = _equity_from_returns(benchmark).to_frame("Strategy")
    market = benchmark.drop(benchmark.index[3])
    actual = build_factor_exposure_table(equity, market).iloc[0]
    assert actual["beta"] == pytest.approx(1)
    assert actual["tracking_error"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("prices", [[100, 0, 102], [100, -1, 102], [100, np.inf, 102]])
def test_benchmark_loader_invalid_prices(tmp_path, prices):
    path = tmp_path / "benchmark.csv"
    pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=3), "SPY": prices}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        load_benchmark_prices_csv(str(path))


def test_benchmark_loader_selection_and_missing_prices(tmp_path):
    path = tmp_path / "benchmark.csv"
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4),
        "SPY": [100, np.nan, 103, 104], "QQQ": [80, 81, 82, 80],
        "Notes": ["a", "b", "c", "d"],
    })
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="multiple numeric columns"):
        load_benchmark_prices_csv(str(path))
    for column in ("Missing", "Notes"):
        with pytest.raises(ValueError, match="missing or non-numeric"):
            load_benchmark_prices_csv(str(path), column=column)
    market = load_benchmark_prices_csv(str(path), column="SPY")
    assert pd.isna(market.iloc[1])
    assert market.pct_change(fill_method=None).iloc[:3].isna().all()
    frame.drop(columns="QQQ").iloc[::-1].to_csv(path, index=False)
    pd.testing.assert_series_equal(load_benchmark_prices_csv(str(path)), market)
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_benchmark_prices_csv(str(path), column="SPY")


def test_factor_plots_normalize_common_date_and_close_figures(tmp_path, benchmark, monkeypatch):
    equity = pd.DataFrame({"Custom": _equity_from_returns(1.5 * benchmark) * 100})
    prices = _equity_from_returns(benchmark).iloc[2:] * 80
    plotted = []
    savefig = plt.savefig

    def capture(*args, **kwargs):
        plotted.append([(line.get_label(), line.get_ydata().copy()) for line in plt.gca().lines])
        savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", capture)
    initial_figures = plt.get_fignums()
    plot_strategy_vs_benchmark(equity, prices, str(tmp_path / "performance.png"))
    assert len(plotted[0]) == 2
    for _, values in plotted[0]:
        assert values[0] == 1
    np.testing.assert_allclose(plotted[0][0][1], equity.loc[prices.index, "Custom"] / equity.loc[prices.index[0], "Custom"])
    rolling = compute_rolling_factor_exposure(1.5 * benchmark, benchmark, window=3)
    plot_rolling_beta(rolling[["beta"]], str(tmp_path / "beta.png"))
    plot_rolling_correlation(rolling[["correlation"]], str(tmp_path / "correlation.png"))
    assert len(plotted[1]) == 2  # Strategy and beta=1 reference line.
    assert plt.get_fignums() == initial_figures
    assert all((tmp_path / name).stat().st_size > 0 for name in ("performance.png", "beta.png", "correlation.png"))


def test_rolling_cli_optional_benchmark_exports_and_preserves_results(tmp_path):
    price_path = tmp_path / "prices.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    dates = pd.date_range("2024-01-01", periods=10)
    pd.DataFrame({
        "Date": dates, "A": [100, 101, 102, 103, 104, 103, 105, 106, 107, 108],
        "B": [200, 198, 202, 205, 207, 206, 210, 212, 211, 214],
    }).to_csv(price_path, index=False)
    pd.DataFrame({
        "Date": dates, "SPY": [100, 101, 99, 102, 104, 103, 105, 106, 104, 108],
    }).to_csv(benchmark_path, index=False)
    shared = [
        "rolling-compare", "--data", str(price_path), "--lookback", "3", "--rebalance", "1",
        "--clusters", "2", "--cap", "0.9", "--cost", "0.001", "--seed", "42",
    ]
    parser = cli.build_parser()
    defaults = parser.parse_args(shared)
    assert (defaults.benchmark_data, defaults.benchmark_column) == (None, None)
    assert (defaults.risk_free_rate, defaults.annualization, defaults.factor_window) == (0, 252, 60)
    baseline, enabled = tmp_path / "baseline", tmp_path / "enabled"
    for directory, extra in (
        (baseline, []),
        (enabled, ["--benchmark-data", str(benchmark_path), "--factor-window", "3", "--risk-free-rate", "0.04", "--annualization", "12"]),
    ):
        args = parser.parse_args(shared + extra + ["--out-dir", str(directory)])
        args.func(args)
    factor_files = ["factor_exposure.csv", "rolling_beta.csv", "rolling_correlation.csv"]
    factor_figures = ["strategy_vs_benchmark.png", "rolling_beta.png", "rolling_correlation.png"]
    for name in factor_files:
        assert not (baseline / name).exists()
        assert (enabled / name).stat().st_size > 0
    for name in factor_figures:
        assert not (baseline / "figures" / name).exists()
        assert (enabled / "figures" / name).stat().st_size > 0
    for name in ("rolling_strategy_metrics.csv", "rolling_equity_curves.csv"):
        assert (baseline / name).read_bytes() == (enabled / name).read_bytes()
    for name in ("rolling_equity_curves.png", "rolling_drawdowns.png"):
        assert (enabled / "figures" / name).stat().st_size > 0
    equity = pd.read_csv(enabled / "rolling_equity_curves.csv", index_col="Date", parse_dates=True)
    market = load_benchmark_prices_csv(str(benchmark_path)).pct_change(fill_method=None)
    expected = build_factor_exposure_table(equity, market, risk_free_rate=0.04, annualization=12)
    saved = pd.read_csv(enabled / "factor_exposure.csv")
    pd.testing.assert_frame_equal(saved, expected, atol=1e-10)
    for name, expected in zip(factor_files[1:], build_rolling_factor_tables(equity, market, window=3)):
        saved = pd.read_csv(enabled / name, index_col="Date", parse_dates=True)
        pd.testing.assert_frame_equal(saved, expected, check_freq=False)
