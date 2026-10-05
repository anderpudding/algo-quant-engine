from __future__ import annotations

import numpy as np
import pandas as pd


# Variance below this threshold cannot support a reliable benchmark slope.
_VARIANCE_TOL = 1e-16
_STD_TOL = 1e-12


def _validate_annualization(annualization: int) -> None:
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, (int, np.integer))
        or annualization < 1
    ):
        raise ValueError("annualization must be a positive integer.")


def _align_returns(
    strategy_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    for index in (strategy_returns.index, benchmark_returns.index):
        if not index.is_unique or index.hasnans:
            raise ValueError("Return dates must be unique and non-missing.")
    strategies, benchmark = strategy_returns.align(benchmark_returns, join="inner", axis=0)
    valid = strategies.notna().all(axis=1) & benchmark.notna()
    strategies = strategies.loc[valid].sort_index()
    benchmark = benchmark.loc[strategies.index]
    if len(strategies) < 2:
        raise ValueError("Factor analysis needs at least 2 common, non-missing return observations.")
    if not (
        np.isfinite(strategies.to_numpy(dtype=float)).all()
        and np.isfinite(benchmark.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Aligned returns must be finite.")
    return strategies, benchmark


def compute_factor_exposure(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> dict[str, float]:
    """Fit an OLS market model on common dates using simple periodic returns.

    Annual effective risk-free rate becomes (1 + rate)**(1 / annualization) - 1.
    Alpha and mean active return use arithmetic annualization. Volatilities use
    sample standard deviation (ddof=1). Information ratio is NaN for effectively
    zero tracking error; correlation and R² are NaN for constant strategies.
    Benchmark variance <= 1e-16 raises ValueError. No observations are filled.
    """
    _validate_annualization(annualization)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1.")
    strategies, benchmark = _align_returns(strategy_returns.to_frame(), benchmark_returns)
    strategy = strategies.iloc[:, 0].to_numpy(dtype=float)
    market = benchmark.to_numpy(dtype=float)
    periodic_rf = float(np.expm1(np.log1p(risk_free_rate) / annualization))
    x = market - periodic_rf
    y = strategy - periodic_rf
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    benchmark_variance = float(x_centered @ x_centered / (len(x) - 1))
    if benchmark_variance <= _VARIANCE_TOL:
        raise ValueError("Benchmark return variance is approximately zero; beta is undefined.")
    beta = float((x_centered @ y_centered) / (x_centered @ x_centered))
    alpha = float(y.mean() - beta * x.mean())
    residuals = y_centered - beta * x_centered
    sst = float(y_centered @ y_centered)
    if sst / (len(y) - 1) <= _VARIANCE_TOL:
        correlation = float("nan")
        r_squared = float("nan")
    else:
        correlation = float(np.clip(
            (x_centered @ y_centered) / np.sqrt((x_centered @ x_centered) * sst), -1, 1
        ))
        r_squared = float(1.0 - (residuals @ residuals) / sst)
    active = strategy - market
    active_return = float(active.mean() * annualization)
    active_std = float(active.std(ddof=1))
    tracking_error = float(active_std * np.sqrt(annualization))
    return {
        "alpha": alpha * annualization,
        "beta": beta,
        "correlation": correlation,
        "r_squared": r_squared,
        "active_return": active_return,
        "tracking_error": tracking_error,
        "information_ratio": active_return / tracking_error if active_std > _STD_TOL else float("nan"),
        "residual_volatility": float(residuals.std(ddof=1) * np.sqrt(annualization)),
    }


def compute_rolling_factor_exposure(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Trailing beta/correlation over full windows of common valid observations.

    Initial incomplete windows and windows with benchmark variance <= 1e-16
    yield NaN. Correlation also yields NaN for a constant strategy. A window
    longer than the aligned sample is allowed and returns all NaN.
    Risk-free subtraction does not change either of these metrics.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("factor window must be an integer >= 2.")
    strategies, benchmark = _align_returns(strategy_returns.to_frame(), benchmark_returns)
    strategy = strategies.iloc[:, 0]
    market_rolling = benchmark.rolling(window, min_periods=window)
    strategy_rolling = strategy.rolling(window, min_periods=window)
    variance = market_rolling.var(ddof=1)
    valid = variance > _VARIANCE_TOL
    beta = (strategy_rolling.cov(benchmark, ddof=1) / variance.where(valid)).where(valid)
    correlation = strategy_rolling.corr(benchmark).clip(-1, 1).where(
        valid & (strategy_rolling.var(ddof=1) > _VARIANCE_TOL)
    )
    return pd.DataFrame({"beta": beta, "correlation": correlation}).rename_axis("Date")


def _equity_factor_returns(
    equity_df: pd.DataFrame, benchmark_returns: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    if equity_df.empty or not equity_df.columns.is_unique:
        raise ValueError("Equity data must have observations and unique strategy columns.")
    if not equity_df.index.is_unique or equity_df.index.hasnans:
        raise ValueError("Equity dates must be unique and non-missing.")
    values = equity_df.to_numpy(dtype=float)
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Equity values must be finite and strictly positive.")
    # Calculate before aligning so a missing benchmark date never aggregates
    # several strategy returns into one return. Missing equity is never filled.
    returns = equity_df.sort_index().pct_change(fill_method=None)
    return _align_returns(returns, benchmark_returns)


def build_factor_exposure_table(
    equity_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> pd.DataFrame:
    """Evaluate every net equity column on the same complete return sample."""
    returns, benchmark = _equity_factor_returns(equity_df, benchmark_returns)
    return pd.DataFrame([
        {"strategy": column, **compute_factor_exposure(
            returns[column], benchmark, risk_free_rate, annualization
        )}
        for column in returns.columns
    ])


def build_rolling_factor_tables(
    equity_df: pd.DataFrame, benchmark_returns: pd.Series, window: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return beta and correlation tables with arbitrary strategy columns."""
    returns, benchmark = _equity_factor_returns(equity_df, benchmark_returns)
    exposures = {
        column: compute_rolling_factor_exposure(returns[column], benchmark, window)
        for column in returns.columns
    }
    beta = pd.DataFrame({column: frame["beta"] for column, frame in exposures.items()})
    correlation = pd.DataFrame({column: frame["correlation"] for column, frame in exposures.items()})
    return beta.rename_axis("Date"), correlation.rename_axis("Date")
