from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def plot_strategy_vs_benchmark(
    equity_df: pd.DataFrame, benchmark_prices: pd.Series, path: str
) -> None:
    """Normalize original net equity and benchmark levels on a common date."""
    equity, benchmark = equity_df.align(benchmark_prices, join="inner", axis=0)
    valid = equity.notna().all(axis=1) & benchmark.notna()
    equity = equity.loc[valid].sort_index()
    benchmark = benchmark.loc[equity.index]
    if equity.empty:
        raise ValueError("No common price/equity dates for benchmark performance plot.")
    values = np.column_stack([equity.to_numpy(dtype=float), benchmark.to_numpy(dtype=float)])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Performance plot requires finite, positive price/equity levels.")
    normalized = equity / equity.iloc[0]
    label = f"Benchmark ({benchmark_prices.name})" if benchmark_prices.name is not None else "Benchmark"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    try:
        for column in normalized.columns:
            plt.plot(normalized.index, normalized[column], label=column)
        plt.plot(benchmark.index, benchmark / benchmark.iloc[0], label=label, linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Normalized equity")
        plt.title("Strategy vs Benchmark Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
    finally:
        plt.close()


def _plot_rolling_exposure(
    exposures: pd.DataFrame, path: str, metric: str, reference: float | None = None
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    try:
        for column in exposures.columns:
            plt.plot(exposures.index, exposures[column], label=column)
        if reference is not None:
            plt.axhline(reference, color="gray", linestyle="--")
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        plt.xlabel("Date")
        plt.ylabel(metric)
        plt.title(f"Rolling Benchmark {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
    finally:
        plt.close()


def plot_rolling_beta(beta_df: pd.DataFrame, path: str) -> None:
    _plot_rolling_exposure(beta_df, path, "Beta", reference=1.0)


def plot_rolling_correlation(correlation_df: pd.DataFrame, path: str) -> None:
    _plot_rolling_exposure(correlation_df, path, "Correlation")
