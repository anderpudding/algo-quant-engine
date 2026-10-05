from __future__ import annotations
import numpy as np
import pandas as pd


def load_prices_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    prices = df.select_dtypes(include=["number"]).copy()
    if prices.shape[1] == 0:
        raise ValueError("No numeric asset columns found in CSV.")

    return prices


def load_benchmark_prices_csv(
    path: str, column: str | None = None, date_col: str = "Date"
) -> pd.Series:
    """Select an independent benchmark price series without filling missing data."""
    prices = load_prices_csv(path, date_col=date_col)
    if column is None:
        if prices.shape[1] != 1:
            raise ValueError(
                "Benchmark CSV has multiple numeric columns; provide --benchmark-column."
            )
        column = prices.columns[0]
    if column not in prices.columns:
        raise ValueError(
            f"Benchmark column '{column}' is missing or non-numeric. "
            f"Numeric columns: {list(prices.columns)}"
        )
    benchmark = prices[column]
    if not benchmark.index.is_unique or benchmark.index.hasnans:
        raise ValueError("Benchmark dates must be unique and non-missing.")
    observed = benchmark.dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Benchmark prices must be finite and strictly positive.")
    return benchmark
