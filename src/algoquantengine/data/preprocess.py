from __future__ import annotations
import numpy as np
import pandas as pd


def clean_prices(prices: pd.DataFrame, fill_method: str = "ffill", drop_thresh: float = 0.05) -> pd.DataFrame:
    if not (0.0 <= drop_thresh < 1.0):
        raise ValueError("drop_thresh must be in [0, 1).")

    missing_frac = prices.isna().mean(axis=0)
    keep = missing_frac <= drop_thresh
    cleaned = prices.loc[:, keep].copy()

    if cleaned.shape[1] == 0:
        raise ValueError("All asset columns dropped due to missingness threshold.")

    if fill_method == "ffill":
        cleaned = cleaned.ffill().bfill()
    elif fill_method == "bfill":
        cleaned = cleaned.bfill().ffill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError("fill_method must be one of: ffill, bfill, none")

    return cleaned


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")

    if (prices <= 0).any().any() and method == "log":
        raise ValueError("Log returns require strictly positive prices.")

    if method == "log":
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()

    rets = rets.dropna(how="any")
    if rets.shape[0] < 2:
        raise ValueError("Not enough data to compute returns.")

    return rets