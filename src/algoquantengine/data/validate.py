from __future__ import annotations

import pandas as pd


def validate_price_frame(prices: pd.DataFrame, min_rows: int = 30, min_assets: int = 5) -> None:
    if prices.empty:
        raise ValueError("Price data is empty.")

    if prices.shape[0] < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {prices.shape[0]}.")

    if prices.shape[1] < min_assets:
        raise ValueError(f"Need at least {min_assets} assets, got {prices.shape[1]}.")

    if prices.isna().mean().max() > 0.25:
        raise ValueError("At least one asset has more than 25% missing values.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")