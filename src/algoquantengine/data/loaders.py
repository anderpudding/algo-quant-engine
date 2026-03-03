from __future__ import annotations
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