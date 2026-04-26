from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw CSV path")
    parser.add_argument("--output", required=True, help="Clean output CSV path")
    parser.add_argument("--date-col", default="Date")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.date_col not in df.columns:
        raise ValueError(f"Missing date column: {args.date_col}")

    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric price columns found.")

    out = df[[args.date_col] + numeric_cols].copy()
    out = out.drop_duplicates(subset=[args.date_col])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved cleaned price CSV: {args.output}")
    print(f"Rows: {out.shape[0]}, Assets: {out.shape[1] - 1}")


if __name__ == "__main__":
    main()