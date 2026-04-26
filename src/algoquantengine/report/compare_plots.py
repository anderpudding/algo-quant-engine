from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _metric_col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns exist: {names}. Available: {list(df.columns)}")


def plot_strategy_return_bar(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)

    ret_col = _metric_col(
        df,
        "return",
        "gross_return",
        "annual_return",
        "expected_return",
        "backtest_return",
    )

    plt.figure()
    plt.bar(df["strategy"], df[ret_col])
    plt.ylabel(ret_col.replace("_", " ").title())
    plt.title("Strategy Return Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_return_bar.png", dpi=200)
    plt.close()


def plot_strategy_risk_return(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)

    ret_col = _metric_col(
        df,
        "return",
        "gross_return",
        "annual_return",
        "expected_return",
        "backtest_return",
    )
    vol_col = _metric_col(
        df,
        "volatility",
        "vol",
        "expected_volatility",
    )

    plt.figure()
    plt.scatter(df[vol_col], df[ret_col])

    for _, row in df.iterrows():
        plt.annotate(row["strategy"], (row[vol_col], row[ret_col]))

    plt.xlabel(vol_col.replace("_", " ").title())
    plt.ylabel(ret_col.replace("_", " ").title())
    plt.title("Risk-Return Strategy Map")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_risk_return.png", dpi=200)
    plt.close()


def plot_strategy_drawdown_bar(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)

    dd_col = _metric_col(df, "max_drawdown", "drawdown")

    plt.figure()
    plt.bar(df["strategy"], df[dd_col])
    plt.ylabel(dd_col.replace("_", " ").title())
    plt.title("Strategy Drawdown Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_drawdown_bar.png", dpi=200)
    plt.close()


def plot_net_vs_gross(df: pd.DataFrame, out_dir: str) -> None:
    if "net_return" not in df.columns and "net_backtest_return" not in df.columns:
        return

    _ensure_dir(out_dir)

    gross_col = _metric_col(
        df,
        "return",
        "gross_return",
        "annual_return",
        "backtest_return",
        "expected_return",
    )
    net_col = _metric_col(df, "net_return", "net_backtest_return")

    x = range(len(df))

    plt.figure()
    plt.plot(x, df[gross_col], marker="o", label="Gross Return")
    plt.plot(x, df[net_col], marker="o", label="Net Return")
    plt.xticks(x, df["strategy"], rotation=30, ha="right")
    plt.ylabel("Return")
    plt.title("Gross vs Net Return After Transaction Costs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_net_vs_gross.png", dpi=200)
    plt.close()


def export_strategy_dashboard(df: pd.DataFrame, out_dir: str) -> None:
    plot_strategy_return_bar(df, out_dir)
    plot_strategy_risk_return(df, out_dir)
    plot_strategy_drawdown_bar(df, out_dir)
    plot_net_vs_gross(df, out_dir)