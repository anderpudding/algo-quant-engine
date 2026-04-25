from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_strategy_return_bar(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    plt.figure()
    plt.bar(df["strategy"], df["return"])
    plt.ylabel("Annualized Return")
    plt.title("Strategy Return Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_return_bar.png", dpi=200)
    plt.close()


def plot_strategy_risk_return(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    plt.figure()
    plt.scatter(df["volatility"], df["return"])

    for _, row in df.iterrows():
        plt.annotate(row["strategy"], (row["volatility"], row["return"]))

    plt.xlabel("Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Risk-Return Strategy Map")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_risk_return.png", dpi=200)
    plt.close()


def plot_strategy_drawdown_bar(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    plt.figure()
    plt.bar(df["strategy"], df["max_drawdown"])
    plt.ylabel("Max Drawdown")
    plt.title("Strategy Drawdown Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "strategy_drawdown_bar.png", dpi=200)
    plt.close()


def plot_net_vs_gross(df: pd.DataFrame, out_dir: str) -> None:
    if "net_return" not in df.columns:
        return

    _ensure_dir(out_dir)
    x = range(len(df))

    plt.figure()
    plt.plot(x, df["return"], marker="o", label="Gross Return")
    plt.plot(x, df["net_return"], marker="o", label="Net Return")
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