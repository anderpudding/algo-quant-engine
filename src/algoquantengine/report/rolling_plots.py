from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_rolling_equity_curves(equity_df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for col in equity_df.columns:
        plt.plot(equity_df.index, equity_df[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Rolling Strategy Equity Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_rolling_drawdowns(equity_df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for col in equity_df.columns:
        peak = equity_df[col].cummax()
        drawdown = equity_df[col] / peak - 1.0
        plt.plot(equity_df.index, drawdown, label=col)

    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Rolling Strategy Drawdowns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()