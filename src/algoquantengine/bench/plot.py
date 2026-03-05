from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_scaling(df: pd.DataFrame, path: str):

    plt.figure()

    plt.plot(df["N"], df["covariance"], label="Covariance")
    plt.plot(df["N"], df["graph_mst"], label="Graph MST")
    plt.plot(df["N"], df["optimizer"], label="Optimizer")

    plt.xlabel("Number of Assets (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling of AlgoQuantEngine Components")

    plt.legend()
    plt.tight_layout()

    plt.savefig(path)
    plt.close()