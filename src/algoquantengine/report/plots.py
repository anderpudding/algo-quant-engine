from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_mst(mst: nx.Graph, path: str | None = None) -> None:
    plt.figure()
    pos = nx.spring_layout(mst, seed=42)
    nx.draw(mst, pos, with_labels=True, node_size=300, font_size=8)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=200)
    else:
        plt.show()
    plt.close()


def plot_cluster_heatmap(corr: np.ndarray, labels: np.ndarray, path: str | None = None) -> None:
    corr = np.asarray(corr, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(labels)
    sorted_corr = corr[np.ix_(order, order)]

    plt.figure()
    plt.imshow(sorted_corr, aspect="auto")
    plt.colorbar()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=200)
    else:
        plt.show()
    plt.close()

def plot_frontier(frontier: list[dict], path: str | None = None) -> None:
    import matplotlib.pyplot as plt
    from pathlib import Path

    vols = [p["vol"] for p in frontier]
    rets = [p["return"] for p in frontier]

    plt.figure()
    plt.plot(vols, rets, marker="o", linestyle="-")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=200)
    else:
        plt.show()
    plt.close()