from __future__ import annotations

import numpy as np
import networkx as nx


def corr_to_distance(corr: np.ndarray) -> np.ndarray:
    # d_ij = sqrt(2(1-rho_ij))
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square N×N matrix")
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)
    return dist


def build_graph_from_corr(
    tickers: list[str],
    corr: np.ndarray,
    threshold: float | None = None,
) -> nx.Graph:
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if corr.shape != (n, n):
        raise ValueError("corr must be square N×N")
    if len(tickers) != n:
        raise ValueError("tickers length must match corr dimension")

    dist = corr_to_distance(corr)

    G = nx.Graph()
    for i, t in enumerate(tickers):
        G.add_node(t, idx=i)

    for i in range(n):
        for j in range(i + 1, n):
            rho = float(corr[i, j])
            if threshold is not None and abs(rho) < threshold:
                continue
            G.add_edge(
                tickers[i],
                tickers[j],
                rho=rho,
                dist=float(dist[i, j]),
            )

    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges (threshold too high?)")

    return G