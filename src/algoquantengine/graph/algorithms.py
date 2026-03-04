from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering


def compute_mst(G: nx.Graph, weight: str = "dist") -> nx.Graph:
    if G.number_of_nodes() == 0:
        raise ValueError("Empty graph")
    mst = nx.minimum_spanning_tree(G, weight=weight, algorithm="kruskal")
    if mst.number_of_edges() != mst.number_of_nodes() - 1:
        raise ValueError("MST not spanning; check graph connectivity")
    return mst


def pagerank_centrality(G: nx.Graph, weight: str = "rho") -> dict[str, float]:
    # PageRank expects non-negative weights; use abs(corr) by default
    H = G.copy()
    for u, v, d in H.edges(data=True):
        w = d.get(weight, 0.0)
        d["pr_weight"] = abs(float(w))
    return nx.pagerank(H, weight="pr_weight")


def spectral_clusters_from_corr(
    corr: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if corr.shape != (n, n):
        raise ValueError("corr must be square N×N")
    if n_clusters < 2 or n_clusters > n:
        raise ValueError("n_clusters must be in [2, N]")

    # affinity must be non-negative; map correlation to [0,1]
    affinity = (corr + 1.0) / 2.0
    np.fill_diagonal(affinity, 1.0)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
    )
    labels = model.fit_predict(affinity)
    return labels.astype(int)