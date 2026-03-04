import numpy as np
import networkx as nx

from algoquantengine.graph.build import corr_to_distance, build_graph_from_corr
from algoquantengine.graph.algorithms import compute_mst, spectral_clusters_from_corr


def test_corr_to_distance_shapes_and_diagonal():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    dist = corr_to_distance(corr)
    assert dist.shape == (2, 2)
    assert np.allclose(np.diag(dist), 0.0)


def test_build_graph_and_mst():
    corr = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.1],
        [0.2, 0.1, 1.0],
    ])
    tickers = ["A", "B", "C"]
    G = build_graph_from_corr(tickers, corr, threshold=0.0)
    mst = compute_mst(G, weight="dist")
    assert mst.number_of_nodes() == 3
    assert mst.number_of_edges() == 2


def test_spectral_clusters_length():
    corr = np.array([
        [1.0, 0.9, 0.1, 0.0],
        [0.9, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.8],
        [0.0, 0.1, 0.8, 1.0],
    ])
    labels = spectral_clusters_from_corr(corr, n_clusters=2, seed=42)
    assert labels.shape == (4,)
    assert set(labels.tolist()) <= {0, 1}