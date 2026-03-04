import numpy as np

from algoquantengine.graph.constraints import cluster_weight_caps
from algoquantengine.opt.solvers import projected_gradient_descent


def test_cluster_weight_caps_structure():
    labels = np.array([0, 0, 1, 1, 1])
    caps = cluster_weight_caps(labels, max_per_cluster=0.6)
    assert len(caps) == 2
    groups = [set(idxs) for idxs, _ in caps]
    assert set([0, 1]) in groups
    assert set([2, 3, 4]) in groups


def test_group_caps_respected_reasonably():
    # Two clusters: {0,1} and {2,3}
    Sigma = np.array([
        [0.10, 0.02, 0.00, 0.00],
        [0.02, 0.11, 0.00, 0.00],
        [0.00, 0.00, 0.08, 0.01],
        [0.00, 0.00, 0.01, 0.09],
    ])
    mu = np.array([0.12, 0.11, 0.10, 0.09])

    caps = [([0, 1], 0.55), ([2, 3], 0.55)]
    w = projected_gradient_descent(Sigma, mu, target_return=0.10, extra_caps=caps, max_iter=4000, lr=0.1)

    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= -1e-10).all()
    assert w[[0, 1]].sum() <= 0.55 + 1e-6
    assert w[[2, 3]].sum() <= 0.55 + 1e-6