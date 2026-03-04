import numpy as np

from algoquantengine.opt.solvers import project_to_simplex, projected_gradient_descent


def test_project_to_simplex_properties():
    v = np.array([0.2, -0.1, 2.0])
    w = project_to_simplex(v)
    assert w.shape == (3,)
    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-10


def test_pgd_returns_valid_weights():
    # simple SPD covariance
    Sigma = np.array([[0.10, 0.02], [0.02, 0.08]])
    mu = np.array([0.12, 0.10])
    w = projected_gradient_descent(Sigma, mu, target_return=0.10, max_iter=2000, lr=0.1)
    assert w.shape == (2,)
    assert (w >= -1e-10).all()
    assert abs(w.sum() - 1.0) < 1e-8