import numpy as np
import pytest

from algoquantengine.opt.strategies import hybrid_graph_constrained


@pytest.fixture
def hybrid_inputs():
    corr = np.array([
        [1.0, 0.9, 0.1, 0.0],
        [0.9, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.8],
        [0.0, 0.1, 0.8, 1.0],
    ])
    vol = np.array([0.2, 0.22, 0.25, 0.3])
    cov = corr * np.outer(vol, vol)
    mu = np.array([0.18, 0.16, 0.10, 0.09])
    return cov, mu, corr


@pytest.mark.parametrize("cap", [0.50, 0.55, 0.90])
def test_hybrid_weights_and_cluster_caps(hybrid_inputs, cap):
    weights, labels, caps = hybrid_graph_constrained(
        *hybrid_inputs, n_clusters=2, max_per_cluster=cap
    )

    assert weights.shape == (4,)
    assert labels.shape == (4,)
    assert np.isfinite(weights).all()
    assert (weights >= -1e-10).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert len(caps) == 2
    assert sorted(i for indices, _ in caps for i in indices) == list(range(4))
    for indices, limit in caps:
        assert limit == cap
        assert weights[indices].sum() <= limit + 1e-6


def test_hybrid_determinism(hybrid_inputs):
    first = hybrid_graph_constrained(*hybrid_inputs, n_clusters=2, max_per_cluster=0.6, seed=7)
    second = hybrid_graph_constrained(*hybrid_inputs, n_clusters=2, max_per_cluster=0.6, seed=7)
    np.testing.assert_allclose(first[0], second[0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_hybrid_caps_with_uneven_clusters():
    corr = np.eye(10)
    corr[:9, :9] = 0.9
    np.fill_diagonal(corr, 1.0)
    mu = np.full(10, 0.15)
    mu[-1] = 0.01
    weights, _, caps = hybrid_graph_constrained(
        corr * 0.04, mu, corr, n_clusters=2, max_per_cluster=0.5
    )
    assert sorted(len(indices) for indices, _ in caps) == [1, 9]
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    for indices, cap in caps:
        assert weights[indices].sum() <= cap + 1e-6


def test_hybrid_infeasible_caps(hybrid_inputs, monkeypatch):
    def unexpected_clustering(*args, **kwargs):
        pytest.fail("Infeasible capacity must be rejected before clustering")

    monkeypatch.setattr(
        "algoquantengine.opt.strategies.spectral_clusters_from_corr", unexpected_clustering
    )
    with pytest.raises(ValueError, match="Infeasible cluster caps"):
        hybrid_graph_constrained(*hybrid_inputs, n_clusters=2, max_per_cluster=0.40)


def test_hybrid_rejects_infeasible_generated_clusters(hybrid_inputs, monkeypatch):
    monkeypatch.setattr(
        "algoquantengine.opt.strategies.spectral_clusters_from_corr",
        lambda *args, **kwargs: np.zeros(4, dtype=int),
    )
    with pytest.raises(ValueError, match="generated clusters"):
        hybrid_graph_constrained(*hybrid_inputs, n_clusters=2, max_per_cluster=0.6)


@pytest.mark.parametrize("n_clusters", [0, 2, 10])
def test_hybrid_tiny_asset_fallback(n_clusters):
    weights, labels, caps = hybrid_graph_constrained(
        np.array([[0.04, 0.01], [0.01, 0.09]]),
        np.array([0.12, 0.08]),
        np.array([[1.0, 1.0 / 6], [1.0 / 6, 1.0]]),
        n_clusters=n_clusters,
        max_per_cluster=0.6,
    )
    np.testing.assert_array_equal(labels, [0, 1])
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    for indices, cap in caps:
        assert weights[indices].sum() <= cap + 1e-6


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mu": np.ones((4, 1))}, "mu must be 1D"),
        ({"mu": np.ones(1)}, "at least 2 assets"),
        ({"Sigma": np.eye(3)}, "shape"),
        ({"corr": np.eye(3)}, "shape"),
        ({"mu": np.array([np.nan, 0.1, 0.1, 0.1])}, "Non-finite"),
        ({"Sigma": np.full((4, 4), np.inf)}, "Non-finite"),
        ({"corr": np.full((4, 4), np.nan)}, "Non-finite"),
        ({"corr": np.triu(np.ones((4, 4)))}, "symmetric"),
        ({"corr": np.full((4, 4), 1.1)}, "unit diagonal"),
        ({"max_per_cluster": 0.0}, "max_per_cluster"),
        ({"max_per_cluster": 1.1}, "max_per_cluster"),
        ({"max_per_cluster": np.nan}, "max_per_cluster"),
        ({"n_clusters": 2.5}, "n_clusters"),
        ({"seed": -1}, "seed"),
    ],
)
def test_hybrid_input_validation(hybrid_inputs, overrides, message):
    cov, mu, corr = hybrid_inputs
    kwargs = {"Sigma": cov, "mu": mu, "corr": corr}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        hybrid_graph_constrained(**kwargs)
