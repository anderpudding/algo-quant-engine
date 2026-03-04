import pandas as pd
import numpy as np

from algoquantengine.data.preprocess import compute_returns, clean_prices
from algoquantengine.data.features import estimate_mu_cov, corr_matrix


def test_compute_returns_log_basic():
    prices = pd.DataFrame(
        {"A": [100, 101, 102], "B": [50, 49, 51]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    rets = compute_returns(prices, method="log")
    assert rets.shape == (2, 2)
    assert np.isfinite(rets.to_numpy()).all()


def test_clean_prices_drops_missing_columns():
    prices = pd.DataFrame(
        {"A": [1.0, None, 3.0], "B": [None, None, None]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    cleaned = clean_prices(prices, drop_thresh=0.5)
    assert "B" not in cleaned.columns
    assert cleaned.isna().sum().sum() == 0


def test_mu_cov_corr_shapes():
    prices = pd.DataFrame(
        {"A": [100, 101, 99, 102], "B": [50, 52, 51, 53]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    rets = compute_returns(prices, method="simple")
    mu, cov = estimate_mu_cov(rets, annualize=252)
    corr = corr_matrix(rets)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert corr.shape == (2, 2)