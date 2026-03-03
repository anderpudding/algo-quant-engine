from __future__ import annotations
import numpy as np
import pandas as pd


def estimate_mu_cov(returns: pd.DataFrame, annualize: int = 252) -> tuple[np.ndarray, np.ndarray]:
    mu_daily = returns.mean(axis=0).to_numpy()
    cov_daily = returns.cov().to_numpy()

    mu = mu_daily * annualize
    cov = cov_daily * annualize

    return mu, cov


def corr_matrix(returns: pd.DataFrame) -> np.ndarray:
    return returns.corr().to_numpy()