from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_return_scenarios(
    returns: pd.DataFrame,
    horizon: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Bootstrap historical daily returns with replacement.
    Output: (n_paths, horizon, N)
    """
    rng = np.random.default_rng(seed)
    R = returns.to_numpy()
    t, n = R.shape
    if horizon < 1 or n_paths < 1:
        raise ValueError("horizon and n_paths must be >= 1")
    if t < 5:
        raise ValueError("Need at least 5 return rows")

    idx = rng.integers(0, t, size=(n_paths, horizon))
    out = R[idx, :]  # (paths, horizon, N)
    return out