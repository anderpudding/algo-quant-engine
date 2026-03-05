from __future__ import annotations

import time
import numpy as np
import pandas as pd

from algoquantengine.graph.algorithms import mst_from_corr
from algoquantengine.opt.meanvar import estimate_mu_cov, efficient_frontier


def random_returns(T: int, N: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.02, size=(T, N))
    cols = [f"A{i}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)


def benchmark_scaling(
    sizes: list[int],
    T: int = 500,
) -> pd.DataFrame:

    rows = []

    for N in sizes:
        rets = random_returns(T, N)

        # covariance
        t0 = time.perf_counter()
        mu, cov = estimate_mu_cov(rets)
        t_cov = time.perf_counter() - t0

        # graph
        corr = rets.corr().to_numpy()
        t0 = time.perf_counter()
        mst_from_corr(corr)
        t_graph = time.perf_counter() - t0

        # optimizer
        t0 = time.perf_counter()
        efficient_frontier(cov, mu, n_points=25)
        t_opt = time.perf_counter() - t0

        rows.append(
            {
                "N": N,
                "covariance": t_cov,
                "graph_mst": t_graph,
                "optimizer": t_opt,
            }
        )

    return pd.DataFrame(rows)