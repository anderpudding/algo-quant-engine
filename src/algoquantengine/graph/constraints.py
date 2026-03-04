from __future__ import annotations

import numpy as np


def cluster_weight_caps(labels: np.ndarray, max_per_cluster: float) -> list[tuple[list[int], float]]:
    """
    Create group cap constraints from cluster labels.

    Returns list of (indices, cap) meaning sum(w[i] for i in indices) <= cap.
    """
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    if not (0.0 < max_per_cluster <= 1.0):
        raise ValueError("max_per_cluster must be in (0, 1]")

    caps: list[tuple[list[int], float]] = []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0].tolist()
        if len(idxs) == 0:
            continue
        caps.append((idxs, float(max_per_cluster)))
    return caps