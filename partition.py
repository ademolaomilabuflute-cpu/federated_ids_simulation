"""
Dirichlet-based non-IID client partitioning (Section 4).

Splits the dataset across clients so that each client's local class
distribution is drawn from a Dirichlet(alpha) prior per class — lower
alpha produces more severe, more realistic client heterogeneity (client
drift), which is what the paper's ablations (Section 5.5) vary.
"""
from __future__ import annotations

import numpy as np


def dirichlet_partition(
    labels: np.ndarray, n_clients: int, alpha: float, seed: int = 42
) -> list[np.ndarray]:
    """Return a list of index arrays, one per client."""
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, idx_split in enumerate(np.split(idx_c, split_points)):
            client_indices[client_id].extend(idx_split.tolist())

    return [np.array(idx) for idx in client_indices]
