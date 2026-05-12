from typing import Iterator, Sequence

import numpy as np
from sklearn.cluster import KMeans

from candidate_pooling.types import ClusteredCandidate, FingerprintedCandidate

N_CLUSTERS = 5


def cluster(
    candidates: Sequence[FingerprintedCandidate],
    n_clusters: int = N_CLUSTERS
) -> Iterator[ClusteredCandidate]:
    F = np.stack(
        [
            np.concatenate(
                [
                    c["loss_deltas"].detach().cpu().numpy(),
                    c["entropy_deltas"].detach().cpu().numpy()
                ]
            )
            for c in candidates
        ]
    )  # [N, 2*n_probe]

    F = (F - F.mean(0)) / (F.std(0) + 1e-8)  # column-standardize
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8  # row-normalize

    labels: np.ndarray = KMeans(  # type: ignore[type-arg]
        n_clusters=n_clusters, random_state=42, n_init="auto"
    ).fit_predict(F)

    for cand, label in zip(candidates, labels):
        yield ClusteredCandidate(**cand, cluster_id=int(label))
