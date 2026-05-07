from typing import Iterator, Sequence

import numpy as np
from braided import strand
from braided.strand import ManyToMany
from sklearn.cluster import KMeans

from candidate_pooling.types import ClusteredCandidate, FingerprintedCandidate

N_CLUSTERS = 5


def make_cluster_strand(n_clusters: int = N_CLUSTERS) -> ManyToMany[ClusteredCandidate]:

    @strand.many_to_many
    def cluster(
        candidates: Sequence[FingerprintedCandidate],
    ) -> Iterator[ClusteredCandidate]:
        F = np.stack(
            [
                np.concatenate(
                    [np.asarray(c["loss_deltas"]), np.asarray(c["entropy_deltas"])]
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

    return cluster
