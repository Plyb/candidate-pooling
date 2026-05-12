from collections import defaultdict
from collections.abc import Iterable
from typing import Iterator

import numpy as np

from candidate_pooling.types import BasisDirection, ClusteredCandidate


def basis(
    candidates: Iterable[ClusteredCandidate],
) -> Iterator[BasisDirection]:
    by_cluster: dict[int, list[ClusteredCandidate]] = defaultdict(list)
    for c in candidates:
        by_cluster[c["cluster_id"]].append(c)

    for cluster_id, members in by_cluster.items():
        loss_fps = np.stack([np.asarray(m["loss_deltas"]) for m in members])
        centroid = loss_fps.mean(0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        scores = [
            float(
                np.linalg.norm(fp)
                * (fp / (np.linalg.norm(fp) + 1e-8)) @ centroid_norm
            )
            for fp in loss_fps
        ]
        best = members[int(np.argmax(scores))]
        yield BasisDirection(
            vector=best["vector"],
            cluster_id=cluster_id,
            loss_fingerprint=best["loss_deltas"],
            entropy_fingerprint=best["entropy_deltas"],
        )
