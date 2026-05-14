from typing import Iterator

import numpy as np

from candidate_pooling.types import BasisDirection, ClusteredCandidates


def basis(
    candidates: ClusteredCandidates,
) -> Iterator[BasisDirection]:
    cluster_ids = candidates["cluster_id"]
    for cluster_id in sorted(set(cluster_ids)):
        indices = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
        loss_fps = candidates["loss_deltas"][indices].detach().cpu().numpy()
        centroid = loss_fps.mean(0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        scores = [
            float(np.linalg.norm(fp) * (fp / (np.linalg.norm(fp) + 1e-8)) @ centroid_norm)
            for fp in loss_fps
        ]
        best = indices[int(np.argmax(scores))]
        yield BasisDirection(
            vector=candidates["vector"][best],
            cluster_id=cluster_id,
            loss_fingerprint=candidates["loss_deltas"][best],
            entropy_fingerprint=candidates["entropy_deltas"][best],
            example_id=candidates["example_id"][best]
        )
