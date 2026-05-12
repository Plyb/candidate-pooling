import numpy as np
from sklearn.cluster import KMeans

from candidate_pooling.types import ClusteredCandidates, FingerprintedCandidates

N_CLUSTERS = 5


def cluster(
    candidates: FingerprintedCandidates,
    n_clusters: int = N_CLUSTERS,
) -> ClusteredCandidates:
    F = np.concatenate(
        [
            candidates["loss_deltas"].detach().cpu().numpy(),
            candidates["entropy_deltas"].detach().cpu().numpy(),
        ],
        axis=1,
    )  # [N, 2*n_probe]

    F = (F - F.mean(0)) / (F.std(0) + 1e-8)  # column-standardize
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8  # row-normalize

    labels: np.ndarray = KMeans(  # type: ignore[type-arg]
        n_clusters=n_clusters, random_state=42, n_init="auto"
    ).fit_predict(F)

    return ClusteredCandidates(**candidates, cluster_id=labels.tolist())
