import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.figure import Figure

from candidate_pooling.types import BasisDirection, ClusteredCandidate


def evaluate(basis_directions: list[BasisDirection]) -> Figure:
    vectors = np.stack([np.asarray(b["vector"]) for b in basis_directions])
    fps = np.stack([np.asarray(b["loss_fingerprint"]) for b in basis_directions])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    fps /= np.linalg.norm(fps, axis=1, keepdims=True) + 1e-8

    geometric_sim = vectors @ vectors.T
    behavioral_sim = fps @ fps.T
    idx = np.triu_indices(len(basis_directions), k=1)

    fig, ax = plt.subplots()
    ax.scatter(geometric_sim[idx], behavioral_sim[idx], alpha=0.7, s=40)
    for i, j, x, y in zip(idx[0], idx[1], geometric_sim[idx], behavioral_sim[idx]):
        ax.annotate(f"({i},{j})", (x, y), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Direction cosine similarity (geometric)")
    ax.set_ylabel("Fingerprint cosine similarity (behavioral)")
    ax.set_title(f"Basis direction diversity  (n={len(basis_directions)} clusters)")
    return fig


def visualize_clusters(clustered_candidates: list[ClusteredCandidate]) -> Figure:
    all_vectors = np.stack([np.asarray(c["vector"]) for c in clustered_candidates])
    cluster_ids = np.array([c["cluster_id"] for c in clustered_candidates])

    embedding: np.ndarray = umap.UMAP(n_components=2, random_state=42).fit_transform(all_vectors)  # type: ignore[attr-defined]

    fig, ax = plt.subplots()
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_ids, cmap="tab10", alpha=0.4, s=5)
    plt.colorbar(sc, ax=ax, label="cluster_id")
    ax.set_title(f"UMAP of {len(clustered_candidates)} candidates")
    return fig
