# Candidate Pooling

Implementation of **label-free candidate pooling** — a bottom-up technique for discovering behavioral subtypes in LLM activation space without requiring human-annotated labels.

## What It Does

Given a language model and a dataset, candidate pooling finds a small set of activation-space directions (steering vectors) that represent *behaviorally distinct* reasoning modes. Unlike supervised approaches (e.g., ReFT), no labels are required: the structure is discovered from the model's own gradients.

The pipeline has five stages:

1. **Mine candidates** — For each training example and each target layer, compute the gradient of the loss w.r.t. the hidden state (`-∇_h L`). Select the top-k token positions by gradient norm. These negated, normalized gradients are candidate steering vectors.

2. **Fingerprint** — Apply each candidate as a steering vector on a held-out probe set. Record per-example loss delta and entropy delta. Concatenate into a fingerprint vector `f ∈ R^{2N}` per candidate.

3. **Cluster** — Column-standardize and row-normalize the fingerprint matrix, then apply KMeans. Candidates that affect the same examples in the same way cluster together; candidates that affect different examples form separate clusters.

4. **Select basis** — From each cluster, pick the candidate with the highest `strength × alignment` score (using loss fingerprints only) as the basis direction.

5. **Evaluate** — Measure geometric diversity (direction cosine similarity), behavioral diversity (fingerprint cosine similarity), and transfer accuracy (e.g., GSM8K).

## Implementation Approach

- **Pipeline**: [`braided`](https://pypi.org/project/braided/) — each stage is a `@strand` transformation in a typed pipeline.
- **Models & Datasets**: `mirror` (from `~/MIRROR-Pipeline`) — used for loading models, tokenizers, and datasets.
