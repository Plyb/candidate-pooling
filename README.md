# candidate-pooling

Label-free **candidate pooling** — discover behaviorally distinct steering directions in an LLM's activation space without supervised labels.

## Pipeline

Adapted from Tommaso Giovannini's unpublished writeup on Candidate Pooling.

1. **Mine** — top-k token positions by `‖∇_h L‖` per example, per layer → candidate steering vectors.
2. **Fingerprint** — apply each candidate on a probe set, record loss-Δ and entropy-Δ.
3. **Cluster** — column-standardize + row-normalize, then KMeans on fingerprints.
4. **Select basis** — one candidate per cluster, ranked by strength × alignment.
5. **Evaluate** — geometric vs. behavioral diversity scatter.

## Prerequisites

- BYU ORC account. [src/launch.py:30-32](src/launch.py#L30-L32) is configured for one H100 on the `cs` QoS, which requires CS-department access to the `cs`/`cs2` partitions. If you don't have that, edit the `SlurmConfig` to a partition/qos you can use.
- `uv` installed.

## Install

```bash
uv sync                                              # creates .venv from uv.lock
mamba create --yes -f environment.yml -p ./.env      # messy, but required because mirror's slurm launcher activates this on the compute node
```

To update the mamba env later: `mamba env update --file environment.yml --prune -p ./.env`.

> `.env/` is a separate mamba environment from the `.venv/` that `uv sync` creates. Mirror's slurm launcher activates `./.env` on the compute node, so it has to be a working environment — not just an empty directory.

## Run

The launcher does login-node prefetch, then submits a SLURM job that runs the pipeline on a compute node:

```bash
# On a login node:
uv run python src/launch.py --prefetch
```

Outputs cache land in `~/nobackup/autodelete/candidate-pooling/`. SLURM stdout/stderr go to [slurm_logs/](slurm_logs/).

Knobs:

- `MODEL_ID`, prefetch dataset → [src/launch.py](src/launch.py)
- `n_train`, `n_probe` → `run_pipeline(...)` call in [src/launch.py](src/launch.py)
- Layer / top-k → [src/candidate_pooling/mining.py](src/candidate_pooling/mining.py)
- SLURM resources (time, GPU, QoS, memory) → `SlurmConfig` in [src/launch.py](src/launch.py)

## Analysis

After the pipeline finishes, the fingerprint / cluster / basis artifacts are cached and can be poked at interactively:

- [src/candidate_pooling/analysis/analysis.py](src/candidate_pooling/analysis/analysis.py) — load cached pipeline outputs.
- [src/candidate_pooling/analysis/dashboard.py](src/candidate_pooling/analysis/dashboard.py) — Reacton widgets (attention-delta, steering-curve, etc.).
- [src/candidate_pooling/analysis/analysis-testing.ipynb](src/candidate_pooling/analysis/analysis-testing.ipynb) — example notebook.

## Layout

```
src/launch.py                          # SLURM entry point
src/candidate_pooling/
  pipeline.py                          # braided wiring of all stages
  mining.py fingerprint.py cluster.py basis.py evaluate.py
  model.py data.py types.py
  lib/                                 # dataset / tensor caching helpers
  analysis/                            # post-run interactive analysis
```

