from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import LlamaForCausalLM

from candidate_pooling.basis import basis
from candidate_pooling.cluster import cluster
from candidate_pooling.fingerprint import make_baseline_strand, make_fingerprint_strand
from candidate_pooling.mining import LAYER, TOP_K, make_mining_strand
from candidate_pooling.model import load_nnsight_model, make_tokenize_strand
from candidate_pooling.types import BasisDirection, ClusteredCandidate

MODEL_ID = "meta-llama/Llama-3.2-1B"
CACHE_DIR = (
    Path.home() / "nobackup" / "autodelete" / "candidate-pooling" / "pipeline_cache_test"
)
_OUTPUT_DIR = Path.home() / "nobackup" / "autodelete" / "candidate-pooling"


def _to_dataset(records: Iterable[dict]) -> Dataset:
    rows = [
        {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in rec.items()
        }
        for rec in records
    ]
    ds = Dataset.from_list(rows)
    ds.set_format("torch", device="cuda")
    return ds


def load_or_compute(cache_path: Path, compute_fn: Callable[[], Dataset]) -> Dataset:
    if cache_path.exists():
        ds: Dataset = load_from_disk(str(cache_path))  # type: ignore[assignment]
        ds.set_format("torch", device="cuda")
        return ds
    ds = compute_fn()
    ds.save_to_disk(str(cache_path))
    ds.set_format("torch", device="cuda")
    return ds


def run_pipeline(n_train: int = 1000, n_probe: int = 200) -> None:
    from candidate_pooling.data import load_mmlu_splits
    from candidate_pooling.evaluate import evaluate, visualize_clusters

    model = load_nnsight_model(MODEL_ID, LlamaForCausalLM)
    train_ds, probe_ds = load_mmlu_splits(n_train=n_train, n_probe=n_probe)

    tokenize_fn = make_tokenize_strand(model)
    mine_fn = make_mining_strand(model, LAYER, TOP_K)
    baseline_fn = make_baseline_strand(model)
    fp_fn = make_fingerprint_strand(model, LAYER)

    def get_tok_train() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "tok_train",
            lambda: _to_dataset(tokenize_fn(ex) for ex in train_ds),  # type: ignore[arg-type]
        )

    def get_tok_probe() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "tok_probe",
            lambda: _to_dataset(tokenize_fn(ex) for ex in probe_ds),  # type: ignore[arg-type]
        )

    def get_mined() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "mined",
            lambda: _to_dataset(cand for ex in get_tok_train() for cand in mine_fn(ex)),  # type: ignore[arg-type]
        )

    def get_baselines() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "baselines",
            lambda: _to_dataset(baseline_fn(ex) for ex in get_tok_probe()),  # type: ignore[arg-type]
        )

    def get_fingerprinted() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "fp",
            lambda: _to_dataset(
                fp_fn(list(get_mined()), list(get_tok_probe()), list(get_baselines()))  # type: ignore[arg-type]
            ),
        )

    def get_clustered() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "cl",
            lambda: _to_dataset(cluster(list(get_fingerprinted()))),  # type: ignore[arg-type]
        )

    def get_basis() -> Dataset:
        return load_or_compute(
            CACHE_DIR / "out",
            lambda: _to_dataset(basis(list(get_clustered()))),  # type: ignore[arg-type]
        )

    basis_directions: list[BasisDirection] = list(get_basis())  # type: ignore[arg-type]
    clustered_candidates: list[ClusteredCandidate] = list(get_clustered())  # type: ignore[arg-type]

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluate(basis_directions).savefig(_OUTPUT_DIR / "scatter.png", dpi=150)
    visualize_clusters(clustered_candidates).savefig(_OUTPUT_DIR / "umap.png", dpi=150)
