from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from transformers import LlamaForCausalLM

from candidate_pooling.basis import basis
from candidate_pooling.cluster import cluster
from candidate_pooling.fingerprint import make_baseline_strand, make_fingerprint_strand
from candidate_pooling.lib.dataset_utils import load_or_compute
from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.mining import LAYER, TOP_K, make_mining_strand
from candidate_pooling.model import load_nnsight_model, make_tokenize_strand
from candidate_pooling.types import BaselineResult, BasisDirection, Candidate, ClusteredCandidate, FingerprintedCandidate, TokenizedExample

MODEL_ID = "meta-llama/Llama-3.2-1B"
CACHE_DIR = (
    Path.home() / "nobackup" / "autodelete" / "candidate-pooling" / "pipeline_cache"
)
_OUTPUT_DIR = Path.home() / "nobackup" / "autodelete" / "candidate-pooling"


def _to_dataset[T : Mapping[str, Any]](records: Iterable[T]) -> TypedDataset[T]:
    def gen():
        yield from records
    return TypedDataset[T](cast(Dataset, Dataset.from_generator(gen)))


def run_pipeline(n_train: int = 1000, n_probe: int = 200) -> None:
    from candidate_pooling.data import load_mmlu_splits
    from candidate_pooling.evaluate import evaluate, visualize_clusters

    model = load_nnsight_model(MODEL_ID, LlamaForCausalLM)
    train_ds, probe_ds = load_mmlu_splits(n_train=n_train, n_probe=n_probe)

    tokenize_fn = make_tokenize_strand(model)
    mine_fn = make_mining_strand(model, LAYER, TOP_K)
    baseline_fn = make_baseline_strand(model)
    fp_fn = make_fingerprint_strand(model, LAYER)

    def get_tok_train() -> TypedDataset[TokenizedExample]:
        return load_or_compute(
            CACHE_DIR / "tok_train",
            lambda: _to_dataset(tokenize_fn(ex) for ex in train_ds),
        )

    def get_tok_probe() -> TypedDataset[TokenizedExample]:
        return load_or_compute(
            CACHE_DIR / "tok_probe",
            lambda: _to_dataset(tokenize_fn(ex) for ex in probe_ds),
        )

    def get_mined() -> TypedDataset[Candidate]:
        return load_or_compute(
            CACHE_DIR / "mined",
            lambda: _to_dataset(cand for ex in get_tok_train() for cand in mine_fn(ex)),
        )

    def get_baselines() -> TypedDataset[BaselineResult]:
        return load_or_compute(
            CACHE_DIR / "baselines",
            lambda: _to_dataset(baseline_fn(ex) for ex in get_tok_probe()),
        )

    def get_fingerprinted() -> TypedDataset[FingerprintedCandidate]:
        return load_or_compute(
            CACHE_DIR / "fp",
            lambda: _to_dataset(fp_fn(get_mined(), get_tok_probe(), get_baselines())),
        )

    def get_clustered() -> TypedDataset[ClusteredCandidate]:
        return load_or_compute(
            CACHE_DIR / "cl",
            lambda: _to_dataset(cluster(get_fingerprinted())),
        )

    def get_basis() -> TypedDataset[BasisDirection]:
        return load_or_compute(
            CACHE_DIR / "out",
            lambda: _to_dataset(basis(get_clustered())),
        )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluate(get_basis()).savefig(_OUTPUT_DIR / "scatter.png", dpi=150)
    visualize_clusters(get_clustered()).savefig(_OUTPUT_DIR / "umap.png", dpi=150)
