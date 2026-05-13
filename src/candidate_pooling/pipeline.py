from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from transformers import LlamaForCausalLM

from candidate_pooling.basis import basis
from candidate_pooling.cluster import cluster
from candidate_pooling.fingerprint import make_baseline_fn, make_fingerprint_fn
from candidate_pooling.lib.dataset_utils import load_or_compute, set_format
from candidate_pooling.lib.tensor_cache import load_or_compute_tensor
from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.mining import LAYER, TOP_K, make_mining_fn
from candidate_pooling.model import load_nnsight_model
from candidate_pooling.types import BaselineResult, BasisDirection, Candidate, ClusteredCandidates, FingerprintedCandidates, TokenizedExample

MODEL_ID = "meta-llama/Llama-3.2-1B"
CACHE_DIR = (
    Path.home() / "nobackup" / "autodelete" / "candidate-pooling" / "pipeline_cache_test"
)
_OUTPUT_DIR = Path.home() / "nobackup" / "autodelete" / "candidate-pooling"


def _to_dataset[T : Mapping[str, Any]](records: Iterable[T]) -> TypedDataset[T]:
    return TypedDataset[T](cast(Dataset, Dataset.from_list(list(records)))) # type: ignore


def run_pipeline(n_train: int = 10, n_probe: int = 5) -> None:
    from candidate_pooling.data import load_mmlu, tokenize_dataset
    from candidate_pooling.evaluate import evaluate, visualize_clusters

    model = load_nnsight_model(MODEL_ID, LlamaForCausalLM)

    mine_fn = make_mining_fn(model, LAYER, TOP_K)
    baseline_fn = make_baseline_fn(model)
    fp_fn = make_fingerprint_fn(model, LAYER)

    train_ds, probe_ds = None, None
    def get_tok_splits() -> tuple[TypedDataset[TokenizedExample], TypedDataset[TokenizedExample]]:
        nonlocal train_ds, probe_ds
        if train_ds is None or probe_ds is None:
            train_ds, probe_ds = tokenize_dataset(model, load_mmlu(), n_train, n_probe)
        return train_ds, probe_ds

    def get_tok_train() -> TypedDataset[TokenizedExample]:
        return load_or_compute(
            CACHE_DIR / "tok_train",
            lambda: get_tok_splits()[0],
        )

    def get_tok_probe() -> TypedDataset[TokenizedExample]:
        return load_or_compute(
            CACHE_DIR / "tok_probe",
            lambda: get_tok_splits()[1],
        )

    def get_mined() -> TypedDataset[Candidate]:
        return load_or_compute(
            CACHE_DIR / "mined",
            lambda: set_format(
                _to_dataset(cand for ex in get_tok_train() for cand in mine_fn(ex)),
                Candidate
            ),
        )

    def get_baselines() -> TypedDataset[BaselineResult]:
        return load_or_compute(
            CACHE_DIR / "baselines",
            lambda: set_format(
                _to_dataset(baseline_fn(ex) for ex in get_tok_probe()),
                BaselineResult
            )
        )

    def get_fingerprinted() -> FingerprintedCandidates:
        return load_or_compute_tensor(
            CACHE_DIR / "fp.pt",
            lambda: fp_fn(get_mined(), get_tok_probe(), get_baselines()),
        )

    def get_clustered() -> ClusteredCandidates:
        return load_or_compute_tensor(
            CACHE_DIR / "cl.pt",
            lambda: cluster(get_fingerprinted()),
        )

    def get_basis() -> TypedDataset[BasisDirection]:
        return load_or_compute(
            CACHE_DIR / "out",
            lambda: set_format(
                _to_dataset(basis(get_clustered())),
                BasisDirection
            )
        )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluate(get_basis()).savefig(_OUTPUT_DIR / "scatter.png", dpi=150)
    visualize_clusters(get_clustered()).savefig(_OUTPUT_DIR / "umap.png", dpi=150)
