from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from transformers import LlamaForCausalLM
from runlog import start_run

from candidate_pooling.basis import basis
from candidate_pooling.cluster import cluster
from candidate_pooling.fingerprint import (
    annotate_with_std_dev,
    make_covariance_fn,
    make_fingerprint_fn,
    make_mean_activation_fn,
)
from candidate_pooling.lib.dataset_utils import load_or_compute, set_format
from candidate_pooling.lib.tensor_cache import load_or_compute_tensor
from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.mining import LAYER, TOP_K, make_mining_fn
from candidate_pooling.model import load_nnsight_model
from candidate_pooling.types import (
    AnnotatedCandidate,
    BasisDirection,
    Candidate,
    ClusteredCandidates,
    FingerprintedCandidates,
    TokenizedExample,
)

MODEL_ID = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = start_run(Path().home() / "nobackup" / "autodelete" / "candidate-pooling", cfg= { "model": MODEL_ID })
CACHE_DIR = OUTPUT_DIR / "pipeline_cache"


def _to_dataset[T : Mapping[str, Any]](records: Iterable[T]) -> TypedDataset[T]:
    return TypedDataset[T](cast(Dataset, Dataset.from_list(list(records)))) # type: ignore


def run_pipeline(n_train: int = 1000, n_probe: int = 200) -> None:
    from candidate_pooling.data import load_arc_easy, tokenize_dataset
    from candidate_pooling.evaluate import evaluate, visualize_clusters

    model = load_nnsight_model(MODEL_ID, LlamaForCausalLM)

    mine_fn = make_mining_fn(model, LAYER, TOP_K)
    fp_fn = make_fingerprint_fn(model, LAYER)
    mean_act_fn = make_mean_activation_fn(model, LAYER)
    cov_fn = make_covariance_fn(model, LAYER)

    train_ds, probe_ds = None, None
    def get_tok_splits() -> tuple[TypedDataset[TokenizedExample], TypedDataset[TokenizedExample]]:
        nonlocal train_ds, probe_ds
        if train_ds is None or probe_ds is None:
            train_ds, probe_ds = tokenize_dataset(model, load_arc_easy(), n_train, n_probe)
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

    def get_probe_mean_activation():
        return load_or_compute_tensor(
            CACHE_DIR / "probe_mean_activation.pt",
            lambda: mean_act_fn(get_tok_probe()),
        )

    def get_probe_covariance():
        return load_or_compute_tensor(
            CACHE_DIR / "probe_covariance.pt",
            lambda: cov_fn(get_tok_probe()),
        )

    def get_annotated() -> TypedDataset[AnnotatedCandidate]:
        return load_or_compute(
            CACHE_DIR / "annotated",
            lambda: set_format(
                _to_dataset(annotate_with_std_dev(get_mined(), get_probe_covariance())),
                AnnotatedCandidate,
            ),
        )

    def get_fingerprinted() -> FingerprintedCandidates:
        return load_or_compute_tensor(
            CACHE_DIR / "fp.pt",
            lambda: fp_fn(get_annotated(), get_tok_probe()),
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    get_probe_mean_activation()
    evaluate(get_basis()).savefig(OUTPUT_DIR / "scatter.png", dpi=150)
    visualize_clusters(get_clustered()).savefig(OUTPUT_DIR / "umap.png", dpi=150)
