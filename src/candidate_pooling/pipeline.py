from pathlib import Path

from braided import Cache, execute_pipeline
from braided.integrations.hf_datasets import (
    DatasetInput,
    HFDatasetSerializer,
    hf_map_funcs,
)
from braided.spec import NodeSpec

from candidate_pooling.basis import select_basis
from candidate_pooling.cluster import make_cluster_strand
from candidate_pooling.fingerprint import make_baseline_strand, make_fingerprint_strand
from candidate_pooling.mining import LAYER, TOP_K, make_mining_strand
from candidate_pooling.model import load_nnsight_model, make_tokenize_strand
from candidate_pooling.types import BasisDirection, ClusteredCandidate, MmluExample

MODEL_ID = "meta-llama/Llama-3.2-1B"
CACHE_DIR = (
    Path.home() / "nobackup" / "autodelete" / "candidate-pooling" / "pipeline_cache"
)
_OUTPUT_DIR = Path.home() / "nobackup" / "autodelete" / "candidate-pooling"


def _make_cache(name: str) -> Cache:
    return Cache(str(CACHE_DIR / name), HFDatasetSerializer())


def build_pipeline(
    model,
    layer: int = LAYER,
    top_k: int = TOP_K,
    n_clusters: int = 5,
    alpha: float = 10.0,
) -> NodeSpec:
    tokenize = make_tokenize_strand(model)
    mine = make_mining_strand(model, layer, top_k)
    baseline = make_baseline_strand(model)
    fp = make_fingerprint_strand(model, layer, alpha)
    cl = make_cluster_strand(n_clusters)
    ba = select_basis()

    return {
        "tokenized_train": {"function": tokenize, "args": ["train"]},
        "cached_tokenized_train": {
            "function": _make_cache("tok_train"),
            "args": ["tokenized_train"],
        },
        "mined": {"function": mine, "args": ["cached_tokenized_train"]},
        "cached_mined": {"function": _make_cache("mined"), "args": ["mined"]},
        "tokenized_probe": {"function": tokenize, "args": ["probe"]},
        "cached_tokenized_probe": {
            "function": _make_cache("tok_probe"),
            "args": ["tokenized_probe"],
        },
        "baselines": {"function": baseline, "args": ["cached_tokenized_probe"]},
        "cached_baselines": {
            "function": _make_cache("baselines"),
            "args": ["baselines"],
        },
        "fingerprinted": {
            "function": fp,
            "args": ["cached_mined", "cached_tokenized_probe", "cached_baselines"],
        },
        "cached_fingerprinted": {
            "function": _make_cache("fp"),
            "args": ["fingerprinted"],
        },
        "clustered": {"function": cl, "args": ["cached_fingerprinted"]},
        "cached_clustered": {
            "function": _make_cache("cl"),
            "args": ["clustered"],
        },
        "basis": {"function": ba, "args": ["cached_clustered"]},
        "out": {"function": _make_cache("out"), "args": ["basis"]},
    }


def run_pipeline() -> None:
    from candidate_pooling.data import load_mmlu_splits
    from candidate_pooling.evaluate import evaluate, visualize_clusters  # type: ignore[import-untyped]

    model = load_nnsight_model(MODEL_ID)
    train_ds, probe_ds = load_mmlu_splits()
    nodes = build_pipeline(model)

    inputs: dict[str, DatasetInput[MmluExample]] = {
        "train": DatasetInput[MmluExample](train_ds),
        "probe": DatasetInput[MmluExample](probe_ds),
    }
    basis_directions: list[BasisDirection] = list(
        execute_pipeline(nodes, inputs, **hf_map_funcs())  # type: ignore[arg-type]
    )

    # Load clustered candidates from cache (written by pipeline above)
    clustered_candidates: list[ClusteredCandidate] = list(
        _make_cache("cl")([])
    )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluate(basis_directions).savefig(_OUTPUT_DIR / "scatter.png", dpi=150)
    visualize_clusters(clustered_candidates).savefig(_OUTPUT_DIR / "umap.png", dpi=150)
