from typing import TypedDict

from jaxtyping import Float, Int
from torch import Tensor


class MmluExample(TypedDict):
    question: str
    choices: list[str]  # always length 4
    answer: int  # 0-3
    subject: str
    example_id: int


class TokenizedExample(TypedDict):
    input_ids: Int[Tensor, "seq"]
    attention_mask: Int[Tensor, "seq"]
    label_id: int  # vocab index of the correct answer letter (A/B/C/D)
    example_id: int


class Candidate(TypedDict):
    vector: Float[Tensor, "d_model"]  # unit-norm negated gradient at token_pos
    layer: int
    example_id: int
    token_pos: int


class BaselineResult(TypedDict):
    loss: float
    entropy: float
    example_id: int


class FingerprintedCandidate(Candidate):
    loss_deltas: Float[Tensor, "n_probe"]
    entropy_deltas: Float[Tensor, "n_probe"]


class ClusteredCandidate(FingerprintedCandidate):
    cluster_id: int


class BasisDirection(TypedDict):
    vector: Float[Tensor, "d_model"]
    cluster_id: int
    loss_fingerprint: Float[Tensor, "n_probe"]
    entropy_fingerprint: Float[Tensor, "n_probe"]
