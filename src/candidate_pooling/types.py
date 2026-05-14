from typing import TypedDict

from jaxtyping import Float, Int
from torch import Tensor


class McqaExample(TypedDict):
    question: str
    choices: list[str]
    answer: int # [0,n) where n is len(choices)

class MmluExample(McqaExample): # always 4 choices
    subject: str

class TransformerInput(TypedDict):
    input_ids: Int[Tensor, 'batch seq']
    attention_mask: Int[Tensor, 'batch seq']

class TokenizedExample(TypedDict):
    input_ids: Int[Tensor, "seq"]
    attention_mask: Int[Tensor, "seq"]
    label_id: int  # vocab index of the correct answer letter (A/B/C/D)
    example_id: int

def to_transformer_input(ex: TokenizedExample) -> TransformerInput:
    return { 'input_ids': ex['input_ids'].unsqueeze(0), 'attention_mask': ex["attention_mask"].unsqueeze(0) }


class Candidate(TypedDict):
    vector: Float[Tensor, "d_model"]  # unit-norm negated gradient at token_pos
    layer: int
    example_id: int
    token_pos: int


class AnnotatedCandidate(Candidate):
    std_dev: float  # sqrt(v^T C v) where C is probe-set activation covariance at layer


class BaselineResult(TypedDict):
    loss: Float[Tensor, "seq"]
    entropy: Float[Tensor, "seq"]
    example_id: int


class FingerprintedCandidates(TypedDict):
    vector: Float[Tensor, "n_candidates d_model"]
    layer: list[int]
    example_id: list[int]
    token_pos: list[int]
    loss_deltas: Float[Tensor, "n_candidates n_tokens_in_probe"]
    entropy_deltas: Float[Tensor, "n_candidates n_tokens_in_probe"]
    std_dev: list[float]


class ClusteredCandidates(FingerprintedCandidates):
    cluster_id: list[int]


class BasisDirection(TypedDict):
    vector: Float[Tensor, "d_model"]
    cluster_id: int
    loss_fingerprint: Float[Tensor, "n_tokens_in_probe"]
    entropy_fingerprint: Float[Tensor, "n_tokens_in_probe"]
    example_id: int
    std_dev: float
