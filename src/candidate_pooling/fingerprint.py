from typing import Iterator, Sequence

import torch
from braided import strand
from braided.strand import ManyToMany, OneToOne
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor

from candidate_pooling.mining import LAYER
from candidate_pooling.types import (
    BaselineResult,
    Candidate,
    FingerprintedCandidate,
    TokenizedExample,
)

_ALPHA_DEFAULT = 10.0


def _compute_delta(
    model: LanguageModel,
    probe: TokenizedExample,
    baseline: BaselineResult,
    layer: int,
    v: Float[Tensor, "d_model"],
    alpha: float,
) -> tuple[float, float]:
    with torch.no_grad(), model.trace(probe):
        model.model.layers[layer].output[0][:] += alpha * v  # type: ignore[attr-defined]
        logits = model.output.logits.save()  # type: ignore[attr-defined]
    steered_loss, steered_entropy = _logits_to_loss_entropy(
        logits.value[0, -1], probe["label_id"]
    )
    return steered_loss - baseline["loss"], steered_entropy - baseline["entropy"]


def _logits_to_loss_entropy(
    logits: Float[Tensor, "vocab"],
    label_id: int,
) -> tuple[float, float]:
    probs = logits.softmax(dim=-1)
    loss = -probs[label_id].log().item()
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum().item()
    return loss, entropy


def make_baseline_strand(model: LanguageModel) -> OneToOne[BaselineResult]:

    @strand
    def compute_baseline(example: TokenizedExample) -> BaselineResult:
        with torch.no_grad(), model.trace(example):
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        loss, entropy = _logits_to_loss_entropy(
            logits.value[0, -1], example["label_id"]
        )
        return BaselineResult(
            loss=loss, entropy=entropy, example_id=example["example_id"]
        )

    return compute_baseline


def make_fingerprint_strand(
    model: LanguageModel,
    layer: int = LAYER,
    alpha: float = _ALPHA_DEFAULT,
) -> ManyToMany[FingerprintedCandidate]:

    @strand.many_to_many
    def fingerprint(
        candidates: Sequence[Candidate],
        probe_examples: Sequence[TokenizedExample],
        baselines: Sequence[BaselineResult],
    ) -> Iterator[FingerprintedCandidate]:
        for candidate in candidates:
            v: Float[Tensor, "d_model"] = torch.as_tensor(candidate["vector"]).cuda()  # type: ignore[attr-defined]
            deltas = [
                _compute_delta(model, probe, baseline, layer, v, alpha)
                for probe, baseline in zip(probe_examples, baselines)
            ]
            loss_deltas, entropy_deltas = zip(*deltas)
            yield FingerprintedCandidate(
                **candidate,
                loss_deltas=torch.as_tensor(list(loss_deltas)),  # type: ignore[attr-defined]
                entropy_deltas=torch.as_tensor(list(entropy_deltas)),  # type: ignore[attr-defined]
            )

    return fingerprint
