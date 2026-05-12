from collections.abc import Callable
from typing import Iterable

import torch
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor

from candidate_pooling.mining import LAYER
from candidate_pooling.types import (
    BaselineResult,
    Candidate,
    FingerprintedCandidates,
    TokenizedExample,
    to_transformer_input,
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
    with torch.no_grad(), model.trace(to_transformer_input(probe)):
        model.model.layers[layer].output[0, -1] += alpha * v  # type: ignore[attr-defined]
        logits = model.output.logits.save()  # type: ignore[attr-defined]
    steered_loss, steered_entropy = _logits_to_loss_entropy(
        logits[0, -1], probe["label_id"]
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


def make_baseline_fn(model: LanguageModel) -> Callable[[TokenizedExample], BaselineResult]:

    def compute_baseline(example: TokenizedExample) -> BaselineResult:
        with torch.no_grad(), model.trace(to_transformer_input(example)):
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        loss, entropy = _logits_to_loss_entropy(
            logits[0, -1], example["label_id"]
        )
        return BaselineResult(
            loss=loss, entropy=entropy, example_id=example["example_id"]
        )

    return compute_baseline


def make_fingerprint_fn(
    model: LanguageModel,
    layer: int = LAYER,
    alpha: float = _ALPHA_DEFAULT,
) -> Callable[[Iterable[Candidate], Iterable[TokenizedExample], Iterable[BaselineResult]], FingerprintedCandidates]:

    def fingerprint(
        candidates: Iterable[Candidate],
        probe_examples: Iterable[TokenizedExample],
        baselines: Iterable[BaselineResult],
    ) -> FingerprintedCandidates:
        vectors: list[Float[Tensor, "d_model"]] = []
        layers: list[int] = []
        example_ids: list[int] = []
        token_positions: list[int] = []
        all_loss_deltas: list[Float[Tensor, "n_probe"]] = []
        all_entropy_deltas: list[Float[Tensor, "n_probe"]] = []

        for candidate in candidates:
            v: Float[Tensor, "d_model"] = torch.as_tensor(candidate["vector"]).cuda()
            deltas = [
                _compute_delta(model, probe, baseline, layer, v, alpha)
                for probe, baseline in zip(probe_examples, baselines)
            ]
            loss_d, entropy_d = zip(*deltas)
            vectors.append(candidate["vector"])
            layers.append(candidate["layer"])
            example_ids.append(candidate["example_id"])
            token_positions.append(candidate["token_pos"])
            all_loss_deltas.append(torch.as_tensor(list(loss_d)))
            all_entropy_deltas.append(torch.as_tensor(list(entropy_d)))
        
        return FingerprintedCandidates(
            vector=torch.stack(vectors),
            layer=layers,
            example_id=example_ids,
            token_pos=token_positions,
            loss_deltas=torch.stack(all_loss_deltas),
            entropy_deltas=torch.stack(all_entropy_deltas),
        )

    return fingerprint
