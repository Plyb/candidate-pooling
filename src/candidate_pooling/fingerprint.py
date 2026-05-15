from collections.abc import Callable
from typing import Collection, Iterable, Iterator

import torch
import torch.nn.functional as F
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor
from tqdm import tqdm

from candidate_pooling.mining import LAYER
from candidate_pooling.types import (
    AnnotatedCandidate,
    BaselineResult,
    Candidate,
    FingerprintedCandidates,
    TokenizedExample,
    to_transformer_input,
)

ALPHA_DEFAULT = 1.0


def _final_token_loss_entropy(
    logits: Float[Tensor, "seq vocab"],
    label_id: int,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    final = logits[-1]
    loss = F.cross_entropy(final.unsqueeze(0), torch.tensor([label_id], device=final.device))
    entropy = torch.distributions.Categorical(logits=final).entropy()
    return loss, entropy


def compute_jacobians(
    model: LanguageModel,
    probe: TokenizedExample,
    layer: int,
) -> tuple[Float[Tensor, "seq d_model"], Float[Tensor, "seq d_model"]]:
    label_id = probe["label_id"]

    def _grad(which: str) -> Float[Tensor, "seq d_model"]:
        with model.trace(to_transformer_input(probe)):
            hidden = model.model.layers[layer].output  # type: ignore[attr-defined]
            hidden.requires_grad_(True)
            logits = model.lm_head.output[0]  # type: ignore[attr-defined]
            loss, entropy = _final_token_loss_entropy(logits, label_id)
            scalar = loss if which == "loss" else entropy
            with scalar.backward():  # type: ignore[union-attr]
                grad = hidden.grad.save()
        return grad[0]

    return _grad("loss"), _grad("entropy")


def compute_delta(
    model: LanguageModel,
    probe: TokenizedExample,
    baseline: BaselineResult,
    layer: int,
    v: Float[Tensor, "d_model"],
    std_dev: float,
    alpha: float = ALPHA_DEFAULT,
) -> tuple[Float[Tensor, "seq"], Float[Tensor, "seq"]]:
    """Validation reference: for each token position t, steer the layer-`layer` activation
    at position t by alpha*std_dev*v and return the resulting change in final-token loss
    and entropy. Loops over positions, so cost is O(seq) forward passes per candidate.
    The Jacobian path in `make_fingerprint_fn` is a linearization of this."""
    seq = int(probe["input_ids"].shape[0])
    baseline_loss = baseline["loss"].cuda()
    baseline_entropy = baseline["entropy"].cuda()
    perturbation = (alpha * std_dev) * v
    delta_loss = torch.empty(seq, device="cuda")
    delta_entropy = torch.empty(seq, device="cuda")
    for t in range(seq):
        with torch.no_grad(), model.trace(to_transformer_input(probe)):
            model.model.layers[layer].output[0, t] += perturbation  # type: ignore[attr-defined]
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        loss_t, entropy_t = _final_token_loss_entropy(logits[0], probe["label_id"])
        delta_loss[t] = loss_t - baseline_loss
        delta_entropy[t] = entropy_t - baseline_entropy
    return delta_loss, delta_entropy


def make_covariance_fn(
    model: LanguageModel,
    layer: int = LAYER,
) -> Callable[[Iterable[TokenizedExample]], Float[Tensor, "d_model d_model"]]:

    def compute_covariance(
        probe_examples: Iterable[TokenizedExample],
    ) -> Float[Tensor, "d_model d_model"]:
        sum_h: Float[Tensor, "d_model"] | None = None
        sum_outer: Float[Tensor, "d_model d_model"] | None = None
        count = 0
        for example in tqdm(probe_examples):
            with torch.no_grad(), model.trace(to_transformer_input(example)):
                hidden = model.model.layers[layer].output[0].save()  # type: ignore[attr-defined]
            mask = example["attention_mask"].to(hidden.device).bool()
            valid = hidden[mask].float()  # fp32 to keep outer-product accumulation numerically stable
            summed = valid.sum(dim=0)
            outer = valid.T @ valid
            sum_h = summed if sum_h is None else sum_h + summed
            sum_outer = outer if sum_outer is None else sum_outer + outer
            count += int(mask.sum().item())
        if sum_h is None or sum_outer is None or count == 0:
            raise ValueError("covariance requires at least one probe token")
        mean = sum_h / count
        return sum_outer / count - torch.outer(mean, mean)

    return compute_covariance


def annotate_with_std_dev(
    candidates: Iterable[Candidate],
    covariance: Float[Tensor, "d_model d_model"],
) -> Iterator[AnnotatedCandidate]:
    cov = covariance.to("cuda").float()
    for candidate in candidates:
        v = torch.as_tensor(candidate["vector"]).cuda().float()
        std_dev = float(torch.sqrt((v @ cov @ v).clamp_min(0.0)).item())
        yield AnnotatedCandidate(**candidate, std_dev=std_dev)


def make_mean_activation_fn(
    model: LanguageModel,
    layer: int = LAYER,
) -> Callable[[Iterable[TokenizedExample]], Float[Tensor, "d_model"]]:

    def compute_mean_activation(
        probe_examples: Iterable[TokenizedExample],
    ) -> Float[Tensor, "d_model"]:
        total: Float[Tensor, "d_model"] | None = None
        count = 0
        for example in tqdm(probe_examples):
            with torch.no_grad(), model.trace(to_transformer_input(example)):
                hidden = model.model.layers[layer].output[0].save()  # type: ignore[attr-defined]
            mask = example["attention_mask"].to(hidden.device).bool()
            valid = hidden[mask]
            summed = valid.sum(dim=0)
            total = summed if total is None else total + summed
            count += int(mask.sum().item())
        if total is None or count == 0:
            raise ValueError("mean activation requires at least one probe token")
        return total / count

    return compute_mean_activation


def make_baseline_fn(model: LanguageModel) -> Callable[[TokenizedExample], BaselineResult]:

    def compute_baseline(example: TokenizedExample) -> BaselineResult:
        with torch.no_grad(), model.trace(to_transformer_input(example)):
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        loss, entropy = _final_token_loss_entropy(logits[0], example["label_id"])
        return BaselineResult(
            loss=loss, entropy=entropy, example_id=example["example_id"]
        )

    return compute_baseline


def make_fingerprint_fn(
    model: LanguageModel,
    layer: int = LAYER,
) -> Callable[[Collection[AnnotatedCandidate], Iterable[TokenizedExample]], FingerprintedCandidates]:

    def fingerprint(
        candidates: Collection[AnnotatedCandidate],
        probe_examples: Iterable[TokenizedExample],
    ) -> FingerprintedCandidates:
        candidate_list = list(candidates)
        vectors: Float[Tensor, "n_candidates d_model"] = torch.stack(
            [torch.as_tensor(c["vector"]).cuda().float() for c in candidate_list]
        )

        loss_blocks: list[Float[Tensor, "n_candidates seq"]] = []
        entropy_blocks: list[Float[Tensor, "n_candidates seq"]] = []
        for probe in tqdm(probe_examples):
            jac_loss, jac_entropy = compute_jacobians(model, probe, layer)
            loss_blocks.append(vectors @ jac_loss.float().T)
            entropy_blocks.append(vectors @ jac_entropy.float().T)

        return FingerprintedCandidates(
            vector=torch.stack([torch.as_tensor(c["vector"]) for c in candidate_list]),
            layer=[c["layer"] for c in candidate_list],
            example_id=[c["example_id"] for c in candidate_list],
            token_pos=[c["token_pos"] for c in candidate_list],
            loss_deltas=torch.cat(loss_blocks, dim=1),
            entropy_deltas=torch.cat(entropy_blocks, dim=1),
            std_dev=[c["std_dev"] for c in candidate_list],
        )

    return fingerprint
