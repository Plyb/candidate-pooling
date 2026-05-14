from collections.abc import Callable
from typing import Iterator

import torch
import torch.nn.functional as F
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor

from candidate_pooling.types import Candidate, TokenizedExample, to_transformer_input

LAYER = 12
TOP_K = 5


def compute_neg_gradients(
    model: LanguageModel,
    example: TokenizedExample,
    layer: int = LAYER,
) -> Float[Tensor, "seq d_model"]:
    label_id_tensor = torch.as_tensor([example["label_id"]], device="cuda")
    with model.trace(to_transformer_input(example)):
        hidden = model.model.layers[layer].output  # type: ignore[attr-defined]
        hidden.requires_grad_(True)
        logits = model.lm_head.output  # type: ignore[attr-defined]
        loss = F.cross_entropy(
            logits[0, -1].unsqueeze(0),
            label_id_tensor,
        )
        with loss.backward():  # type: ignore
            hidden_grad = hidden.grad.save()  # [seq, d_model]
    return -hidden_grad[0]


def make_mining_fn(
    model: LanguageModel,
    layer: int = LAYER,
    top_k: int = TOP_K,
) -> Callable[[TokenizedExample], Iterator[Candidate]]:

    def mine(example: TokenizedExample) -> Iterator[Candidate]:
        neg_grad: Float[Tensor, "seq d_model"] = compute_neg_gradients(model, example, layer)
        norms: Float[Tensor, "seq"] = neg_grad.norm(dim=-1)
        top_positions = norms.topk(top_k).indices.tolist()
        for pos in top_positions:
            v = neg_grad[pos]
            yield Candidate(
                vector=v / v.norm(),
                layer=layer,
                example_id=example["example_id"],
                token_pos=int(pos),
            )

    return mine
