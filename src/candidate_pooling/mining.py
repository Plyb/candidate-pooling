from typing import Iterator

import torch
import torch.nn.functional as F
from braided import strand
from braided.strand import OneToMany
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor

from candidate_pooling.types import Candidate, TokenizedExample

LAYER = 12
TOP_K = 5


def make_mining_strand(
    model: LanguageModel,
    layer: int = LAYER,
    top_k: int = TOP_K,
) -> OneToMany[Candidate]:

    @strand.one_to_many
    def mine(example: TokenizedExample) -> Iterator[Candidate]:
        with model.trace(example) as tracer:
            hidden = model.model.layers[layer].output[0]  # type: ignore[attr-defined]
            logits = model.output.logits  # type: ignore[attr-defined]
            loss = F.cross_entropy(
                logits[0, -1].unsqueeze(0),
                torch.as_tensor([example["label_id"]], device="cuda"),  # type: ignore[attr-defined]
            )
            loss.backward()
            neg_grad = (-hidden[0].grad).save()  # [seq, d_model]

        neg_grad_val: Float[Tensor, "seq d_model"] = neg_grad.value
        norms: Float[Tensor, "seq"] = neg_grad_val.norm(dim=-1)
        top_positions = norms.topk(top_k).indices.tolist()
        for pos in top_positions:
            v = neg_grad_val[pos]
            yield Candidate(
                vector=v / v.norm(),
                layer=layer,
                example_id=example["example_id"],
                token_pos=int(pos),
            )

    return mine
