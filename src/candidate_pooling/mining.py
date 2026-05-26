from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float
from nnsight import LanguageModel
from sae_lens import SAE
from torch import Tensor

from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.types import Candidate, TokenizedExample, to_transformer_input
from candidate_pooling.util import to_dataset

LAYER = 24


type MineFn = Callable[[Callable[[], TypedDataset[TokenizedExample]]], TypedDataset[Candidate]]


class MiningStrategy(Protocol):

    def make_mine_fn(self, model: LanguageModel) -> MineFn: ...

    @property
    def run_cfg(self) -> dict[str, Any]: ... # includes the strategy name and any extra cfg needed for the strategy


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


@dataclass
class TopKGradsStrategy(MiningStrategy):
    top_k: int

    @property
    def run_cfg(self) -> dict[str, Any]:
        return {"strategy": "top_k_grads", "top_k": self.top_k}

    def make_mine_fn(self, model: LanguageModel) -> MineFn:

        def mine_iter(examples: TypedDataset[TokenizedExample]) -> Iterator[Candidate]:
            for example in examples:
                neg_grad: Float[Tensor, "seq d_model"] = compute_neg_gradients(model, example, LAYER)
                norms: Float[Tensor, "seq"] = neg_grad.norm(dim=-1)
                top_positions = norms.topk(self.top_k).indices.tolist()
                for pos in top_positions:
                    v = neg_grad[pos]
                    yield Candidate(
                        vector=v / v.norm(),
                        layer=LAYER,
                        example_id=example["example_id"],
                        token_pos=int(pos),
                    )

        def mine(examples: Callable[[], TypedDataset[TokenizedExample]]) -> TypedDataset[Candidate]:
            return to_dataset(mine_iter(examples()))

        return mine


@dataclass
class SaeStrategy(MiningStrategy):
    release: str
    sae_id: str


    @property
    def run_cfg(self) -> dict[str, Any]:
        return {"strategy": "sae", "release": self.release, "sae_id": self.sae_id, "layer": LAYER}

    def make_mine_fn(self, model: LanguageModel) -> MineFn:
        sae = SAE.from_pretrained(self.release, self.sae_id, device="cuda")
        W_dec: Float[Tensor, "d_sae d_model"] = sae.W_dec.detach()
        directions: Float[Tensor, "d_sae d_model"] = W_dec / W_dec.norm(dim=-1, keepdim=True)

        def mine_iter() -> Iterator[Candidate]:
            for v in directions:
                yield Candidate(
                    vector=v,
                    layer=LAYER,
                    example_id=-1,
                    token_pos=-1,
                )

        def mine(examples: Callable[[], TypedDataset[TokenizedExample]]) -> TypedDataset[Candidate]:
            return to_dataset(mine_iter())

        return mine
