import logging
from typing import Any, cast

import torch
from datasets import Dataset
from nnsight import LanguageModel

from byutils import load_dataset

from candidate_pooling.lib.dataset_utils import set_format
from candidate_pooling.lib.typed_dataset import TypedDataset, TypedIterableDataset
from candidate_pooling.model import make_tokenize_fn
from candidate_pooling.types import McqaExample, MmluExample, TokenizedExample, to_transformer_input

logger = logging.getLogger(__name__)


def load_mmlu(seed: int = 42) -> TypedIterableDataset[MmluExample]:
    ds_dict = load_dataset("cais/mmlu", "auxiliary_train")
    split: Dataset = ds_dict["train"]  # type: ignore[assignment]
    shuffled = TypedDataset[Any](split).shuffle(seed=seed)

    def unwrap(item: dict[str, Any]) -> MmluExample:
        return MmluExample(**item["train"])

    return shuffled.to_iterable_dataset().map(unwrap)


def load_arc_easy(seed: int = 42) -> TypedIterableDataset[McqaExample]:
    ds_dict = load_dataset("allenai/ai2_arc", "ARC-Easy")
    split: Dataset = ds_dict["train"]  # type: ignore[assignment]
    shuffled = TypedDataset[Any](split).shuffle(seed=seed)

    def to_mcqa(item: dict[str, Any]) -> McqaExample:
        labels: list[str] = item["choices"]["label"]
        texts: list[str] = item["choices"]["text"]
        return McqaExample(
            question=item["question"],
            choices=texts,
            answer=labels.index(item["answerKey"]),
        )

    return shuffled.to_iterable_dataset().map(to_mcqa)


def tokenize_dataset(
    model: LanguageModel,
    dataset: TypedIterableDataset[McqaExample],
    n_train: int,
    n_probe: int,
) -> tuple[TypedDataset[TokenizedExample], TypedDataset[TokenizedExample]]:
    tokenize = make_tokenize_fn(model)

    def filter_fn(item: TokenizedExample) -> bool:
        with torch.no_grad(), model.trace(to_transformer_input(item)):
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        return bool(logits[0, -1].argmax().item() == item["label_id"])

    target = n_train + n_probe
    lazy_pipeline = dataset.map(tokenize, with_indices=True).filter(filter_fn).take(target)
    combined = TypedDataset[TokenizedExample](
        cast(Dataset, Dataset.from_generator(lambda: iter(lazy_pipeline)))
    )

    if len(combined) < target:
        logger.warning(
            "tokenize_dataset retained %d / %d examples after filtering",
            len(combined),
            target,
        )

    train_count = min(n_train, len(combined))
    train_ds = combined.take(train_count)
    probe_ds = combined.skip(train_count)

    return (
        set_format(train_ds, TokenizedExample),
        set_format(probe_ds, TokenizedExample),
    )
