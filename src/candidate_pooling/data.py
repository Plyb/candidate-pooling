from collections.abc import Callable
from typing import Any, Mapping, cast

import torch
from datasets import Dataset
from nnsight import LanguageModel

from byutils import load_dataset

from candidate_pooling.lib.dataset_utils import set_format
from candidate_pooling.lib.typed_dataset import TypedDataset
from candidate_pooling.model import make_tokenize_fn
from candidate_pooling.types import MmluExample, TokenizedExample, to_transformer_input


def map_and_filter_take_n[InRowT: Mapping[str, Any], OutRowT: Mapping[str, Any]](
    dataset: TypedDataset[InRowT],
    map_fn: Callable[[Any], OutRowT],
    filter_fn: Callable[[OutRowT], bool],
    target_count: int,
) -> TypedDataset[OutRowT]:
    lazy_pipeline = dataset.to_iterable_dataset().map(map_fn).filter(filter_fn).take(target_count)
    return TypedDataset[OutRowT](cast(Dataset, Dataset.from_generator(lambda: iter(lazy_pipeline))))


def load_mmlu_splits(
    model: LanguageModel,
    n_train: int = 1000,
    n_probe: int = 200,
    seed: int = 42,
) -> tuple[TypedDataset[TokenizedExample], TypedDataset[TokenizedExample]]:
    ds_dict = load_dataset("cais/mmlu", "auxiliary_train")
    split: Dataset = ds_dict["train"]  # type: ignore[assignment]
    shuffled = TypedDataset[Any](split.shuffle(seed=seed))

    tokenize = make_tokenize_fn(model)

    def map_fn(item: dict[str, Any]) -> TokenizedExample:
        mmlu_ex: MmluExample = {**item["train"], "example_id": 0}  # type: ignore[typeddict-item]
        return tokenize(mmlu_ex)

    def filter_fn(item: TokenizedExample) -> bool:
        with torch.no_grad(), model.trace(to_transformer_input(item)):
            logits = model.output.logits.save()  # type: ignore[attr-defined]
        return bool(logits[0, -1].argmax().item() == item["label_id"])

    def assign_ids(offset: int) -> Callable[[TokenizedExample, int], TokenizedExample]:
        def fn(item: TokenizedExample, idx: int) -> TokenizedExample:
            return TokenizedExample(
                input_ids=item["input_ids"],
                attention_mask=item["attention_mask"],
                label_id=item["label_id"],
                example_id=offset + idx,
            )
        return fn

    train_raw = map_and_filter_take_n(shuffled, map_fn, filter_fn, n_train)
    train_ds = train_raw.map(assign_ids(0), with_indices=True)
    probe_start = train_ds[-1]['example_id']

    probe_raw = map_and_filter_take_n(shuffled.skip(probe_start), map_fn, filter_fn, n_probe)
    probe_ds = probe_raw.map(assign_ids(probe_start), with_indices=True)

    return set_format(train_ds, TokenizedExample), set_format(probe_ds, TokenizedExample)
