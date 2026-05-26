
from typing import Any, Iterable, Mapping, cast

from datasets import Dataset

from candidate_pooling.lib.typed_dataset import TypedDataset


def to_dataset[T : Mapping[str, Any]](records: Iterable[T]) -> TypedDataset[T]:
    return TypedDataset[T](cast(Dataset, Dataset.from_list(list(records)))) # type: ignore