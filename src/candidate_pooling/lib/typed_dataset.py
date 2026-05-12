from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal, overload

import datasets
from datasets.utils.typing import PathLike


class TypedDataset[RowT: Mapping[str, Any]]:
    def __init__(self, dataset: datasets.Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[RowT]:
        return iter(self._dataset)  # type: ignore[return-value]

    def __contains__(self, item: object) -> bool:
        return item in self._dataset

    def __getitem__(self, index: int) -> RowT:
        return self._dataset[index]  # type: ignore[return-value]

    def skip(self, n: int) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.select(range(n, len(self._dataset))))

    def take(self, n: int) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.select(range(n)))

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedDataset[OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedDataset[OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedDataset[OutRow]:
        return TypedDataset(self._dataset.map(fn, with_indices=with_indices, **kwargs))
    
    def save_to_disk(
        self,
        dataset_path: PathLike,
        max_shard_size: str | int | None = None,
        num_shards: int | None = None,
        num_proc: int | None = None,
        storage_options: dict | None = None,
    ):
        self._dataset.save_to_disk(dataset_path, max_shard_size, num_shards, num_proc, storage_options)
