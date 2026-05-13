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
        return TypedDataset[RowT](self._dataset.select(range(n, len(self._dataset))))

    def take(self, n: int) -> TypedDataset[RowT]:
        return TypedDataset[RowT](self._dataset.select(range(n)))

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedDataset[OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedDataset[OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedDataset[OutRow]:
        return TypedDataset[OutRow](self._dataset.map(fn, with_indices=with_indices, **kwargs))

    def filter(self, fn: Callable[[RowT], bool], **kwargs: Any) -> TypedDataset[RowT]:
        return TypedDataset[RowT](self._dataset.filter(fn, **kwargs))

    def shuffle(self, seed: int | None = None) -> TypedDataset[RowT]:
        return TypedDataset[RowT](self._dataset.shuffle(seed=seed))

    def to_iterable_dataset(self, num_shards: int | None = 1) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset[RowT](self._dataset.to_iterable_dataset(num_shards=num_shards))

    def save_to_disk(
        self,
        dataset_path: PathLike,
        max_shard_size: str | int | None = None,
        num_shards: int | None = None,
        num_proc: int | None = None,
        storage_options: dict | None = None,
    ):
        self._dataset.save_to_disk(dataset_path, max_shard_size, num_shards, num_proc, storage_options)


class TypedIterableDataset[RowT: Mapping[str, Any]]:
    def __init__(self, dataset: datasets.IterableDataset) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterator[RowT]:
        return iter(self._dataset)  # type: ignore[return-value]

    def skip(self, n: int) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset[RowT](self._dataset.skip(n))

    def take(self, n: int) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset[RowT](self._dataset.take(n))

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedIterableDataset[OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedIterableDataset[OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedIterableDataset[OutRow]:
        return TypedIterableDataset[OutRow](self._dataset.map(fn, with_indices=with_indices, **kwargs))

    def filter(self, fn: Callable[[RowT], bool], **kwargs: Any) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset[RowT](self._dataset.filter(fn, **kwargs))

    def shuffle(self, seed: int | None = None, buffer_size: int = 1000) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset[RowT](self._dataset.shuffle(seed=seed, buffer_size=buffer_size))
