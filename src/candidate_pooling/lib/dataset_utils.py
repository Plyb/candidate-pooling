from pathlib import Path
from typing import Any, Callable, Mapping, cast

from datasets import Dataset, load_from_disk

from candidate_pooling.lib.typed_dataset import TypedDataset


def map_and_filter_take_n[InRowT: Mapping[str, Any], OutRowT: Mapping[str, Any]](
    dataset: TypedDataset[InRowT],
    map_fn: Callable[[Any], OutRowT],
    filter_fn: Callable[[OutRowT], bool],
    target_count: int,
) -> TypedDataset[OutRowT]:
    lazy_pipeline = dataset._dataset.to_iterable_dataset().map(map_fn).filter(filter_fn).take(target_count)
    return TypedDataset(cast(Dataset, Dataset.from_generator(lambda: iter(lazy_pipeline))))


def load_or_compute[T: Mapping[str, Any]](cache_path: Path, compute_fn: Callable[[], TypedDataset[T]]) -> TypedDataset[T]:
    if cache_path.exists():
        ds = cast(Dataset, load_from_disk(str(cache_path)))
        return TypedDataset[T](ds)

    ds = compute_fn()
    ds.save_to_disk(str(cache_path))
    return ds
