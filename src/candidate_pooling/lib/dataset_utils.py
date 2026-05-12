from pathlib import Path
from typing import Any, Callable, Mapping, cast

from datasets import Dataset, load_from_disk

from candidate_pooling.lib.typed_dataset import TypedDataset


def load_or_compute[T: Mapping[str, Any]](cache_path: Path, compute_fn: Callable[[], TypedDataset[T]]) -> TypedDataset[T]:
    if cache_path.exists():
        ds = cast(Dataset, load_from_disk(str(cache_path)))
        return TypedDataset[T](ds)

    ds = compute_fn()
    ds.save_to_disk(str(cache_path))
    return ds
