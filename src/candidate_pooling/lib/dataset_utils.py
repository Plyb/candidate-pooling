import typing
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch
from datasets import Dataset, load_from_disk

from candidate_pooling.lib.typed_dataset import TypedDataset


def _is_torch_tensor_type(tp: Any) -> bool:
    if tp is torch.Tensor:
        return True
    if typing.get_origin(tp) is typing.Annotated:
        return typing.get_args(tp)[0] is torch.Tensor
    # jaxtyping tensor types (Float[torch.Tensor, ...], etc.) expose array_type
    return getattr(tp, "array_type", None) is torch.Tensor


def set_format[RowT: Mapping[str, Any]]( #TODO move this out
    dataset: TypedDataset[RowT],
    row_type: type[RowT],
) -> TypedDataset[RowT]:
    hints = typing.get_type_hints(row_type, include_extras=True)
    tensor_columns = [col for col, tp in hints.items() if _is_torch_tensor_type(tp)]
    dataset._dataset.set_format(type="torch", columns=tensor_columns, output_all_columns=True, device='cuda')
    return dataset


def load_or_compute[T: Mapping[str, Any]](cache_path: Path, compute_fn: Callable[[], TypedDataset[T]]) -> TypedDataset[T]:
    if cache_path.exists():
        ds = cast(Dataset, load_from_disk(str(cache_path)))
        return TypedDataset[T](ds)

    ds = compute_fn()
    ds.save_to_disk(str(cache_path))
    return ds
