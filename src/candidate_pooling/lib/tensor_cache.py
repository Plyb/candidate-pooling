from pathlib import Path
from typing import Callable

import torch


def load_or_compute_tensor[T](cache_path: Path, compute_fn: Callable[[], T]) -> T:
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)  # type: ignore[return-value]
    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result
