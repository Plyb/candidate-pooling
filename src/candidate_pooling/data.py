from datasets import Dataset
from byutils import load_dataset


def load_mmlu_splits(
    n_train: int = 1000,
    n_probe: int = 200,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    ds_dict = load_dataset("cais/mmlu", "auxiliary_train")
    split: Dataset = ds_dict["train"]  # type: ignore[assignment]

    shuffled = split.shuffle(seed=seed)

    train_ds = shuffled.select(range(n_train)).map(
        lambda item, idx: {**item['train'], "example_id": idx}, with_indices=True
    )
    probe_ds = shuffled.select(range(n_train, n_train + n_probe)).map(
        lambda item, idx: {**item['train'], "example_id": n_train + idx}, with_indices=True
    )

    return train_ds, probe_ds
