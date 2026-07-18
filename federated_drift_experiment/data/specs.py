"""データセット固有の構造とモデル条件を一元管理する。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    input_dim: int
    num_concepts: int
    num_classes: int = 2
    hidden_dims: tuple[int, ...] = (32, 32)
    learning_rate: float | None = None


DATASET_SPECS = {
    "blobs": DatasetSpec(2, 4),
    "sea": DatasetSpec(3, 4),
    "circle": DatasetSpec(2, 2),
    "sine": DatasetSpec(2, 2),
    "sea2": DatasetSpec(3, 2),
    "mnist2": DatasetSpec(
        784, 2, num_classes=10,
        hidden_dims=(1568,), learning_rate=1e-3,
    ),
    "mnist4": DatasetSpec(
        784, 4, num_classes=10,
        hidden_dims=(1568,), learning_rate=1e-3,
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset!r}") from exc
