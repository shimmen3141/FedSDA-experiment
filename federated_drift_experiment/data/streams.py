"""データセット選択とクライアントストリームの構築。"""
import numpy as np
import torch

from .. import config
from .mnist import sample_mnist
from .names import normalize_dataset_name
from .synthetic import (
    generate_blobs,
    generate_circle2,
    generate_sea4,
    generate_sine2,
)


_GENERATORS = {
    "blobs": generate_blobs,
    "sea4": generate_sea4,
    "sea2": generate_sea4,
    "circle2": generate_circle2,
    "sine2": generate_sine2,
    "mnist2": sample_mnist,
    "mnist4": sample_mnist,
}


def generate_data(concept_id, n_samples=1, dataset=None):
    """指定データセット・コンセプトからテンソル形式の標本を生成する。"""
    dataset = config.DATASET if dataset is None else dataset
    dataset = normalize_dataset_name(dataset)
    if concept_id not in range(config.num_concepts(dataset)):
        raise ValueError(f"Unknown concept_id {concept_id} for dataset {dataset!r}")
    try:
        generator = _GENERATORS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset!r}") from exc
    x_list, y_list = generator(concept_id, n_samples)
    if n_samples == 1:
        return torch.FloatTensor(x_list[0]), torch.FloatTensor([y_list[0]])
    return (
        torch.FloatTensor(np.asarray(x_list)),
        torch.FloatTensor(np.asarray(y_list)).unsqueeze(1),
    )


def build_data_streams(schedules):
    """概念系列に従って全クライアントのデータストリームを生成する。"""
    return [[generate_data(concept_id) for concept_id in schedule]
            for schedule in schedules]
