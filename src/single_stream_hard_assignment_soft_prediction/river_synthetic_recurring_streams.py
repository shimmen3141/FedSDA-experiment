"""Riverの合成データの生成器（STAGGER・Agrawal・Random RBF）から、再来の系列の配列を作る。

概念列は`random_concept_schedule`（製品のランダムな概念列と同じ規則）を`random.Random(系列のseed)`で作る。
概念ごとに専用の生成器（seedは`1000 × 系列のseed + 概念の番号`）を持ち、その概念の標本は生成器の続きから
取る（再来で同じ標本を繰り返さない）。概念の中身（STAGGER・Agrawalの分類の関数、Random RBFの中心の配置）は
系列のseedによらず固定で、標本だけがseedで変わる。製品のデータセットの定義とは別で、名前は重ならない。
"""

import random
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from river.datasets import synth

from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from single_stream_hard_assignment_soft_prediction.random_concept_schedule import (
    generate_random_concept_schedule,
)

_SEED_RULE = "1000 * random_seed + concept_id"


@dataclass(frozen=True, kw_only=True)
class RiverSyntheticStreamDefinition:
    """1つのデータの定義。`fixed_generator_arguments`は記録用（生成器の名前・固定の引数・seedの決め方）。"""

    dataset_name: str
    concept_count: int
    class_count: int
    feature_names: tuple[str, ...]
    generator_name: str
    fixed_generator_arguments: MappingProxyType


def _definition(*, dataset_name, concept_count, feature_names, generator_name, arguments):
    return RiverSyntheticStreamDefinition(
        dataset_name=dataset_name,
        concept_count=concept_count,
        class_count=2,
        feature_names=tuple(feature_names),
        generator_name=generator_name,
        fixed_generator_arguments=MappingProxyType(
            dict(generator=generator_name, seed_rule=_SEED_RULE, **arguments)
        ),
    )


RIVER_SYNTHETIC_STREAM_DEFINITIONS = MappingProxyType(
    {
        "stagger3": _definition(
            dataset_name="stagger3",
            concept_count=3,
            feature_names=("size", "color", "shape"),
            generator_name="STAGGER",
            arguments=dict(classification_function="concept_id", balance_classes=False),
        ),
        "agrawal4": _definition(
            dataset_name="agrawal4",
            concept_count=4,
            feature_names=(
                "salary",
                "commission",
                "age",
                "elevel",
                "car",
                "zipcode",
                "hvalue",
                "hyears",
                "loan",
            ),
            generator_name="Agrawal",
            arguments=dict(
                classification_function="concept_id", balance_classes=False, perturbation=0.05
            ),
        ),
        "randomrbf4": _definition(
            dataset_name="randomrbf4",
            concept_count=4,
            feature_names=tuple(str(i) for i in range(10)),
            generator_name="RandomRBF",
            arguments=dict(seed_model="concept_id", n_classes=2, n_features=10, n_centroids=50),
        ),
    }
)


def _create_concept_generator(*, dataset_name: str, concept_id: int, seed: int):
    if dataset_name == "stagger3":
        return synth.STAGGER(classification_function=concept_id, seed=seed, balance_classes=False)
    if dataset_name == "agrawal4":
        return synth.Agrawal(
            classification_function=concept_id, seed=seed, balance_classes=False, perturbation=0.05
        )
    if dataset_name == "randomrbf4":
        return synth.RandomRBF(
            seed_model=concept_id, seed_sample=seed, n_classes=2, n_features=10, n_centroids=50
        )
    raise ValueError(f"知らないデータです: {dataset_name}")


def _validate_arguments(
    *, dataset_name, random_seed, sample_count
) -> RiverSyntheticStreamDefinition:
    if dataset_name not in RIVER_SYNTHETIC_STREAM_DEFINITIONS:
        raise ValueError(
            f"知らないデータです: {dataset_name}（{'・'.join(RIVER_SYNTHETIC_STREAM_DEFINITIONS)}）"
        )
    for name, value in (("random_seed", random_seed), ("sample_count", sample_count)):
        if type(value) is not int:
            raise TypeError(f"{name}はbool以外のbuiltin intが必要です。")
        if value < 0:
            raise ValueError(f"{name}は0以上が必要です。")
    try:
        get_dataset_definition(dataset_name=dataset_name)
    except (KeyError, ValueError):
        pass
    else:
        raise ValueError(f"製品のデータセットと同じ名前です: {dataset_name}")
    return RIVER_SYNTHETIC_STREAM_DEFINITIONS[dataset_name]


def generate_river_synthetic_stream(
    *, dataset_name: str, random_seed: int, sample_count: int
) -> dict[str, np.ndarray]:
    """`features`（float64）・`labels`（int64）・`concept_ids`（int64）の配列を返す。"""
    definition = _validate_arguments(
        dataset_name=dataset_name, random_seed=random_seed, sample_count=sample_count
    )
    concept_ids = generate_random_concept_schedule(
        sample_count=sample_count,
        concept_count=definition.concept_count,
        python_random_generator=random.Random(random_seed),
    )
    iterators = {}
    features = np.empty((sample_count, len(definition.feature_names)), dtype=np.float64)
    labels = np.empty(sample_count, dtype=np.int64)
    for index, concept_id in enumerate(concept_ids):
        if concept_id not in iterators:
            iterators[concept_id] = iter(
                _create_concept_generator(
                    dataset_name=dataset_name,
                    concept_id=concept_id,
                    seed=1000 * random_seed + concept_id,
                )
            )
        x, y = next(iterators[concept_id])
        if tuple(str(key) for key in x) != definition.feature_names:
            raise RuntimeError(f"生成した標本の特徴の名前が定義と違います: {list(x)}")
        features[index] = [float(x[key]) for key in x]
        labels[index] = int(y)
    return dict(
        features=features,
        labels=labels,
        concept_ids=np.array(concept_ids, dtype=np.int64),
    )
