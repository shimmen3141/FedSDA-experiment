"""製品のSTAGGER・Agrawal・Random RBFの生成器が、Riverの定義と同じであることを確かめる（Riverのある環境だけ。WSL）。

(i) 分類の関数: STAGGERは27通りの全ての入力、Agrawalは摂動なしのRiverの生成器の1万標本で、ラベルが全て一致する。
(ii) Random RBFの中心: Riverの`centroids[i]`・`centroid_weights[i]`と、製品の中心のi番目が一致する。
(iii) 分布: 各概念で2万標本ずつ、クラス1の割合と各特徴の平均（Riverの特徴は製品と同じ尺度へ写す）の差が、
差の標準誤差の4倍以内。標本の乱数の列はRiverと違う（製品はrunのNumPyの乱数）ので、分布で比べる。
このmoduleはtorchをimportしない。
"""

import itertools
import math

import numpy as np
import pytest

pytest.importorskip("river")

from river.datasets import synth  # noqa: E402

from federated_learning_experiments.data.agrawal.agrawal_sample_generation import (  # noqa: E402
    AgrawalSampleGenerator,
    compute_agrawal_class_label,
)
from federated_learning_experiments.data.random_rbf.random_rbf_sample_generation import (  # noqa: E402
    RandomRbfSampleGenerator,
    generate_random_rbf_centroids,
)
from federated_learning_experiments.data.stagger.stagger_sample_generation import (  # noqa: E402
    StaggerSampleGenerator,
    compute_stagger_class_label,
)

DISTRIBUTION_SAMPLE_COUNT = 20000
STANDARD_ERROR_MULTIPLE = 4.0
AGRAWAL_FEATURE_SCALES = (
    ("salary", 20000.0, 130000.0),
    ("commission", 0.0, 75000.0),
    ("age", 20.0, 60.0),
    ("elevel", 0.0, 4.0),
    ("car", 1.0, 19.0),
    ("zipcode", 0.0, 8.0),
    ("hvalue", 0.0, 900000.0),
    ("hyears", 1.0, 29.0),
    ("loan", 0.0, 500000.0),
)


@pytest.mark.parametrize("concept_id", range(3))
def test_stagger_labels_match_river_on_all_inputs(concept_id):
    river_function = synth.STAGGER(classification_function=concept_id)._functions[concept_id]
    for size, color, shape in itertools.product(range(3), repeat=3):
        assert compute_stagger_class_label(
            size=size, color=color, shape=shape, concept_id=concept_id
        ) == river_function(size, color, shape)


@pytest.mark.parametrize("concept_id", range(4))
def test_agrawal_labels_match_river_without_perturbation(concept_id):
    river_stream = synth.Agrawal(
        classification_function=concept_id, seed=100 + concept_id, perturbation=0.0
    )
    for x, y in river_stream.take(10000):
        assert compute_agrawal_class_label(**x, concept_id=concept_id) == y


@pytest.mark.parametrize("concept_id", range(4))
def test_random_rbf_centroids_match_river(concept_id):
    river_generator = synth.RandomRBF(
        seed_model=concept_id, seed_sample=0, n_classes=2, n_features=10, n_centroids=50
    )
    river_generator._generate_centroids()
    centroids = generate_random_rbf_centroids(concept_id=concept_id)
    assert len(centroids) == len(river_generator.centroids) == 50
    for centroid, river_centroid, river_weight in zip(
        centroids, river_generator.centroids, river_generator.centroid_weights, strict=True
    ):
        assert centroid.centre == tuple(river_centroid.centre)
        assert centroid.class_label == river_centroid.class_label
        assert centroid.standard_deviation == river_centroid.std_dev
        assert centroid.weight == river_weight


def _river_rows(dataset_name, concept_id, seed):
    if dataset_name == "stagger3":
        stream = synth.STAGGER(classification_function=concept_id, seed=seed)
        return [
            ([x["size"] / 2, x["color"] / 2, x["shape"] / 2], y)
            for x, y in stream.take(DISTRIBUTION_SAMPLE_COUNT)
        ]
    if dataset_name == "agrawal4":
        stream = synth.Agrawal(classification_function=concept_id, seed=seed, perturbation=0.05)
        return [
            ([(x[name] - lowest) / width for name, lowest, width in AGRAWAL_FEATURE_SCALES], y)
            for x, y in stream.take(DISTRIBUTION_SAMPLE_COUNT)
        ]
    stream = synth.RandomRBF(
        seed_model=concept_id, seed_sample=seed, n_classes=2, n_features=10, n_centroids=50
    )
    return [([x[i] for i in range(10)], y) for x, y in stream.take(DISTRIBUTION_SAMPLE_COUNT)]


_PRODUCT_GENERATORS = {
    "stagger3": (StaggerSampleGenerator, 3),
    "agrawal4": (AgrawalSampleGenerator, 4),
    "randomrbf4": (RandomRbfSampleGenerator, 4),
}


def _product_rows(dataset_name, concept_id, seed):
    generator_type, concept_count = _PRODUCT_GENERATORS[dataset_name]
    generator = generator_type(
        numpy_random_generator=np.random.RandomState(seed), concept_count=concept_count
    )
    rows = []
    for _ in range(DISTRIBUTION_SAMPLE_COUNT):
        sample = generator.generate_sample(concept_id=concept_id)
        rows.append((list(sample.feature_values), sample.class_label))
    return rows


def _distribution_violations(dataset_name, concept_id, seed):
    """クラス1の割合と各特徴の平均で、差が差の標準誤差の4倍を超える比較を返す。"""
    product = _product_rows(dataset_name, concept_id, seed)
    river = _river_rows(dataset_name, concept_id, seed)
    columns = {
        "class_1_rate": (
            np.array([y for _, y in product], dtype=float),
            np.array([y for _, y in river], dtype=float),
        )
    }
    for feature_index in range(len(product[0][0])):
        columns[f"feature_{feature_index}"] = (
            np.array([x[feature_index] for x, _ in product]),
            np.array([x[feature_index] for x, _ in river]),
        )
    violations = []
    for name, (product_values, river_values) in columns.items():
        difference = float(product_values.mean() - river_values.mean())
        standard_error = math.sqrt(
            product_values.var(ddof=1) / len(product_values)
            + river_values.var(ddof=1) / len(river_values)
        )
        if standard_error == 0.0:
            if difference != 0.0:
                violations.append((name, difference, standard_error))
        elif abs(difference) > STANDARD_ERROR_MULTIPLE * standard_error:
            violations.append((name, difference, standard_error))
    return violations


@pytest.mark.parametrize(
    ("dataset_name", "concept_id"),
    [("stagger3", c) for c in range(3)]
    + [("agrawal4", c) for c in range(4)]
    + [("randomrbf4", c) for c in range(4)],
)
def test_product_distribution_matches_river(dataset_name, concept_id):
    assert _distribution_violations(dataset_name, concept_id, seed=0) == []
