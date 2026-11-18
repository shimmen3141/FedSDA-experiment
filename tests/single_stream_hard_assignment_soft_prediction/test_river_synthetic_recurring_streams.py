"""Riverの合成データの再来の系列: 定義、概念列、概念ごとの生成器の続きから取ること、決定性（WSL）。"""

import random

import numpy as np
import pytest

pytest.importorskip("river")

from river.datasets import synth  # noqa: E402

from federated_learning_experiments.data.dataset_definitions import (  # noqa: E402
    get_dataset_definition,
)
from single_stream_hard_assignment_soft_prediction.random_concept_schedule import (  # noqa: E402
    generate_random_concept_schedule,
)
from single_stream_hard_assignment_soft_prediction.river_synthetic_recurring_streams import (  # noqa: E402
    RIVER_SYNTHETIC_STREAM_DEFINITIONS,
    generate_river_synthetic_stream,
)
from single_stream_hard_assignment_soft_prediction.single_stream_input import (  # noqa: E402
    compute_stream_array_digest,
)

SAMPLE_COUNT = 3000
DATASETS = ("stagger3", "agrawal4", "randomrbf4")


def independent_generator(dataset_name, concept_id, seed, perturbation=0.05):
    """testの側で、同じ概念・同じseedの生成器を、定義を使わずに作る。"""
    if dataset_name == "stagger3":
        return synth.STAGGER(classification_function=concept_id, seed=seed)
    if dataset_name == "agrawal4":
        return synth.Agrawal(
            classification_function=concept_id, seed=seed, perturbation=perturbation
        )
    return synth.RandomRBF(
        seed_model=concept_id, seed_sample=seed, n_classes=2, n_features=10, n_centroids=50
    )


def test_definitions():
    assert set(RIVER_SYNTHETIC_STREAM_DEFINITIONS) == set(DATASETS)
    expected = {"stagger3": (3, 3), "agrawal4": (4, 9), "randomrbf4": (4, 10)}
    for name, definition in RIVER_SYNTHETIC_STREAM_DEFINITIONS.items():
        assert definition.dataset_name == name
        assert (definition.concept_count, len(definition.feature_names)) == expected[name]
        assert definition.class_count == 2
        assert all(type(n) is str for n in definition.feature_names)
        arguments = definition.fixed_generator_arguments
        assert (
            "1000" in arguments["seed_rule"] and arguments["generator"] == definition.generator_name
        )
        with pytest.raises((KeyError, ValueError)):
            get_dataset_definition(dataset_name=name)
    assert (
        RIVER_SYNTHETIC_STREAM_DEFINITIONS["agrawal4"].fixed_generator_arguments["perturbation"]
        == 0.05
    )
    assert RIVER_SYNTHETIC_STREAM_DEFINITIONS["randomrbf4"].feature_names == tuple(
        str(i) for i in range(10)
    )


@pytest.fixture(scope="module", params=DATASETS)
def generated(request):
    name = request.param
    return name, generate_river_synthetic_stream(
        dataset_name=name, random_seed=4, sample_count=SAMPLE_COUNT
    )


def test_shapes_types_and_label_range(generated):
    name, arrays = generated
    definition = RIVER_SYNTHETIC_STREAM_DEFINITIONS[name]
    assert set(arrays) == {"features", "labels", "concept_ids"}
    assert arrays["features"].shape == (SAMPLE_COUNT, len(definition.feature_names))
    assert arrays["features"].dtype == np.float64
    assert arrays["labels"].dtype == np.int64 and arrays["concept_ids"].dtype == np.int64
    assert set(np.unique(arrays["labels"])) <= {0, 1}


def test_concepts_follow_the_schedule_from_the_seed(generated):
    name, arrays = generated
    expected = generate_random_concept_schedule(
        sample_count=SAMPLE_COUNT,
        concept_count=RIVER_SYNTHETIC_STREAM_DEFINITIONS[name].concept_count,
        python_random_generator=random.Random(4),
    )
    assert tuple(arrays["concept_ids"].tolist()) == expected


def test_each_concept_continues_its_own_generator(generated):
    """概念ごとの標本の並びは、その概念の生成器を単独で回した最初のn件と同じ（再来は続きから）。"""
    name, arrays = generated
    feature_names = RIVER_SYNTHETIC_STREAM_DEFINITIONS[name].feature_names
    concept_ids = arrays["concept_ids"]
    recurring = False
    for concept_id in np.unique(concept_ids):
        mask = concept_ids == concept_id
        count = int(mask.sum())
        samples = list(
            independent_generator(name, int(concept_id), 1000 * 4 + int(concept_id)).take(count)
        )
        expected_features = np.array([[float(x[k]) for k in x] for x, _ in samples])
        expected_labels = np.array([int(y) for _, y in samples])
        assert [str(k) for k in samples[0][0]] == list(feature_names)
        np.testing.assert_array_equal(arrays["features"][mask], expected_features)
        np.testing.assert_array_equal(arrays["labels"][mask], expected_labels)
        runs = np.flatnonzero(np.diff(mask.astype(int)) == 1)
        recurring |= len(runs) + int(mask[0]) >= 2
    # この系列には再来があり、上の比較が「続きから取る」ことを確かめている。
    assert recurring


def test_same_arguments_give_the_same_stream_and_seeds_differ():
    for name in DATASETS:
        a = generate_river_synthetic_stream(dataset_name=name, random_seed=1, sample_count=800)
        b = generate_river_synthetic_stream(dataset_name=name, random_seed=1, sample_count=800)
        c = generate_river_synthetic_stream(dataset_name=name, random_seed=2, sample_count=800)
        assert compute_stream_array_digest(a) == compute_stream_array_digest(b)
        assert compute_stream_array_digest(a) != compute_stream_array_digest(c)


def test_agrawal_perturbation_is_passed_to_the_generator():
    arrays = generate_river_synthetic_stream(dataset_name="agrawal4", random_seed=0, sample_count=1)
    ((x_with, y_with),) = independent_generator("agrawal4", 0, 0).take(1)
    ((x_without, y_without),) = independent_generator("agrawal4", 0, 0, perturbation=0.0).take(1)
    assert int(arrays["labels"][0]) == y_with == y_without
    np.testing.assert_array_equal(arrays["features"][0], [float(v) for v in x_with.values()])
    # 摂動するのは数値の特徴（salary・commission・age・hvalue・hyears・loan）。全体で違うことを確かめる。
    numeric = ("salary", "commission", "age", "hvalue", "hyears", "loan")
    assert [x_with[k] for k in numeric] != [x_without[k] for k in numeric]
    assert x_with["salary"] != x_without["salary"]


@pytest.mark.parametrize(
    "arguments",
    [
        dict(dataset_name="sine2"),
        dict(dataset_name="unknown"),
        dict(random_seed=-1),
        dict(random_seed=True),
        dict(sample_count=-1),
        dict(sample_count=2.0),
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    full = dict(dataset_name="stagger3", random_seed=0, sample_count=10)
    full.update(arguments)
    with pytest.raises((TypeError, ValueError)):
        generate_river_synthetic_stream(**full)
