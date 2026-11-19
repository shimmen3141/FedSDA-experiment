"""STAGGER・Agrawal・Random RBF（stagger3・agrawal4・randomrbf4）の生成器の定義を確かめる。

Riverとの照合は`tests/single_stream_hard_assignment_soft_prediction/test_recurring_synthetic_generators_match_river.py`
（Riverのある環境だけ）。ここは、Riverの無い環境でも確かめられる定義（値の範囲・尺度・ラベルの規則・乱数の扱い）を見る。
"""

import itertools
import math

import numpy as np
import pytest

from federated_learning_experiments.data.agrawal.agrawal_sample_generation import (
    AgrawalSampleGenerator,
    compute_agrawal_class_label,
)
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.data.random_rbf.random_rbf_sample_generation import (
    RandomRbfCentroid,
    RandomRbfSampleGenerator,
    generate_random_rbf_centroids,
)
from federated_learning_experiments.data.stagger.stagger_sample_generation import (
    StaggerSampleGenerator,
    compute_stagger_class_label,
)

GENERATOR_TYPES_AND_CONCEPT_COUNTS = (
    (StaggerSampleGenerator, 3),
    (AgrawalSampleGenerator, 4),
    (RandomRbfSampleGenerator, 4),
)
FEATURE_COUNT_BY_GENERATOR_TYPE = {
    StaggerSampleGenerator: 3,
    AgrawalSampleGenerator: 9,
    RandomRbfSampleGenerator: 10,
}


def _random_state_snapshot(random_state):
    kind, keys, position, has_gauss, cached = random_state.get_state()
    return kind, keys.copy(), position, has_gauss, cached


def _assert_same_random_state(before, random_state):
    after = random_state.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def _samples(generator_type, concept_count, concept_id, sample_count, seed=0):
    generator = generator_type(
        numpy_random_generator=np.random.RandomState(seed), concept_count=concept_count
    )
    return [generator.generate_sample(concept_id=concept_id) for _ in range(sample_count)]


@pytest.mark.parametrize(("generator_type", "concept_count"), GENERATOR_TYPES_AND_CONCEPT_COUNTS)
def test_construction_does_not_consume_the_run_random_state(generator_type, concept_count):
    random_state = np.random.RandomState(3)
    before = _random_state_snapshot(random_state)
    generator = generator_type(numpy_random_generator=random_state, concept_count=concept_count)
    _assert_same_random_state(before, random_state)
    assert generator.numpy_random_generator is random_state
    assert generator.concept_count == concept_count


@pytest.mark.parametrize(("generator_type", "concept_count"), GENERATOR_TYPES_AND_CONCEPT_COUNTS)
@pytest.mark.parametrize("invalid", [0, -1, "maximum+1", True, 2.0, None])
def test_construction_rejects_invalid_concept_counts(generator_type, concept_count, invalid):
    value = concept_count + 1 if invalid == "maximum+1" else invalid
    expected = TypeError if type(value) is not int else ValueError
    with pytest.raises(expected):
        generator_type(numpy_random_generator=np.random.RandomState(0), concept_count=value)


@pytest.mark.parametrize(("generator_type", "concept_count"), GENERATOR_TYPES_AND_CONCEPT_COUNTS)
@pytest.mark.parametrize("invalid", [-1, "concept_count", True, 0.0, None])
def test_invalid_concepts_are_rejected_without_consuming_randomness(
    generator_type, concept_count, invalid
):
    random_state = np.random.RandomState(5)
    generator = generator_type(numpy_random_generator=random_state, concept_count=concept_count)
    value = concept_count if invalid == "concept_count" else invalid
    before = _random_state_snapshot(random_state)
    expected = TypeError if type(value) is not int else ValueError
    with pytest.raises(expected):
        generator.generate_sample(concept_id=value)
    _assert_same_random_state(before, random_state)


@pytest.mark.parametrize(("generator_type", "concept_count"), GENERATOR_TYPES_AND_CONCEPT_COUNTS)
def test_same_seed_and_concepts_give_the_same_samples(generator_type, concept_count):
    concept_ids = [c % concept_count for c in range(40)]

    def generate(seed):
        generator = generator_type(
            numpy_random_generator=np.random.RandomState(seed), concept_count=concept_count
        )
        return [generator.generate_sample(concept_id=c) for c in concept_ids]

    first, second, other = generate(11), generate(11), generate(12)
    assert first == second
    assert first != other
    for sample in first:
        assert type(sample) is ObservedSample
        assert len(sample.feature_values) == FEATURE_COUNT_BY_GENERATOR_TYPE[generator_type]
        assert sample.class_label in (0, 1)
        # float32へ丸めてある。
        for value in sample.feature_values:
            assert float(np.float32(value)) == value


# ---- STAGGER ----


@pytest.mark.parametrize(
    ("size", "color", "shape", "expected_labels"),
    [
        (0, 0, 0, (1, 1, 0)),
        (0, 0, 1, (1, 0, 0)),
        (0, 1, 1, (0, 0, 0)),
        (1, 1, 1, (0, 0, 1)),
        (2, 2, 1, (0, 1, 1)),
        (1, 0, 0, (0, 1, 1)),
    ],
)
def test_stagger_class_labels_follow_the_three_concepts(size, color, shape, expected_labels):
    labels = tuple(
        compute_stagger_class_label(size=size, color=color, shape=shape, concept_id=concept_id)
        for concept_id in range(3)
    )
    assert labels == expected_labels


@pytest.mark.parametrize("concept_id", range(3))
def test_stagger_features_are_halved_category_values_with_their_concept_label(concept_id):
    samples = _samples(StaggerSampleGenerator, 3, concept_id, 600)
    seen = set()
    for sample in samples:
        assert all(value in (0.0, 0.5, 1.0) for value in sample.feature_values)
        size, color, shape = (int(value * 2) for value in sample.feature_values)
        seen.add((size, color, shape))
        assert sample.class_label == compute_stagger_class_label(
            size=size, color=color, shape=shape, concept_id=concept_id
        )
    assert seen == set(itertools.product(range(3), repeat=3))


def test_stagger_label_function_rejects_values_outside_its_categories():
    with pytest.raises(ValueError):
        compute_stagger_class_label(size=3, color=0, shape=0, concept_id=0)
    with pytest.raises(ValueError):
        compute_stagger_class_label(size=0, color=0, shape=0, concept_id=3)


# ---- Agrawal ----

_AGRAWAL_SCALE = (
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


def _unscaled_agrawal(sample):
    return {
        name: offset + width * value
        for (name, offset, width), value in zip(_AGRAWAL_SCALE, sample.feature_values, strict=True)
    }


@pytest.mark.parametrize("concept_id", range(4))
def test_agrawal_features_stay_in_their_scaled_ranges_after_perturbation(concept_id):
    samples = _samples(AgrawalSampleGenerator, 4, concept_id, 3000)
    labels = []
    for sample in samples:
        assert all(0.0 <= value <= 1.0 for value in sample.feature_values)
        raw = _unscaled_agrawal(sample)
        for name in ("age", "elevel", "car", "zipcode", "hyears"):
            assert abs(raw[name] - round(raw[name])) < 1e-3, name
        zipcode = round(raw["zipcode"])
        # Riverの引数の並びのため、hvalueは(9 − zipcode) × 100000か0。
        assert min(abs(raw["hvalue"]), abs(raw["hvalue"] - (9 - zipcode) * 100000)) < 1.0
        if raw["commission"] > 1.0:
            assert raw["commission"] >= 10000 - 1.0
            # commissionは摂動の前のsalaryが75000未満のときだけ。摂動は最大±6500。
            assert raw["salary"] <= 75000 + 6500 + 1.0
        else:
            assert raw["salary"] >= 75000 - 6500 - 1.0
        labels.append(sample.class_label)
    assert 0 < sum(labels) < len(labels)


@pytest.mark.parametrize(
    ("values", "expected_labels"),
    [
        # age 30、salary 60000、elevel 0: 関数0〜3とも1。
        (dict(salary=60000, commission=20000, age=30, elevel=0, loan=150000), (1, 1, 1, 1)),
        # age 50、salary 60000、elevel 4: 関数0は0、1は0、2は0、3はsalary 75000〜125000でないので0。
        (dict(salary=60000, commission=20000, age=50, elevel=4, loan=150000), (0, 0, 0, 0)),
        # age 70、salary 100000、elevel 3: 0は1、1は0、2は1、3は50000〜100000なので1。
        (dict(salary=100000, commission=0, age=70, elevel=3, loan=150000), (1, 0, 1, 1)),
        # age 45、salary 90000、elevel 1: 0は0、1は1、2は1、3は1。
        (dict(salary=90000, commission=0, age=45, elevel=1, loan=150000), (0, 1, 1, 1)),
    ],
)
def test_agrawal_class_labels_follow_functions_zero_to_three(values, expected_labels):
    common = dict(car=5, zipcode=3, hvalue=300000.0, hyears=10)
    labels = tuple(
        compute_agrawal_class_label(**values, **common, concept_id=concept_id)
        for concept_id in range(4)
    )
    assert labels == expected_labels


def test_agrawal_label_function_rejects_concepts_outside_zero_to_three():
    with pytest.raises(ValueError):
        compute_agrawal_class_label(
            salary=60000,
            commission=0,
            age=30,
            elevel=0,
            car=1,
            zipcode=0,
            hvalue=0.0,
            hyears=1,
            loan=0.0,
            concept_id=4,
        )


# ---- Random RBF ----


def test_random_rbf_centroids_are_fixed_per_concept_and_independent_of_the_run():
    first = generate_random_rbf_centroids(concept_id=2)
    assert first == generate_random_rbf_centroids(concept_id=2)
    assert first != generate_random_rbf_centroids(concept_id=1)
    assert len(first) == 50
    for centroid in first:
        assert type(centroid) is RandomRbfCentroid
        assert len(centroid.centre) == 10
        assert all(0.0 <= value < 1.0 for value in centroid.centre)
        assert centroid.class_label in (0, 1)
        assert 0.0 <= centroid.standard_deviation < 1.0
        assert 0.0 <= centroid.weight < 1.0
    generators = [
        RandomRbfSampleGenerator(
            numpy_random_generator=np.random.RandomState(seed), concept_count=4
        )
        for seed in (0, 1)
    ]
    assert generators[0].centroids_by_concept == generators[1].centroids_by_concept
    assert generators[0].centroids_by_concept[2] == first


@pytest.mark.parametrize("concept_id", range(4))
def test_random_rbf_samples_lie_near_a_centroid_with_its_label(concept_id):
    centroids = generate_random_rbf_centroids(concept_id=concept_id)
    for sample in _samples(RandomRbfSampleGenerator, 4, concept_id, 400):
        # 中心からの距離は |標準正規| × 標準偏差。ラベルは、その距離で説明できる中心の中にある。
        candidates = [
            centroid
            for centroid in centroids
            if math.dist(sample.feature_values, centroid.centre)
            <= 6.0 * centroid.standard_deviation + 1e-5
        ]
        assert sample.class_label in {centroid.class_label for centroid in candidates}
