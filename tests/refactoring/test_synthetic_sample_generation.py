"""合成データ（sine2・sea2・sea4・circle2）の生成器と概念列を、実旧と照合する。"""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.circle.circle_sample_generation import (
    CircleSampleGenerator,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_generation import (
    generate_random_client_concept_traces,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.dataset_definitions import (
    DEFINED_DATASET_NAMES,
    get_dataset_definition,
)
from federated_learning_experiments.data.observed_sample_generation import (
    OBSERVED_SAMPLE_GENERATOR_TYPES,
    build_client_observed_streams,
    create_observed_sample_generator,
    is_observed_sample_generator_of_dataset,
)
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ObservedSample,
)
from federated_learning_experiments.data.sea.sea_sample_generation import SeaSampleGenerator
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator

EXPECTED_GENERATOR_TYPE_BY_DATASET_NAME = {
    "sine2": SineSampleGenerator,
    "sea2": SeaSampleGenerator,
    "sea4": SeaSampleGenerator,
    "circle2": CircleSampleGenerator,
}
DATASET_NAME_AND_CONCEPT_ID_PAIRS = [
    (dataset_name, concept_id)
    for dataset_name in DEFINED_DATASET_NAMES
    for concept_id in range(get_dataset_definition(dataset_name=dataset_name).concept_count)
]


def assert_numpy_random_states_equal(actual_numpy_random_state, expected_numpy_random_state):
    """生成器の種類・キー配列・位置・正規分布キャッシュを全て比較する。"""
    assert actual_numpy_random_state[0] == expected_numpy_random_state[0]
    np.testing.assert_array_equal(actual_numpy_random_state[1], expected_numpy_random_state[1])
    assert actual_numpy_random_state[2:] == expected_numpy_random_state[2:]


def borrow_legacy_numpy_random_generator(monkeypatch, legacy_numpy_random_generator):
    """実旧の合成データの生成が使う乱数を、渡したRandomStateへ差し替える。"""
    from federated_drift_experiment.data import synthetic as legacy_synthetic_module

    monkeypatch.setattr(
        legacy_synthetic_module,
        "np",
        SimpleNamespace(random=legacy_numpy_random_generator, sin=np.sin),
    )


def test_expected_generator_types_cover_all_defined_datasets():
    """定義のある全datasetに生成器があり、検査用の型の一覧と一致する。"""
    assert tuple(EXPECTED_GENERATOR_TYPE_BY_DATASET_NAME) == DEFINED_DATASET_NAMES
    assert set(OBSERVED_SAMPLE_GENERATOR_TYPES) == set(
        EXPECTED_GENERATOR_TYPE_BY_DATASET_NAME.values()
    )


@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
def test_created_generator_borrows_the_random_generator_without_consuming_it(dataset_name):
    """datasetに対応する型の生成器を返し、渡した乱数をそのまま持ち、進めない。"""
    numpy_random_generator = np.random.RandomState(3)
    initial_numpy_random_state = numpy_random_generator.get_state()
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
    )
    assert type(sample_generator) is EXPECTED_GENERATOR_TYPE_BY_DATASET_NAME[dataset_name]
    assert sample_generator.numpy_random_generator is numpy_random_generator
    assert_numpy_random_states_equal(numpy_random_generator.get_state(), initial_numpy_random_state)


@pytest.mark.parametrize("generator_dataset_name", DEFINED_DATASET_NAMES)
@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
def test_generator_belongs_only_to_its_own_dataset(generator_dataset_name, dataset_name):
    """生成器は、作ったdatasetのものとだけ判定される（sea2とsea4は、同じ型でも別）。"""
    sample_generator = create_observed_sample_generator(
        dataset_name=generator_dataset_name, numpy_random_generator=np.random.RandomState(0)
    )
    assert is_observed_sample_generator_of_dataset(
        sample_generator=sample_generator, dataset_name=dataset_name
    ) == (generator_dataset_name == dataset_name)


@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
def test_other_objects_are_not_generators_of_a_dataset(dataset_name):
    """生成器でないもの・派生型は、どのdatasetの生成器でもない。定義のないdatasetは拒否する。"""
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=np.random.RandomState(0)
    )
    derived_generator_type = type("DerivedGenerator", (type(sample_generator),), {})
    derived_generator = derived_generator_type.__new__(derived_generator_type)
    derived_generator.__dict__.update(sample_generator.__dict__)
    for other_object in (None, np.random.RandomState(0), derived_generator):
        assert not is_observed_sample_generator_of_dataset(
            sample_generator=other_object, dataset_name=dataset_name
        )
    with pytest.raises(ValueError, match="dataset_name"):
        is_observed_sample_generator_of_dataset(
            sample_generator=sample_generator, dataset_name="mnist2"
        )
    with pytest.raises(TypeError):
        is_observed_sample_generator_of_dataset(sample_generator, dataset_name)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("argument_name", "invalid_value", "expected_error_type"),
    [
        ("dataset_name", "blobs", ValueError),
        ("dataset_name", "mnist2", ValueError),
        ("dataset_name", None, TypeError),
        ("numpy_random_generator", None, TypeError),
        ("numpy_random_generator", np.random, TypeError),
        ("numpy_random_generator", np.random.default_rng(0), TypeError),
        ("numpy_random_generator", random.Random(0), TypeError),
    ],
)
def test_generator_creation_rejects_invalid_arguments(
    argument_name, invalid_value, expected_error_type
):
    """定義のないdatasetと、exactなRandomState以外の乱数を拒否する。"""
    arguments = {
        "dataset_name": "sea2",
        "numpy_random_generator": np.random.RandomState(0),
    }
    arguments[argument_name] = invalid_value
    with pytest.raises(expected_error_type, match=argument_name):
        create_observed_sample_generator(**arguments)
    with pytest.raises(TypeError):
        create_observed_sample_generator("sea2", np.random.RandomState(0))  # type: ignore[misc]


@pytest.mark.parametrize("random_seed", [0, 17])
@pytest.mark.parametrize(("dataset_name", "concept_id"), DATASET_NAME_AND_CONCEPT_ID_PAIRS)
def test_samples_match_legacy_generation_and_random_state(
    monkeypatch, dataset_name, concept_id, random_seed
):
    """同じ乱数から、実旧の`generate_data`と、特徴（float32）・ラベル・生成後の乱数が一致する。"""
    from federated_drift_experiment.data import streams as legacy_streams_module

    numpy_random_generator = np.random.RandomState(random_seed)
    legacy_numpy_random_generator = np.random.RandomState(random_seed)
    borrow_legacy_numpy_random_generator(monkeypatch, legacy_numpy_random_generator)
    global_numpy_random_state = np.random.get_state()
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
    )
    observed_class_labels = set()
    for _sample_index in range(1500):
        legacy_feature_tensor, legacy_label_tensor = legacy_streams_module.generate_data(
            concept_id, dataset=dataset_name
        )
        observed_sample = sample_generator.generate_sample(concept_id=concept_id)
        assert type(observed_sample) is ObservedSample
        assert observed_sample.feature_values == tuple(legacy_feature_tensor.tolist())
        assert len(observed_sample.feature_values) == dataset_definition.input_feature_count
        assert all(type(feature_value) is float for feature_value in observed_sample.feature_values)
        assert type(observed_sample.class_label) is int
        assert observed_sample.class_label == int(legacy_label_tensor.item())
        assert 0 <= observed_sample.class_label < dataset_definition.class_count
        observed_class_labels.add(observed_sample.class_label)
    # 片方のラベルだけの照合になっていないことを確かめる。
    assert observed_class_labels == set(range(dataset_definition.class_count))
    assert_numpy_random_states_equal(
        numpy_random_generator.get_state(), legacy_numpy_random_generator.get_state()
    )
    assert_numpy_random_states_equal(np.random.get_state(), global_numpy_random_state)


@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
@pytest.mark.parametrize("invalid_concept_id", [-1, "count", True, False, 0.0, "0", None, []])
def test_samples_reject_invalid_concepts_like_legacy_without_consuming_randomness(
    monkeypatch, dataset_name, invalid_concept_id
):
    """datasetの概念数以上・負・整数以外の概念IDを、乱数を進めずに拒否する。"""
    from federated_drift_experiment.data import streams as legacy_streams_module

    concept_count = get_dataset_definition(dataset_name=dataset_name).concept_count
    if invalid_concept_id == "count":
        invalid_concept_id = concept_count
    numpy_random_generator = np.random.RandomState(0)
    initial_numpy_random_state = numpy_random_generator.get_state()
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
    )
    with pytest.raises((TypeError, ValueError), match="concept_id"):
        sample_generator.generate_sample(concept_id=invalid_concept_id)
    assert_numpy_random_states_equal(numpy_random_generator.get_state(), initial_numpy_random_state)
    # 範囲外の整数は、実旧も拒否する（boolなどの型は、新のほうが厳しい）。
    if type(invalid_concept_id) is int:
        borrow_legacy_numpy_random_generator(monkeypatch, np.random.RandomState(0))
        with pytest.raises(ValueError, match="concept_id"):
            legacy_streams_module.generate_data(invalid_concept_id, dataset=dataset_name)


@pytest.mark.parametrize(("dataset_name", "last_concept_id"), [("sea2", 1), ("sea4", 3)])
def test_sea_generator_accepts_only_the_concepts_of_its_dataset(dataset_name, last_concept_id):
    """sea2とsea4は同じ生成器だが、受け取る概念IDはdatasetの概念数まで。"""
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=np.random.RandomState(0)
    )
    assert type(sample_generator.generate_sample(concept_id=last_concept_id)) is ObservedSample
    with pytest.raises(ValueError, match="concept_id"):
        sample_generator.generate_sample(concept_id=last_concept_id + 1)


@pytest.mark.parametrize("invalid_concept_count", [0, 5, -1, True, 2.0, "2", None])
def test_sea_generator_rejects_invalid_concept_counts(invalid_concept_count):
    """SEAの生成器は、閾値のある1〜4の厳密な整数の概念数だけを受け取る。"""
    with pytest.raises((TypeError, ValueError), match="concept_count"):
        SeaSampleGenerator(
            numpy_random_generator=np.random.RandomState(0),
            concept_count=invalid_concept_count,
        )


class FixedDrawRandomState(np.random.RandomState):
    """決めた特徴と、決めた雑音の乱数を返す。"""

    def __init__(self, *, feature_values, label_noise_draw=None):
        super().__init__(0)
        self.fixed_feature_values = feature_values
        self.label_noise_draw = label_noise_draw

    def uniform(self, lower_bound, upper_bound, *, size):
        assert size == len(self.fixed_feature_values)
        return np.array(self.fixed_feature_values, dtype=np.float64)

    def rand(self):
        assert self.label_noise_draw is not None
        return self.label_noise_draw


@pytest.mark.parametrize(
    ("concept_id", "expected_threshold"), [(0, 9.0), (1, 8.0), (2, 7.0), (3, 9.5)]
)
@pytest.mark.parametrize(
    ("label_noise_draw", "label_is_flipped"),
    [(0.0, True), (np.nextafter(0.10, 0.0), True), (0.10, False), (0.99, False)],
)
def test_sea_labels_use_concept_threshold_before_rounding_and_strict_noise_bound(
    concept_id, expected_threshold, label_noise_draw, label_is_flipped
):
    """閾値ちょうどはラベル1、float32へ丸める前に判定し、雑音は0.10未満のときだけ反転する。"""
    just_above_threshold = float(np.nextafter(expected_threshold, np.inf))
    for second_feature_value, label_before_noise in (
        (expected_threshold, 1),
        (just_above_threshold, 0),
    ):
        sample_generator = SeaSampleGenerator(
            numpy_random_generator=FixedDrawRandomState(
                feature_values=(0.0, second_feature_value, 9.75),
                label_noise_draw=label_noise_draw,
            ),
            concept_count=4,
        )
        observed_sample = sample_generator.generate_sample(concept_id=concept_id)
        assert observed_sample.class_label == (
            1 - label_before_noise if label_is_flipped else label_before_noise
        )
        assert observed_sample.feature_values == (
            0.0,
            expected_threshold,
            9.75,
        )


@pytest.mark.parametrize(("concept_id", "center_x", "radius"), [(0, 0.2, 0.15), (1, 0.6, 0.25)])
def test_circle_labels_are_one_strictly_outside_the_concept_circle(concept_id, center_x, radius):
    """円の外だけがラベル1（円の中と中心は0）。float32へ丸める前に判定する。"""
    for first_feature_value, expected_class_label in (
        (center_x, 0),
        (center_x + radius * 0.5, 0),
        (center_x + radius * 1.5, 1),
        (float(np.nextafter(center_x + radius * 1.5, np.inf)), 1),
    ):
        sample_generator = CircleSampleGenerator(
            numpy_random_generator=FixedDrawRandomState(feature_values=(first_feature_value, 0.5))
        )
        observed_sample = sample_generator.generate_sample(concept_id=concept_id)
        assert observed_sample.class_label == expected_class_label
        assert observed_sample.feature_values == (float(np.float32(first_feature_value)), 0.5)


@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
def test_client_observed_streams_match_legacy_streams_in_client_and_sample_order(
    monkeypatch, dataset_name
):
    """概念列から、client順・標本位置順に生成する（実旧の`build_data_streams`と一致）。"""
    from federated_drift_experiment import config as legacy_config_module
    from federated_drift_experiment.data import streams as legacy_streams_module

    concept_count = get_dataset_definition(dataset_name=dataset_name).concept_count
    concept_schedules = [
        [(client_id + sample_index // 3) % concept_count for sample_index in range(40)]
        for client_id in range(3)
    ]
    numpy_random_generator = np.random.RandomState(5)
    legacy_numpy_random_generator = np.random.RandomState(5)
    borrow_legacy_numpy_random_generator(monkeypatch, legacy_numpy_random_generator)
    monkeypatch.setattr(legacy_config_module, "DATASET", dataset_name)
    legacy_data_streams = legacy_streams_module.build_data_streams(concept_schedules)
    observed_client_streams = build_client_observed_streams(
        evaluation_concept_traces=tuple(
            ClientConceptTrace(
                client_id=client_id, concept_ids_by_sample_index=tuple(concept_schedule)
            )
            for client_id, concept_schedule in enumerate(concept_schedules)
        ),
        sample_generator=create_observed_sample_generator(
            dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
        ),
    )
    assert len(observed_client_streams) == 3
    for client_id, observed_client_stream in enumerate(observed_client_streams):
        assert observed_client_stream.client_id == client_id
        assert len(observed_client_stream.observed_samples) == 40
        for observed_sample, (legacy_feature_tensor, legacy_label_tensor) in zip(
            observed_client_stream.observed_samples, legacy_data_streams[client_id], strict=True
        ):
            assert observed_sample.feature_values == tuple(legacy_feature_tensor.tolist())
            assert observed_sample.class_label == int(legacy_label_tensor.item())
    assert_numpy_random_states_equal(
        numpy_random_generator.get_state(), legacy_numpy_random_generator.get_state()
    )


@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
@pytest.mark.parametrize("random_seed", [0, 17])
@pytest.mark.parametrize("per_client_sample_count", [305, 1500])
@pytest.mark.parametrize("per_eligible_sample_concept_change_probability", [0.015, 1.0])
def test_random_client_concept_traces_match_legacy_for_each_dataset(
    monkeypatch,
    dataset_name,
    random_seed,
    per_client_sample_count,
    per_eligible_sample_concept_change_probability,
):
    """概念数をdatasetの定義から取り、実旧の概念列・生成後の乱数と一致する（sea4は4概念）。"""
    from federated_drift_experiment.data import schedules as legacy_schedules_module

    concept_count = get_dataset_definition(dataset_name=dataset_name).concept_count
    python_random_generator = random.Random(random_seed)
    legacy_python_random_generator = random.Random(random_seed)
    monkeypatch.setattr(legacy_schedules_module, "random", legacy_python_random_generator)
    global_python_random_state = random.getstate()
    legacy_concept_schedules = legacy_schedules_module.make_concept_schedules(
        3,
        per_client_sample_count,
        min_stable_period=100,
        drift_prob=per_eligible_sample_concept_change_probability,
        schedule_type="random",
        dataset=dataset_name,
    )
    evaluation_concept_traces = generate_random_client_concept_traces(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset_name,
            random_seed=random_seed,
            client_count=3,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=50,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=(
                per_eligible_sample_concept_change_probability
            ),
        ),
        python_random_generator=python_random_generator,
    )
    assert [
        list(client_concept_trace.concept_ids_by_sample_index)
        for client_concept_trace in evaluation_concept_traces
    ] == legacy_concept_schedules
    visited_concept_ids = {
        concept_id
        for client_concept_trace in evaluation_concept_traces
        for concept_id in client_concept_trace.concept_ids_by_sample_index
    }
    assert visited_concept_ids <= set(range(concept_count))
    if per_client_sample_count == 1500 and per_eligible_sample_concept_change_probability == 1.0:
        # 全概念を通る条件で照合していることを確かめる。
        assert visited_concept_ids == set(range(concept_count))
    assert python_random_generator.getstate() == legacy_python_random_generator.getstate()
    assert random.getstate() == global_python_random_state
