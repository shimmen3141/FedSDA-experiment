"""SINEデータ供給の固定条件と不変記録を確認する。"""

from dataclasses import FrozenInstanceError
import random

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_generation import (
    generate_random_client_concept_traces,
)
from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
    ObservedSample,
)


@pytest.fixture
def valid_concept_schedule_values():
    """各テストへ独立した正常値の辞書を返す。"""
    return {
        "concept_schedule_strategy": "random_changes_after_minimum_index_gap",
        "minimum_sample_index_gap_before_change_trial": 100,
        "per_eligible_sample_concept_change_probability": 0.01,
    }


@pytest.mark.parametrize("minimum_sample_index_gap_before_change_trial", [0, 1, 100, 10**400])
@pytest.mark.parametrize("per_eligible_sample_concept_change_probability", [0, 0.0, 0.01, 0.5, 1.0, 1])
def test_random_concept_schedule_settings_accept_valid_values(
    valid_concept_schedule_values,
    minimum_sample_index_gap_before_change_trial,
    per_eligible_sample_concept_change_probability,
):
    """位置差と確率の境界を受理し、指定値を変換せず保持する。"""
    valid_concept_schedule_values["minimum_sample_index_gap_before_change_trial"] = (
        minimum_sample_index_gap_before_change_trial
    )
    valid_concept_schedule_values["per_eligible_sample_concept_change_probability"] = (
        per_eligible_sample_concept_change_probability
    )
    settings_instance = RandomConceptScheduleSettings(**valid_concept_schedule_values)
    assert settings_instance.concept_schedule_strategy == "random_changes_after_minimum_index_gap"
    assert settings_instance.minimum_sample_index_gap_before_change_trial is (
        minimum_sample_index_gap_before_change_trial
    )
    assert settings_instance.per_eligible_sample_concept_change_probability is (
        per_eligible_sample_concept_change_probability
    )


@pytest.mark.parametrize(
    "configuration_parameter_name,specified_parameter_value,expected_validation_failure_reason",
    [
        *[("concept_schedule_strategy", specified_parameter_value,
           "文字列でrandom_changes_after_minimum_index_gapのいずれかを指定してください。")
          for specified_parameter_value in (
              "unknown", "random", "RANDOM_CHANGES_AFTER_MINIMUM_INDEX_GAP",
              " random_changes_after_minimum_index_gap", "random_changes_after_minimum_index_gap ",
              True, False, 0, None, [],
          )],
        *[("minimum_sample_index_gap_before_change_trial", specified_parameter_value,
           "整数、0以上を指定してください。bool・数値文字列は受理しません。")
          for specified_parameter_value in (
              -1, -10**400, 0.0, 1.5, True, False, "100", None, [],
              float("nan"), float("inf"), float("-inf"),
          )],
        *[("per_eligible_sample_concept_change_probability", specified_parameter_value,
           "有限の実数（整数も可）、0以上、1以下を指定してください。bool・数値文字列は受理しません。")
          for specified_parameter_value in (
              -0.01, 1.01, -1, 2, 10**400, True, False, "0.01", None, [],
              float("nan"), float("inf"), float("-inf"),
          )],
    ],
)
def test_random_concept_schedule_settings_reject_invalid_values(
    valid_concept_schedule_values, configuration_parameter_name,
    specified_parameter_value, expected_validation_failure_reason,
):
    """不正な指定の項目・元の値・日本語の許容条件を報告する。"""
    valid_concept_schedule_values[configuration_parameter_name] = specified_parameter_value
    with pytest.raises(RunSettingsValidationError) as exception_info:
        RandomConceptScheduleSettings(**valid_concept_schedule_values)
    assert exception_info.value.configuration_parameter_name == configuration_parameter_name
    assert exception_info.value.specified_parameter_value is specified_parameter_value
    assert exception_info.value.validation_failure_reason == expected_validation_failure_reason
    assert configuration_parameter_name in str(exception_info.value)
    assert repr(specified_parameter_value) in str(exception_info.value)
    assert expected_validation_failure_reason in str(exception_info.value)


@pytest.mark.parametrize("missing_parameter_name", [
    "concept_schedule_strategy", "minimum_sample_index_gap_before_change_trial",
    "per_eligible_sample_concept_change_probability",
])
def test_random_concept_schedule_settings_require_all_fields(
    valid_concept_schedule_values, missing_parameter_name,
):
    """どの項目にも既定値を設けず、欠落した項目を示す。"""
    del valid_concept_schedule_values[missing_parameter_name]
    with pytest.raises(TypeError, match=missing_parameter_name):
        RandomConceptScheduleSettings(**valid_concept_schedule_values)


def test_random_concept_schedule_settings_are_frozen_and_keyword_only(
    valid_concept_schedule_values,
):
    """構築後の全項目の変更・削除と位置引数による構築を拒否する。"""
    settings_instance = RandomConceptScheduleSettings(**valid_concept_schedule_values)
    for configuration_parameter_name in valid_concept_schedule_values:
        with pytest.raises(FrozenInstanceError):
            setattr(settings_instance, configuration_parameter_name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(settings_instance, configuration_parameter_name)
    with pytest.raises(TypeError):
        RandomConceptScheduleSettings(*valid_concept_schedule_values.values())


@pytest.mark.parametrize("class_label", [0, 1])
def test_observed_sample_preserves_features_and_binary_label(class_label):
    """観測標本は指定した特徴とクラスだけを持ち、評価用真値を持たない。"""
    feature_values = (0.125, 0.75)
    observed_sample = ObservedSample(feature_values=feature_values, class_label=class_label)
    assert observed_sample.feature_values is feature_values
    assert observed_sample.class_label == class_label
    assert set(observed_sample.__dataclass_fields__) == {"feature_values", "class_label"}
    assert not hasattr(observed_sample, "concept_id")
    assert not hasattr(observed_sample, "concept_ids_by_sample_index")


@pytest.mark.parametrize("client_id", [0, 2, 10**400])
def test_client_data_records_preserve_ids_positions_and_counts(client_id):
    """位置の異なる標本と評価用概念を同じclientに対応付け、順序を保持する。"""
    observed_samples = (
        ObservedSample(feature_values=(0.125, 0.75), class_label=1),
        ObservedSample(feature_values=(0.5, 0.25), class_label=0),
        ObservedSample(feature_values=(0.875, 0.625), class_label=1),
    )
    concept_ids_by_sample_index = (0, 1, 1)
    client_observed_stream = ClientObservedStream(
        client_id=client_id, observed_samples=observed_samples,
    )
    client_concept_trace = ClientConceptTrace(
        client_id=client_id, concept_ids_by_sample_index=concept_ids_by_sample_index,
    )
    assert client_observed_stream.client_id == client_concept_trace.client_id == client_id
    assert client_observed_stream.observed_samples is observed_samples
    assert client_concept_trace.concept_ids_by_sample_index is concept_ids_by_sample_index
    assert len(client_observed_stream.observed_samples) == len(
        client_concept_trace.concept_ids_by_sample_index
    ) == 3
    assert client_observed_stream.observed_samples[1].feature_values == (0.5, 0.25)
    assert client_observed_stream.observed_samples[1].class_label == 0
    assert client_concept_trace.concept_ids_by_sample_index[1] == 1
    assert ClientObservedStream(client_id=client_id, observed_samples=()).observed_samples == ()
    assert ClientConceptTrace(
        client_id=client_id, concept_ids_by_sample_index=(),
    ).concept_ids_by_sample_index == ()


@pytest.mark.parametrize("record_type,record_field_values", [
    (ObservedSample, {"feature_values": (0.125, 0.75), "class_label": 1}),
    (ClientObservedStream, {"client_id": 0, "observed_samples": (
        ObservedSample(feature_values=(0.125, 0.75), class_label=1),
    )}),
    (ClientConceptTrace, {"client_id": 0, "concept_ids_by_sample_index": (0, 1)}),
])
def test_observed_data_records_are_frozen_and_keyword_only(record_type, record_field_values):
    """全フィールドの変更・削除と、tuple内の要素変更を拒否する。"""
    record_instance = record_type(**record_field_values)
    for record_field_name in record_field_values:
        with pytest.raises(FrozenInstanceError):
            setattr(record_instance, record_field_name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(record_instance, record_field_name)
    with pytest.raises(TypeError):
        record_type(*record_field_values.values())
    if record_type is ObservedSample:
        with pytest.raises(TypeError):
            record_instance.feature_values[0] = 0.0
    elif record_type is ClientObservedStream:
        with pytest.raises(TypeError):
            record_instance.observed_samples[0] = None
        with pytest.raises(FrozenInstanceError):
            record_instance.observed_samples[0].class_label = 0
        with pytest.raises(TypeError):
            record_instance.observed_samples[0].feature_values[0] = 0.0
    else:
        with pytest.raises(TypeError):
            record_instance.concept_ids_by_sample_index[0] = 1


@pytest.mark.parametrize("record_type,record_field_values,record_field_name", [
    (ObservedSample, {"feature_values": [0.125, 0.75], "class_label": 1}, "feature_values"),
    (ObservedSample, {"feature_values": ([], 0.75), "class_label": 1}, "feature_values"),
    (ClientObservedStream, {"client_id": 0, "observed_samples": []}, "observed_samples"),
    (ClientObservedStream, {"client_id": 0, "observed_samples": ([],)}, "observed_samples"),
    (ClientConceptTrace, {"client_id": 0, "concept_ids_by_sample_index": [0, 1]},
     "concept_ids_by_sample_index"),
    (ClientConceptTrace, {"client_id": 0, "concept_ids_by_sample_index": ([],)},
     "concept_ids_by_sample_index"),
])
def test_observed_data_records_reject_mutable_collections(
    record_type, record_field_values, record_field_name,
):
    """可変コレクションを保持せず、コピーによる暗黙の受理もしない。"""
    with pytest.raises(TypeError, match=record_field_name):
        record_type(**record_field_values)


@pytest.mark.parametrize("record_field_name,invalid_field_value,expected_exception_type", [
    ("feature_values", (), ValueError),
    ("feature_values", (0.5,), ValueError),
    ("feature_values", (0.5, 0.5, 0.5), ValueError),
    ("feature_values", None, TypeError),
    ("feature_values", (1, 0.5), TypeError),
    ("feature_values", (True, 0.5), TypeError),
    ("feature_values", (0.5, "0.5"), TypeError),
    ("class_label", -1, ValueError),
    ("class_label", 2, ValueError),
    ("class_label", True, TypeError),
    ("class_label", False, TypeError),
    ("class_label", 0.0, TypeError),
    ("class_label", "1", TypeError),
    ("class_label", None, TypeError),
])
def test_observed_sample_rejects_invalid_values(
    record_field_name, invalid_field_value, expected_exception_type,
):
    """2特徴のfloat tupleと厳密な整数の二値クラスだけを受理する。"""
    record_field_values = {"feature_values": (0.125, 0.75), "class_label": 1}
    record_field_values[record_field_name] = invalid_field_value
    with pytest.raises(expected_exception_type, match=record_field_name):
        ObservedSample(**record_field_values)


@pytest.mark.parametrize("record_type,record_field_values,record_field_name,invalid_field_value,expected_exception_type", [
    *[(record_type, record_field_values, "client_id", invalid_field_value, expected_exception_type)
      for record_type, record_field_values in (
          (ClientObservedStream, {"client_id": 0, "observed_samples": ()}),
          (ClientConceptTrace, {"client_id": 0, "concept_ids_by_sample_index": ()}),
      )
      for invalid_field_value, expected_exception_type in (
          (-1, ValueError), (True, TypeError), (False, TypeError),
          (0.0, TypeError), ("0", TypeError), (None, TypeError),
      )],
    *[(ClientObservedStream, {"client_id": 0, "observed_samples": ()},
       "observed_samples", invalid_field_value, TypeError)
      for invalid_field_value in (None, (None,), (1,), ((0.5, 0.5),))],
    *[(ClientConceptTrace, {"client_id": 0, "concept_ids_by_sample_index": ()},
       "concept_ids_by_sample_index", invalid_field_value, expected_exception_type)
      for invalid_field_value, expected_exception_type in (
          (None, TypeError), ((-1,), ValueError), ((2,), ValueError),
          ((True,), TypeError), ((False,), TypeError), ((0.0,), TypeError),
          (("0",), TypeError), ((None,), TypeError),
      )],
])
def test_client_data_records_reject_invalid_ids_and_items(
    record_type, record_field_values, record_field_name, invalid_field_value,
    expected_exception_type,
):
    """非負整数ID、観測標本、厳密な整数の二値概念だけを受理する。"""
    record_field_values = {**record_field_values, record_field_name: invalid_field_value}
    with pytest.raises(expected_exception_type, match=record_field_name):
        record_type(**record_field_values)


@pytest.mark.parametrize("record_type,record_field_values,record_field_name", [
    (record_type, record_field_values, record_field_name)
    for record_type, record_field_values in (
        (ObservedSample, {"feature_values": (0.125, 0.75), "class_label": 1}),
        (ClientObservedStream, {"client_id": 0, "observed_samples": ()}),
        (ClientConceptTrace, {"client_id": 0, "concept_ids_by_sample_index": ()}),
    )
    for record_field_name in record_field_values
])
def test_observed_data_records_require_all_fields(record_type, record_field_values, record_field_name):
    """各記録のどちらのフィールドにも既定値を設けない。"""
    record_field_values = dict(record_field_values)
    del record_field_values[record_field_name]
    with pytest.raises(TypeError, match=record_field_name):
        record_type(**record_field_values)


@pytest.mark.parametrize("random_seed", [0, 17])
@pytest.mark.parametrize("client_count", [1, 3])
@pytest.mark.parametrize("per_client_sample_count", [1, 101, 305, 1500])
@pytest.mark.parametrize("minimum_sample_index_gap_before_change_trial", [0, 100])
@pytest.mark.parametrize("per_eligible_sample_concept_change_probability", [0.0, 0.015, 1.0])
@pytest.mark.parametrize("preparation_random_draw_count", [0, 11])
def test_random_client_concept_traces_match_reference_and_random_state(
    monkeypatch, random_seed, client_count, per_client_sample_count,
    minimum_sample_index_gap_before_change_trial,
    per_eligible_sample_concept_change_probability, preparation_random_draw_count,
):
    """旧系列と生成後の乱数を照合し、各clientの件数・順序を維持する。"""
    from federated_drift_experiment.data import schedules as reference_schedules_module

    experiment_run_conditions = ExperimentRunConditions(
        dataset_name="sine2", random_seed=random_seed, client_count=client_count,
        per_client_sample_count=per_client_sample_count,
        server_aggregation_interval_per_client_samples=50,
    )
    settings_instance = RandomConceptScheduleSettings(
        concept_schedule_strategy="random_changes_after_minimum_index_gap",
        minimum_sample_index_gap_before_change_trial=minimum_sample_index_gap_before_change_trial,
        per_eligible_sample_concept_change_probability=per_eligible_sample_concept_change_probability,
    )
    python_random_generator = random.Random(random_seed)
    for random_draw_index in range(preparation_random_draw_count):
        python_random_generator.random()
    initial_python_random_state = python_random_generator.getstate()
    reference_python_random_generator = random.Random()
    reference_python_random_generator.setstate(initial_python_random_state)
    monkeypatch.setattr(reference_schedules_module, "random", reference_python_random_generator)
    global_python_random_state = random.getstate()
    reference_concept_schedules = reference_schedules_module.make_random_schedules(
        client_count, per_client_sample_count, 2,
        minimum_sample_index_gap_before_change_trial,
        per_eligible_sample_concept_change_probability,
    )
    evaluation_concept_traces = generate_random_client_concept_traces(
        experiment_run_conditions=experiment_run_conditions,
        concept_schedule_settings=settings_instance,
        python_random_generator=python_random_generator,
    )
    assert isinstance(evaluation_concept_traces, tuple)
    assert len(evaluation_concept_traces) == client_count
    for client_id, client_concept_trace in enumerate(evaluation_concept_traces):
        assert isinstance(client_concept_trace, ClientConceptTrace)
        assert client_concept_trace.client_id == client_id
        assert client_concept_trace.concept_ids_by_sample_index == tuple(
            reference_concept_schedules[client_id]
        )
        assert len(client_concept_trace.concept_ids_by_sample_index) == per_client_sample_count
        assert client_concept_trace.concept_ids_by_sample_index[0] == 0
    assert python_random_generator.getstate() == reference_python_random_generator.getstate()
    assert random.getstate() == global_python_random_state


def test_random_client_concept_traces_use_strict_minimum_index_gap():
    """位置差100では101で初めて変化し、次の変化は202になる。"""
    evaluation_concept_traces = generate_random_client_concept_traces(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2", random_seed=0, client_count=3,
            per_client_sample_count=204, server_aggregation_interval_per_client_samples=50,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=1.0,
        ),
        python_random_generator=random.Random(0),
    )
    for client_concept_trace in evaluation_concept_traces:
        assert client_concept_trace.concept_ids_by_sample_index == (0,) * 101 + (1,) * 101 + (0,) * 2


def test_random_client_concept_traces_consume_eligible_trials_with_zero_probability():
    """確率0でも試行乱数を消費し、借りた乱数以外は変更しない。"""
    python_random_generator = random.Random(0)
    expected_python_random_generator = random.Random(0)
    global_python_random_state = random.getstate()
    evaluation_concept_traces = generate_random_client_concept_traces(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2", random_seed=0, client_count=3,
            per_client_sample_count=1500, server_aggregation_interval_per_client_samples=50,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=0.0,
        ),
        python_random_generator=python_random_generator,
    )
    eligible_sample_trial_count = 3 * (1500 - 101)
    for random_draw_index in range(eligible_sample_trial_count):
        expected_python_random_generator.random()
    assert all(
        client_concept_trace.concept_ids_by_sample_index == (0,) * 1500
        for client_concept_trace in evaluation_concept_traces
    )
    assert python_random_generator.getstate() == expected_python_random_generator.getstate()
    assert random.getstate() == global_python_random_state


def test_random_client_concept_trace_generation_requires_keyword_arguments():
    """公開関数は設定と借りる乱数をkeywordで明示する。"""
    experiment_run_conditions = ExperimentRunConditions(
        dataset_name="sine2", random_seed=0, client_count=1,
        per_client_sample_count=1, server_aggregation_interval_per_client_samples=50,
    )
    settings_instance = RandomConceptScheduleSettings(
        concept_schedule_strategy="random_changes_after_minimum_index_gap",
        minimum_sample_index_gap_before_change_trial=100,
        per_eligible_sample_concept_change_probability=0.015,
    )
    with pytest.raises(TypeError):
        generate_random_client_concept_traces(
            experiment_run_conditions, settings_instance, random.Random(0),
        )
