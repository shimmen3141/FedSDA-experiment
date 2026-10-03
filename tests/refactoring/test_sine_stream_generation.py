"""SINEデータ供給の固定条件を確認する。"""

from dataclasses import FrozenInstanceError

import pytest

from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
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
