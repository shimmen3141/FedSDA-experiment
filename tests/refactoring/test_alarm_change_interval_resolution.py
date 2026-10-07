"""警報の変化区間の解決を、結果recordと実旧の警報処理の両方で検証する。"""

from dataclasses import FrozenInstanceError, fields

import pytest

from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
)
from federated_learning_experiments.runtime import (
    alarm_change_interval_resolution as resolution_module,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES,
    AlarmChangeIntervalResolution,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)


def test_alarm_change_interval_resolution_is_immutable():
    assert ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES == (
        "alarm_interval_held_model_reused",
        "alarm_interval_current_model_maintained",
        "alarm_interval_candidate_validation_started",
    )
    assert type(ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES) is tuple
    assert tuple(field.name for field in fields(AlarmChangeIntervalResolution)) == (
        "resolution_outcome",
        "alarm_interval_reuse_assessment",
        "assigned_model_id",
        "training_model_assignment_change",
        "started_validation_session",
    )
    assert all(field.kw_only for field in fields(AlarmChangeIntervalResolution))
    alarm_interval_reuse_assessment = AlarmIntervalModelReuseAssessment(
        baseline_supported_interval_mean_losses_by_model_id=((4, 0.5), (-103, 0.25)),
        reusable_mean_losses_by_model_id=((-103, 0.25),),
    )
    training_model_assignment_change = TrainingModelAssignmentChange(
        previous_model_id=4, current_model_id=-103
    )
    started_validation_session = object.__new__(PostAlarmCandidateValidationSession)
    # 3つの結果種別それぞれのfieldの組を、そのまま保持する。
    for (
        resolution_outcome,
        assigned_model_id,
        field_value,
        expected_started_validation_session,
    ) in (
        ("alarm_interval_held_model_reused", -103, training_model_assignment_change, None),
        ("alarm_interval_current_model_maintained", 4, None, None),
        ("alarm_interval_candidate_validation_started", None, None, started_validation_session),
    ):
        alarm_change_interval_resolution = AlarmChangeIntervalResolution(
            resolution_outcome=resolution_outcome,
            alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
            assigned_model_id=assigned_model_id,
            training_model_assignment_change=field_value,
            started_validation_session=expected_started_validation_session,
        )
        assert alarm_change_interval_resolution.resolution_outcome == resolution_outcome
        assert (
            alarm_change_interval_resolution.alarm_interval_reuse_assessment
            is alarm_interval_reuse_assessment
        )
        assert alarm_change_interval_resolution.assigned_model_id == assigned_model_id
        assert alarm_change_interval_resolution.training_model_assignment_change is field_value
        assert (
            alarm_change_interval_resolution.started_validation_session
            is expected_started_validation_session
        )
        for field_name in (
            "resolution_outcome",
            "alarm_interval_reuse_assessment",
            "assigned_model_id",
            "training_model_assignment_change",
            "started_validation_session",
        ):
            with pytest.raises(FrozenInstanceError):
                setattr(alarm_change_interval_resolution, field_name, None)
    # 正式値以外の結果種別（候補検証の確定処理の値や旧のaction名を含む）は拒否する。
    for invalid_case in (
        "held_reference_model_reused",
        "current_model_maintained",
        "candidate_validation_started",
        "reuse",
        "",
        None,
        1,
    ):
        with pytest.raises(ValueError):
            AlarmChangeIntervalResolution(
                resolution_outcome=invalid_case,
                alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
                assigned_model_id=4,
                training_model_assignment_change=None,
                started_validation_session=None,
            )
    # 全fieldがkeyword必須。
    with pytest.raises(TypeError):
        AlarmChangeIntervalResolution(
            "alarm_interval_current_model_maintained",
            alarm_interval_reuse_assessment,
            4,
            None,
            None,
        )
    with pytest.raises(TypeError):
        AlarmChangeIntervalResolution(
            resolution_outcome="alarm_interval_current_model_maintained",
            alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
            assigned_model_id=4,
        )
    assert not hasattr(resolution_module, "__all__")
