"""新しい実験設定の検証契約を確認する。"""

import ast
import sys
from dataclasses import MISSING, FrozenInstanceError, dataclass, field, fields, replace
from importlib.util import resolve_name
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.configuration.run_settings import (
    ValidatedExperimentRunSettingsSubset,
)
from federated_learning_experiments.configuration.run_settings_validation import (
    validate_experiment_run_settings,
)
from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_consolidation_settings import (
    ModelConsolidationSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)


@pytest.fixture
def valid_run_settings_mapping():
    """初回の有効な機能設定を揃え、実験ごとに独立した辞書を返す。"""
    return {
        "method_name": "fedsda",
        "experiment_run_conditions": ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=1,
            per_client_sample_count=10,
            server_aggregation_interval_per_client_samples=5,
        ),
        "model_architecture_settings": ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        "loss_change_detection_settings": LossChangeDetectionSettings(
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=0.05,
        ),
        "prediction_combination_settings": PredictionCombinationSettings(
            prediction_combination_strategy="fixed_share_weighted_prediction",
            prediction_mixture_activation_policy="always",
            prediction_weight_recalibration_after_aggregation_policy=(
                "recompute_buffer_losses_and_replay_weight_updates"
            ),
            prediction_state_reset_on_training_assignment_change_policy=(
                "restart_adahedge_preserve_fixed_share_prediction_state"
            ),
            fixed_share_weight_redistribution_time_scale_samples=2,
        ),
        "local_training_settings": LocalTrainingSettings(
            local_model_parameter_update_strategy=("joint_backbone_adapter_and_head_training"),
            shared_backbone_gradient_combination_strategy=(
                "sample_weighted_mean_per_concept_gradients"
            ),
        ),
        "training_data_assignment_settings": TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=2,
        ),
        "candidate_model_training_and_acceptance_settings": (
            CandidateModelTrainingAndAcceptanceSettings(
                candidate_model_acceptance_policy=(
                    "current_model_first_reuse_then_two_segment_candidate_validation"
                ),
                candidate_post_alarm_validation_sample_count=3,
            )
        ),
        "model_consolidation_settings": ModelConsolidationSettings(
            model_clustering_trigger_policy="on_new_model_registration",
            model_pair_comparison_strategy=("classwise_unique_correctness_lower_confidence_bound"),
            model_clustering_linkage="average_linkage",
            model_consolidation_policy="weighted_parameter_average_and_merge_ids",
        ),
    }


@pytest.mark.parametrize(
    "configuration_parameter_name,specified_parameter_value",
    [
        *[(None, specified_parameter_value) for specified_parameter_value in (2, 3, 100)],
        *[
            (configuration_parameter_name, specified_parameter_value)
            for configuration_parameter_name in (
                "method_name",
                "experiment_run_conditions",
                "model_architecture_settings",
                "loss_change_detection_settings",
                "prediction_combination_settings",
                "local_training_settings",
                "training_data_assignment_settings",
                "candidate_model_training_and_acceptance_settings",
                "model_consolidation_settings",
            )
            for specified_parameter_value in (None, False)
        ],
        *[
            ("method_name", specified_parameter_value)
            for specified_parameter_value in (
                "unknown",
                "FedSDA",
                " fedsda",
                "fedsda_residual_adapter_switching",
            )
        ],
        *[
            (
                "training_data_assignment_settings",
                TrainingDataAssignmentSettings(
                    pending_assignment_buffer_capacity_samples=specified_parameter_value,
                ),
            )
            for specified_parameter_value in (1, 3, 100)
        ],
    ],
)
def test_direct_subset_construction_uses_the_same_validation(
    valid_run_settings_mapping,
    configuration_parameter_name,
    specified_parameter_value,
):
    """直接構築でも同じ方式・機能型・組合せを受理または拒否する。"""
    unvalidated_run_settings = valid_run_settings_mapping.copy()
    if configuration_parameter_name is None:
        unvalidated_run_settings["prediction_combination_settings"] = replace(
            unvalidated_run_settings["prediction_combination_settings"],
            fixed_share_weight_redistribution_time_scale_samples=specified_parameter_value,
        )
        unvalidated_run_settings["training_data_assignment_settings"] = (
            TrainingDataAssignmentSettings(
                pending_assignment_buffer_capacity_samples=specified_parameter_value,
            )
        )
        assert validate_experiment_run_settings(unvalidated_run_settings) is None
        validated_settings_subset = ValidatedExperimentRunSettingsSubset(
            **unvalidated_run_settings,
        )
        assert tuple(
            settings_field.name for settings_field in fields(validated_settings_subset)
        ) == (tuple(valid_run_settings_mapping))
        assert get_type_hints(ValidatedExperimentRunSettingsSubset) == {
            component_settings_name: type(component_settings)
            for component_settings_name, component_settings in unvalidated_run_settings.items()
        }
        for settings_field in fields(validated_settings_subset):
            assert settings_field.kw_only
            assert settings_field.default is MISSING
            assert settings_field.default_factory is MISSING
            assert (
                getattr(validated_settings_subset, settings_field.name)
                is (unvalidated_run_settings[settings_field.name])
            )
            valid_settings_values = unvalidated_run_settings.copy()
            del valid_settings_values[settings_field.name]
            with pytest.raises(TypeError, match=settings_field.name):
                ValidatedExperimentRunSettingsSubset(**valid_settings_values)
        with pytest.raises(TypeError):
            ValidatedExperimentRunSettingsSubset(*unvalidated_run_settings.values())
        return

    unvalidated_run_settings[configuration_parameter_name] = specified_parameter_value
    with pytest.raises(RunSettingsValidationError) as validation_error:
        validate_experiment_run_settings(unvalidated_run_settings)
    expected_failure_reason = validation_error.value.validation_failure_reason
    if isinstance(specified_parameter_value, TrainingDataAssignmentSettings):
        configuration_parameter_name = "fixed_share_weight_redistribution_time_scale_samples"
        specified_parameter_value = 2
    with pytest.raises(RunSettingsValidationError) as validation_error:
        ValidatedExperimentRunSettingsSubset(**unvalidated_run_settings)
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert validation_error.value.validation_failure_reason == expected_failure_reason


def test_subset_does_not_share_mutable_input_mapping(valid_run_settings_mapping):
    """入力辞書の差替えや削除は、構築済みの不変条件へ波及しない。"""
    unvalidated_run_settings = valid_run_settings_mapping.copy()
    validated_settings_subset = ValidatedExperimentRunSettingsSubset(
        **unvalidated_run_settings,
    )
    assert unvalidated_run_settings == valid_run_settings_mapping
    for configuration_parameter_name, component_settings in valid_run_settings_mapping.items():
        assert unvalidated_run_settings[configuration_parameter_name] is component_settings
        unvalidated_run_settings[configuration_parameter_name] = None
        assert (
            getattr(validated_settings_subset, configuration_parameter_name) is component_settings
        )
        with pytest.raises(FrozenInstanceError):
            setattr(validated_settings_subset, configuration_parameter_name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(validated_settings_subset, configuration_parameter_name)
    unvalidated_run_settings.clear()
    for configuration_parameter_name, component_settings in valid_run_settings_mapping.items():
        assert (
            getattr(validated_settings_subset, configuration_parameter_name) is component_settings
        )


@pytest.mark.parametrize("specified_parameter_value", [2, 3, 100])
def test_run_settings_validation_accepts_valid_component_mapping(
    valid_run_settings_mapping,
    specified_parameter_value,
):
    valid_run_settings_mapping["prediction_combination_settings"] = replace(
        valid_run_settings_mapping["prediction_combination_settings"],
        fixed_share_weight_redistribution_time_scale_samples=specified_parameter_value,
    )
    valid_run_settings_mapping["training_data_assignment_settings"] = (
        TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=specified_parameter_value,
        )
    )
    unvalidated_run_settings = MappingProxyType(valid_run_settings_mapping.copy())
    assert validate_experiment_run_settings(unvalidated_run_settings) is None
    assert dict(unvalidated_run_settings) == valid_run_settings_mapping
    assert all(
        unvalidated_run_settings[configuration_parameter_name] is component_settings
        for configuration_parameter_name, component_settings in valid_run_settings_mapping.items()
    )


@pytest.mark.parametrize(
    "configuration_parameter_name",
    [
        "method_name",
        "experiment_run_conditions",
        "model_architecture_settings",
        "loss_change_detection_settings",
        "prediction_combination_settings",
        "local_training_settings",
        "training_data_assignment_settings",
        "candidate_model_training_and_acceptance_settings",
        "model_consolidation_settings",
        "mode",
        "routing_settings",
        "unknown",
    ],
)
@pytest.mark.parametrize(
    "specified_parameter_value",
    [
        MISSING,
        None,
        {},
        [],
        True,
        1,
        "fedsda",
        " fedsda",
        "unknown",
        "fedsda_residual_adapter_switching",
    ],
)
def test_run_settings_validation_rejects_invalid_component_mapping(
    valid_run_settings_mapping,
    configuration_parameter_name,
    specified_parameter_value,
):
    if configuration_parameter_name == "method_name" and specified_parameter_value == "fedsda":
        specified_parameter_value = "FedSDA"
    if (
        configuration_parameter_name not in valid_run_settings_mapping
        and specified_parameter_value is MISSING
    ):
        specified_parameter_value = None
    unvalidated_run_settings = valid_run_settings_mapping.copy()
    if specified_parameter_value is MISSING:
        del unvalidated_run_settings[configuration_parameter_name]
    else:
        unvalidated_run_settings[configuration_parameter_name] = specified_parameter_value
    with pytest.raises(RunSettingsValidationError) as validation_error:
        validate_experiment_run_settings(MappingProxyType(unvalidated_run_settings))
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name
    assert validation_error.value.specified_parameter_value is (
        None if specified_parameter_value is MISSING else specified_parameter_value
    )
    assert validation_error.value.validation_failure_reason
    assert valid_run_settings_mapping["method_name"] == "fedsda"


@pytest.mark.parametrize("specified_parameter_value", [1, 3, 100])
def test_fixed_share_time_scale_must_match_assignment_buffer_capacity(
    valid_run_settings_mapping,
    specified_parameter_value,
):
    unvalidated_run_settings = valid_run_settings_mapping.copy()
    unvalidated_run_settings["training_data_assignment_settings"] = TrainingDataAssignmentSettings(
        pending_assignment_buffer_capacity_samples=specified_parameter_value,
    )
    with pytest.raises(RunSettingsValidationError) as validation_error:
        validate_experiment_run_settings(unvalidated_run_settings)
    assert validation_error.value.configuration_parameter_name == (
        "fixed_share_weight_redistribution_time_scale_samples"
    )
    assert validation_error.value.specified_parameter_value == 2
    assert "pending_assignment_buffer_capacity_samples" in (
        validation_error.value.validation_failure_reason
    )
    assert str(specified_parameter_value) in validation_error.value.validation_failure_reason
    assert (
        unvalidated_run_settings[
            "training_data_assignment_settings"
        ].pending_assignment_buffer_capacity_samples
        == specified_parameter_value
    )


@pytest.mark.parametrize(
    "configuration_parameter_name",
    [
        "method_name",
        "experiment_run_conditions",
        "model_architecture_settings",
        "loss_change_detection_settings",
        "prediction_combination_settings",
        "local_training_settings",
        "training_data_assignment_settings",
        "candidate_model_training_and_acceptance_settings",
        "model_consolidation_settings",
        "aaa_unknown",
    ],
)
@pytest.mark.parametrize("specified_parameter_value", [False, True])
def test_validation_reports_failures_in_declaration_order(
    valid_run_settings_mapping,
    configuration_parameter_name,
    specified_parameter_value,
):
    unvalidated_run_settings = valid_run_settings_mapping.copy()
    if configuration_parameter_name == "aaa_unknown":
        unvalidated_run_settings.update({"zzz_unknown": 1, "aaa_unknown": 2})
    else:
        for component_settings_name in tuple(unvalidated_run_settings)[
            tuple(unvalidated_run_settings).index(configuration_parameter_name) :
        ]:
            unvalidated_run_settings[component_settings_name] = None
    if specified_parameter_value:
        unvalidated_run_settings = dict(reversed(tuple(unvalidated_run_settings.items())))
    with pytest.raises(RunSettingsValidationError) as validation_error:
        validate_experiment_run_settings(unvalidated_run_settings)
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name


@pytest.mark.parametrize(
    (
        "configuration_parameter_name",
        "specified_parameter_value",
        "expected_failure_reason",
    ),
    [
        ("client_count", -1, "1以上の整数を指定してください。"),
        ("dataset_name", "unknown", "sine2・sea2・mnist2のいずれかを指定してください。"),
        ("client_count", [1], "リストではなく1以上の整数を指定してください。"),
    ],
)
def test_validation_error_preserves_parameter_details(
    configuration_parameter_name,
    specified_parameter_value,
    expected_failure_reason,
):
    """項目・元の指定値・日本語の許容条件を例外から取得できる。"""
    with pytest.raises(ValueError) as validation_error:
        raise RunSettingsValidationError(
            configuration_parameter_name=configuration_parameter_name,
            specified_parameter_value=specified_parameter_value,
            validation_failure_reason=expected_failure_reason,
        )

    assert isinstance(validation_error.value, RunSettingsValidationError)
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert validation_error.value.validation_failure_reason == expected_failure_reason
    assert configuration_parameter_name in str(validation_error.value)
    assert repr(specified_parameter_value) in str(validation_error.value)
    assert expected_failure_reason in str(validation_error.value)


@pytest.mark.parametrize("specified_parameter_value", ["sine2", "sea2", "mnist2"])
def test_existing_dataset_identifiers_are_accepted(specified_parameter_value):
    settings_instance = ExperimentRunConditions(
        dataset_name=specified_parameter_value,
        random_seed=0,
        client_count=1,
        per_client_sample_count=1,
        server_aggregation_interval_per_client_samples=1,
    )
    assert settings_instance.dataset_name == specified_parameter_value


@pytest.mark.parametrize(
    "specified_parameter_value",
    [
        "unknown",
        "mnist4",
        "sine_two_concepts",
        "sea_two_concepts",
        "mnist_two_concepts",
        "SINE2",
        "SEA2",
        "MNIST2",
        " sine2",
        True,
        1,
        None,
        [],
    ],
)
def test_unknown_dataset_identifiers_are_rejected(specified_parameter_value):
    with pytest.raises(RunSettingsValidationError) as validation_error:
        ExperimentRunConditions(
            dataset_name=specified_parameter_value,
            random_seed=0,
            client_count=1,
            per_client_sample_count=1,
            server_aggregation_interval_per_client_samples=1,
        )
    assert validation_error.value.configuration_parameter_name == "dataset_name"
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "文字列" in validation_error.value.validation_failure_reason
    assert all(
        value in validation_error.value.validation_failure_reason
        for value in ("sine2", "sea2", "mnist2")
    )


@pytest.mark.parametrize(
    "settings_type,valid_settings_values,configuration_parameter_name",
    [
        (
            ExperimentRunConditions,
            dict(
                dataset_name="sine2",
                random_seed=0,
                client_count=1,
                per_client_sample_count=1,
                server_aggregation_interval_per_client_samples=1,
            ),
            configuration_parameter_name,
        )
        for configuration_parameter_name in (
            "random_seed",
            "client_count",
            "per_client_sample_count",
            "server_aggregation_interval_per_client_samples",
        )
    ]
    + [
        (
            ModelArchitectureSettings,
            dict(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=1,
            ),
            "residual_adapter_requested_rank",
        ),
        (
            PredictionCombinationSettings,
            dict(
                prediction_combination_strategy="fixed_share_weighted_prediction",
                prediction_mixture_activation_policy="always",
                prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
                prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
                fixed_share_weight_redistribution_time_scale_samples=2,
            ),
            "fixed_share_weight_redistribution_time_scale_samples",
        ),
        (
            TrainingDataAssignmentSettings,
            dict(
                pending_assignment_buffer_capacity_samples=1,
            ),
            "pending_assignment_buffer_capacity_samples",
        ),
        (
            CandidateModelTrainingAndAcceptanceSettings,
            dict(
                candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
                candidate_post_alarm_validation_sample_count=2,
            ),
            "candidate_post_alarm_validation_sample_count",
        ),
    ],
)
@pytest.mark.parametrize("specified_parameter_value", [True, False, "1", 1.0, None, [], -1])
def test_integer_settings_reject_invalid_types_and_ranges(
    settings_type,
    valid_settings_values,
    configuration_parameter_name,
    specified_parameter_value,
):
    valid_settings_values = dict(valid_settings_values)
    valid_settings_values[configuration_parameter_name] = specified_parameter_value
    with pytest.raises(RunSettingsValidationError) as validation_error:
        settings_type(**valid_settings_values)
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "整数" in validation_error.value.validation_failure_reason
    assert (
        "0以上"
        if configuration_parameter_name == "random_seed"
        else "2以上"
        if configuration_parameter_name
        in (
            "fixed_share_weight_redistribution_time_scale_samples",
            "candidate_post_alarm_validation_sample_count",
        )
        else "1以上"
    ) in (validation_error.value.validation_failure_reason)


@pytest.mark.parametrize(
    "settings_type,valid_settings_values,configuration_parameter_name",
    [
        (
            ExperimentRunConditions,
            dict(
                dataset_name="sine2",
                random_seed=0,
                client_count=1,
                per_client_sample_count=1,
                server_aggregation_interval_per_client_samples=1,
            ),
            configuration_parameter_name,
        )
        for configuration_parameter_name in (
            "random_seed",
            "client_count",
            "per_client_sample_count",
            "server_aggregation_interval_per_client_samples",
        )
    ]
    + [
        (
            ModelArchitectureSettings,
            dict(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=1,
            ),
            "residual_adapter_requested_rank",
        ),
        (
            PredictionCombinationSettings,
            dict(
                prediction_combination_strategy="fixed_share_weighted_prediction",
                prediction_mixture_activation_policy="always",
                prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
                prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
                fixed_share_weight_redistribution_time_scale_samples=2,
            ),
            "fixed_share_weight_redistribution_time_scale_samples",
        ),
        (
            TrainingDataAssignmentSettings,
            dict(
                pending_assignment_buffer_capacity_samples=1,
            ),
            "pending_assignment_buffer_capacity_samples",
        ),
        (
            CandidateModelTrainingAndAcceptanceSettings,
            dict(
                candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
                candidate_post_alarm_validation_sample_count=2,
            ),
            "candidate_post_alarm_validation_sample_count",
        ),
    ],
)
@pytest.mark.parametrize("specified_parameter_value", [0, 1, 2, 10**400])
def test_integer_settings_validate_boundary_values(
    settings_type,
    valid_settings_values,
    configuration_parameter_name,
    specified_parameter_value,
):
    valid_settings_values = dict(valid_settings_values)
    valid_settings_values[configuration_parameter_name] = specified_parameter_value
    if (
        configuration_parameter_name != "random_seed"
        and specified_parameter_value == 0
        or configuration_parameter_name
        in (
            "fixed_share_weight_redistribution_time_scale_samples",
            "candidate_post_alarm_validation_sample_count",
        )
        and specified_parameter_value == 1
    ):
        with pytest.raises(RunSettingsValidationError) as validation_error:
            settings_type(**valid_settings_values)
        assert validation_error.value.configuration_parameter_name == configuration_parameter_name
        assert validation_error.value.specified_parameter_value is specified_parameter_value
    else:
        settings_instance = settings_type(**valid_settings_values)
        assert getattr(settings_instance, configuration_parameter_name) is specified_parameter_value


def test_aggregation_interval_can_exceed_stream_length():
    settings_instance = ExperimentRunConditions(
        dataset_name="sea2",
        random_seed=0,
        client_count=1,
        per_client_sample_count=1,
        server_aggregation_interval_per_client_samples=100,
    )
    assert settings_instance.server_aggregation_interval_per_client_samples == 100


@pytest.mark.parametrize(
    "settings_type,valid_settings_values",
    [
        (
            ExperimentRunConditions,
            dict(
                dataset_name="mnist2",
                random_seed=0,
                client_count=1,
                per_client_sample_count=1,
                server_aggregation_interval_per_client_samples=1,
            ),
        ),
        (
            ModelArchitectureSettings,
            dict(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=1,
            ),
        ),
        *[
            (
                LossChangeDetectionSettings,
                dict(
                    drift_detector_name="e_sr",
                    loss_monitoring_scope="overall_and_true_class_losses",
                    e_sr_false_alarm_control_alpha=specified_parameter_value,
                ),
            )
            for specified_parameter_value in (0.05, 5e-324, 0.9999999999999999)
        ],
        (
            PredictionCombinationSettings,
            dict(
                prediction_combination_strategy="fixed_share_weighted_prediction",
                prediction_mixture_activation_policy="always",
                prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
                prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
                fixed_share_weight_redistribution_time_scale_samples=2,
            ),
        ),
        (
            TrainingDataAssignmentSettings,
            dict(
                pending_assignment_buffer_capacity_samples=1,
            ),
        ),
        (
            LocalTrainingSettings,
            dict(
                local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
            ),
        ),
        (
            CandidateModelTrainingAndAcceptanceSettings,
            dict(
                candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
                candidate_post_alarm_validation_sample_count=2,
            ),
        ),
        (
            ModelConsolidationSettings,
            dict(
                model_clustering_trigger_policy="on_new_model_registration",
                model_pair_comparison_strategy="classwise_unique_correctness_lower_confidence_bound",
                model_clustering_linkage="average_linkage",
                model_consolidation_policy="weighted_parameter_average_and_merge_ids",
            ),
        ),
    ],
)
def test_settings_instances_are_immutable(settings_type, valid_settings_values):
    settings_instance = settings_type(**valid_settings_values)
    for configuration_parameter_name in valid_settings_values:
        with pytest.raises(TypeError, match=configuration_parameter_name):
            settings_type(
                **{
                    name: value
                    for name, value in valid_settings_values.items()
                    if name != configuration_parameter_name
                }
            )
    with pytest.raises(TypeError, match="unknown"):
        settings_type(**valid_settings_values, unknown="unknown")
    for settings_field in fields(settings_instance):
        assert (
            getattr(settings_instance, settings_field.name)
            is valid_settings_values[settings_field.name]
        )
        with pytest.raises(FrozenInstanceError):
            setattr(settings_instance, settings_field.name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(settings_instance, settings_field.name)


@pytest.mark.parametrize(
    "settings_type",
    [
        ExperimentRunConditions,
        ModelArchitectureSettings,
        LossChangeDetectionSettings,
        PredictionCombinationSettings,
        TrainingDataAssignmentSettings,
        LocalTrainingSettings,
        CandidateModelTrainingAndAcceptanceSettings,
        ModelConsolidationSettings,
    ],
)
def test_field_annotations_and_metadata_declare_parameter_constraints(settings_type):
    assert get_type_hints(settings_type) == (
        {
            "dataset_name": str,
            "random_seed": int,
            "client_count": int,
            "per_client_sample_count": int,
            "server_aggregation_interval_per_client_samples": int,
        }
        if settings_type is ExperimentRunConditions
        else {
            "model_architecture_name": str,
            "residual_adapter_requested_rank": int,
        }
        if settings_type is ModelArchitectureSettings
        else {
            "drift_detector_name": str,
            "loss_monitoring_scope": str,
            "e_sr_false_alarm_control_alpha": float,
        }
        if settings_type is LossChangeDetectionSettings
        else {
            "prediction_combination_strategy": str,
            "prediction_mixture_activation_policy": str,
            "prediction_weight_recalibration_after_aggregation_policy": str,
            "prediction_state_reset_on_training_assignment_change_policy": str,
            "fixed_share_weight_redistribution_time_scale_samples": int,
        }
        if settings_type is PredictionCombinationSettings
        else {
            "pending_assignment_buffer_capacity_samples": int,
        }
        if settings_type is TrainingDataAssignmentSettings
        else {
            "local_model_parameter_update_strategy": str,
            "shared_backbone_gradient_combination_strategy": str,
        }
        if settings_type is LocalTrainingSettings
        else {
            "candidate_model_acceptance_policy": str,
            "candidate_post_alarm_validation_sample_count": int,
        }
        if settings_type is CandidateModelTrainingAndAcceptanceSettings
        else {
            "model_clustering_trigger_policy": str,
            "model_pair_comparison_strategy": str,
            "model_clustering_linkage": str,
            "model_consolidation_policy": str,
        }
    )
    for settings_field in fields(settings_type):
        assert settings_field.kw_only
        assert settings_field.default is MISSING
        assert settings_field.default_factory is MISSING
        if get_type_hints(settings_type)[settings_field.name] is str:
            assert settings_field.metadata["allowed_parameter_values"] == (
                ("sine2", "sea2", "sea4", "circle2", "mnist2")
                if settings_field.name == "dataset_name"
                else ("shared_backbone_residual_adapter",)
                if settings_field.name == "model_architecture_name"
                else ("e_sr",)
                if settings_field.name == "drift_detector_name"
                else ("overall_and_true_class_losses",)
                if settings_field.name == "loss_monitoring_scope"
                else ("fixed_share_weighted_prediction",)
                if settings_field.name == "prediction_combination_strategy"
                else ("always",)
                if settings_field.name == "prediction_mixture_activation_policy"
                else ("recompute_buffer_losses_and_replay_weight_updates",)
                if settings_field.name == "prediction_weight_recalibration_after_aggregation_policy"
                else ("joint_backbone_adapter_and_head_training",)
                if settings_field.name == "local_model_parameter_update_strategy"
                else ("sample_weighted_mean_per_concept_gradients",)
                if settings_field.name == "shared_backbone_gradient_combination_strategy"
                else ("current_model_first_reuse_then_two_segment_candidate_validation",)
                if settings_field.name == "candidate_model_acceptance_policy"
                else ("on_new_model_registration",)
                if settings_field.name == "model_clustering_trigger_policy"
                else ("classwise_unique_correctness_lower_confidence_bound",)
                if settings_field.name == "model_pair_comparison_strategy"
                else ("average_linkage",)
                if settings_field.name == "model_clustering_linkage"
                else ("weighted_parameter_average_and_merge_ids",)
                if settings_field.name == "model_consolidation_policy"
                else ("restart_adahedge_preserve_fixed_share_prediction_state",)
            )
        elif settings_field.name == "e_sr_false_alarm_control_alpha":
            assert settings_field.metadata["parameter_unit"] == "dimensionless"
            assert settings_field.metadata["minimum_allowed_value"] == 0
            assert settings_field.metadata["minimum_value_is_inclusive"] is False
            assert settings_field.metadata["maximum_allowed_value"] == 1
            assert settings_field.metadata["maximum_value_is_inclusive"] is False
        else:
            assert settings_field.metadata["minimum_allowed_value"] == (
                0
                if settings_field.name == "random_seed"
                else 2
                if settings_field.name
                in (
                    "fixed_share_weight_redistribution_time_scale_samples",
                    "candidate_post_alarm_validation_sample_count",
                )
                else 1
            )
            assert settings_field.metadata["minimum_value_is_inclusive"] is True
            assert "maximum_allowed_value" not in settings_field.metadata
            assert isinstance(settings_field.metadata["parameter_unit"], str)
            assert settings_field.metadata["parameter_unit"]
            if settings_field.name == "residual_adapter_requested_rank":
                assert settings_field.metadata["parameter_unit"] == "rank"
            if settings_field.name in (
                "fixed_share_weight_redistribution_time_scale_samples",
                "pending_assignment_buffer_capacity_samples",
                "candidate_post_alarm_validation_sample_count",
            ):
                assert settings_field.metadata["parameter_unit"] == "sample/client"
    with pytest.raises(TypeError):
        settings_type()
    with pytest.raises(TypeError):
        if settings_type is ExperimentRunConditions:
            settings_type("sine2", 0, 1, 1, 1)
        elif settings_type is ModelArchitectureSettings:
            settings_type("shared_backbone_residual_adapter", 1)
        elif settings_type is LossChangeDetectionSettings:
            settings_type("e_sr", "overall_and_true_class_losses", 0.05)
        elif settings_type is TrainingDataAssignmentSettings:
            settings_type(1)
        elif settings_type is LocalTrainingSettings:
            settings_type(
                "joint_backbone_adapter_and_head_training",
                "sample_weighted_mean_per_concept_gradients",
            )
        elif settings_type is CandidateModelTrainingAndAcceptanceSettings:
            settings_type("current_model_first_reuse_then_two_segment_candidate_validation", 2)
        elif settings_type is ModelConsolidationSettings:
            settings_type(
                "on_new_model_registration",
                "classwise_unique_correctness_lower_confidence_bound",
                "average_linkage",
                "weighted_parameter_average_and_merge_ids",
            )
        else:
            settings_type(
                "fixed_share_weighted_prediction",
                "always",
                "recompute_buffer_losses_and_replay_weight_updates",
                "restart_adahedge_preserve_fixed_share_prediction_state",
                2,
            )


@pytest.mark.parametrize("specified_parameter_value", [1, 2, 1000, 10**400])
def test_requested_adapter_rank_is_preserved(specified_parameter_value):
    """構造の正式名を受理し、要求rankを丸めず保持する。"""
    settings_instance = ModelArchitectureSettings(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=specified_parameter_value,
    )
    assert settings_instance.model_architecture_name == "shared_backbone_residual_adapter"
    assert settings_instance.residual_adapter_requested_rank is specified_parameter_value


@pytest.mark.parametrize(
    "settings_type,valid_settings_values,configuration_parameter_name",
    [
        (
            ModelArchitectureSettings,
            dict(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=1,
            ),
            "model_architecture_name",
        ),
        *[
            (
                LossChangeDetectionSettings,
                dict(
                    drift_detector_name="e_sr",
                    loss_monitoring_scope="overall_and_true_class_losses",
                    e_sr_false_alarm_control_alpha=0.05,
                ),
                configuration_parameter_name,
            )
            for configuration_parameter_name in (
                "drift_detector_name",
                "loss_monitoring_scope",
            )
        ],
        (
            CandidateModelTrainingAndAcceptanceSettings,
            dict(
                candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
                candidate_post_alarm_validation_sample_count=2,
            ),
            "candidate_model_acceptance_policy",
        ),
        *[
            (
                ModelConsolidationSettings,
                dict(
                    model_clustering_trigger_policy="on_new_model_registration",
                    model_pair_comparison_strategy="classwise_unique_correctness_lower_confidence_bound",
                    model_clustering_linkage="average_linkage",
                    model_consolidation_policy="weighted_parameter_average_and_merge_ids",
                ),
                configuration_parameter_name,
            )
            for configuration_parameter_name in (
                "model_clustering_trigger_policy",
                "model_pair_comparison_strategy",
                "model_clustering_linkage",
                "model_consolidation_policy",
            )
        ],
        *[
            (
                PredictionCombinationSettings,
                dict(
                    prediction_combination_strategy="fixed_share_weighted_prediction",
                    prediction_mixture_activation_policy="always",
                    prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
                    prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
                    fixed_share_weight_redistribution_time_scale_samples=2,
                ),
                configuration_parameter_name,
            )
            for configuration_parameter_name in (
                "prediction_combination_strategy",
                "prediction_mixture_activation_policy",
                "prediction_weight_recalibration_after_aggregation_policy",
                "prediction_state_reset_on_training_assignment_change_policy",
            )
        ],
        *[
            (
                LocalTrainingSettings,
                dict(
                    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                ),
                configuration_parameter_name,
            )
            for configuration_parameter_name in (
                "local_model_parameter_update_strategy",
                "shared_backbone_gradient_combination_strategy",
            )
        ],
    ],
)
@pytest.mark.parametrize(
    "specified_parameter_value",
    [
        "unknown",
        "residual_adapter",
        "shared_backbone",
        "SHARED_BACKBONE_RESIDUAL_ADAPTER",
        " shared_backbone_residual_adapter",
        "ClassESR",
        "class_esr",
        "ESR",
        "esr",
        "E_SR",
        " e_sr",
        "overall",
        "class",
        "overall_and_class_losses",
        "OVERALL_AND_TRUE_CLASS_LOSSES",
        " overall_and_true_class_losses",
        "switching",
        "FIXED_SHARE_WEIGHTED_PREDICTION",
        " fixed_share_weighted_prediction",
        "ALWAYS",
        " always",
        "RECOMPUTE_BUFFER_LOSSES_AND_REPLAY_WEIGHT_UPDATES",
        " recompute_buffer_losses_and_replay_weight_updates",
        "reset_all_routing_state",
        "reset_adahedge",
        "reset_all",
        "off",
        "RESTART_ADAHEDGE_PRESERVE_FIXED_SHARE_PREDICTION_STATE",
        " restart_adahedge_preserve_fixed_share_prediction_state",
        "SWITCHING_FIXED_SHARE_MIXTURE",
        " switching_fixed_share_mixture",
        "FIFO_LOSS_REPLAY",
        " fifo_loss_replay",
        "RESTART_ADAHEDGE_PRESERVE_SWITCHING",
        " restart_adahedge_preserve_switching",
        "switching_fixed_share_mixture",
        "fifo_loss_replay",
        "restart_adahedge_preserve_switching",
        "joint",
        "mean",
        "pcgrad",
        "joint_shared_backbone_updates",
        "sample_weighted_mean",
        "JOINT_BACKBONE_ADAPTER_AND_HEAD_TRAINING",
        " joint_backbone_adapter_and_head_training",
        "SAMPLE_WEIGHTED_MEAN_PER_CONCEPT_GRADIENTS",
        " sample_weighted_mean_per_concept_gradients",
        "CURRENT_MODEL_FIRST_REUSE_THEN_TWO_SEGMENT_CANDIDATE_VALIDATION",
        " current_model_first_reuse_then_two_segment_candidate_validation",
        "refit_compare_two_window",
        "candidate_refit_compare_two_window",
        "two_segment_candidate_validation",
        "ON_NEW_MODEL_REGISTRATION",
        " on_new_model_registration",
        "CLASSWISE_UNIQUE_CORRECTNESS_LOWER_CONFIDENCE_BOUND",
        " classwise_unique_correctness_lower_confidence_bound",
        "AVERAGE_LINKAGE",
        " average_linkage",
        "WEIGHTED_PARAMETER_AVERAGE_AND_MERGE_IDS",
        " weighted_parameter_average_and_merge_ids",
        True,
        False,
        1,
        None,
        [],
    ],
)
def test_component_options_reject_unknown_names(
    settings_type,
    valid_settings_values,
    configuration_parameter_name,
    specified_parameter_value,
):
    """旧名・大小文字差・型違いを正式な機能の選択肢として受理しない。"""
    expected_failure_reason = valid_settings_values[configuration_parameter_name]
    valid_settings_values = dict(valid_settings_values)
    valid_settings_values[configuration_parameter_name] = specified_parameter_value
    with pytest.raises(RunSettingsValidationError) as validation_error:
        settings_type(**valid_settings_values)
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "文字列" in validation_error.value.validation_failure_reason
    assert expected_failure_reason in validation_error.value.validation_failure_reason


@pytest.mark.parametrize("specified_parameter_value", [3, 5, 101, 10**400 + 1])
def test_candidate_post_alarm_validation_accepts_odd_sample_counts(specified_parameter_value):
    """将来検証件数は2以上の奇数を受理し、丸めず保持する。"""
    settings_instance = CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
        candidate_post_alarm_validation_sample_count=specified_parameter_value,
    )
    assert settings_instance.candidate_model_acceptance_policy == (
        "current_model_first_reuse_then_two_segment_candidate_validation"
    )
    assert (
        settings_instance.candidate_post_alarm_validation_sample_count is specified_parameter_value
    )


@pytest.mark.parametrize(
    "specified_parameter_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0.05",
        None,
        [],
        0,
        1,
        -0.05,
        1.05,
        10**400,
    ],
)
def test_monitoring_alpha_rejects_nonfinite_and_out_of_range_values(specified_parameter_value):
    """誤警報制御値はboolを除く有限実数の開区間だけを受理する。"""
    with pytest.raises(RunSettingsValidationError) as validation_error:
        LossChangeDetectionSettings(
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=specified_parameter_value,
        )
    assert validation_error.value.configuration_parameter_name == "e_sr_false_alarm_control_alpha"
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "有限の実数" in validation_error.value.validation_failure_reason
    assert "0より大きい" in validation_error.value.validation_failure_reason
    assert "1未満" in validation_error.value.validation_failure_reason


@pytest.mark.parametrize("settings_field_type", [int, float])
@pytest.mark.parametrize(
    "specified_parameter_value,settings_field_metadata",
    [
        (0, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": True}),
        (1, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": False}),
        (1, {"maximum_allowed_value": 1, "maximum_value_is_inclusive": True}),
        (0, {"maximum_allowed_value": 1, "maximum_value_is_inclusive": False}),
        (10**400, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": False}),
    ],
)
def test_settings_field_validation_accepts_supported_numeric_values(
    settings_field_type,
    specified_parameter_value,
    settings_field_metadata,
):
    @dataclass(frozen=True, kw_only=True)
    class NumericSettingsForValidation:
        random_seed: settings_field_type = field(metadata=settings_field_metadata)

    settings_instance = NumericSettingsForValidation(random_seed=specified_parameter_value)
    assert validate_settings_field_values(settings_instance) is None
    assert settings_instance.random_seed is specified_parameter_value
    if settings_field_type is float:
        settings_instance = NumericSettingsForValidation(random_seed=0.5)
        assert validate_settings_field_values(settings_instance) is None
        assert settings_instance.random_seed == 0.5


@pytest.mark.parametrize(
    "specified_parameter_value,settings_field_metadata",
    [
        (float("nan"), {}),
        (float("inf"), {}),
        (float("-inf"), {}),
        (True, {}),
        (False, {}),
        ("0.5", {}),
        (None, {}),
        ([], {}),
        (0, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": False}),
        (-1, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": True}),
        (1, {"maximum_allowed_value": 1, "maximum_value_is_inclusive": False}),
        (2, {"maximum_allowed_value": 1, "maximum_value_is_inclusive": True}),
    ],
)
def test_settings_field_validation_rejects_invalid_values(
    specified_parameter_value,
    settings_field_metadata,
):
    @dataclass(frozen=True, kw_only=True)
    class NumericSettingsForValidation:
        random_seed: float = field(metadata=settings_field_metadata)
        client_count: int

    settings_instance = NumericSettingsForValidation(
        random_seed=specified_parameter_value,
        client_count=False,
    )
    with pytest.raises(RunSettingsValidationError) as validation_error:
        validate_settings_field_values(settings_instance)
    assert validation_error.value.configuration_parameter_name == "random_seed"
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "有限の実数" in validation_error.value.validation_failure_reason
    if "minimum_allowed_value" in settings_field_metadata:
        assert (
            "以上" if settings_field_metadata["minimum_value_is_inclusive"] else "より大きい"
        ) in validation_error.value.validation_failure_reason
    if "maximum_allowed_value" in settings_field_metadata:
        assert (
            "以下" if settings_field_metadata["maximum_value_is_inclusive"] else "未満"
        ) in validation_error.value.validation_failure_reason


@pytest.mark.parametrize("settings_field_type", [bool, list[int], object, int | None])
def test_settings_field_validation_rejects_unsupported_annotations(settings_field_type):
    @dataclass(frozen=True, kw_only=True)
    class UnsupportedSettingsForValidation:
        random_seed: settings_field_type

    settings_instance = UnsupportedSettingsForValidation(random_seed=0)
    with pytest.raises(TypeError, match="random_seed.*宣言型.*未対応"):
        validate_settings_field_values(settings_instance)


def test_configuration_foundation_imports_only_allowed_dependencies():
    """設定基盤の旧依存・数値依存と、設定層の逆向きimportを検出する。"""
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    assert package_source_directory.is_dir()
    for source_file_path in sorted(package_source_directory.rglob("*.py")):
        source_module_path = source_file_path.relative_to(package_source_directory).as_posix()
        if not (
            source_module_path.startswith(("core/", "configuration/"))
            or (
                source_module_path.startswith(("learning/", "methods/"))
                and source_module_path.endswith("_settings.py")
            )
        ):
            continue
        importing_package_name = "federated_learning_experiments"
        if source_file_path.parent != package_source_directory:
            importing_package_name += "." + (
                source_file_path.parent.relative_to(package_source_directory)
                .as_posix()
                .replace("/", ".")
            )
        parsed_source_module = ast.parse(source_file_path.read_text(encoding="utf-8"))
        for import_statement in ast.walk(parsed_source_module):
            if isinstance(import_statement, ast.Import):
                imported_module_names = tuple(
                    imported_module_alias.name for imported_module_alias in import_statement.names
                )
            elif isinstance(import_statement, ast.ImportFrom):
                imported_module_names = (
                    resolve_name(
                        "." * import_statement.level + (import_statement.module or ""),
                        importing_package_name,
                    ),
                )
            else:
                continue
            for imported_module_name in imported_module_names:
                if imported_module_name.split(".")[0] in sys.stdlib_module_names:
                    continue
                assert imported_module_name.startswith("federated_learning_experiments."), (
                    source_module_path,
                    imported_module_name,
                )
                if source_module_path.startswith(("core/", "learning/", "methods/")) or (
                    source_module_path == "configuration/experiment_run_conditions.py"
                ):
                    assert imported_module_name.startswith(
                        "federated_learning_experiments.core."
                    ), (source_module_path, imported_module_name)
                if source_module_path == "configuration/run_settings_validation.py":
                    assert imported_module_name.startswith(
                        (
                            "federated_learning_experiments.core.",
                            "federated_learning_experiments.learning.",
                            "federated_learning_experiments.methods.",
                        )
                    ) or imported_module_name == (
                        "federated_learning_experiments.configuration.experiment_run_conditions"
                    ), (source_module_path, imported_module_name)
