"""clientの設定の束: 値の型と範囲、機能別の設定の型、賭け率、最終構成で同じ値でなければならない組。"""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientScalarSettings,
    FedsdaRunClientSettings,
)

VALID_SCALAR_VALUES = dict(
    maximum_tolerated_mean_loss_increase=0.1,
    minimum_candidate_mean_loss_improvement=0.0001,
    new_model_upload_delay_round_count=1,
    minimum_change_interval_sample_count=3,
    local_training_batch_sample_count=4,
    maximum_stored_evaluation_sample_count_per_model=20,
    added_evaluation_batch_sample_count=5,
    maximum_cross_evaluation_sample_count=50,
    loss_monitor_maximum_retained_candidate_count=1000,
    detector_name="overall + class-conditional e-SR mixture",
)


def make_valid_run_client_settings(valid_run_settings_mapping, **replaced_fields):
    """有効な束。機能別の設定は、既存の有効なrun設定のものを使う。指定したfieldだけ置き換える。"""
    return FedsdaRunClientSettings(
        **dict(
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            prediction_combination_settings=valid_run_settings_mapping[
                "prediction_combination_settings"
            ],
            local_training_settings=valid_run_settings_mapping["local_training_settings"],
            local_training_schedule_settings=LocalTrainingScheduleSettings(
                training_requests_per_update_interval=2,
                joint_update_iterations_per_training_request=1,
            ),
            training_data_assignment_settings=valid_run_settings_mapping[
                "training_data_assignment_settings"
            ],
            candidate_model_training_and_acceptance_settings=valid_run_settings_mapping[
                "candidate_model_training_and_acceptance_settings"
            ],
            candidate_parameter_initialization_settings=CandidateParameterInitializationSettings(
                candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
            ),
            candidate_epoch_training_settings=CandidateEpochTrainingSettings(
                candidate_training_strategy="validation_loss_early_stopping",
                maximum_epoch_count=5,
                maximum_batch_sample_count=4,
                validation_sample_fraction=0.2,
                consecutive_non_improving_epoch_limit=3,
                minimum_validation_loss_decrease=0.0001,
            ),
            parameter_optimizer_settings=AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant="amsgrad"
            ),
            rebuilt_model_parameter_optimizer_settings=AdamParameterOptimizerSettings(
                learning_rate=0.02, weight_decay=0.001, adam_variant="amsgrad"
            ),
            scalar_settings=FedsdaRunClientScalarSettings(**VALID_SCALAR_VALUES),
            loss_monitor_betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
        )
        | replaced_fields
    )


def make_subclass_instance(settings_instance):
    """同じfieldの値を持つ、派生型の設定。"""
    return type("SettingsSubclass", (type(settings_instance),), {})(**vars(settings_instance))


def test_valid_settings_bundle_is_accepted_and_frozen(valid_run_settings_mapping):
    run_client_settings = make_valid_run_client_settings(valid_run_settings_mapping)
    assert run_client_settings.scalar_settings == FedsdaRunClientScalarSettings(
        **VALID_SCALAR_VALUES
    )
    with pytest.raises(AttributeError):
        run_client_settings.loss_monitor_betting_fractions = (0.5,)
    with pytest.raises(AttributeError):
        run_client_settings.scalar_settings.detector_name = "other"
    # SGDの設定も受け入れる（2つのoptimizerの設定は、同じ種類にする。学習率は違ってよい）。
    make_valid_run_client_settings(
        valid_run_settings_mapping,
        parameter_optimizer_settings=SgdParameterOptimizerSettings(learning_rate=0.01),
        rebuilt_model_parameter_optimizer_settings=SgdParameterOptimizerSettings(learning_rate=0.5),
    )


def test_settings_bundle_requires_same_kind_of_two_optimizer_settings(valid_run_settings_mapping):
    for replaced_field_name in (
        "parameter_optimizer_settings",
        "rebuilt_model_parameter_optimizer_settings",
    ):
        with pytest.raises(RunSettingsValidationError) as raised_error:
            make_valid_run_client_settings(
                valid_run_settings_mapping,
                **{replaced_field_name: SgdParameterOptimizerSettings(learning_rate=0.01)},
            )
        assert (
            raised_error.value.configuration_parameter_name
            == "rebuilt_model_parameter_optimizer_settings"
        )


@pytest.mark.parametrize(
    "scalar_field_name,accepted_value",
    [
        # 受け入れる範囲の端。増加量と改善量は、整数も受け入れる（値を使う部品と同じ）。
        ("maximum_tolerated_mean_loss_increase", 0.0),
        ("maximum_tolerated_mean_loss_increase", 0),
        ("maximum_tolerated_mean_loss_increase", 2.5),
        ("minimum_candidate_mean_loss_improvement", 0.0),
        ("new_model_upload_delay_round_count", 1),
        ("minimum_change_interval_sample_count", 1),
        ("local_training_batch_sample_count", 1),
        ("maximum_stored_evaluation_sample_count_per_model", 1),
        ("added_evaluation_batch_sample_count", 0),
        ("loss_monitor_maximum_retained_candidate_count", 1),
        ("detector_name", "e"),
    ],
)
def test_scalar_settings_accept_boundary_values(scalar_field_name, accepted_value):
    scalar_settings = FedsdaRunClientScalarSettings(
        **VALID_SCALAR_VALUES | {scalar_field_name: accepted_value}
    )
    assert getattr(scalar_settings, scalar_field_name) == accepted_value


INVALID_NUMBERS = (True, "1", None, math.nan, math.inf)


@pytest.mark.parametrize(
    "scalar_field_name,rejected_value",
    [
        *[
            (scalar_field_name, rejected_value)
            for scalar_field_name in (
                "maximum_tolerated_mean_loss_increase",
                "minimum_candidate_mean_loss_improvement",
            )
            # numpyのfloat64はfloatの派生型。builtinの型だけを受け入れる。
            for rejected_value in (*INVALID_NUMBERS, -1e-9, -1, np.float64(0.5))
        ],
        *[
            (scalar_field_name, rejected_value)
            for scalar_field_name in (
                "new_model_upload_delay_round_count",
                "minimum_change_interval_sample_count",
                "local_training_batch_sample_count",
                "maximum_stored_evaluation_sample_count_per_model",
                "loss_monitor_maximum_retained_candidate_count",
            )
            for rejected_value in (*INVALID_NUMBERS, 0, -1, 1.0, np.int64(3))
        ],
        *[
            ("added_evaluation_batch_sample_count", rejected_value)
            for rejected_value in (*INVALID_NUMBERS, -1, 0.0)
        ],
        ("detector_name", ""),
        ("detector_name", "  \t"),
        ("detector_name", None),
        ("detector_name", 3),
        ("detector_name", type("StrSubclass", (str,), {})("detector")),
    ],
)
def test_scalar_settings_reject_invalid_values(scalar_field_name, rejected_value):
    with pytest.raises(RunSettingsValidationError) as raised_error:
        FedsdaRunClientScalarSettings(**VALID_SCALAR_VALUES | {scalar_field_name: rejected_value})
    assert raised_error.value.configuration_parameter_name == scalar_field_name


SETTINGS_FIELD_NAMES = (
    "loss_change_detection_settings",
    "prediction_combination_settings",
    "local_training_settings",
    "local_training_schedule_settings",
    "training_data_assignment_settings",
    "candidate_model_training_and_acceptance_settings",
    "candidate_parameter_initialization_settings",
    "candidate_epoch_training_settings",
    "parameter_optimizer_settings",
    "rebuilt_model_parameter_optimizer_settings",
    "scalar_settings",
)


@pytest.mark.parametrize("settings_field_name", SETTINGS_FIELD_NAMES)
@pytest.mark.parametrize("invalid_kind", ("none", "other_settings_type", "subclass"))
def test_settings_bundle_rejects_other_settings_types(
    settings_field_name, invalid_kind, valid_run_settings_mapping
):
    valid_settings = make_valid_run_client_settings(valid_run_settings_mapping)
    # 2つ先のfieldの値を使う（optimizerの設定の2つは同じ型なので、隣では別の型にならない）。
    other_field_name = SETTINGS_FIELD_NAMES[
        (SETTINGS_FIELD_NAMES.index(settings_field_name) + 2) % len(SETTINGS_FIELD_NAMES)
    ]
    assert type(getattr(valid_settings, other_field_name)) is not type(
        getattr(valid_settings, settings_field_name)
    )
    invalid_value = {
        "none": None,
        "other_settings_type": getattr(valid_settings, other_field_name),
        "subclass": make_subclass_instance(getattr(valid_settings, settings_field_name)),
    }[invalid_kind]
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping, **{settings_field_name: invalid_value}
        )
    assert raised_error.value.configuration_parameter_name == settings_field_name


@pytest.mark.parametrize(
    "invalid_betting_fractions",
    [
        (),
        [0.1, 0.4],
        None,
        (0.0, 0.4),
        (0.4, 1.0),
        (0.4, -0.1),
        (0.4, math.nan),
        (0.4, 1),
        (0.4, True),
        (0.4, np.float64(0.2)),
        type("TupleSubclass", (tuple,), {})((0.4,)),
    ],
)
def test_settings_bundle_rejects_invalid_betting_fractions(
    invalid_betting_fractions, valid_run_settings_mapping
):
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping, loss_monitor_betting_fractions=invalid_betting_fractions
        )
    assert raised_error.value.configuration_parameter_name == "loss_monitor_betting_fractions"


def test_settings_bundle_accepts_single_and_repeated_betting_fractions(valid_run_settings_mapping):
    for betting_fractions in ((0.4,), (0.4, 0.4), (1e-9, 1.0 - 1e-9)):
        assert (
            make_valid_run_client_settings(
                valid_run_settings_mapping, loss_monitor_betting_fractions=betting_fractions
            ).loss_monitor_betting_fractions
            == betting_fractions
        )


def test_settings_bundle_requires_fixed_share_time_scale_equal_to_pending_capacity(
    valid_run_settings_mapping,
):
    prediction_combination_settings = valid_run_settings_mapping["prediction_combination_settings"]
    pending_capacity = valid_run_settings_mapping[
        "training_data_assignment_settings"
    ].pending_assignment_buffer_capacity_samples
    assert (
        prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        == pending_capacity
    )
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping,
            prediction_combination_settings=replace(
                prediction_combination_settings,
                fixed_share_weight_redistribution_time_scale_samples=pending_capacity + 1,
            ),
        )
    assert (
        raised_error.value.configuration_parameter_name
        == "fixed_share_weight_redistribution_time_scale_samples"
    )
    assert raised_error.value.specified_parameter_value == pending_capacity + 1


def test_settings_bundle_requires_equal_minimum_improvement_and_early_stopping_decrease(
    valid_run_settings_mapping,
):
    valid_settings = make_valid_run_client_settings(valid_run_settings_mapping)
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping,
            scalar_settings=replace(
                valid_settings.scalar_settings, minimum_candidate_mean_loss_improvement=0.5
            ),
        )
    assert (
        raised_error.value.configuration_parameter_name == "minimum_candidate_mean_loss_improvement"
    )
    with pytest.raises(RunSettingsValidationError):
        make_valid_run_client_settings(
            valid_run_settings_mapping,
            candidate_epoch_training_settings=replace(
                valid_settings.candidate_epoch_training_settings,
                minimum_validation_loss_decrease=0.5,
            ),
        )
    # 両方を同じ値に変えれば受け入れる。
    make_valid_run_client_settings(
        valid_run_settings_mapping,
        scalar_settings=replace(
            valid_settings.scalar_settings, minimum_candidate_mean_loss_improvement=0.5
        ),
        candidate_epoch_training_settings=replace(
            valid_settings.candidate_epoch_training_settings, minimum_validation_loss_decrease=0.5
        ),
    )


def test_settings_bundle_requires_equal_local_and_candidate_batch_sample_counts(
    valid_run_settings_mapping,
):
    valid_settings = make_valid_run_client_settings(valid_run_settings_mapping)
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping,
            scalar_settings=replace(
                valid_settings.scalar_settings, local_training_batch_sample_count=5
            ),
        )
    assert raised_error.value.configuration_parameter_name == "local_training_batch_sample_count"
    assert raised_error.value.specified_parameter_value == 5
    make_valid_run_client_settings(
        valid_run_settings_mapping,
        scalar_settings=replace(
            valid_settings.scalar_settings, local_training_batch_sample_count=5
        ),
        candidate_epoch_training_settings=replace(
            valid_settings.candidate_epoch_training_settings, maximum_batch_sample_count=5
        ),
    )


def test_settings_bundle_rejects_component_settings_mutated_around_frozen(
    valid_run_settings_mapping,
):
    """frozenを回避して不正にした設定は、束の生成で、その設定自身の検査が拒否する。"""
    valid_settings = make_valid_run_client_settings(valid_run_settings_mapping)
    mutated_scalar_settings = replace(valid_settings.scalar_settings)
    object.__setattr__(mutated_scalar_settings, "new_model_upload_delay_round_count", 0)
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping, scalar_settings=mutated_scalar_settings
        )
    assert raised_error.value.configuration_parameter_name == "new_model_upload_delay_round_count"
    mutated_assignment_settings = replace(valid_settings.training_data_assignment_settings)
    object.__setattr__(mutated_assignment_settings, "pending_assignment_buffer_capacity_samples", 0)
    with pytest.raises(RunSettingsValidationError):
        make_valid_run_client_settings(
            valid_run_settings_mapping,
            training_data_assignment_settings=mutated_assignment_settings,
        )
