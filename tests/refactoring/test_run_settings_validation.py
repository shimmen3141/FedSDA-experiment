"""新しい実験設定の検証契約を確認する。"""

from dataclasses import FrozenInstanceError, MISSING, dataclass, field, fields
from typing import get_type_hints

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
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
    assert (
        validation_error.value.configuration_parameter_name
        == configuration_parameter_name
    )
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
    ["unknown", "mnist4", "sine_two_concepts", "sea_two_concepts",
     "mnist_two_concepts", "SINE2", "SEA2", "MNIST2", " sine2", True, 1, None, []],
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
    assert all(value in validation_error.value.validation_failure_reason
               for value in ("sine2", "sea2", "mnist2"))


@pytest.mark.parametrize(
    "settings_type,valid_settings_values,configuration_parameter_name",
    [
        (ExperimentRunConditions, dict(
            dataset_name="sine2", random_seed=0, client_count=1,
            per_client_sample_count=1, server_aggregation_interval_per_client_samples=1,
        ), configuration_parameter_name)
        for configuration_parameter_name in (
            "random_seed", "client_count", "per_client_sample_count",
            "server_aggregation_interval_per_client_samples",
        )
    ] + [(ModelArchitectureSettings, dict(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=1,
    ), "residual_adapter_requested_rank")],
)
@pytest.mark.parametrize("specified_parameter_value", [True, False, "1", 1.0, None, [], -1])
def test_integer_settings_reject_invalid_types_and_ranges(
    settings_type, valid_settings_values, configuration_parameter_name,
    specified_parameter_value,
):
    valid_settings_values = dict(valid_settings_values)
    valid_settings_values[configuration_parameter_name] = specified_parameter_value
    with pytest.raises(RunSettingsValidationError) as validation_error:
        settings_type(**valid_settings_values)
    assert validation_error.value.configuration_parameter_name == configuration_parameter_name
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "整数" in validation_error.value.validation_failure_reason
    assert ("0以上" if configuration_parameter_name == "random_seed" else "1以上") in (
        validation_error.value.validation_failure_reason
    )


@pytest.mark.parametrize(
    "settings_type,valid_settings_values,configuration_parameter_name",
    [
        (ExperimentRunConditions, dict(
            dataset_name="sine2", random_seed=0, client_count=1,
            per_client_sample_count=1, server_aggregation_interval_per_client_samples=1,
        ), configuration_parameter_name)
        for configuration_parameter_name in (
            "random_seed", "client_count", "per_client_sample_count",
            "server_aggregation_interval_per_client_samples",
        )
    ] + [(ModelArchitectureSettings, dict(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=1,
    ), "residual_adapter_requested_rank")],
)
@pytest.mark.parametrize("specified_parameter_value", [0, 1, 10**400])
def test_integer_settings_accept_boundary_values(
    settings_type, valid_settings_values, configuration_parameter_name,
    specified_parameter_value,
):
    valid_settings_values = dict(valid_settings_values)
    valid_settings_values[configuration_parameter_name] = specified_parameter_value
    if configuration_parameter_name != "random_seed" and specified_parameter_value == 0:
        with pytest.raises(RunSettingsValidationError):
            settings_type(**valid_settings_values)
    else:
        settings_instance = settings_type(**valid_settings_values)
        assert getattr(settings_instance, configuration_parameter_name) == specified_parameter_value


def test_aggregation_interval_can_exceed_stream_length():
    settings_instance = ExperimentRunConditions(
        dataset_name="sea2", random_seed=0, client_count=1,
        per_client_sample_count=1, server_aggregation_interval_per_client_samples=100,
    )
    assert settings_instance.server_aggregation_interval_per_client_samples == 100


@pytest.mark.parametrize(
    "settings_type,valid_settings_values",
    [
        (ExperimentRunConditions, dict(
            dataset_name="mnist2", random_seed=0, client_count=1,
            per_client_sample_count=1, server_aggregation_interval_per_client_samples=1,
        )),
        (ModelArchitectureSettings, dict(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=1,
        )),
    ],
)
def test_settings_instances_are_immutable(settings_type, valid_settings_values):
    settings_instance = settings_type(**valid_settings_values)
    for settings_field in fields(settings_instance):
        with pytest.raises(FrozenInstanceError):
            setattr(settings_instance, settings_field.name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(settings_instance, settings_field.name)


@pytest.mark.parametrize("settings_type", [ExperimentRunConditions, ModelArchitectureSettings])
def test_field_annotations_and_metadata_declare_parameter_constraints(settings_type):
    assert get_type_hints(settings_type) == (
        {
            "dataset_name": str, "random_seed": int, "client_count": int,
            "per_client_sample_count": int,
            "server_aggregation_interval_per_client_samples": int,
        } if settings_type is ExperimentRunConditions else {
            "model_architecture_name": str, "residual_adapter_requested_rank": int,
        }
    )
    for settings_field in fields(settings_type):
        assert settings_field.kw_only
        assert settings_field.default is MISSING
        assert settings_field.default_factory is MISSING
        if get_type_hints(settings_type)[settings_field.name] is str:
            assert settings_field.metadata["allowed_parameter_values"] == (
                ("sine2", "sea2", "mnist2")
                if settings_field.name == "dataset_name"
                else ("shared_backbone_residual_adapter",)
            )
        else:
            assert settings_field.metadata["minimum_allowed_value"] == (
                0 if settings_field.name == "random_seed" else 1
            )
            assert settings_field.metadata["minimum_value_is_inclusive"] is True
            assert "maximum_allowed_value" not in settings_field.metadata
            assert isinstance(settings_field.metadata["parameter_unit"], str)
            assert settings_field.metadata["parameter_unit"]
            if settings_field.name == "residual_adapter_requested_rank":
                assert settings_field.metadata["parameter_unit"] == "rank"
    with pytest.raises(TypeError):
        settings_type()
    with pytest.raises(TypeError):
        if settings_type is ExperimentRunConditions:
            settings_type("sine2", 0, 1, 1, 1)
        else:
            settings_type("shared_backbone_residual_adapter", 1)


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
    "specified_parameter_value",
    ["unknown", "residual_adapter", "shared_backbone", "SHARED_BACKBONE_RESIDUAL_ADAPTER",
     " shared_backbone_residual_adapter", True, 1, None, []],
)
def test_component_options_reject_unknown_names(specified_parameter_value):
    """旧名・大小文字差・型違いを正式なモデル構造として受理しない。"""
    with pytest.raises(RunSettingsValidationError) as validation_error:
        ModelArchitectureSettings(
            model_architecture_name=specified_parameter_value,
            residual_adapter_requested_rank=1,
        )
    assert validation_error.value.configuration_parameter_name == "model_architecture_name"
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "文字列" in validation_error.value.validation_failure_reason
    assert "shared_backbone_residual_adapter" in validation_error.value.validation_failure_reason


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
    settings_field_type, specified_parameter_value, settings_field_metadata,
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
        (float("nan"), {}), (float("inf"), {}), (float("-inf"), {}),
        (True, {}), (False, {}), ("0.5", {}), (None, {}), ([], {}),
        (0, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": False}),
        (-1, {"minimum_allowed_value": 0, "minimum_value_is_inclusive": True}),
        (1, {"maximum_allowed_value": 1, "maximum_value_is_inclusive": False}),
        (2, {"maximum_allowed_value": 1, "maximum_value_is_inclusive": True}),
    ],
)
def test_settings_field_validation_rejects_invalid_values(
    specified_parameter_value, settings_field_metadata,
):
    @dataclass(frozen=True, kw_only=True)
    class NumericSettingsForValidation:
        random_seed: float = field(metadata=settings_field_metadata)
        client_count: int

    settings_instance = NumericSettingsForValidation(
        random_seed=specified_parameter_value, client_count=False,
    )
    with pytest.raises(RunSettingsValidationError) as validation_error:
        validate_settings_field_values(settings_instance)
    assert validation_error.value.configuration_parameter_name == "random_seed"
    assert validation_error.value.specified_parameter_value is specified_parameter_value
    assert "有限の実数" in validation_error.value.validation_failure_reason
    if "minimum_allowed_value" in settings_field_metadata:
        assert ("以上" if settings_field_metadata["minimum_value_is_inclusive"]
                else "より大きい") in validation_error.value.validation_failure_reason
    if "maximum_allowed_value" in settings_field_metadata:
        assert ("以下" if settings_field_metadata["maximum_value_is_inclusive"]
                else "未満") in validation_error.value.validation_failure_reason


@pytest.mark.parametrize("settings_field_type", [bool, list[int], object, int | None])
def test_settings_field_validation_rejects_unsupported_annotations(settings_field_type):
    @dataclass(frozen=True, kw_only=True)
    class UnsupportedSettingsForValidation:
        random_seed: settings_field_type

    settings_instance = UnsupportedSettingsForValidation(random_seed=0)
    with pytest.raises(TypeError, match="random_seed.*宣言型.*未対応"):
        validate_settings_field_values(settings_instance)
