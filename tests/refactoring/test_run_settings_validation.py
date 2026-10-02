"""新しい実験設定の検証契約を確認する。"""

import pytest

from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
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
