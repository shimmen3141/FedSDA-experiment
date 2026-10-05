"""設定型の型注釈・metadataに従い、フィールド値を宣言順で検証する。"""

from dataclasses import fields
from math import isfinite
from typing import get_type_hints

from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)


def validate_settings_field_values(settings_instance: object) -> None:
    """設定を変更せず、最初の不正値の項目・元の値・許容条件を報告する。"""
    # fields自身がdataclassかを実行時検査する。汎用入口のobject型を保持する。
    for settings_field in fields(settings_instance):  # pyright: ignore[reportArgumentType]
        configuration_parameter_name = settings_field.name
        settings_field_type = get_type_hints(type(settings_instance))[configuration_parameter_name]
        specified_parameter_value = getattr(settings_instance, configuration_parameter_name)
        settings_field_metadata = settings_field.metadata
        if settings_field_type not in (str, int, float):
            raise TypeError(
                f"設定項目 {configuration_parameter_name!r} の宣言型 "
                f"{settings_field_type!r} は未対応です。"
            )

        validation_failure_reason = ""
        if settings_field_type is str:
            allowed_parameter_values = settings_field_metadata.get("allowed_parameter_values")
            validation_failure_reason = "文字列を指定してください。"
            if allowed_parameter_values is not None:
                validation_failure_reason = (
                    "文字列で"
                    + "・".join(allowed_parameter_values)
                    + "のいずれかを指定してください。"
                )
            if isinstance(specified_parameter_value, str) and (
                allowed_parameter_values is None
                or specified_parameter_value in allowed_parameter_values
            ):
                continue
        else:
            minimum_allowed_value = settings_field_metadata.get("minimum_allowed_value")
            maximum_allowed_value = settings_field_metadata.get("maximum_allowed_value")
            validation_failure_reason = (
                "整数" if settings_field_type is int else "有限の実数（整数も可）"
            )
            if minimum_allowed_value is not None:
                validation_failure_reason += f"、{minimum_allowed_value}" + (
                    "以上"
                    if settings_field_metadata["minimum_value_is_inclusive"]
                    else "より大きい値"
                )
            if maximum_allowed_value is not None:
                validation_failure_reason += f"、{maximum_allowed_value}" + (
                    "以下" if settings_field_metadata["maximum_value_is_inclusive"] else "未満"
                )
            validation_failure_reason += "を指定してください。bool・数値文字列は受理しません。"
            if (
                not isinstance(specified_parameter_value, bool)
                and isinstance(
                    specified_parameter_value,
                    int if settings_field_type is int else (int, float),
                )
                # 整数は有限なので、巨大整数をfloatに変換しない。
                and (
                    not isinstance(specified_parameter_value, float)
                    or isfinite(specified_parameter_value)
                )
                and (
                    minimum_allowed_value is None
                    or (
                        specified_parameter_value >= minimum_allowed_value
                        if settings_field_metadata["minimum_value_is_inclusive"]
                        else specified_parameter_value > minimum_allowed_value
                    )
                )
                and (
                    maximum_allowed_value is None
                    or (
                        specified_parameter_value <= maximum_allowed_value
                        if settings_field_metadata["maximum_value_is_inclusive"]
                        else specified_parameter_value < maximum_allowed_value
                    )
                )
            ):
                continue
        raise RunSettingsValidationError(
            configuration_parameter_name=configuration_parameter_name,
            specified_parameter_value=specified_parameter_value,
            validation_failure_reason=validation_failure_reason,
        )
