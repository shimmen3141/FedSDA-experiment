"""dataclassの設定を、保存用の入れ子の辞書（JSONにできる値だけ）へ変換する。"""

from dataclasses import fields, is_dataclass
from math import isfinite
from typing import cast

# 設定の型の名前を入れるキー（同じ場所に、種類の違う設定が入りうるので、型の名前も保存する）。
_SETTINGS_TYPE_KEY = "settings_type"


def _convert_value(*, value: object, value_path: str) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _convert_settings(settings=value, settings_path=value_path)
    if isinstance(value, (tuple, list)):
        elements = cast("tuple[object, ...] | list[object]", value)
        return [
            _convert_value(value=element, value_path=f"{value_path}[{element_index}]")
            for element_index, element in enumerate(elements)
        ]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        float_value = cast(float, value)
        if not isfinite(float_value):
            raise ValueError(f"value at {value_path} must be finite: {value!r}")
        return float_value
    raise TypeError(
        f"value at {value_path} cannot be saved: {value!r} ({type(value).__name__}). "
        "Only settings dataclasses, tuples, lists, str, int, float, bool and None are supported"
    )


def _convert_settings(*, settings: object, settings_path: str) -> dict[str, object]:
    plain_mapping: dict[str, object] = {_SETTINGS_TYPE_KEY: type(settings).__name__}
    # 呼出し側が、dataclassの値であることを確かめている。
    for settings_field in fields(settings):  # type: ignore[arg-type]
        if settings_field.name == _SETTINGS_TYPE_KEY:
            raise ValueError(
                f"settings at {settings_path} must not have a field named {_SETTINGS_TYPE_KEY!r}"
            )
        plain_mapping[settings_field.name] = _convert_value(
            value=getattr(settings, settings_field.name),
            value_path=f"{settings_path}.{settings_field.name}",
        )
    return plain_mapping


def convert_settings_to_plain_mapping(settings: object) -> dict[str, object]:
    """dataclassの設定を、型の名前と、全部のfieldの値を持つ、入れ子の辞書へ変換する。

    dataclassは`{"settings_type": 型の名前, field名: 値, ...}`（fieldの宣言の順）、tuple・listはlist、
    `str`・`int`・`bool`・`None`と、有限の`float`は、そのまま。ほかの値は拒否する。
    """
    if not is_dataclass(settings) or isinstance(settings, type):
        raise TypeError("settings must be a dataclass instance")
    return _convert_settings(settings=settings, settings_path=type(settings).__name__)
