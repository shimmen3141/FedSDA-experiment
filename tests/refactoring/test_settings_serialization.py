"""dataclassの設定を、保存用の入れ子の辞書へ変換する規則。"""

import json
import math
from dataclasses import dataclass, replace

import pytest

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)


@dataclass(frozen=True, kw_only=True)
class InnerSettings:
    learning_rate: float
    variant_name: str


@dataclass(frozen=True, kw_only=True)
class OtherInnerSettings:
    learning_rate: float
    variant_name: str


@dataclass(frozen=True, kw_only=True)
class OuterSettings:
    sample_count: int
    is_enabled: bool
    optional_limit: int | None
    layer_widths: tuple[int, ...]
    inner_settings: InnerSettings | OtherInnerSettings
    inner_settings_sequence: tuple[InnerSettings, ...]


def make_outer_settings(**replaced_fields):
    return replace(
        OuterSettings(
            sample_count=3,
            is_enabled=True,
            optional_limit=None,
            layer_widths=(32, 16),
            inner_settings=InnerSettings(learning_rate=0.01, variant_name="amsgrad"),
            inner_settings_sequence=(
                InnerSettings(learning_rate=0.5, variant_name="plain"),
                InnerSettings(learning_rate=0.25, variant_name="plain"),
            ),
        ),
        **replaced_fields,
    )


def test_settings_are_converted_to_nested_plain_mapping_with_type_names():
    """dataclassは、型の名前と、宣言の順のfieldを持つ辞書になる。tupleはlistになる。"""
    plain_mapping = convert_settings_to_plain_mapping(make_outer_settings())
    assert plain_mapping == {
        "settings_type": "OuterSettings",
        "sample_count": 3,
        "is_enabled": True,
        "optional_limit": None,
        "layer_widths": [32, 16],
        "inner_settings": {
            "settings_type": "InnerSettings",
            "learning_rate": 0.01,
            "variant_name": "amsgrad",
        },
        "inner_settings_sequence": [
            {"settings_type": "InnerSettings", "learning_rate": 0.5, "variant_name": "plain"},
            {"settings_type": "InnerSettings", "learning_rate": 0.25, "variant_name": "plain"},
        ],
    }
    assert list(plain_mapping) == [
        "settings_type",
        "sample_count",
        "is_enabled",
        "optional_limit",
        "layer_widths",
        "inner_settings",
        "inner_settings_sequence",
    ]
    # 値の型は、そのまま（boolがintに、intがfloatにならない）。
    assert type(plain_mapping["sample_count"]) is int
    assert type(plain_mapping["is_enabled"]) is bool
    assert type(plain_mapping["layer_widths"]) is list
    # JSONにして、戻しても、同じ値。
    assert json.loads(json.dumps(plain_mapping)) == plain_mapping


def test_conversion_is_deterministic_and_distinguishes_every_value_and_type():
    """同じ設定からは同じ辞書、どれか1つの値（や、同じfieldを持つ別の型）が違えば、違う辞書。"""
    settings = make_outer_settings()
    plain_mapping = convert_settings_to_plain_mapping(settings)
    assert convert_settings_to_plain_mapping(make_outer_settings()) == plain_mapping
    # 変換は、設定を変えない。戻り値を変えても、次の変換に影響しない。
    plain_mapping["layer_widths"].append(8)
    assert convert_settings_to_plain_mapping(settings) != plain_mapping
    assert settings == make_outer_settings()
    plain_mapping = convert_settings_to_plain_mapping(settings)
    for changed_settings in (
        make_outer_settings(sample_count=4),
        make_outer_settings(is_enabled=False),
        make_outer_settings(optional_limit=0),
        make_outer_settings(layer_widths=(32,)),
        make_outer_settings(layer_widths=(16, 32)),
        make_outer_settings(
            inner_settings=InnerSettings(learning_rate=0.02, variant_name="amsgrad")
        ),
        # 同じfield・同じ値で、型だけが違う。
        make_outer_settings(
            inner_settings=OtherInnerSettings(learning_rate=0.01, variant_name="amsgrad")
        ),
        make_outer_settings(inner_settings_sequence=()),
    ):
        assert convert_settings_to_plain_mapping(changed_settings) != plain_mapping


@dataclass(frozen=True)
class SettingsWithAnyValue:
    value: object


@dataclass(frozen=True)
class SettingsWithReservedFieldName:
    settings_type: str


@pytest.mark.parametrize(
    ("invalid_value", "expected_error_type"),
    [
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
        ({"key": 1}, TypeError),
        ({1, 2}, TypeError),
        (b"bytes", TypeError),
        (1 + 2j, TypeError),
        (object(), TypeError),
        (InnerSettings, TypeError),
        ((1, (2, object())), TypeError),
        ((InnerSettings(learning_rate=math.nan, variant_name="x"),), ValueError),
    ],
)
def test_values_that_cannot_be_saved_are_rejected(invalid_value, expected_error_type):
    """JSONにできない値と、非有限のfloatは、入れ子の中にあっても拒否する。"""
    with pytest.raises(expected_error_type, match="value"):
        convert_settings_to_plain_mapping(SettingsWithAnyValue(value=invalid_value))


@pytest.mark.parametrize("invalid_settings", [None, 1, "settings", (1, 2), {"a": 1}, InnerSettings])
def test_only_settings_instances_are_converted(invalid_settings):
    """最上位は、dataclassの値だけ（型そのものや、ほかの値は、拒否する）。"""
    with pytest.raises(TypeError):
        convert_settings_to_plain_mapping(invalid_settings)


def test_reserved_field_name_is_rejected():
    """型の名前を入れるキーと同じ名前のfieldを持つ設定は、拒否する。"""
    with pytest.raises(ValueError, match="settings_type"):
        convert_settings_to_plain_mapping(SettingsWithReservedFieldName(settings_type="x"))


def test_nested_lists_and_plain_values_are_converted():
    assert convert_settings_to_plain_mapping(
        SettingsWithAnyValue(value=[1, (2.5, "a", None, False), []])
    ) == {"settings_type": "SettingsWithAnyValue", "value": [1, [2.5, "a", None, False], []]}
