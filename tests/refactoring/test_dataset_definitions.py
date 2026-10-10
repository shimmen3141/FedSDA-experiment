"""datasetの定義を、実旧の定義と照合する。"""

from dataclasses import FrozenInstanceError, fields

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.dataset_definitions import (
    DEFINED_DATASET_NAMES,
    DatasetDefinition,
    get_dataset_definition,
)

DATASET_NAMES = ("sine2", "sea2", "sea4", "circle2", "mnist2", "mnist4")


def test_defined_dataset_names_are_all_datasets_except_blobs():
    """定義のあるdatasetは、実旧のdatasetのうち、blobs以外の全部（定義した順）。実行条件が受け取る名前と同じ。"""
    from federated_drift_experiment.data.specs import DATASET_SPECS

    assert DEFINED_DATASET_NAMES == DATASET_NAMES
    assert set(DEFINED_DATASET_NAMES) == set(DATASET_SPECS) - {"blobs"}
    dataset_name_field = next(
        settings_field
        for settings_field in fields(ExperimentRunConditions)
        if settings_field.name == "dataset_name"
    )
    assert dataset_name_field.metadata["allowed_parameter_values"] == DEFINED_DATASET_NAMES


@pytest.mark.parametrize("dataset_name", DATASET_NAMES)
def test_dataset_definition_matches_legacy_dataset_spec(dataset_name):
    """特徴数・概念数・クラス数が、実旧の`DATASET_SPECS`と一致する。"""
    from federated_drift_experiment.data.specs import DATASET_SPECS

    legacy_dataset_spec = DATASET_SPECS[dataset_name]
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    assert type(dataset_definition) is DatasetDefinition
    assert dataset_definition == DatasetDefinition(
        dataset_name=dataset_name,
        input_feature_count=legacy_dataset_spec.input_dim,
        concept_count=legacy_dataset_spec.num_concepts,
        class_count=legacy_dataset_spec.num_classes,
    )
    assert get_dataset_definition(dataset_name=dataset_name) is dataset_definition


def test_dataset_definition_is_frozen_and_keyword_only():
    """定義は変更できず、名前付き引数だけで受け取る。"""
    dataset_definition = get_dataset_definition(dataset_name="sea4")
    with pytest.raises(FrozenInstanceError):
        dataset_definition.concept_count = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        get_dataset_definition("sea4")  # type: ignore[misc]
    with pytest.raises(TypeError):
        DatasetDefinition("sea4", 3, 4, 2)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("invalid_dataset_name", "expected_error_type"),
    [
        ("blobs", ValueError),
        ("mnist", ValueError),
        ("MNIST2", ValueError),
        ("SINE2", ValueError),
        ("sine", ValueError),
        ("", ValueError),
        (None, TypeError),
        (0, TypeError),
        (("sine2",), TypeError),
    ],
)
def test_undefined_dataset_names_are_rejected(invalid_dataset_name, expected_error_type):
    """定義のない名前（blobs、別表記）と、str以外を拒否する。"""
    with pytest.raises(expected_error_type, match="dataset_name"):
        get_dataset_definition(dataset_name=invalid_dataset_name)
