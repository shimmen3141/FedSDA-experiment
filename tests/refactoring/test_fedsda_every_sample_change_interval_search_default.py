"""新しい既定、比較用構成、過去の識別子と版15の設定を守る。"""

from dataclasses import replace

import pytest
from test_fedsda_run_client import use_legacy_compatible_policies

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.sweep import experiment_configurations as configurations
from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier
from tests.refactoring import behavior_golden_cases as golden

FORWARD = "minimum_interval_multiples_with_forward_prequential_code"
REVERSE = "every_sample_with_reverse_prequential_code"
COMPARISON = "fedsda-minimum-interval-multiples-change-interval-search"
CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=0,
    client_count=10,
    per_client_sample_count=600,
    server_aggregation_interval_per_client_samples=100,
)


def _assignment(settings):
    return settings.run_participant_settings.run_client_settings.training_data_assignment_settings


def _build(name, conditions=CONDITIONS):
    return configurations.build_experiment_run_settings(
        configuration_name=name, experiment_run_conditions=conditions
    )


def test_final_configuration_uses_every_sample_search_by_default():
    assert (
        _assignment(
            build_final_configuration_fedsda_run_settings(experiment_run_conditions=CONDITIONS)
        ).change_interval_search
        == REVERSE
    )


@pytest.mark.parametrize(
    "name", [n for n in configurations.EXPERIMENT_CONFIGURATION_NAMES if n.startswith("fedsda")]
)
def test_all_fedsda_configurations_inherit_search_default_except_comparison(name):
    assert _assignment(_build(name)).change_interval_search == (
        FORWARD if name == COMPARISON else REVERSE
    )


@pytest.mark.parametrize(
    "dataset,identifier",
    [
        ("circle2", "circle2_interval100_seed0_f07c86311252"),
        ("fmnist2", "fmnist2_interval100_seed0_35a843b0567c"),
    ],
)
def test_existing_every_sample_configuration_keeps_settings_and_completed_identifier(
    dataset, identifier
):
    conditions = replace(CONDITIONS, dataset_name=dataset, per_client_sample_count=5000)
    name = configurations.EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME
    settings = _build(name, conditions)
    default = _build("fedsda", conditions)
    assert settings == default
    assert make_run_identifier(configuration_name=name, run_settings=settings) == identifier
    assert make_run_identifier(configuration_name="fedsda", run_settings=default) != identifier


def test_comparison_configuration_changes_only_search_and_is_protected_without_golden():
    assert (
        configurations.MINIMUM_INTERVAL_MULTIPLES_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME
        == COMPARISON
    )
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-1] == COMPARISON
    default, comparison = _build("fedsda"), _build(COMPARISON)
    participant = default.run_participant_settings
    client = participant.run_client_settings
    assert comparison == replace(
        default,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                training_data_assignment_settings=replace(
                    _assignment(default), change_interval_search=FORWARD
                ),
            ),
        ),
    )
    assert golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[COMPARISON] == (
        "tests/refactoring/test_fedsda_every_sample_change_interval_search_default.py"
    )


def test_comparison_settings_match_version15_fedsda_settings(monkeypatch):
    expected = golden.build_case_settings("fedsda", baseline_version=15)
    original = golden.build_experiment_run_settings

    def build(**kwargs):
        return original(**{**kwargs, "configuration_name": COMPARISON})

    monkeypatch.setattr(golden, "build_experiment_run_settings", build)
    assert golden.build_case_settings("fedsda", baseline_version=15) == expected
    assert _assignment(_build(COMPARISON)).change_interval_search == FORWARD


def test_non_mdl_legacy_comparison_explicitly_keeps_forward_search():
    client = use_legacy_compatible_policies(
        _build("fedsda")
    ).run_participant_settings.run_client_settings
    assert client.training_data_assignment_settings.alarm_interval_assignment_policy == (
        "detector_change_span_and_mean_loss_increase_tolerance"
    )
    assert client.training_data_assignment_settings.change_interval_search == FORWARD
