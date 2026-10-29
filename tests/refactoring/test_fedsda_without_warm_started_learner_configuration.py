"""warm startの学習者だけを外す構成と、学習者1つの符号長を守る。"""

from dataclasses import replace

import pytest

from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    NewModelLearnerSelection,
    select_new_model_learner,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep import experiment_configurations as configurations
from tests.refactoring import behavior_golden_cases as golden

CONFIGURATION = "fedsda-without-warm-started-learner"
CASE = "fedsda-sine2-adoption-without-validation"


def test_configuration_changes_only_new_model_hypothesis_learners():
    conditions = golden.ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=3,
        per_client_sample_count=1500,
        server_aggregation_interval_per_client_samples=50,
    )
    default = configurations.build_experiment_run_settings(
        configuration_name="fedsda", experiment_run_conditions=conditions
    )
    actual = configurations.build_experiment_run_settings(
        configuration_name=CONFIGURATION, experiment_run_conditions=conditions
    )
    participant = default.run_participant_settings
    client = participant.run_client_settings
    assert actual == replace(
        default,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                training_data_assignment_settings=replace(
                    client.training_data_assignment_settings,
                    new_model_hypothesis_learners="reinitialized_learner",
                ),
            ),
        ),
    )
    assert configurations.WITHOUT_WARM_STARTED_LEARNER_CONFIGURATION_NAME == CONFIGURATION
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-1] == CONFIGURATION
    assert golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[CONFIGURATION] == (
        "tests/refactoring/test_fedsda_without_warm_started_learner_configuration.py"
    )


@pytest.mark.slow
def test_fixed_version16_case_selects_new_models_without_warm_started_learner(monkeypatch):
    original = golden.build_experiment_run_settings

    def build(**kwargs):
        return original(**{**kwargs, "configuration_name": CONFIGURATION})

    monkeypatch.setattr(golden, "build_experiment_run_settings", build)
    settings = golden.build_case_settings(CASE, baseline_version=16)
    run = execute_fedsda_experiment_run(run_settings=settings)
    tables = build_fedsda_run_record_tables(experiment_run=run)
    hypotheses = tables["alarm_interval_assignment_selection.selected_hypothesis"].tolist()
    names = tables["alarm_interval_assignment_selection.new_model_learner_name"].tolist()
    assert hypotheses.count("new_model") > 0
    assert "warm_started_learner" not in names
    assert all(
        name == ("reinitialized_learner" if hypothesis == "new_model" else "")
        for hypothesis, name in zip(hypotheses, names, strict=True)
    )


def test_single_learner_does_not_add_learner_selection_code_length():
    assert select_new_model_learner(
        prequential_code_lengths_by_learner_name=(("reinitialized_learner", 7.5),)
    ) == NewModelLearnerSelection(learner_name="reinitialized_learner", description_length=7.5)
