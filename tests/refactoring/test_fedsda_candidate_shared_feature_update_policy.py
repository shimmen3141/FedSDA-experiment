"""候補の学習で、共有部を固定する設定（`candidate_shared_feature_update_policy`）の動作と、組合せの検査。

「候補と一緒に更新する」値の動作は、これまでの実旧との照合（`test_candidate_epoch_training.py`ほか）が、
この値を明示して守る。ここでは、「固定する」値が、共有部だけを変えないことと、乱数の消費を変えないことを確かめる。
"""

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_candidate_epoch_training import build_candidate_epoch_training_oracle
from test_joint_model_parameter_update import assert_nested_state_equal

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.learning.training.candidate_epoch_training import (
    train_candidate_classifier_epochs,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)

UPDATE = "update_shared_features_with_candidate"
FIXED = "keep_shared_features_fixed"
VALIDATION = "current_model_first_reuse_then_two_segment_candidate_validation"
WITHOUT_VALIDATION = "current_model_first_reuse_then_candidate_adoption_without_validation"


def make_settings(*, candidate_training_strategy, policy):
    return CandidateEpochTrainingSettings(
        candidate_training_strategy=candidate_training_strategy,
        maximum_epoch_count=4,
        maximum_batch_sample_count=3,
        validation_sample_fraction=0.2,
        consecutive_non_improving_epoch_limit=10,
        minimum_validation_loss_decrease=0.0,
        candidate_shared_feature_update_policy=policy,
    )


def capture_random_states():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def assert_random_states_equal(actual, expected):
    assert torch.equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert actual[2][0] == expected[2][0]
    assert np.array_equal(actual[2][1], expected[2][1])
    assert actual[2][2:] == expected[2][2:]


def train_with_policy(*, candidate_training_state, candidate_training_strategy, policy, seed):
    """同じ初期状態の候補の写しを、指定した値で学習し、学習の後の候補と乱数の状態を返す。"""
    training_state = deepcopy(candidate_training_state)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    input_features = torch.arange(22, dtype=torch.float32).reshape(11, 2) / 7
    observed_class_labels = (torch.arange(11) % 2).float().reshape(11, 1)
    result = train_candidate_classifier_epochs(
        candidate_classifier=training_state.candidate_classifier,
        candidate_shared_parameter_optimizer_state=training_state.candidate_shared_parameter_optimizer_state,
        candidate_concept_specific_parameter_optimizer_state=training_state.candidate_concept_specific_parameter_optimizer_state,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        candidate_epoch_training_settings=make_settings(
            candidate_training_strategy=candidate_training_strategy, policy=policy
        ),
    )
    return training_state, result, capture_random_states()


@pytest.mark.parametrize(
    "candidate_training_strategy", ["fixed_epoch_training", "validation_loss_early_stopping"]
)
@pytest.mark.parametrize("optimizer_variant", ["standard", "sgd"])
def test_fixed_shared_features_keep_shared_parameters_and_optimizer_and_random_consumption(
    candidate_training_strategy, optimizer_variant, monkeypatch
):
    """固定すると、共有部の値とoptimizerの状態が変わらず、概念固有部は学習され、乱数の消費は変わらない。"""
    initial_random_states = capture_random_states()
    try:
        candidate_training_state, _, _ = build_candidate_epoch_training_oracle(
            class_count=2, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        classifier = candidate_training_state.candidate_classifier
        shared_before = [
            value.detach().clone() for value in classifier.feature_extractor.parameters()
        ]
        concept_before = [
            value.detach().clone()
            for value in tuple(classifier.model_owned_feature_transform.parameters())
            + tuple(classifier.classification_layer.parameters())
        ]
        shared_optimizer_before = deepcopy(
            candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer.state_dict()
        )
        concept_optimizer_before = deepcopy(
            candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
        )
        fixed_state, fixed_result, fixed_random_states = train_with_policy(
            candidate_training_state=candidate_training_state,
            candidate_training_strategy=candidate_training_strategy,
            policy=FIXED,
            seed=11,
        )
        updated_state, updated_result, updated_random_states = train_with_policy(
            candidate_training_state=candidate_training_state,
            candidate_training_strategy=candidate_training_strategy,
            policy=UPDATE,
            seed=11,
        )
        fixed_classifier = fixed_state.candidate_classifier
        for value, expected in zip(
            fixed_classifier.feature_extractor.parameters(), shared_before, strict=True
        ):
            assert torch.equal(value, expected)
        fixed_concept_values = tuple(
            fixed_classifier.model_owned_feature_transform.parameters()
        ) + tuple(fixed_classifier.classification_layer.parameters())
        assert any(
            not torch.equal(value, expected)
            for value, expected in zip(fixed_concept_values, concept_before, strict=True)
        )
        assert_nested_state_equal(
            fixed_state.candidate_shared_parameter_optimizer_state.parameter_optimizer.state_dict(),
            shared_optimizer_before,
        )
        if optimizer_variant != "sgd":
            # Adamは、stepで状態（step数・moment）が進む（SGDは、状態を持たない）。
            assert (
                fixed_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()[
                    "state"
                ]
                != concept_optimizer_before["state"]
            )
        # 一緒に更新する値では、共有部が変わる（上の確認が、常に成り立つ比較ではない）。
        assert any(
            not torch.equal(value, expected)
            for value, expected in zip(
                updated_state.candidate_classifier.feature_extractor.parameters(),
                shared_before,
                strict=True,
            )
        )
        # 学習の量と、乱数の消費（DataLoaderのshuffle、早期終了の分割）は、2つの値で同じ。
        assert fixed_result.completed_epoch_count > 0
        assert (
            fixed_result.candidate_trained_sample_count
            == updated_result.candidate_trained_sample_count
        )
        assert (
            fixed_result.candidate_parameter_update_step_count
            == updated_result.candidate_parameter_update_step_count
        )
        assert_random_states_equal(fixed_random_states, updated_random_states)
    finally:
        torch.set_rng_state(initial_random_states[0])
        random.setstate(initial_random_states[1])
        np.random.set_state(initial_random_states[2])


def build_small_settings(configuration_name, **changed_epoch_training_fields):
    settings = build_experiment_run_settings(
        configuration_name=configuration_name,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=2,
            per_client_sample_count=900,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    participant = settings.run_participant_settings
    client = participant.run_client_settings
    # 概念の変化を、goldenの大きい条件のケースと同じ頻度にする（警報と候補の学習が起きるように）。
    execution_settings = settings.execution_settings
    return replace(
        settings,
        execution_settings=replace(
            execution_settings,
            concept_schedule_settings=replace(
                execution_settings.concept_schedule_settings,
                minimum_sample_index_gap_before_change_trial=100,
                per_eligible_sample_concept_change_probability=0.015,
            ),
        ),
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                candidate_model_training_and_acceptance_settings=replace(
                    client.candidate_model_training_and_acceptance_settings,
                    candidate_model_acceptance_policy=WITHOUT_VALIDATION,
                ),
                candidate_epoch_training_settings=replace(
                    client.candidate_epoch_training_settings, **changed_epoch_training_fields
                ),
            ),
        ),
    )


def run_outputs(run_settings):
    run = execute_fedsda_experiment_run(run_settings=run_settings)
    tables = build_fedsda_run_record_tables(experiment_run=run)
    return run, tables


@pytest.mark.slow
def test_independent_models_give_the_same_run_with_either_value():
    """共有部を持たない構造では、共有部にパラメータが無いので、2つの値で、runの指標と記録が同じ。"""
    outputs = {}
    for policy in (UPDATE, FIXED):
        torch.manual_seed(0)
        outputs[policy] = run_outputs(
            build_small_settings(
                "fedsda-independent-models", candidate_shared_feature_update_policy=policy
            )
        )
    (update_run, update_tables), (fixed_run, fixed_tables) = outputs[UPDATE], outputs[FIXED]
    adaptation = update_run.run_metrics.change_response_metrics.adaptation_action_metrics
    # 候補の学習を通る条件であること（比較が空にならない）。
    assert adaptation.adopt_candidate_model_count > 0
    assert update_run.run_metrics == fixed_run.run_metrics
    assert update_tables.keys() == fixed_tables.keys()
    for name, column in update_tables.items():
        assert np.array_equal(column, fixed_tables[name], equal_nan=column.dtype.kind == "f"), name


@pytest.mark.parametrize(
    "acceptance_policy,shared_feature_policy,accepted",
    [
        (VALIDATION, UPDATE, True),
        (VALIDATION, FIXED, False),
        (WITHOUT_VALIDATION, UPDATE, True),
        (WITHOUT_VALIDATION, FIXED, True),
    ],
)
def test_candidate_validation_cannot_be_combined_with_fixed_shared_features(
    acceptance_policy, shared_feature_policy, accepted
):
    """候補検証ありと、共有部の固定の組合せだけを、run設定の組立てで拒否する。"""
    settings = build_experiment_run_settings(
        configuration_name="fedsda",
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sea2",
            random_seed=0,
            client_count=2,
            per_client_sample_count=40,
            server_aggregation_interval_per_client_samples=10,
        ),
    )
    client = settings.run_participant_settings.run_client_settings

    def build():
        return replace(
            client,
            candidate_model_training_and_acceptance_settings=replace(
                client.candidate_model_training_and_acceptance_settings,
                candidate_model_acceptance_policy=acceptance_policy,
            ),
            candidate_epoch_training_settings=replace(
                client.candidate_epoch_training_settings,
                candidate_shared_feature_update_policy=shared_feature_policy,
            ),
            # 候補検証は、最小記述長の原理で選ぶ扱いとも組み合わせられないので、ここでは今の扱いにする
            # （このtestは、共有部の固定との組合せだけを確かめる）。
            training_data_assignment_settings=replace(
                client.training_data_assignment_settings,
                alarm_interval_assignment_policy="detector_change_span_and_mean_loss_increase_tolerance",
            ),
        )

    if accepted:
        built = build()
        assert (
            built.candidate_epoch_training_settings.candidate_shared_feature_update_policy
            == shared_feature_policy
        )
    else:
        with pytest.raises(RunSettingsValidationError) as error_information:
            build()
        assert (
            error_information.value.configuration_parameter_name
            == "candidate_shared_feature_update_policy"
        )


def test_legacy_compatible_entry_fixes_and_checks_candidate_shared_feature_update():
    """旧と比べる入口は、候補の共有部を一緒に更新する値に明示し、入口の検査は、固定する値を拒否する。"""
    from test_fedsda_run_client import (
        assert_run_client_uses_legacy_compatible_policies,
        use_legacy_compatible_policies,
    )

    fixed_settings = build_small_settings("fedsda", candidate_shared_feature_update_policy=FIXED)
    legacy_compatible = use_legacy_compatible_policies(fixed_settings)
    assert (
        legacy_compatible.run_participant_settings.run_client_settings.candidate_epoch_training_settings.candidate_shared_feature_update_policy
        == UPDATE
    )
    for run_settings, accepted in ((legacy_compatible, True), (fixed_settings, False)):
        run = execute_fedsda_experiment_run(run_settings=run_settings)
        (run_client, *_) = run.measured_run.participants.client_operations
        if accepted:
            assert_run_client_uses_legacy_compatible_policies(run_client)
        else:
            with pytest.raises(AssertionError, match="実旧との照合には"):
                assert_run_client_uses_legacy_compatible_policies(run_client)
