"""「新しいモデル」の仮説の学習者の組と、初期モデルの損失の統計の推定の仕方の、設定・接続・記録・計算回数を検証する。"""

from dataclasses import replace

import pytest
import torch
from test_fedsda_minimum_description_length_alarm_interval_assignment import build_small_settings
from test_initial_model_pretraining import make_pretraining_arguments

import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as selection_module
import federated_learning_experiments.runtime.fedsda_run_participant_factory as factory_module
from federated_learning_experiments.core.settings_field_validation import (
    RunSettingsValidationError,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    measure_model_computation,
)
from federated_learning_experiments.learning.training.prequential_code_length import (
    compute_reinitialized_learner_prequential_code_length,
    compute_warm_started_learner_prequential_code_length,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    NO_NEW_MODEL_LEARNER_NAME,
    REINITIALIZED_LEARNER_NAME,
    WARM_STARTED_LEARNER_NAME,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME,
    NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES,
    REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME,
    REINITIALIZED_LEARNER_POLICY_NAME,
    validate_new_model_hypothesis_learners,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.runtime.initial_model_pretraining import (
    FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME,
    RESUBSTITUTION_ESTIMATION_NAME,
    estimate_cross_validated_loss_statistics,
    pretrain_initial_model,
)

WARM = REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME
CROSS_VALIDATION = FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME


def with_policies(
    settings,
    *,
    new_model_hypothesis_learners=None,
    initial_model_loss_statistics_estimation=None,
    alarm_interval_assignment_policy=None,
):
    participant = settings.run_participant_settings
    client = participant.run_client_settings
    assignment_changes = {
        name: value
        for name, value in (
            ("new_model_hypothesis_learners", new_model_hypothesis_learners),
            ("alarm_interval_assignment_policy", alarm_interval_assignment_policy),
        )
        if value is not None
    }
    participant_changes = {}
    if initial_model_loss_statistics_estimation is not None:
        participant_changes["initial_model_loss_statistics_estimation"] = (
            initial_model_loss_statistics_estimation
        )
    return replace(
        settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                training_data_assignment_settings=replace(
                    client.training_data_assignment_settings, **assignment_changes
                ),
            ),
            **participant_changes,
        ),
    )


def test_learner_policy_names_and_validation():
    assert NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES == (
        REINITIALIZED_LEARNER_POLICY_NAME,
        REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME,
    )
    assert REINITIALIZED_LEARNER_POLICY_NAME == "reinitialized_learner"
    assert WARM == "reinitialized_and_warm_started_learners"
    for policy in NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES:
        validate_new_model_hypothesis_learners(new_model_hypothesis_learners=policy)
    with pytest.raises(ValueError):
        validate_new_model_hypothesis_learners(new_model_hypothesis_learners="other")

    class DerivedStr(str):
        pass

    with pytest.raises(TypeError):
        validate_new_model_hypothesis_learners(new_model_hypothesis_learners=DerivedStr(WARM))
    assignment = build_small_settings().run_participant_settings.run_client_settings.training_data_assignment_settings
    for policy in NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES:
        assert (
            replace(assignment, new_model_hypothesis_learners=policy).new_model_hypothesis_learners
            == policy
        )
    with pytest.raises(RunSettingsValidationError):
        replace(assignment, new_model_hypothesis_learners="other")


def test_warm_started_learners_are_rejected_without_the_minimum_description_length_selection():
    with pytest.raises(RunSettingsValidationError) as error_information:
        with_policies(
            build_small_settings(),
            new_model_hypothesis_learners=WARM,
            alarm_interval_assignment_policy=DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME,
        )
    assert error_information.value.configuration_parameter_name == "new_model_hypothesis_learners"
    # 最小記述長の選択とは組み合わせられる。
    assert (
        with_policies(
            build_small_settings(), new_model_hypothesis_learners=WARM
        ).run_participant_settings.run_client_settings.training_data_assignment_settings.new_model_hypothesis_learners
        == WARM
    )


def test_initial_model_loss_statistics_estimation_accepts_only_declared_values():
    settings = build_small_settings()
    for estimation in (RESUBSTITUTION_ESTIMATION_NAME, CROSS_VALIDATION):
        assert (
            with_policies(
                settings, initial_model_loss_statistics_estimation=estimation
            ).run_participant_settings.initial_model_loss_statistics_estimation
            == estimation
        )
    for rejected in ("other", None, True):
        with pytest.raises(RunSettingsValidationError) as error_information:
            replace(
                settings.run_participant_settings, initial_model_loss_statistics_estimation=rejected
            )
        assert (
            error_information.value.configuration_parameter_name
            == "initial_model_loss_statistics_estimation"
        )


def test_cross_validation_needs_at_least_as_many_pretraining_samples_as_folds():
    participant = with_policies(
        build_small_settings(), initial_model_loss_statistics_estimation=CROSS_VALIDATION
    ).run_participant_settings
    for sample_count, accepted in ((4, False), (5, True)):
        pretraining = replace(
            participant.initial_model_pretraining_settings, pretraining_sample_count=sample_count
        )
        if accepted:
            assert (
                replace(
                    participant, initial_model_pretraining_settings=pretraining
                ).initial_model_pretraining_settings.pretraining_sample_count
                == 5
            )
        else:
            with pytest.raises(RunSettingsValidationError):
                replace(participant, initial_model_pretraining_settings=pretraining)
    # 再代入の推定では、標本が少なくても受け入れる。
    resubstitution = replace(participant, initial_model_loss_statistics_estimation="resubstitution")
    assert (
        replace(
            resubstitution,
            initial_model_pretraining_settings=replace(
                participant.initial_model_pretraining_settings, pretraining_sample_count=4
            ),
        ).initial_model_loss_statistics_estimation
        == "resubstitution"
    )


def test_previous_policies_never_build_the_warm_started_learner_or_cross_validation(monkeypatch):
    def fail(**arguments):
        raise AssertionError("the new computations must not be called")

    monkeypatch.setattr(
        selection_module, "compute_warm_started_learner_prequential_code_length", fail
    )
    monkeypatch.setattr(factory_module, "estimate_cross_validated_loss_statistics", fail)
    run = execute_fedsda_experiment_run(
        run_settings=with_policies(
            build_small_settings(),
            new_model_hypothesis_learners=REINITIALIZED_LEARNER_POLICY_NAME,
            initial_model_loss_statistics_estimation=RESUBSTITUTION_ESTIMATION_NAME,
        )
    )
    tables = build_fedsda_run_record_tables(experiment_run=run)
    selected = tables["alarm_interval_assignment_selection.selected_hypothesis"].tolist()
    names = tables["alarm_interval_assignment_selection.new_model_learner_name"].tolist()
    assert "new_model" in selected
    assert all(
        name
        == (REINITIALIZED_LEARNER_NAME if hypothesis == "new_model" else NO_NEW_MODEL_LEARNER_NAME)
        for hypothesis, name in zip(selected, names, strict=True)
    )


def test_new_policies_record_the_learner_and_replace_the_initial_statistics(monkeypatch):
    warm_started_calls = []
    original_warm_started = selection_module.compute_warm_started_learner_prequential_code_length

    def recording_warm_started(**arguments):
        warm_started_calls.append(arguments["input_features"].shape[0])
        return original_warm_started(**arguments)

    estimated = []
    original_estimate = factory_module.estimate_cross_validated_loss_statistics

    def recording_estimate(**arguments):
        statistics = original_estimate(**arguments)
        estimated.append((arguments["fold_count"], statistics))
        return statistics

    assembled_statistics = []
    original_assemble = factory_module.assemble_fedsda_run_client

    def recording_assemble(**arguments):
        assembled_statistics.append(arguments["initial_loss_statistics"])
        return original_assemble(**arguments)

    pretrained = []
    original_pretrain = factory_module.pretrain_initial_model

    def recording_pretrain(**arguments):
        result = original_pretrain(**arguments)
        pretrained.append(result)
        return result

    monkeypatch.setattr(
        selection_module,
        "compute_warm_started_learner_prequential_code_length",
        recording_warm_started,
    )
    monkeypatch.setattr(
        factory_module, "estimate_cross_validated_loss_statistics", recording_estimate
    )
    monkeypatch.setattr(factory_module, "assemble_fedsda_run_client", recording_assemble)
    monkeypatch.setattr(factory_module, "pretrain_initial_model", recording_pretrain)
    run = execute_fedsda_experiment_run(
        run_settings=with_policies(
            build_small_settings(),
            new_model_hypothesis_learners=WARM,
            initial_model_loss_statistics_estimation=CROSS_VALIDATION,
        )
    )
    assert warm_started_calls
    assert len(estimated) == 1 and estimated[0][0] == 5
    statistics = estimated[0][1]
    assert statistics != pretrained[0].loss_statistics
    assert len(assembled_statistics) == 2
    assert all(assembled is statistics for assembled in assembled_statistics)
    tables = build_fedsda_run_record_tables(experiment_run=run)
    selected = tables["alarm_interval_assignment_selection.selected_hypothesis"].tolist()
    names = tables["alarm_interval_assignment_selection.new_model_learner_name"].tolist()
    assert len(names) == len(selected) > 0
    for hypothesis, name in zip(selected, names, strict=True):
        if hypothesis == "new_model":
            assert name in (REINITIALIZED_LEARNER_NAME, WARM_STARTED_LEARNER_NAME)
        else:
            assert name == NO_NEW_MODEL_LEARNER_NAME


def test_warm_started_learner_is_counted_like_the_reinitialized_learner():
    torch.manual_seed(3)
    arguments = make_pretraining_arguments(class_count=4)
    template = pretrain_initial_model(**arguments).classifier
    features = torch.randn(7, 2)
    labels = torch.randint(0, 4, (7, 1)).float()
    with measure_model_computation() as warm_meter:
        compute_warm_started_learner_prequential_code_length(
            template_classifier=template,
            input_features=features,
            observed_class_labels=labels,
            parameter_optimizer_settings=arguments["parameter_optimizer_settings"],
        )
    with measure_model_computation() as reinitialized_meter:
        compute_reinitialized_learner_prequential_code_length(
            template_classifier=template,
            input_features=features,
            observed_class_labels=labels,
            parameter_optimizer_settings=arguments["parameter_optimizer_settings"],
            initialization_seed=1,
        )
    warm = warm_meter.get_model_computation_counts()
    assert warm == reinitialized_meter.get_model_computation_counts()
    assert warm.concept_specific_part_training_example_count == 7
    assert warm.concept_specific_part_inference_example_count == 7


def test_cross_validation_counts_the_fold_training_and_held_out_evaluation():
    arguments = make_pretraining_arguments()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        pretrained = pretrain_initial_model(**arguments)
    settings = arguments["initial_model_pretraining_settings"]
    sample_count = settings.pretraining_sample_count
    with measure_model_computation() as meter:
        estimate_cross_validated_loss_statistics(
            pretraining_samples=pretrained.pretraining_samples,
            fold_count=5,
            initial_model_pretraining_settings=settings,
            model_architecture_settings=arguments["model_architecture_settings"],
            input_feature_count=arguments["input_feature_count"],
            hidden_layer_widths=arguments["hidden_layer_widths"],
            class_count=arguments["class_count"],
            parameter_optimizer_settings=arguments["parameter_optimizer_settings"],
        )
    counts = meter.get_model_computation_counts()
    # 各標本は、5組のうち4組で学習に使い、1組で評価する。
    assert (
        counts.shared_part_training_example_count
        == 4 * sample_count * settings.pretraining_epoch_count
    )
    assert counts.shared_part_inference_example_count == sample_count
    assert counts.concept_specific_part_inference_example_count == sample_count


def test_cross_validation_is_counted_in_the_preparation_and_not_in_the_metrics():
    runs = {
        estimation: execute_fedsda_experiment_run(
            run_settings=with_policies(
                build_small_settings(), initial_model_loss_statistics_estimation=estimation
            )
        )
        for estimation in (RESUBSTITUTION_ESTIMATION_NAME, CROSS_VALIDATION)
    }
    preparation = {
        estimation: run.measured_run.preparation_model_computation_counts
        for estimation, run in runs.items()
    }
    pretraining = runs[
        CROSS_VALIDATION
    ].run_settings.run_participant_settings.initial_model_pretraining_settings
    assert (
        preparation[CROSS_VALIDATION].shared_part_training_example_count
        - preparation[RESUBSTITUTION_ESTIMATION_NAME].shared_part_training_example_count
        == 4 * pretraining.pretraining_sample_count * pretraining.pretraining_epoch_count
    )
    # 交差検証は評価だけを行い（学習に使った標本での統計の評価は行わない）、再代入の推定の評価も同じ数。
    assert (
        preparation[CROSS_VALIDATION].shared_part_inference_example_count
        - preparation[RESUBSTITUTION_ESTIMATION_NAME].shared_part_inference_example_count
        == pretraining.pretraining_sample_count
    )
    # 指標は、初期準備の後から数える（初期準備の計算を含まない）。
    for run in runs.values():
        counts = run.measured_run
        total = (
            sum(
                round_counts.local_processing_model_computation_counts.shared_part_training_example_count
                + round_counts.synchronization_model_computation_counts.shared_part_training_example_count
                for round_counts in counts.round_model_computation_counts
            )
            + counts.finalization_model_computation_counts.shared_part_training_example_count
        )
        assert counts.model_computation_counts.shared_part_training_example_count == total


def test_every_fedsda_configuration_inherits_the_new_defaults_except_candidate_validation():
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
        build_experiment_run_settings,
    )

    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=2,
        per_client_sample_count=200,
        server_aggregation_interval_per_client_samples=50,
    )
    checked = 0
    for configuration_name in EXPERIMENT_CONFIGURATION_NAMES:
        settings = build_experiment_run_settings(
            configuration_name=configuration_name, experiment_run_conditions=conditions
        )
        if type(settings) is not FedsdaRunSettings:
            continue
        checked += 1
        participant = settings.run_participant_settings
        learners = participant.run_client_settings.training_data_assignment_settings.new_model_hypothesis_learners
        if configuration_name == "fedsda-with-candidate-validation":
            assert (learners, participant.initial_model_loss_statistics_estimation) == (
                REINITIALIZED_LEARNER_POLICY_NAME,
                RESUBSTITUTION_ESTIMATION_NAME,
            ), configuration_name
        else:
            assert (learners, participant.initial_model_loss_statistics_estimation) == (
                WARM,
                CROSS_VALIDATION,
            ), configuration_name
    assert checked > 2


def test_warm_started_learner_computation_is_counted_in_the_run_metrics(monkeypatch):
    """warm startの学習者の計算を1回余分に行わせると、全体runの指標が、余分な計算の計数だけ増える。

    余分な計算の結果は捨てるので、runの軌跡は変わらない（指標の増え方が、余分な計算を内側で計った計数と一致する）。
    """
    import dataclasses

    settings = with_policies(
        build_small_settings(),
        new_model_hypothesis_learners=WARM,
        initial_model_loss_statistics_estimation=CROSS_VALIDATION,
    )
    baseline = execute_fedsda_experiment_run(run_settings=settings)
    original = selection_module.compute_warm_started_learner_prequential_code_length
    extra_counts = []

    def computing_twice(**arguments):
        with measure_model_computation() as meter:
            original(**arguments)
        extra_counts.append(meter.get_model_computation_counts())
        return original(**arguments)

    monkeypatch.setattr(
        selection_module, "compute_warm_started_learner_prequential_code_length", computing_twice
    )
    doubled = execute_fedsda_experiment_run(run_settings=settings)
    assert extra_counts
    assert doubled.run_metrics.prediction_accuracy == baseline.run_metrics.prediction_accuracy
    baseline_counts = dataclasses.asdict(baseline.run_metrics.model_computation_counts)
    doubled_counts = dataclasses.asdict(doubled.run_metrics.model_computation_counts)
    for name, value in baseline_counts.items():
        assert doubled_counts[name] - value == sum(
            getattr(counts, name) for counts in extra_counts
        ), name
    assert sum(counts.concept_specific_part_training_example_count for counts in extra_counts) > 0
