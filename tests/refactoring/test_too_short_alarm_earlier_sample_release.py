"""短い警報のあと、現行モデルへ取り込んだ前区間を、保留から外す扱い（LEGACY-002の修正）。

実旧には、この扱いの正解が無い（旧は、保留を全部残す）。警報応答そのもの（区間の準備と、監視の再開）は、
実旧と照合し、保留の消費と記録は、状態を直接確かめる。全体runでは、同じ標本が2回取り込まれないことを確かめる。
"""

from dataclasses import replace

import behavior_golden_cases as golden
import pytest
import torch
from test_alarm_buffer_response import assert_buffer_response_matches_legacy
from test_alarm_response_completion import (
    MONITORED_LOSSES_AFTER_COMPLETION,
    build_response_completion_oracle,
    observe_monitored_losses_in_both_implementations,
    run_legacy_alarm_with_real_completion,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.alarm_occurrence_handling as alarm_occurrence_module
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
    PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES,
    RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME,
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    respond_to_alarm_with_buffered_samples,
)
from federated_learning_experiments.runtime.alarm_response_completion import (
    complete_alarm_buffer_response,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

# ---- 設定と、保留の位置のowner ----


def make_pending_buffer(*, policy_name, capacity=3, sample_indices=()):
    pending_buffer = PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=capacity,
            pending_sample_retention_after_too_short_change_interval_policy=policy_name,
        )
    )
    for sample_index in sample_indices:
        pending_buffer.append_observed_sample_index(sample_index=sample_index)
    return pending_buffer


def test_policy_names_are_declared_and_owner_reports_the_selected_policy():
    assert PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES == (
        RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME,
        KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
    )
    assert (
        RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME
        == "release_earlier_samples_assigned_to_current_model"
    )
    assert KEEP_ALL_PENDING_SAMPLES_POLICY_NAME == "keep_all_pending_samples"
    assert make_pending_buffer(
        policy_name=RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME
    ).releases_earlier_samples_after_too_short_change_interval
    assert not make_pending_buffer(
        policy_name=KEEP_ALL_PENDING_SAMPLES_POLICY_NAME
    ).releases_earlier_samples_after_too_short_change_interval
    with pytest.raises(AttributeError):
        make_pending_buffer(
            policy_name=KEEP_ALL_PENDING_SAMPLES_POLICY_NAME
        ).releases_earlier_samples_after_too_short_change_interval = True


@pytest.mark.parametrize(
    "policy_name", PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES
)
@pytest.mark.parametrize(
    ("released_sample_indices", "remaining_sample_indices"),
    [((), (10, 11, 12, 13)), ((10,), (11, 12, 13)), ((10, 11, 12), (13,)), ((10, 11, 12, 13), ())],
)
def test_release_earliest_pending_sample_indices_releases_exactly_the_given_prefix(
    policy_name, released_sample_indices, remaining_sample_indices
):
    """古い側を外す操作は、方針に依らない（方針を読むのは、警報応答の完了処理）。"""
    pending_buffer = make_pending_buffer(policy_name=policy_name, sample_indices=(10, 11, 12, 13))
    assert (
        pending_buffer.release_earliest_pending_sample_indices(
            released_sample_indices=released_sample_indices
        )
        == released_sample_indices
    )
    state = pending_buffer.get_state_snapshot()
    assert state.pending_sample_indices == remaining_sample_indices
    # 最後の観測位置は変わらず、次の位置を続けて足せる。容量の判定は、残った件数で行う。
    assert state.last_observed_sample_index == 13
    pending_buffer.append_observed_sample_index(sample_index=14)
    assert pending_buffer.get_state_snapshot().pending_sample_indices == (
        *remaining_sample_indices,
        14,
    )
    assert (
        pending_buffer.get_sample_indices_exceeding_capacity()
        == (
            *remaining_sample_indices,
            14,
        )[: max(0, len(remaining_sample_indices) + 1 - 3)]
    )


@pytest.mark.parametrize(
    ("invalid_released_sample_indices", "expected_error"),
    [
        ((11,), ValueError),
        ((10, 12), ValueError),
        ((11, 10), ValueError),
        ((10, 11, 12, 13, 14), ValueError),
        ((-1,), ValueError),
        ([10], TypeError),
        (None, TypeError),
        ((True,), TypeError),
        ((10.0,), TypeError),
    ],
)
def test_release_earliest_pending_sample_indices_rejects_other_indices_without_change(
    invalid_released_sample_indices, expected_error
):
    pending_buffer = make_pending_buffer(
        policy_name=RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME,
        sample_indices=(10, 11, 12, 13),
    )
    state_before = pending_buffer.get_state_snapshot()
    with pytest.raises(expected_error):
        pending_buffer.release_earliest_pending_sample_indices(
            released_sample_indices=invalid_released_sample_indices
        )
    assert pending_buffer.get_state_snapshot() == state_before


def test_settings_reject_undeclared_policy():
    for invalid_policy in ("", "keep_all", "release", None, 1):
        with pytest.raises(Exception) as raised_error:
            TrainingDataAssignmentSettings(
                pending_assignment_buffer_capacity_samples=3,
                pending_sample_retention_after_too_short_change_interval_policy=invalid_policy,
            )
        assert type(raised_error.value).__name__ == "RunSettingsValidationError"


# ---- 警報応答の完了処理 ----


def replace_pending_buffer_with_policy(*, response_arguments, completion_arguments, policy_name):
    """応答と完了の引数の、保留の位置のownerを、同じ位置・同じ容量で、方針だけが違うものへ替える。"""
    previous_buffer = response_arguments["pending_training_assignment_buffer"]
    assert completion_arguments["pending_training_assignment_buffer"] is previous_buffer
    replacing_buffer = make_pending_buffer(
        policy_name=policy_name,
        capacity=previous_buffer.capacity_sample_count,
        sample_indices=previous_buffer.get_state_snapshot().pending_sample_indices,
    )
    assert replacing_buffer.get_state_snapshot() == previous_buffer.get_state_snapshot()
    response_arguments["pending_training_assignment_buffer"] = replacing_buffer
    completion_arguments["pending_training_assignment_buffer"] = replacing_buffer


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "policy_name", PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES
)
@pytest.mark.parametrize(
    "earlier_sample_count,estimated_change_span_sample_count,minimum_change_interval_sample_count,expected_outcome_is_too_short",
    [(3, 2, 3, True), (3, 6, 7, True), (3, 99, 99, True), (3, 3, 3, False), (0, 99, 3, False)],
)
def test_completion_releases_earlier_interval_after_too_short_change_interval(
    class_count,
    policy_name,
    earlier_sample_count,
    estimated_change_span_sample_count,
    minimum_change_interval_sample_count,
    expected_outcome_is_too_short,
    monkeypatch,
    valid_run_settings_mapping,
):
    """短い警報: 前区間を外す方針では、前区間の位置だけを外す。全部残す方針では、外さない。

    区間の準備（前区間の取込みと評価標本の保存）と、監視の再開は、どちらの方針でも、実旧と同じ。
    短い警報でない警報は、どちらの方針でも、保留を全部消費する。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            response_arguments,
            completion_arguments,
            preparation_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            class_count=class_count,
            alarm_interval_resolution_case="current_model_maintained",
            earlier_sample_count=earlier_sample_count,
            estimated_change_span_sample_count=estimated_change_span_sample_count,
            minimum_change_interval_sample_count=minimum_change_interval_sample_count,
        )
        replace_pending_buffer_with_policy(
            response_arguments=response_arguments,
            completion_arguments=completion_arguments,
            policy_name=policy_name,
        )
        for arguments in (preparation_arguments, resolution_arguments):
            if "pending_training_assignment_buffer" in arguments:
                arguments["pending_training_assignment_buffer"] = response_arguments[
                    "pending_training_assignment_buffer"
                ]
        initial_training_model_id = legacy_client.current_model_id
        initial_torch_random_state = torch.get_rng_state().clone()
        legacy_result = run_legacy_alarm_with_real_completion(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        torch.set_rng_state(initial_torch_random_state)
        alarm_buffer_response = respond_to_alarm_with_buffered_samples(**response_arguments)
        pending_buffer = completion_arguments["pending_training_assignment_buffer"]
        state_before_completion = pending_buffer.get_state_snapshot()
        prepared_intervals = alarm_buffer_response.prepared_alarm_training_intervals
        earlier_sample_indices = tuple(
            observation.sample_index for observation in prepared_intervals.earlier_observations
        )
        change_interval_sample_indices = tuple(
            observation.sample_index
            for observation in prepared_intervals.change_interval_observations
        )
        assert (
            alarm_buffer_response.response_outcome == "alarm_change_interval_too_short"
        ) is expected_outcome_is_too_short
        response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=alarm_buffer_response, **completion_arguments
        )
        state_after_completion = pending_buffer.get_state_snapshot()
        if not expected_outcome_is_too_short:
            expected_drained_sample_indices = state_before_completion.pending_sample_indices
            expected_remaining_sample_indices = ()
        elif policy_name == RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME:
            expected_drained_sample_indices = earlier_sample_indices
            expected_remaining_sample_indices = change_interval_sample_indices
        else:
            expected_drained_sample_indices = ()
            expected_remaining_sample_indices = state_before_completion.pending_sample_indices
        assert response_completion.drained_pending_sample_indices == expected_drained_sample_indices
        assert state_after_completion.pending_sample_indices == expected_remaining_sample_indices
        assert (
            state_after_completion.last_observed_sample_index
            == state_before_completion.last_observed_sample_index
        )
        if expected_outcome_is_too_short:
            # 条件が、前区間のある場合と無い場合の両方を覆う。
            assert bool(earlier_sample_indices) is (estimated_change_span_sample_count != 99)
            assert response_completion.previous_training_model_id == initial_training_model_id
            assert response_completion.current_training_model_id == initial_training_model_id
        # 区間の準備と解決（取込み、評価標本、統計、学習帰属）は、方針に依らず、実旧と同じ。
        assert_buffer_response_matches_legacy(
            response=alarm_buffer_response,
            response_arguments=response_arguments,
            preparation_arguments=preparation_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            initial_training_model_id=initial_training_model_id,
        )
        # 再開した監視は、方針に依らず、実旧と同じ値で進む。
        observe_monitored_losses_in_both_implementations(
            completion_arguments=completion_arguments,
            legacy_client=legacy_client,
            class_count=class_count,
            first_sample_index=response_arguments["proposal_sample_index"] + 1,
            monitored_losses=MONITORED_LOSSES_AFTER_COMPLETION,
        )
        # 記録の検査: 短い警報では、空か、前区間の位置。それ以外の並びは、拒否する。
        if expected_outcome_is_too_short:
            for accepted_sample_indices in ((), earlier_sample_indices):
                replace(response_completion, drained_pending_sample_indices=accepted_sample_indices)
            for rejected_sample_indices in (
                change_interval_sample_indices,
                state_before_completion.pending_sample_indices,
                earlier_sample_indices[:-1] if len(earlier_sample_indices) > 1 else (10**6,),
            ):
                if rejected_sample_indices in ((), earlier_sample_indices):
                    continue
                with pytest.raises(ValueError):
                    replace(
                        response_completion, drained_pending_sample_indices=rejected_sample_indices
                    )


# ---- 全体run: 同じ標本が、2回取り込まれない ----

SINE2_EVENT_CASE_NAME = "fedsda-sine2-events"
ADWIN_CONFIGURATION_NAMES = (
    "fedsda-classwise-adwin-detection",
    "fedsda-overall-loss-adwin-detection",
)


def replace_too_short_change_interval_policy(run_settings, policy_name):
    run_participant_settings = run_settings.run_participant_settings
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            run_participant_settings,
            run_client_settings=replace(
                run_client_settings,
                training_data_assignment_settings=replace(
                    run_client_settings.training_data_assignment_settings,
                    pending_sample_retention_after_too_short_change_interval_policy=policy_name,
                ),
            ),
        ),
    )


def build_sine2_event_run_settings(*, policy_name, detection_configuration_name=None):
    """基準版2のケース`fedsda-sine2-events`の設定（登録の扱いは既定）で、短い警報のあとの扱いを選ぶ。

    `detection_configuration_name`を渡すと、検出の設定だけを、その構成（ADWINで監視する構成）のものへ替える。
    """
    run_settings = golden.build_case_settings(SINE2_EVENT_CASE_NAME, 2)
    if detection_configuration_name is not None:
        detection_client_settings = build_experiment_run_settings(
            configuration_name=detection_configuration_name,
            experiment_run_conditions=run_settings.execution_settings.experiment_run_conditions,
        ).run_participant_settings.run_client_settings
        run_participant_settings = run_settings.run_participant_settings
        run_client_settings = run_participant_settings.run_client_settings
        run_settings = replace(
            run_settings,
            run_participant_settings=replace(
                run_participant_settings,
                run_client_settings=replace(
                    run_client_settings,
                    loss_change_detection_settings=detection_client_settings.loss_change_detection_settings,
                    scalar_settings=replace(
                        run_client_settings.scalar_settings,
                        detector_name=detection_client_settings.scalar_settings.detector_name,
                    ),
                ),
            ),
        )
    return replace_too_short_change_interval_policy(run_settings, policy_name)


def execute_run_observing_alarm_responses(run_settings):
    """全体runを実行し、警報のたびに、保留の標本のうち、すでに学習標本へ取り込まれているものの数を観測する。

    警報の推定変化点の算出と、ADWINの補助の判定は、警報の応答へ渡す保留の標本と、同じ保留を読む。
    """
    observation = dict(
        alarm_count=0,
        too_short_alarm_count=0,
        too_short_alarm_with_earlier_interval_count=0,
        pending_sample_already_assigned_count=0,
    )
    real_response = alarm_occurrence_module.respond_to_alarm_with_buffered_samples

    def observe_response(**response_arguments):
        assigned_sample_object_ids = {
            id(training_sample)
            for collection in response_arguments[
                "training_sample_store"
            ].snapshot_ordered_model_training_samples()
            for training_sample in collection.training_samples
        }
        observation["alarm_count"] += 1
        observation["pending_sample_already_assigned_count"] += sum(
            id(indexed_observation.training_sample) in assigned_sample_object_ids
            for indexed_observation in response_arguments["pending_sample_observations"]
        )
        alarm_buffer_response = real_response(**response_arguments)
        if alarm_buffer_response.response_outcome == "alarm_change_interval_too_short":
            observation["too_short_alarm_count"] += 1
            observation["too_short_alarm_with_earlier_interval_count"] += bool(
                alarm_buffer_response.prepared_alarm_training_intervals.earlier_observations
            )
        return alarm_buffer_response

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            alarm_occurrence_module, "respond_to_alarm_with_buffered_samples", observe_response
        )
        with torch.random.fork_rng(devices=[]):
            experiment_run = execute_fedsda_experiment_run(run_settings=run_settings)
    return experiment_run, observation


@pytest.fixture(scope="module")
def sine2_event_runs_by_policy():
    return {
        policy_name: execute_run_observing_alarm_responses(
            build_sine2_event_run_settings(policy_name=policy_name)
        )
        for policy_name in PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES
    }


def test_whole_run_assigns_each_sample_at_most_once(sine2_event_runs_by_policy):
    """既定の扱い: 短い警報が起きても、同じ標本が2回取り込まれない。警報が読む保留は、未取込みの標本だけ。"""
    experiment_run, observation = sine2_event_runs_by_policy[
        RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME
    ]
    assert observation["too_short_alarm_with_earlier_interval_count"] > 0
    assert golden.count_repeated_training_sample_assignments(experiment_run) == 0
    assert observation["pending_sample_already_assigned_count"] == 0
    # 観測した標本は、学習標本（どれかのモデル）か、runの終わりの保留か、進行中の候補検証の、どれかにある。
    conditions = experiment_run.run_settings.execution_settings.experiment_run_conditions
    assigned_or_pending_sample_count = 0
    for run_client in experiment_run.measured_run.participants.client_operations:
        owners = run_client.owners
        assigned_or_pending_sample_count += sum(
            len(collection.training_samples)
            for collection in owners.training_sample_store.snapshot_ordered_model_training_samples()
        ) + len(owners.pending_sample_observation_store.snapshot_pending_sample_observations())
    assert assigned_or_pending_sample_count <= (
        conditions.client_count * conditions.per_client_sample_count
    )


def test_whole_run_with_keep_all_policy_assigns_samples_twice(sine2_event_runs_by_policy):
    """旧と同じ扱い: 同じ条件で、同じ標本の2回目の取込みが起きる（この条件が、LEGACY-002の経路を通ることの確認）。"""
    experiment_run, observation = sine2_event_runs_by_policy[KEEP_ALL_PENDING_SAMPLES_POLICY_NAME]
    assert observation["too_short_alarm_with_earlier_interval_count"] > 0
    assert golden.count_repeated_training_sample_assignments(experiment_run) == 487
    assert observation["pending_sample_already_assigned_count"] > 0


def test_whole_runs_of_both_too_short_alarm_policies_produce_different_metrics(
    sine2_event_runs_by_policy,
):
    release_run, _ = sine2_event_runs_by_policy[
        RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME
    ]
    keep_run, _ = sine2_event_runs_by_policy[KEEP_ALL_PENDING_SAMPLES_POLICY_NAME]
    assert release_run.run_metrics != keep_run.run_metrics


@pytest.mark.parametrize("detection_configuration_name", ADWIN_CONFIGURATION_NAMES)
def test_adwin_monitoring_run_reads_only_unassigned_pending_samples(detection_configuration_name):
    """ADWINで監視する構成: 補助の判定と推定変化点が読む保留に、取込み済みの標本が入らない（既定の扱い）。"""
    experiment_run, observation = execute_run_observing_alarm_responses(
        build_sine2_event_run_settings(
            policy_name=RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME,
            detection_configuration_name=detection_configuration_name,
        )
    )
    assert observation["alarm_count"] > 0
    assert observation["pending_sample_already_assigned_count"] == 0
    assert golden.count_repeated_training_sample_assignments(experiment_run) == 0


def test_every_fedsda_configuration_defaults_to_releasing_earlier_samples():
    fedsda_configuration_count = 0
    for configuration_name in EXPERIMENT_CONFIGURATION_NAMES:
        run_settings = golden.build_case_settings(configuration_name, golden.BASELINE_VERSION)
        if type(run_settings) is not FedsdaRunSettings:
            continue
        fedsda_configuration_count += 1
        default_run_settings = build_experiment_run_settings(
            configuration_name=configuration_name,
            experiment_run_conditions=run_settings.execution_settings.experiment_run_conditions,
        )
        assert (
            default_run_settings.run_participant_settings.run_client_settings.training_data_assignment_settings.pending_sample_retention_after_too_short_change_interval_policy
            == RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME
        ), configuration_name
    assert fedsda_configuration_count == 11
