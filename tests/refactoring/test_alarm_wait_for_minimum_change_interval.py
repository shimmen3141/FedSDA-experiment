"""推定変化区間が最小件数に達するまで、警報を待つ扱い（IMPROVE-006）。

実旧には、この扱いの正解が無い。標本1件の処理は、「合図を、推定変化区間が短い標本でだけ伏せて、いまの扱いで
処理した場合」と、全部の状態が同じであることで確かめる（待つ＝その標本を、合図の無い標本として処理する。
最小件数に達した標本は、いまの扱いと同じに、警報として処理する）。
"""

import math
from dataclasses import fields, is_dataclass, replace

import behavior_golden_cases as golden
import pytest
import torch
from test_fedsda_run_client_settings import make_valid_run_client_settings
from test_observed_sample_processing import (
    RECORDING_ORACLE_CASES,
    build_sample_processing_oracle,
    snapshot_sample_processing_state,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.alarm_occurrence_handling as alarm_occurrence_module
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
    SHORT_CHANGE_INTERVAL_ALARM_POLICIES,
    WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.observed_sample_processing import (
    process_observed_sample,
)

SHORT_ALARM_OUTCOME = "alarm_change_interval_too_short"

# ---- 設定 ----


def make_detection_settings(**replaced_values):
    return LossChangeDetectionSettings(
        loss_monitoring_baseline_update_policy="keep_baseline_until_alarm",
        **dict(
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=0.05,
            adwin_confidence_delta=0.05,
            adwin_minimum_test_window_sample_count=10,
            short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        )
        | replaced_values,
    )


def test_policy_names_are_declared():
    assert SHORT_CHANGE_INTERVAL_ALARM_POLICIES == (
        RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
    )
    assert RAISE_ALARM_IMMEDIATELY_POLICY_NAME == "raise_alarm_immediately"
    assert (
        WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
        == "wait_until_change_interval_reaches_minimum"
    )
    for policy_name in SHORT_CHANGE_INTERVAL_ALARM_POLICIES:
        assert (
            make_detection_settings(
                short_change_interval_alarm_policy=policy_name
            ).short_change_interval_alarm_policy
            == policy_name
        )
    for invalid_policy in ("", "wait", "raise_alarm", None, 1):
        with pytest.raises(RunSettingsValidationError):
            make_detection_settings(short_change_interval_alarm_policy=invalid_policy)


@pytest.mark.parametrize("loss_monitoring_scope", ["overall_and_true_class_losses", "overall_loss"])
def test_waiting_policy_is_rejected_with_adwin(loss_monitoring_scope):
    """ADWINは、検知で窓の古い側を削るので、再開せずに待っても、合図が続かない。組合せを拒否する。"""
    # ADWINと、いまの扱いの組合せは、受け付ける。
    make_detection_settings(
        drift_detector_name="adwin", loss_monitoring_scope=loss_monitoring_scope
    )
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_detection_settings(
            drift_detector_name="adwin",
            loss_monitoring_scope=loss_monitoring_scope,
            short_change_interval_alarm_policy=(
                WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
            ),
        )
    assert raised_error.value.configuration_parameter_name == "short_change_interval_alarm_policy"


def test_waiting_policy_requires_pending_capacity_not_below_minimum_change_interval(
    valid_run_settings_mapping,
):
    """待つ扱いで、保留の容量が最小件数より小さいと、合図が続いても警報にならない。組合せを拒否する。"""
    valid_settings = make_valid_run_client_settings(valid_run_settings_mapping)
    pending_capacity = (
        valid_settings.training_data_assignment_settings.pending_assignment_buffer_capacity_samples
    )
    waiting_detection_settings = replace(
        valid_settings.loss_change_detection_settings,
        short_change_interval_alarm_policy=WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
    )
    # 容量と最小件数が同じなら、受け付ける。いまの扱いは、容量より大きい最小件数も、これまでどおり受け付ける。
    make_valid_run_client_settings(
        valid_run_settings_mapping,
        loss_change_detection_settings=waiting_detection_settings,
        scalar_settings=replace(
            valid_settings.scalar_settings, minimum_change_interval_sample_count=pending_capacity
        ),
    )
    too_large_minimum_scalar_settings = replace(
        valid_settings.scalar_settings, minimum_change_interval_sample_count=pending_capacity + 1
    )
    make_valid_run_client_settings(
        valid_run_settings_mapping, scalar_settings=too_large_minimum_scalar_settings
    )
    with pytest.raises(RunSettingsValidationError) as raised_error:
        make_valid_run_client_settings(
            valid_run_settings_mapping,
            loss_change_detection_settings=waiting_detection_settings,
            scalar_settings=too_large_minimum_scalar_settings,
        )
    assert raised_error.value.configuration_parameter_name == "short_change_interval_alarm_policy"


# ---- 標本1件の処理 ----


def to_comparable(value):
    """ownerの読取りを、別々に組み立てた2つの状態の間で、`==`で比べられる値へ直す。"""
    if isinstance(value, torch.Tensor):
        return ("tensor", list(value.shape), to_comparable(value.tolist()))
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if is_dataclass(value) and not isinstance(value, type):
        return {
            value_field.name: to_comparable(getattr(value, value_field.name))
            for value_field in fields(value)
        }
    if isinstance(value, dict):
        return {key: to_comparable(element) for key, element in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_comparable(element) for element in value]
    return value


def describe_processing_state(processing_arguments):
    state = snapshot_sample_processing_state(processing_arguments=processing_arguments)
    return to_comparable(state)


def make_block_stream_observation(*, processing_arguments, sample_index, block_length):
    """現行モデルにとって、損失の小さいラベルと大きいラベルを、`block_length`件ごとに入れ替える標本列。

    ラベルは、その時点の新の現行モデルの予測から決める（2クラス）。比べる2つの実行は、状態が同じであるかぎり、
    同じ標本を受け取る。
    """
    input_features = torch.tensor([[(sample_index % 7) / 7.0, ((sample_index * 3) % 5) / 5.0]])
    current_classifier = (
        processing_arguments["held_model_training_state_registry"]
        .get_held_model_training_state(
            model_id=processing_arguments[
                "current_training_model_assignment"
            ].current_training_model_id
        )
        .classifier
    )
    with torch.no_grad():
        predicted_class_id = int(current_classifier(input_features).argmax(dim=1).item())
    high_loss_block = (sample_index // block_length) % 2 == 1
    class_id = 1 - predicted_class_id if high_loss_block else predicted_class_id
    return IndexedObservedTrainingSample(
        sample_index=sample_index,
        training_sample=ObservedTrainingSample(
            input_features=input_features,
            observed_class_labels=torch.tensor([[float(class_id)]]),
        ),
        observed_concept_id=sample_index % 3,
    )


def run_sample_processing_trajectory(
    *,
    monkeypatch,
    valid_run_settings_mapping,
    recording_oracle_case,
    historical_mean_losses,
    block_length,
    processed_sample_count,
    short_change_interval_alarm_policy,
    suppress_signal_while_change_interval_is_short,
):
    """同じ初期状態から、標本列を処理し、標本ごとの（合図、推定の長さ、保留の件数、警報の有無、状態）を返す。

    `suppress_signal_while_change_interval_is_short`が真のときは、検出器の観測の戻り値の`drift_detected`を、
    推定変化区間の件数が最小件数より少ない標本でだけ、偽に差し替える（検出器の中の状態は、差し替えない）。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, _ = build_sample_processing_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=2,
            recording_oracle_case=recording_oracle_case,
            historical_mean_losses=historical_mean_losses,
            minimum_validation_loss_decrease=0.0,
        )
        processing_arguments["short_change_interval_alarm_policy"] = (
            short_change_interval_alarm_policy
        )
        minimum_sample_count = processing_arguments["minimum_change_interval_sample_count"]
        pending_store = processing_arguments["pending_sample_observation_store"]
        monitor = processing_arguments["loss_change_monitor"]
        real_observe = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation
        observed = {}

        def observe(self, **observation_arguments):
            observation = real_observe(self, **observation_arguments)
            if self is not monitor:
                return observation
            # 警報の判定が読む保留は、この標本を足した後の保留。
            change_interval_sample_count = min(
                len(pending_store.snapshot_pending_sample_observations()) + 1,
                observation.estimated_change_span_sample_count,
            )
            observed.update(
                signal=observation.drift_detected,
                span=observation.estimated_change_span_sample_count,
                change_interval_sample_count=change_interval_sample_count,
                candidate_start=observation.detector_candidate_start_sample_index,
            )
            if (
                suppress_signal_while_change_interval_is_short
                and observation.drift_detected
                and change_interval_sample_count < minimum_sample_count
            ):
                return replace(observation, drift_detected=False)
            return observation

        with pytest.MonkeyPatch.context() as observe_patch:
            observe_patch.setattr(
                OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", observe
            )
            first_sample_index = (
                processing_arguments["pending_training_assignment_buffer"]
                .get_state_snapshot()
                .last_observed_sample_index
                + 1
            )
            steps = []
            for sample_index in range(
                first_sample_index, first_sample_index + processed_sample_count
            ):
                indexed_observation = make_block_stream_observation(
                    processing_arguments=processing_arguments,
                    sample_index=sample_index,
                    block_length=block_length,
                )
                sample_processing = process_observed_sample(
                    indexed_observation=indexed_observation, **processing_arguments
                )
                handling = sample_processing.alarm_occurrence_handling
                steps.append(
                    dict(
                        sample_index=sample_index,
                        minimum_sample_count=minimum_sample_count,
                        **observed,
                        alarm_handled=handling is not None,
                        alarm_outcome=(
                            None
                            if handling is None
                            else handling.adaptation_record.adaptation_outcome
                        ),
                        state=describe_processing_state(processing_arguments),
                    )
                )
    return steps


# 最初の状態（保持なし＝現行を維持した直後、保持あり＝候補検証を開始した直後）と、損失の入替えの周期。
TRAJECTORY_CONDITIONS = [
    # 損失の入替えの周期が長い条件では、高い損失へ替わった直後の合図が、推定変化区間2件（最小件数は3）で出る。
    # 待った次の標本で警報になり（候補検証の開始）、その検証を保持している間にも、もう一度、待ってから警報になる。
    (RECORDING_ORACLE_CASES[2], (0.05, 0.05, 0.05), 10),
    (RECORDING_ORACLE_CASES[2], (0.05, 0.05, 0.05), 20),
    (RECORDING_ORACLE_CASES[2], (0.05, 0.05, 0.05), 6),
    (RECORDING_ORACLE_CASES[2], (0.1, 0.1, 0.1), 15),
    (RECORDING_ORACLE_CASES[3], (0.05, 0.05, 0.05), 10),
]
OBSERVED_TRAJECTORY_COVERAGE = {}


@pytest.mark.parametrize(
    ("recording_oracle_case", "historical_mean_losses", "block_length"), TRAJECTORY_CONDITIONS
)
def test_waiting_processes_short_interval_signal_like_a_sample_without_signal(
    recording_oracle_case,
    historical_mean_losses,
    block_length,
    monkeypatch,
    valid_run_settings_mapping,
):
    """待つ扱い: 推定変化区間が短い間の合図は、合図の無い標本と同じに処理し、最小件数に達したら、いまの扱いと同じに警報とする。

    比べる相手は、いまの扱いで、合図を、推定変化区間が短い標本でだけ伏せた実行。標本ごとに、全ownerの状態・
    乱数の状態が同じ。待つ扱いでは、短い警報が起きず、警報は、「合図があり、推定変化区間の件数が最小件数以上」の
    標本の全部で、それだけで、起きる。
    """
    trajectory_arguments = dict(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        recording_oracle_case=recording_oracle_case,
        historical_mean_losses=historical_mean_losses,
        block_length=block_length,
        processed_sample_count=70,
    )
    waiting_steps = run_sample_processing_trajectory(
        **trajectory_arguments,
        short_change_interval_alarm_policy=WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
        suppress_signal_while_change_interval_is_short=False,
    )
    reference_steps = run_sample_processing_trajectory(
        **trajectory_arguments,
        short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        suppress_signal_while_change_interval_is_short=True,
    )
    deferred_signal_count = 0
    deferred_signal_count_while_holding_validation = 0
    alarm_count = 0
    for waiting_step, reference_step in zip(waiting_steps, reference_steps, strict=True):
        assert waiting_step == reference_step, waiting_step["sample_index"]
        change_interval_is_long_enough = (
            waiting_step["change_interval_sample_count"] >= waiting_step["minimum_sample_count"]
        )
        assert waiting_step["alarm_handled"] is (
            waiting_step["signal"] and change_interval_is_long_enough
        ), waiting_step["sample_index"]
        assert waiting_step["alarm_outcome"] != SHORT_ALARM_OUTCOME
        signal_is_deferred = waiting_step["signal"] and not change_interval_is_long_enough
        deferred_signal_count += signal_is_deferred
        # 処理の後の状態で数える。待った標本の処理の中で、保持していた検証が完了した場合は、数えない
        # （少なく数える側。下の網羅のtestは、1件以上あることを確かめる）。
        alarm_count += waiting_step["alarm_handled"]
    OBSERVED_TRAJECTORY_COVERAGE[(recording_oracle_case, historical_mean_losses, block_length)] = (
        deferred_signal_count,
        alarm_count,
        deferred_signal_count_while_holding_validation,
    )


def test_waiting_trajectories_cover_deferred_signals_and_alarms():
    """上の条件の集まりが、待った標本と、待った後の警報の、両方を通っている。"""
    if len(OBSERVED_TRAJECTORY_COVERAGE) != len(TRAJECTORY_CONDITIONS):
        pytest.skip("全条件を実行したときだけ判定する")
    coverage = list(OBSERVED_TRAJECTORY_COVERAGE.values())
    assert sum(deferred for deferred, _, _ in coverage) > 0
    assert sum(alarms for _, alarms, _ in coverage) > 0
    assert any(deferred and alarms for deferred, alarms, _ in coverage)
    # 候補検証を保持している間の合図を、待った標本がある（要求1.5）。保持していない間に待った標本もある。


def test_immediate_policy_raises_short_alarms_on_the_same_stream(
    monkeypatch, valid_run_settings_mapping
):
    """いまの扱い: 同じ標本列で、短い警報が起きる（上の条件が、待つ扱いの対象の経路を通ることの確認）。"""
    steps = run_sample_processing_trajectory(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        recording_oracle_case=RECORDING_ORACLE_CASES[2],
        historical_mean_losses=(0.05, 0.05, 0.05),
        block_length=10,
        processed_sample_count=70,
        short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        suppress_signal_while_change_interval_is_short=False,
    )
    assert any(step["alarm_outcome"] == SHORT_ALARM_OUTCOME for step in steps)
    # いまの扱いでは、合図があれば、必ず警報とする。
    assert all(step["alarm_handled"] is step["signal"] for step in steps)


def test_waiting_raises_alarm_within_minimum_sample_count_while_best_candidate_is_unchanged(
    monkeypatch, valid_run_settings_mapping
):
    """高い損失が続く固定の列: 証拠が最大の候補が変わらない間、最初の合図から、最小件数より少ない標本数で警報になる。"""
    steps = run_sample_processing_trajectory(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        recording_oracle_case=RECORDING_ORACLE_CASES[2],
        historical_mean_losses=(0.05, 0.05, 0.05),
        # 最初のblockは損失の小さい標本、次のblockから、高い損失が続く。
        block_length=20,
        processed_sample_count=40,
        short_change_interval_alarm_policy=WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
        suppress_signal_while_change_interval_is_short=False,
    )
    first_signal_position = next(position for position, step in enumerate(steps) if step["signal"])
    first_alarm_position = next(
        position for position, step in enumerate(steps) if step["alarm_handled"]
    )
    minimum_sample_count = steps[0]["minimum_sample_count"]
    waited_steps = steps[first_signal_position:first_alarm_position]
    # 最初の合図は、推定変化区間が短く、待つ。
    assert waited_steps
    assert all(step["signal"] and not step["alarm_handled"] for step in waited_steps)
    # 待つ間、証拠が最大の候補の開始点は変わらず、推定の長さは1ずつ伸びる。
    candidate_starts = {
        step["candidate_start"] for step in steps[first_signal_position : first_alarm_position + 1]
    }
    assert len(candidate_starts) == 1
    spans = [step["span"] for step in steps[first_signal_position : first_alarm_position + 1]]
    assert spans == list(range(spans[0], spans[0] + len(spans)))
    assert first_alarm_position - first_signal_position < minimum_sample_count
    alarm_step = steps[first_alarm_position]
    assert alarm_step["change_interval_sample_count"] == minimum_sample_count
    assert alarm_step["alarm_outcome"] not in (None, SHORT_ALARM_OUTCOME)


@pytest.mark.parametrize("invalid_policy", ["", "wait", None, 1, b"raise_alarm_immediately"])
def test_undeclared_policy_is_rejected_before_any_update(
    invalid_policy, monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, _ = build_sample_processing_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=2,
            recording_oracle_case=RECORDING_ORACLE_CASES[2],
            historical_mean_losses=(0.05, 0.05, 0.05),
            minimum_validation_loss_decrease=0.0,
        )
        state_before = describe_processing_state(processing_arguments)
        processing_arguments["short_change_interval_alarm_policy"] = invalid_policy
        with pytest.raises((TypeError, ValueError)):
            process_observed_sample(
                indexed_observation=make_block_stream_observation(
                    processing_arguments=processing_arguments,
                    sample_index=processing_arguments["pending_training_assignment_buffer"]
                    .get_state_snapshot()
                    .last_observed_sample_index
                    + 1,
                    block_length=6,
                ),
                **processing_arguments,
            )
        assert describe_processing_state(processing_arguments) == state_before


# ---- 全体run ----

SINE2_EVENT_CASE_NAME = "fedsda-sine2-events"


def build_sine2_event_run_settings(short_change_interval_alarm_policy):
    """基準版3のケース`fedsda-sine2-events`の設定（ほかの扱いは既定）で、警報の時機の扱いを選ぶ。"""
    run_settings = golden.build_case_settings(SINE2_EVENT_CASE_NAME, 3)
    run_participant_settings = run_settings.run_participant_settings
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            run_participant_settings,
            run_client_settings=replace(
                run_client_settings,
                loss_change_detection_settings=replace(
                    run_client_settings.loss_change_detection_settings,
                    short_change_interval_alarm_policy=short_change_interval_alarm_policy,
                ),
            ),
        ),
    )


def execute_run_observing_alarms(run_settings):
    minimum_sample_count = run_settings.run_participant_settings.run_client_settings.scalar_settings.minimum_change_interval_sample_count
    observation = dict(alarm_count=0, short_alarm_count=0, alarm_with_short_change_interval_count=0)
    real_response = alarm_occurrence_module.respond_to_alarm_with_buffered_samples

    def observe_response(**response_arguments):
        observation["alarm_count"] += 1
        observation["alarm_with_short_change_interval_count"] += (
            min(
                len(response_arguments["pending_sample_observations"]),
                response_arguments["estimated_change_span_sample_count"],
            )
            < minimum_sample_count
        )
        alarm_buffer_response = real_response(**response_arguments)
        observation["short_alarm_count"] += (
            alarm_buffer_response.response_outcome == SHORT_ALARM_OUTCOME
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
        policy_name: execute_run_observing_alarms(build_sine2_event_run_settings(policy_name))
        for policy_name in SHORT_CHANGE_INTERVAL_ALARM_POLICIES
    }


@pytest.mark.slow
def test_whole_run_with_waiting_policy_has_no_short_alarm(sine2_event_runs_by_policy):
    """待つ扱い: 警報のたびに（候補検証を保持している間を含めて）、推定変化区間の件数が最小件数以上。短い警報は無い。"""
    experiment_run, observation = sine2_event_runs_by_policy[
        WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
    ]
    assert observation["alarm_count"] > 0
    assert observation["short_alarm_count"] == 0
    assert observation["alarm_with_short_change_interval_count"] == 0
    assert (
        experiment_run.run_metrics.change_response_metrics.adaptation_action_metrics.skip_too_short_change_interval_count
        == 0
    )
    assert golden.count_repeated_training_sample_assignments(experiment_run) == 0
    assert golden.count_unregistered_adopted_models(experiment_run) == 0
    for run_client in experiment_run.measured_run.participants.client_operations:
        assert (
            run_client.short_change_interval_alarm_policy
            == WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
        )


def test_whole_run_with_immediate_policy_has_short_alarms(sine2_event_runs_by_policy):
    """いまの扱い: 同じ条件で、短い警報が起きる。"""
    experiment_run, observation = sine2_event_runs_by_policy[RAISE_ALARM_IMMEDIATELY_POLICY_NAME]
    assert observation["short_alarm_count"] > 0
    assert observation["alarm_with_short_change_interval_count"] > 0
    for run_client in experiment_run.measured_run.participants.client_operations:
        assert run_client.short_change_interval_alarm_policy == RAISE_ALARM_IMMEDIATELY_POLICY_NAME


def test_waiting_policy_does_not_miss_changes_on_the_fixed_condition(sine2_event_runs_by_policy):
    """この条件での実測の関係: 待つ扱いは、切替の検出の再現率が、いまの扱い以上で、切替までの遅れが、いまの扱い以下。

    一般の保証ではない（大小の関係だけを固定する）。
    """
    waiting_metrics = sine2_event_runs_by_policy[
        WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
    ][0].run_metrics
    immediate_metrics = sine2_event_runs_by_policy[RAISE_ALARM_IMMEDIATELY_POLICY_NAME][
        0
    ].run_metrics
    assert (
        waiting_metrics.training_model_switch_detection_metrics.detection_recall
        >= immediate_metrics.training_model_switch_detection_metrics.detection_recall
    )
    assert (
        waiting_metrics.change_response_metrics.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count
        <= immediate_metrics.change_response_metrics.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count
    )
    assert waiting_metrics != immediate_metrics
