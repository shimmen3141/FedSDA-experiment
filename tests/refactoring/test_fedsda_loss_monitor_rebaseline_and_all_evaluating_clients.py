"""損失の低下による基準の引下げと、評価clientの上限の契約。"""

from dataclasses import replace
from math import exp
from unittest.mock import patch

import pytest

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)


def _monitor(baseline=0.6, policy="lower_baseline_on_loss_decrease", scope="overall_loss"):
    return OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=LossChangeDetectionSettings(
            drift_detector_name="e_sr",
            loss_monitoring_scope=scope,
            e_sr_false_alarm_control_alpha=0.001,
            adwin_confidence_delta=0.002,
            adwin_minimum_test_window_sample_count=5,
            short_change_interval_alarm_policy="raise_alarm_immediately",
            loss_monitoring_baseline_update_policy=policy,
        ),
        class_count=2,
        initial_baseline_loss_mean=baseline,
        maximum_retained_candidate_count=1000,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )


def _observe(monitor, index, loss=0.1, current=0.1):
    return monitor.observe_loss_after_label_observation(
        observed_loss=loss,
        observed_class_id=index % 2,
        sample_index=index,
        current_model_baseline_loss_mean=current,
    )


@pytest.mark.parametrize("baseline", [0.01, 0.6, 1.0 - 1e-6])
@pytest.mark.parametrize("loss", [0.0, 0.2, 1.0])
def test_decrease_detector_increment_and_counts(baseline, loss):
    monitor = _monitor(baseline)
    detector = monitor._decrease_detector
    observation = _observe(monitor, 0, loss=loss, current=baseline)
    state = detector.get_state_snapshot()
    for fraction, capital in zip(monitor._betting_fractions, state.candidate_log_capitals[0]):
        assert exp(capital) == pytest.approx(1.0 + fraction * (baseline - loss) / (1.0 - baseline))
    assert observation.component_update_count == 2
    assert observation.decrease_detector_component_update_count == 1
    assert observation.evaluated_candidate_bet_count == 10
    assert observation.decrease_detector_evaluated_candidate_bet_count == 5


@pytest.mark.parametrize("scope", ["overall_loss", "overall_and_true_class_losses"])
def test_lowering_returns_original_observation_and_clears_monitor_only(scope):
    monitor = _monitor(scope=scope)
    unchanged = _monitor(policy="keep_baseline_until_alarm", scope=scope)
    lowered = []
    for index in range(120):
        expected = _observe(unchanged, index)
        actual = _observe(monitor, index)
        if actual.baseline_lowered:
            lowered.append(index)
            assert actual.previous_baseline_loss_mean == 0.6
            assert actual.lowered_baseline_loss_mean == 0.1
            assert (
                replace(
                    actual,
                    component_update_count=expected.component_update_count,
                    evaluated_candidate_bet_count=expected.evaluated_candidate_bet_count,
                    decrease_detector_component_update_count=0,
                    decrease_detector_evaluated_candidate_bet_count=0,
                    baseline_lowered=False,
                    previous_baseline_loss_mean=None,
                    lowered_baseline_loss_mean=None,
                )
                == expected
            )
            snapshot = monitor.get_state_snapshot()
            assert snapshot.last_sample_index is None
            assert snapshot.last_observation is None
            assert snapshot.class_esr_states_by_class_id == ()
            assert snapshot.overall_esr_state.baseline_loss_mean == 0.1
            assert snapshot.decrease_esr_state.baseline_loss_mean == 0.9
    assert len(lowered) == 1


def test_equal_losses_do_not_lower_and_reset_restarts_decrease_detector():
    monitor = _monitor()
    for index in range(80):
        assert not _observe(monitor, index, loss=0.6, current=0.6).baseline_lowered
    monitor.reset(baseline_loss_mean=0.2)
    assert monitor.get_state_snapshot().decrease_esr_state.baseline_loss_mean == 0.8
    assert monitor.last_observation is None


def test_decrease_signal_without_lower_saved_mean_restarts_only_decrease_detector():
    monitor = _monitor()
    detector = monitor._decrease_detector
    for index in range(80):
        assert not _observe(monitor, index, current=0.7).baseline_lowered
        if monitor._decrease_detector is not detector:
            break
    else:
        pytest.fail("下がる向きの検出器が再開されなかった")
    assert monitor.get_state_snapshot().overall_esr_state.baseline_loss_mean == 0.6
    assert monitor.last_observation.sample_index == index


def test_upward_signal_prevents_lowering_even_when_decrease_detector_signals():
    monitor = _monitor(0.01)
    detector_type = type(monitor._decrease_detector)
    original = detector_type.observe_loss

    def observe(detector, **arguments):
        return replace(original(detector, **arguments), drift_detected=True)

    with patch.object(detector_type, "observe_loss", observe):
        observation = _observe(monitor, 0, loss=0.9, current=0.001)
    assert observation.drift_detected
    assert not observation.baseline_lowered
    assert monitor.last_observation == observation


def test_keep_policy_has_no_decrease_detector_or_extra_computation():
    monitor = _monitor(policy="keep_baseline_until_alarm")
    for index in range(80):
        observation = _observe(monitor, index)
        assert not observation.baseline_lowered
        assert observation.component_update_count == 1
        assert observation.decrease_detector_component_update_count == 0
        assert observation.decrease_detector_evaluated_candidate_bet_count == 0
    assert monitor.get_state_snapshot().decrease_esr_state is None


@pytest.mark.parametrize(
    "policy", ["unknown", True, type("PolicyString", (str,), {})("keep_baseline_until_alarm")]
)
def test_invalid_baseline_policy_is_rejected(policy):
    with pytest.raises(RunSettingsValidationError):
        _monitor(policy=policy)


def test_adwin_cannot_lower_baseline():
    with pytest.raises(RunSettingsValidationError, match="loss_monitoring_baseline_update_policy"):
        replace(_monitor()._loss_change_detection_settings, drift_detector_name="adwin")


def test_baseline_lowering_records_validate_before_mutation():
    from federated_learning_experiments.evaluation.loss_change_alarm_record_store import (
        LossChangeAlarmRecordStore,
    )

    store = LossChangeAlarmRecordStore()
    store.append_baseline_lowering_record(
        sample_index=9, previous_baseline_loss_mean=0.6, lowered_baseline_loss_mean=0.1
    )
    snapshot = store.get_state_snapshot()
    assert snapshot.baseline_lowering_sample_indices == (9,)
    assert snapshot.previous_baseline_loss_means == (0.6,)
    assert snapshot.lowered_baseline_loss_means == (0.1,)
    with pytest.raises(ValueError):
        store.append_baseline_lowering_record(
            sample_index=10, previous_baseline_loss_mean=0.1, lowered_baseline_loss_mean=0.6
        )
    assert store.get_state_snapshot() == snapshot


def test_decrease_computation_counts_are_a_validated_subset_of_total():
    from federated_learning_experiments.evaluation.loss_monitoring_computation_count_store import (
        LossMonitoringComputationCountStore,
    )

    store = LossMonitoringComputationCountStore()
    store.record_loss_monitoring_computation(
        detector_component_update_count=3,
        evaluated_candidate_bet_count=15,
        decrease_detector_component_update_count=1,
        decrease_detector_evaluated_candidate_bet_count=5,
    )
    counts = store.get_loss_monitoring_computation_counts()
    assert counts.decrease_detector_component_update_count == 1
    assert counts.decrease_detector_evaluated_candidate_bet_count == 5
    with pytest.raises(ValueError):
        store.record_loss_monitoring_computation(
            detector_component_update_count=1,
            evaluated_candidate_bet_count=5,
            decrease_detector_component_update_count=2,
        )
    assert store.get_loss_monitoring_computation_counts() == counts


def test_final_configuration_and_all_fedsda_configurations_use_explicit_policies():
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
        build_experiment_run_settings,
    )

    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=7,
        per_client_sample_count=600,
        server_aggregation_interval_per_client_samples=100,
    )
    for name in EXPERIMENT_CONFIGURATION_NAMES:
        if not name.startswith("fedsda"):
            continue
        settings = build_experiment_run_settings(
            configuration_name=name, experiment_run_conditions=conditions
        ).run_participant_settings
        expected_policy = (
            "keep_baseline_until_alarm"
            if name
            in (
                "fedsda-classwise-adwin-detection",
                "fedsda-overall-loss-adwin-detection",
            )
            else "lower_baseline_on_loss_decrease"
        )
        assert (
            settings.run_client_settings.loss_change_detection_settings.loss_monitoring_baseline_update_policy
            == expected_policy
        ), name
        assert settings.maximum_evaluating_client_count_per_model == 7, name
