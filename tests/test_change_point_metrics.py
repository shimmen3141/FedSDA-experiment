from types import SimpleNamespace

from federated_drift_experiment.metrics import compute_metrics


def test_change_point_metrics_use_detector_estimates_not_alarm_times():
    client = SimpleNamespace(
        history_accuracy=[1] * 100,
        local_switch_positions=[65],
        detected_event_positions=[65],
        estimated_drift_start_positions=[52],
    )

    results = compute_metrics(
        [client], {0: [50]}, delay_tolerance=20, stable_window=0
    )

    assert results["change_point_estimate_count"] == 1
    assert results["change_point_mae"] == 2.0
    assert results["change_point_bias"] == 2.0
