from types import SimpleNamespace

from federated_drift_experiment.adaptation_events import AdaptationEvent
from federated_drift_experiment.detection_episode import DetectionEpisodeController
from federated_drift_experiment.metrics import compute_metrics


def test_detection_episode_allows_at_most_one_operation_per_window():
    controller = DetectionEpisodeController(enabled=True, length=30)

    allowed, first_episode = controller.observe_detection(100)
    assert allowed is True
    controller.mark_operation()

    allowed, same_episode = controller.observe_detection(120)
    assert allowed is False
    assert same_episode == first_episode

    allowed, next_episode = controller.observe_detection(130)
    assert allowed is True
    assert next_episode != first_episode


def test_disabled_detection_episode_preserves_existing_behavior():
    controller = DetectionEpisodeController(enabled=False, length=30)
    assert controller.observe_detection(100) == (True, None)
    controller.mark_operation()
    assert controller.observe_detection(101) == (True, None)


def test_metrics_classify_switch_false_positives_and_actions():
    positions = [50, 110, 120, 250, 400, 610]
    events = [
        AdaptationEvent(110, "ESR", "reuse", 0, 1),
        AdaptationEvent(120, "ESR", "episode_suppressed", 1, 1),
        AdaptationEvent(400, "server", "server_merge", 1, 0),
        AdaptationEvent(610, "ESR", "create", 0, -100),
        AdaptationEvent(700, "ESR", "create_rejected", -100, -100),
    ]
    client = SimpleNamespace(
        history_accuracy=[1] * 800,
        local_switch_positions=positions,
        detected_event_positions=positions,
        estimated_drift_start_positions=positions,
        adaptation_events=events,
        mapping_change_positions=[400],
    )

    metrics = compute_metrics(
        [client], {0: [100, 600]}, delay_tolerance=100, stable_window=0
    )

    assert metrics["tp"] == 2
    assert metrics["switch_fp_early"] == 1
    assert metrics["switch_fp_duplicate"] == 1
    assert metrics["switch_fp_late"] == 1
    assert metrics["switch_fp_isolated"] == 1
    assert metrics["adaptation_reuse_precision"] == 1.0
    assert metrics["adaptation_create_precision"] == 1.0
    assert metrics["adaptation_episode_suppressed_count"] == 1
    assert metrics["adaptation_create_rejected_count"] == 1
    assert metrics["server_mapping_change_count"] == 1
