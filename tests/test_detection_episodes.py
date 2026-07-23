from types import SimpleNamespace

from federated_drift_experiment.adaptation_events import AdaptationEvent
from federated_drift_experiment.detection_episode import DetectionEpisodeController
from federated_drift_experiment.metrics import compute_metrics
from federated_drift_experiment.provisional_model import ProvisionalModelDecision


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
        provisional_model_decisions=[
            ProvisionalModelDecision(
                position=610,
                detector="ESR",
                accepted=True,
                reason="accepted",
                interval_count=30,
                training_count=24,
                validation_count=6,
                reference_model_id=0,
                candidate_mean_loss=0.2,
                reference_mean_loss=0.5,
                candidate_recent_loss=0.3,
                reference_recent_loss=0.4,
            ),
            ProvisionalModelDecision(
                position=700,
                detector="ESR",
                accepted=False,
                reason="full_and_recent",
                interval_count=20,
                training_count=16,
                validation_count=4,
                reference_model_id=0,
                candidate_mean_loss=0.7,
                reference_mean_loss=0.5,
                candidate_recent_loss=0.8,
                reference_recent_loss=0.4,
            ),
        ],
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
    assert metrics["provisional_proposal_count"] == 2
    assert metrics["provisional_acceptance_rate"] == 0.5
    assert metrics["provisional_matched_true_count"] == 1
    assert metrics["provisional_accepted_matched_true_count"] == 1
    assert metrics["provisional_rejected_matched_true_count"] == 0
    assert metrics["provisional_accepted_precision"] == 1.0
    assert metrics["provisional_reject_full_and_recent_count"] == 1
    assert abs(metrics["provisional_accepted_full_margin_mean"] - 0.3) < 1e-12
