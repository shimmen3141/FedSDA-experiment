import pytest

from federated_drift_experiment.experiment_spec.metrics import (
    METRICS,
    METRICS_BY_ID,
    METRIC_PROFILES,
    SCALAR_METRIC_IDS,
    metric,
    metrics_in_group,
    metrics_in_profile,
)


def test_metric_ids_are_unique_and_preserve_registry_order():
    assert len(SCALAR_METRIC_IDS) == len(set(SCALAR_METRIC_IDS))
    assert tuple(METRICS_BY_ID) == SCALAR_METRIC_IDS
    assert tuple(item.id for item in METRICS) == SCALAR_METRIC_IDS


def test_metric_metadata_is_complete():
    for item in METRICS:
        assert item.group
        assert item.tier in {"primary", "secondary", "diagnostic"}
        assert item.applicability
        assert item.description
        assert item.storage == "csv"


def test_metric_queries_and_profiles_reference_registered_metrics():
    assert metric("accuracy").group == "predictive_performance"
    assert {item.id for item in metrics_in_group("communication")} == {
        "comm_models_up", "comm_models_down", "comm_models_total",
        "comm_messages_up", "comm_messages_down", "comm_messages_total",
    }
    for profile_name, ids in METRIC_PROFILES.items():
        assert ids
        assert tuple(item.id for item in metrics_in_profile(profile_name)) == ids
        assert set(ids) <= set(SCALAR_METRIC_IDS)
    assert {
        "routing_loo_evaluation_count",
        "routing_loo_bounded_delta_mean",
        "routing_loo_active_unassigned_nonpositive_rate",
        "routing_loo_active_joint_nonpositive_rate",
        "routing_archive_shadow_accuracy_delta",
        "routing_archive_shadow_retained_global_model_rate",
        "routing_active_set_retained_global_model_rate",
        "routing_active_set_apply_retained_global_model_rate",
    } <= {
        item.id for item in metrics_in_group("routing_contribution")
    }


def test_unknown_metric_and_profile_are_rejected():
    with pytest.raises(KeyError, match="Unknown metric id"):
        metric("unknown")
    with pytest.raises(KeyError, match="Unknown metric profile"):
        metrics_in_profile("unknown")
