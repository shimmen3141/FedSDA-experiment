"""手法の設定: 「製品と同じ」の値が製品のFedHASPの既定と同じこと、範囲外の拒否を確かめる（riverを使わない）。"""

from dataclasses import replace

import pytest

from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
    SingleStreamHardAssignmentSoftPredictionSettings,
    build_default_single_stream_settings,
)


def product_client_settings():
    pytest.importorskip("torch")
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
        build_final_configuration_fedsda_run_settings,
    )

    settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=1,
            per_client_sample_count=100,
            server_aggregation_interval_per_client_samples=100,
        )
    )
    return settings.run_participant_settings.run_client_settings


def test_values_shared_with_the_product_match_the_product_defaults():
    product = product_client_settings()
    settings = build_default_single_stream_settings()
    assert settings.loss_change_detection_settings == product.loss_change_detection_settings
    assert settings.loss_monitor_betting_fractions == product.loss_monitor_betting_fractions
    assert (
        settings.loss_monitor_maximum_retained_candidate_count
        == product.scalar_settings.loss_monitor_maximum_retained_candidate_count
    )
    assert (
        settings.minimum_change_interval_sample_count
        == product.scalar_settings.minimum_change_interval_sample_count
    )
    assert (
        settings.pending_assignment_buffer_capacity_samples
        == product.training_data_assignment_settings.pending_assignment_buffer_capacity_samples
    )
    prediction = product.prediction_combination_settings
    assert (
        settings.prediction_combination_settings.prediction_combination_strategy
        == prediction.prediction_combination_strategy
    )
    assert (
        settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        == prediction.fixed_share_weight_redistribution_time_scale_samples
    )
    data_assignment = product.training_data_assignment_settings
    assert settings.new_model_hypothesis_learners == data_assignment.new_model_hypothesis_learners
    assert settings.change_interval_search == data_assignment.change_interval_search


def test_values_specific_to_the_single_stream_method():
    settings = build_default_single_stream_settings()
    assert settings.warm_up_sample_count == 200
    assert settings.probability_smoothing_weight == 0.01
    assert settings.switching_probability == 1.0 / 120
    assert settings.pending_assignment_buffer_capacity_samples == 30
    assert settings.minimum_change_interval_sample_count == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("pending_assignment_buffer_capacity_samples", 0),
        ("minimum_change_interval_sample_count", 0),
        ("warm_up_sample_count", 0),
        ("probability_smoothing_weight", 0.0),
        ("probability_smoothing_weight", 1.0),
        ("loss_monitor_maximum_retained_candidate_count", 0),
        ("loss_monitor_betting_fractions", ()),
        ("warm_up_sample_count", True),
        ("probability_smoothing_weight", "0.01"),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        replace(build_default_single_stream_settings(), **{field: value})


def test_settings_type_is_frozen():
    settings = build_default_single_stream_settings()
    assert type(settings) is SingleStreamHardAssignmentSoftPredictionSettings
    with pytest.raises(AttributeError):
        settings.warm_up_sample_count = 10
