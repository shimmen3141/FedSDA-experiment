"""固定値の置換の寿命、消費側、小規模runと既定値での完全一致を検査する。"""

import json
import math
from dataclasses import replace

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.sweep import (
    experiment_configurations as configurations,
)
from tools.research_provisional_candidate_expert import (
    fixed_value_sensitivity as sensitivity,
)
from tools.research_provisional_candidate_expert import run as research


def _job(dataset, variant="product-default"):
    return {
        "variant": variant,
        "dataset": dataset,
        "seed": 0,
        "interval": 50,
        "sample_count": 1000,
        "check": False,
    }


def _default_client_settings():
    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=10,
        per_client_sample_count=1000,
        server_aggregation_interval_per_client_samples=50,
    )
    return configurations.build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=conditions,
    ).run_participant_settings.run_client_settings


def _normalize(value):
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items() if key != "seconds"}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


@pytest.fixture(scope="module", params=["sine2", "mnist4"])
def baseline(request):
    return request.param, research._execute_job(_job(request.param))


@pytest.mark.parametrize("fail", [False, True])
def test_settings_builder_restored_even_after_exception(fail):
    original = configurations.build_final_configuration_fedsda_run_settings
    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=10,
        per_client_sample_count=1000,
        server_aggregation_interval_per_client_samples=50,
    )
    values = sensitivity.FixedValues(
        time_scale_samples=60, esr_alpha=0.01, fifo_capacity_samples=15
    )
    try:
        with sensitivity.fixed_value_settings(values):
            settings = configurations.build_final_configuration_fedsda_run_settings(
                experiment_run_conditions=conditions
            )
            client = settings.run_participant_settings.run_client_settings
            assert (
                client.training_data_assignment_settings.pending_assignment_buffer_capacity_samples
                == 15
            )
            assert (
                client.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
                == 60
            )
            assert client.loss_change_detection_settings.e_sr_false_alarm_control_alpha == 0.01
            if fail:
                raise LookupError("寿命の検査")
    except LookupError:
        assert fail
    assert configurations.build_final_configuration_fedsda_run_settings is original
    restored = original(
        experiment_run_conditions=conditions
    ).run_participant_settings.run_client_settings
    assert (
        restored.training_data_assignment_settings.pending_assignment_buffer_capacity_samples == 30
    )
    assert (
        restored.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        == 120
    )
    assert restored.loss_change_detection_settings.e_sr_false_alarm_control_alpha == 0.001


@pytest.mark.parametrize(
    "values",
    [
        sensitivity.FixedValues(time_scale_samples=0),
        sensitivity.FixedValues(esr_alpha=1.0),
        sensitivity.FixedValues(fifo_capacity_samples=0),
    ],
)
def test_replacement_runs_product_settings_validation(values):
    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=10,
        per_client_sample_count=1000,
        server_aggregation_interval_per_client_samples=50,
    )
    settings = configurations.build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=conditions
    )
    with pytest.raises(ValueError):
        sensitivity.replace_fixed_values(settings, values)


def test_default_values_are_exactly_product_default(baseline, monkeypatch, record_property):
    dataset, expected = baseline
    # 同じ変種を使い、ビルダーの戻り値の置換の有無だけを変える。
    monkeypatch.setitem(
        sensitivity.FIXED_VALUE_SENSITIVITY_VARIANTS,
        "product-default",
        sensitivity.FixedValues(),
    )
    actual = research._execute_job(_job(dataset))
    assert _normalize(actual) == _normalize(expected)
    record_property("small_run", json.dumps(actual, allow_nan=True))


@pytest.mark.parametrize("variant", tuple(sensitivity.FIXED_VALUE_SENSITIVITY_VARIANTS))
def test_small_run_consumers_use_requested_values(baseline, variant, record_property):
    dataset, default = baseline
    result = research._execute_job(_job(dataset, variant))
    actual = result["fixed_value_sensitivity"]
    values = sensitivity.FIXED_VALUE_SENSITIVITY_VARIANTS[variant]
    assert actual["fixed_share_rate"] == 1.0 / values.time_scale_samples
    assert actual["mdl_switching_probability"] == 1.0 / values.time_scale_samples
    assert actual["synchronized_local_share_rate"] == 1.0 / values.time_scale_samples
    assert actual["detector_alpha"] == values.esr_alpha
    assert actual["fifo_capacity_samples"] == values.fifo_capacity_samples
    assert actual["maximum_pending_sample_count"] == values.fifo_capacity_samples + 1
    for name in (
        "fixed_share_rate",
        "mdl_switching_probability",
        "synchronized_local_share_rate",
        "detector_alpha",
        "fifo_capacity_samples",
    ):
        assert actual[name + "_observation_count"] > 0
    # 警報があるなら、MDLの選択関数に渡された確率・探索範囲も照合する。
    if actual["mdl_selection_count"]:
        assert actual["mdl_selection_switching_probability"] == 1.0 / values.time_scale_samples
        assert actual["maximum_search_sample_count"] <= values.fifo_capacity_samples + 1
        assert actual["change_interval_grid_sample_count"] == 5
    record_property("small_run", json.dumps(result, allow_nan=True))
    record_property(
        "differs_from_default",
        _normalize(result["alarm_interval_assignment_selection"])
        != _normalize(default["alarm_interval_assignment_selection"])
        or result["server_parameter_digest"] != default["server_parameter_digest"]
        or result["prediction_record_digest"] != default["prediction_record_digest"],
    )


def test_stale_fifo_capacity_stops_and_restores_consumers():
    from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
        PendingTrainingAssignmentBuffer,
    )

    append = PendingTrainingAssignmentBuffer.append_observed_sample_index
    buffer = PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=_default_client_settings().training_data_assignment_settings
    )
    assert buffer.capacity_sample_count == 30
    for capacity in (15, 60):
        with (
            pytest.raises(RuntimeError, match="fifo_capacity_samples"),
            sensitivity.observe_fixed_value_consumers(
                values=replace(sensitivity.FixedValues(), fifo_capacity_samples=capacity),
                diagnostics={},
            ),
        ):
            buffer.append_observed_sample_index(sample_index=0)
        assert PendingTrainingAssignmentBuffer.append_observed_sample_index is append


@pytest.mark.parametrize("consumer", ["fixed_share", "mdl", "detector"])
def test_stale_consumer_values_stop_and_all_patches_restore(consumer):
    from federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection import (
        BoundedLossESRDetector,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.runtime import (
        fedsda_run_client,
        observed_sample_processing,
    )

    targets = [
        (BoundedLossESRDetector, "__init__"),
        (FixedShareWeightController, "update_weights_after_loss_observation"),
        (fedsda_run_client, "process_observed_sample"),
        (observed_sample_processing, "select_alarm_interval_assignment"),
    ]
    originals = [getattr(target, name) for target, name in targets]
    values = sensitivity.FixedValues(time_scale_samples=60, esr_alpha=0.01)
    expected = {
        "fixed_share": "fixed_share_rate",
        "mdl": "mdl_selection_switching_probability",
        "detector": "detector_alpha",
    }[consumer]
    with (
        pytest.raises(RuntimeError, match=expected),
        sensitivity.observe_fixed_value_consumers(values=values, diagnostics={}),
    ):
        if consumer == "fixed_share":
            controller = FixedShareWeightController(
                prediction_combination_settings=_default_client_settings().prediction_combination_settings
            )
            controller.update_weights_after_loss_observation()
        elif consumer == "mdl":
            observed_sample_processing.select_alarm_interval_assignment(
                switching_probability=1.0 / 120
            )
        else:
            BoundedLossESRDetector(false_alarm_control_alpha=0.001)
    assert [getattr(target, name) for target, name in targets] == originals
