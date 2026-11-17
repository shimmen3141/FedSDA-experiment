"""DWM（構成`dwm-neural-network`）の設定・参加者の準備・サーバ・全体run・指標・記録の表・保存を確かめる。"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    calculate_detection_metrics,
    extract_concept_change_sample_indices,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_experiment_run import (
    execute_dynamic_weighted_majority_experiment_run,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_run_artifact_files import (
    load_dynamic_weighted_majority_run_artifacts,
    save_dynamic_weighted_majority_experiment_run,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_run_record_tables import (
    build_dynamic_weighted_majority_run_record_tables,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_run_settings import (
    DynamicWeightedMajorityRunParticipantSettings,
    build_default_dynamic_weighted_majority_run_settings,
)
from federated_learning_experiments.runtime.fedavg_run_settings import (
    build_default_fedavg_run_settings,
)


def conditions(dataset="sine2", clients=1, samples=300, interval=50):
    return ExperimentRunConditions(
        dataset_name=dataset,
        random_seed=0,
        client_count=clients,
        per_client_sample_count=samples,
        server_aggregation_interval_per_client_samples=interval,
    )


def settings_with_period(run_conditions, period):
    settings = build_default_dynamic_weighted_majority_run_settings(
        experiment_run_conditions=run_conditions
    )
    return replace(
        settings,
        run_participant_settings=replace(
            settings.run_participant_settings, period_sample_count=period
        ),
    )


@pytest.mark.parametrize("dataset", ["sine2", "mnist4"])
def test_default_settings_take_the_fedavg_learner_and_add_the_paper_values(dataset):
    run_conditions = conditions(dataset=dataset)
    dwm = build_default_dynamic_weighted_majority_run_settings(
        experiment_run_conditions=run_conditions
    )
    fedavg = build_default_fedavg_run_settings(experiment_run_conditions=run_conditions)
    assert dwm.execution_settings == fedavg.execution_settings
    assert dwm.run_metric_settings == fedavg.run_metric_settings
    d, f = dwm.run_participant_settings, fedavg.run_participant_settings
    for name in (
        "initial_model_pretraining_settings",
        "hidden_layer_widths",
        "parameter_optimizer_settings",
        "local_training_batch_sample_count",
        "updates_per_sample",
    ):
        assert getattr(d, name) == getattr(f, name)
    assert (d.weight_decrease_factor, d.removal_weight_threshold, d.period_sample_count) == (
        0.5,
        0.01,
        50,
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(weight_decrease_factor=0.0),
        dict(weight_decrease_factor=1.0),
        dict(removal_weight_threshold=0.0),
        dict(removal_weight_threshold=1.0),
        dict(weight_decrease_factor=1),
        dict(period_sample_count=0),
        dict(period_sample_count=True),
    ],
)
def test_participant_settings_reject_out_of_range_values(changes):
    base = build_default_dynamic_weighted_majority_run_settings(
        experiment_run_conditions=conditions()
    ).run_participant_settings
    with pytest.raises((TypeError, ValueError)):
        DynamicWeightedMajorityRunParticipantSettings(
            **{**{f: getattr(base, f) for f in base.__dataclass_fields__}, **changes}
        )


def run_small(dataset="sine2", clients=2, period=10):
    return execute_dynamic_weighted_majority_experiment_run(
        run_settings=settings_with_period(conditions(dataset=dataset, clients=clients), period)
    )


@pytest.mark.parametrize("dataset", ["sine2", "sea2"])
def test_small_full_run_metrics_match_the_records(dataset):
    run = run_small(dataset=dataset)
    clients = run.measured_run.participants.client_operations
    server = run.measured_run.participants.server_operations
    metrics = run.run_metrics
    # clientごとに独立で、同じ初期状態（事前学習の直後）から始まる。
    first = [client.expert_lifetimes[0] for client in clients]
    assert all(life.created_sample_index == -1 for life in first)
    correctness = [p.prediction_is_correct for c in clients for p in c.sample_predictions]
    assert metrics.prediction_accuracy == pytest.approx(sum(correctness) / len(correctness))
    assert metrics.final_global_model_count == sum(len(c.experts) for c in clients)
    assert metrics.held_model_sample_count == sum(
        p.held_model_count_at_prediction for c in clients for p in c.sample_predictions
    )
    assert metrics.processed_sample_count == 300 * len(clients)
    assert metrics.final_parameter_value_count == sum(
        p.numel() for c in clients for e in c.experts for p in e.classifier.parameters()
    )
    for field_name in convert_settings_to_plain_mapping(metrics.communication_volume):
        if field_name != "settings_type":
            assert getattr(metrics.communication_volume, field_name) == 0
    assert metrics.server_computation_counts.aggregation_parameter_multiply_accumulate_count == 0
    assert metrics.client_consolidation_parameter_multiply_accumulate_count == 0
    changes = tuple(
        extract_concept_change_sample_indices(
            concept_ids_by_sample_index=t.concept_ids_by_sample_index
        )
        for t in run.measured_run.run_result.evaluation_concept_traces
    )
    assert metrics.training_model_switch_detection_metrics == calculate_detection_metrics(
        concept_change_sample_indices_by_client=changes,
        detection_sample_indices_by_client=tuple(() for _ in clients),
        maximum_delay_sample_count=run.run_settings.run_metric_settings.maximum_detection_delay_sample_count,
    )
    assert len(server.synchronized_client_expert_counts) == 300 // 50


def test_clients_start_from_the_same_pretrained_state_and_run_independently():
    run = run_small(clients=2, period=10_000)
    a, b = run.measured_run.participants.client_operations
    # 周期が来ないので専門家は1つのまま。clientごとに別の系列で学習し、パラメータは別のobject。
    assert len(a.experts) == len(b.experts) == 1
    assert a.experts[0].classifier is not b.experts[0].classifier
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(
            a.experts[0].classifier.parameters(), b.experts[0].classifier.parameters(), strict=True
        )
    )


def test_record_tables_follow_the_fedavg_names_and_round_trip_through_files(tmp_path):
    run = run_small(dataset="sine2", clients=1, period=10)
    tables = build_dynamic_weighted_majority_run_record_tables(experiment_run=run)
    client = run.measured_run.participants.client_operations[0]
    assert "model_loss_statistics.mean_loss" not in tables
    assert tables["sample_prediction.prediction_is_correct"].shape == (1, 300)
    assert tables["sample_prediction.held_model_count_at_prediction"].tolist() == [
        [p.held_model_count_at_prediction for p in client.sample_predictions]
    ]
    assert tables["expert_lifetime.expert_id"].tolist() == [
        life.expert_id for life in client.expert_lifetimes
    ]
    assert tables["model_training.model_id"].tolist() == [
        life.expert_id for life in client.expert_lifetimes
    ]
    assert set(tables["model_training.statistics_forward_call_count"].tolist()) == {0}
    paths = save_dynamic_weighted_majority_experiment_run(
        experiment_run=run,
        output_directory=tmp_path,
        file_stem="dwm",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    summary, loaded = load_dynamic_weighted_majority_run_artifacts(
        summary_file_path=paths.summary_file_path
    )
    assert summary["method_name"] == "dynamic_weighted_majority"
    assert summary["run_settings"]["run_participant_settings"]["period_sample_count"] == 10
    assert loaded.keys() == tables.keys()
    for name, values in tables.items():
        np.testing.assert_array_equal(loaded[name], values)
    with pytest.raises(FileExistsError):
        save_dynamic_weighted_majority_experiment_run(
            experiment_run=run,
            output_directory=tmp_path,
            file_stem="dwm",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    text = paths.summary_file_path.read_text(encoding="utf-8").replace(
        '"dynamic_weighted_majority"', '"fedavg"'
    )
    paths.summary_file_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_dynamic_weighted_majority_run_artifacts(summary_file_path=paths.summary_file_path)
