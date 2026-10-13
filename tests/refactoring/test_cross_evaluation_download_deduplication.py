"""LEGACY-018の修正: 転送イベントの独立観測と呼出し単位の重複除外。"""

from dataclasses import replace
from random import Random
from types import SimpleNamespace

import pytest
import torch
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment.servers.clustering import CrossEvaluationClusteringServer
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    ClientModelCrossEvaluation,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.model_cross_evaluation import (
    cross_evaluate_global_models,
)


def observe_legacy_duplicate_downloads(monkeypatch):
    """旧の元処理・計数を維持して、実際の候補IDとclient IDから重複転送だけを積算する。"""
    original_begin = CrossEvaluationClusteringServer._begin_cross_evaluation_model_transfers
    original_record = CrossEvaluationClusteringServer._record_cross_evaluation_model_transfer

    def begin(server):
        server._observed_sent_candidate_client_pairs = set()
        if not hasattr(server, "_observed_duplicate_download_counts"):
            server._observed_duplicate_download_counts = [0, 0, 0]
        return original_begin(server)

    def record(server, model_id, params, target_clients):
        transferred_parameters = [
            parameter for parameter in params.values() if isinstance(parameter, torch.Tensor)
        ]
        value_count = sum(parameter.numel() for parameter in transferred_parameters)
        byte_count = sum(
            parameter.numel() * parameter.element_size() for parameter in transferred_parameters
        )
        for client in target_clients:
            key = (model_id, client.client_id)
            if key in server._observed_sent_candidate_client_pairs:
                for index, count in enumerate((1, value_count, byte_count)):
                    server._observed_duplicate_download_counts[index] += count
            server._observed_sent_candidate_client_pairs.add(key)
        return original_record(server, model_id, params, target_clients)

    monkeypatch.setattr(
        CrossEvaluationClusteringServer, "_begin_cross_evaluation_model_transfers", begin
    )
    monkeypatch.setattr(
        CrossEvaluationClusteringServer, "_record_cross_evaluation_model_transfer", record
    )


def expected_legacy_communication_volume_after_deduplication(legacy_server):
    """未観測の構成は旧との完全一致。観測した構成は下り3項目だけ重複イベント分を引く。"""
    model_count, value_count, byte_count = getattr(
        legacy_server, "_observed_duplicate_download_counts", (0, 0, 0)
    )
    return CommunicationVolumeSnapshot(
        uploaded_model_count=legacy_server.comm_models_up,
        downloaded_model_count=legacy_server.comm_models_down - model_count,
        uploaded_message_count=legacy_server.comm_messages_up,
        downloaded_message_count=legacy_server.comm_messages_down,
        uploaded_parameter_value_count=legacy_server.comm_parameter_values_up,
        downloaded_parameter_value_count=legacy_server.comm_parameter_values_down - value_count,
        uploaded_byte_count=legacy_server.comm_bytes_up,
        downloaded_byte_count=legacy_server.comm_bytes_down - byte_count,
    )


def test_legacy_observer_preserves_counts_and_distinguishes_identical_candidates(monkeypatch):
    observe_legacy_duplicate_downloads(monkeypatch)
    server = CrossEvaluationClusteringServer(distance_threshold=0.1, verbose=False)
    parameters = {"weight": torch.ones(2), "bias": torch.ones(1, dtype=torch.float64)}
    clients = [SimpleNamespace(client_id=0), SimpleNamespace(client_id=1)]
    for _ in range(2):
        server._begin_cross_evaluation_model_transfers()
        server._record_cross_evaluation_model_transfer(0, parameters, clients)
        server._record_cross_evaluation_model_transfer(1, parameters, clients)
        server._record_cross_evaluation_model_transfer(0, parameters, clients[:1])
    assert (server.comm_models_down, server.comm_parameter_values_down, server.comm_bytes_down) == (
        10,
        30,
        160,
    )
    assert server._observed_duplicate_download_counts == [2, 6, 32]
    expected = expected_legacy_communication_volume_after_deduplication(server)
    assert (
        expected.downloaded_model_count,
        expected.downloaded_parameter_value_count,
        expected.downloaded_byte_count,
    ) == (8, 24, 128)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("client_limit", [1, 3])
@pytest.mark.parametrize("held_ids", [((0, 1), (0, 1), (0,)), ((0,), (1,), ()), ((), (), ())])
def test_downloads_are_unique_per_candidate_client_and_per_call(
    monkeypatch, valid_run_settings_mapping, shared, client_limit, held_ids
):
    from test_server_model_registration_and_aggregation import build_server_round_oracle

    oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    clients = oracle["run_clients"]
    # 同値の候補でも、正式IDが違えば別モデル。dtypeの違うパラメータでバイト計数も検証する。
    parameters = {
        "prediction_head.weight": torch.ones(2, dtype=torch.float32),
        "prediction_head.bias": torch.ones(1, dtype=torch.float64),
    }
    if shared:
        parameters["feature_extractor.weight"] = torch.ones(3, dtype=torch.float32)
    repository = GlobalModelRepository(
        initial_model_id=0,
        initial_parameter_snapshot=parameters,
        initial_loss_statistics=oracle["global_model_repository"].get_global_model_loss_statistics(
            model_id=0
        ),
    )
    for model_id in (1, 2):
        repository.set_global_model_parameters(model_id=model_id, parameter_snapshot=parameters)
    monkeypatch.setattr(
        FedsdaRunClient,
        "get_cross_evaluation_held_model_ids",
        lambda client: held_ids[client.client_id],
    )
    evaluations = []

    def evaluate(client, **kwargs):
        evaluations.append((client.client_id, kwargs["target_model_id"]))
        return ClientModelCrossEvaluation(
            evaluated_sample_count=0,
            bounded_loss_sum=0.0,
            squared_bounded_loss_sum=0.0,
            correctness_counts=None,
            class_correctness_counts=(),
        )

    monkeypatch.setattr(
        FedsdaRunClient, "evaluate_candidate_model_on_target_model_samples", evaluate
    )
    volume = CommunicationVolumeRecordStore()
    records = CrossEvaluationRecordStore()
    generator, expected_generator = Random(37), Random(37)
    expected_models = expected_values = expected_bytes = expected_messages = 0
    for round_index in (0, 1):
        sent_pairs, sent_clients, expected_evaluations = set(), set(), []
        for candidate_id in (0, 1, 2):
            for target_id in (0, 1, 2):
                selected = [
                    client.client_id
                    for client in clients
                    if target_id in held_ids[client.client_id]
                ]
                if len(selected) > client_limit:
                    selected = expected_generator.sample(selected, client_limit)
                expected_evaluations.extend((client_id, target_id) for client_id in selected)
                sent_pairs.update((candidate_id, client_id) for client_id in selected)
                sent_clients.update(selected)
        expected_models += len(sent_pairs)
        expected_values += 3 * len(sent_pairs) + (3 * len(sent_clients) if shared else 0)
        expected_bytes += 16 * len(sent_pairs) + (12 * len(sent_clients) if shared else 0)
        expected_messages += len(expected_evaluations)
        evaluations.clear()
        cross_evaluate_global_models(
            run_clients=clients,
            global_model_repository=repository,
            communication_volume_record_store=volume,
            cross_evaluation_record_store=records,
            cross_evaluated_model_ids=(0, 1, 2),
            round_index=round_index,
            maximum_evaluating_client_count_per_model=client_limit,
            python_random_generator=generator,
        )
        assert evaluations == expected_evaluations
        assert generator.getstate() == expected_generator.getstate()
        assert volume.get_state_snapshot() == CommunicationVolumeSnapshot(
            uploaded_model_count=0,
            downloaded_model_count=expected_models,
            uploaded_message_count=expected_messages,
            downloaded_message_count=expected_messages,
            uploaded_parameter_value_count=0,
            downloaded_parameter_value_count=expected_values,
            uploaded_byte_count=0,
            downloaded_byte_count=expected_bytes,
        )
        assert len(records.snapshot_client_cross_evaluation_records()) == expected_messages


@pytest.mark.slow
def test_independent_run_saves_corrected_communication_volume(
    monkeypatch, tmp_path, valid_run_settings_mapping
):
    """helper内で実旧から重複分を引いた値を検証し、その値が単独APIのJSONにも保存される。"""
    from test_fedsda_configuration_variants import CONFIGURATION_VARIANTS
    from test_fedsda_run_metric_derivation import compare_small_run_with_real_legacy
    from test_fedsda_stream_protocol_run import (
        make_execution_settings,
        make_run_participant_settings,
    )

    from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_artifact_files import (
        load_fedsda_run_artifacts,
        save_fedsda_experiment_run,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    variant = CONFIGURATION_VARIANTS["fedsda-independent-models"]
    conditions = dict(
        random_seed=0,
        client_count=3,
        per_client_sample_count=300,
        aggregation_interval=10,
        minimum_change_gap=30,
        concept_change_probability=0.05,
        update_interval=2,
    )
    comparison = compare_small_run_with_real_legacy(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        valid_run_settings_mapping=valid_run_settings_mapping,
        **conditions,
        maximum_delay=100,
        recovery_window=50,
        configuration_variant=variant,
    )
    execution = make_execution_settings(
        **{key: value for key, value in conditions.items() if key != "update_interval"}
    )
    settings = build_experiment_run_settings(
        configuration_name="fedsda-independent-models",
        experiment_run_conditions=execution.experiment_run_conditions,
    )
    settings = replace(
        settings,
        execution_settings=execution,
        run_participant_settings=variant.modify_run_participant_settings(
            make_run_participant_settings(valid_run_settings_mapping, update_interval=2)
        ),
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=100, post_change_recovery_window_sample_count=50
        ),
    )
    run = execute_fedsda_experiment_run(run_settings=settings)
    assert run.run_metrics.communication_volume == comparison.run_metrics.communication_volume
    paths = save_fedsda_experiment_run(
        experiment_run=run, output_directory=tmp_path, file_stem="independent"
    )
    summary, tables = load_fedsda_run_artifacts(summary_file_path=paths.summary_file_path)
    from dataclasses import asdict

    assert summary["run_metrics"]["communication_volume"] == asdict(
        comparison.run_metrics.communication_volume
    ) | {"settings_type": "CommunicationVolumeSnapshot"}
    # NPZは通信量を持たない。評価の診断列が保存され、重複除外で評価そのものが省略されないことを確認する。
    records = run.measured_run.participants.server_operations.owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    assert tables["client_cross_evaluation.client_id"].size == len(records)
