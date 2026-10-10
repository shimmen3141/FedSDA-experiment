"""1 runの結果を、表ごと・列ごとの配列（保存する形）にする。

キーは`表の名前.列の名前`。列の名前は、新実装の記録のfieldの名前。事象の表（概念の変更、適応ほか）は、
行ごとに`client_id`を持つ1次元の列で、client順・起きた順に並ぶ。標本ごとの表は、client×標本の2次元。
ラウンドごとの表は、ラウンド（×client）。整数は`int64`、真偽は`bool`、文字列はNumPyのUnicode。
読取りだけを行い、参加者の状態と乱数を変えない。
"""

from dataclasses import fields
from typing import cast

from numpy import asarray, bool_, int64, ndarray, str_

from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import FedsdaExperimentRun
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.fedsda_run_server import FedsdaRunServer

# 適応の結果種別から、粗い区分への対応。旧の結果（粗い区分しか持たない）を変換した表と、同じ列を持てるようにする。
ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME: dict[str, str] = {
    "alarm_during_candidate_validation": "defer_during_candidate_validation",
    "alarm_change_interval_too_short": "skip_too_short_change_interval",
    "alarm_interval_held_model_reused": "reuse_held_model",
    "alarm_interval_current_model_maintained": "maintain_current_model",
    "alarm_interval_candidate_validation_started": "start_candidate_validation",
    "post_alarm_validation_candidate_adopted": "adopt_candidate_model",
    "post_alarm_validation_held_model_reused": "reuse_held_model",
    "post_alarm_validation_current_model_maintained": "maintain_current_model",
    "post_alarm_validation_candidate_rejected": "reject_candidate_model",
    "post_alarm_validation_incomplete_candidate_rejected": "reject_candidate_model",
    "server_consolidation_training_model_remapped": "remap_by_server_consolidation",
}
# 初期モデル（登録のラウンドがない）の、登録ラウンドの値。
INITIAL_MODEL_REGISTERED_ROUND_INDEX = -1


def _integer_column(values: list[int] | list[list[int]]) -> ndarray:
    return asarray(values, dtype=int64)


def _boolean_column(values: list[bool] | list[list[bool]]) -> ndarray:
    return asarray(values, dtype=bool_)


def _text_column(values: list[str]) -> ndarray:
    return asarray(values, dtype=str_)


def _build_round_computation_table(
    *, table_name: str, round_computation_counts: list[ModelComputationCounts]
) -> dict[str, ndarray]:
    return {
        f"{table_name}.{count_field.name}": _integer_column(
            [getattr(counts, count_field.name) for counts in round_computation_counts]
        )
        for count_field in fields(ModelComputationCounts)
    }


def build_fedsda_run_record_tables(*, experiment_run: FedsdaExperimentRun) -> dict[str, ndarray]:
    """1 runの結果から、保存する表（`表の名前.列の名前`→配列）を作る。"""
    if type(experiment_run) is not FedsdaExperimentRun:
        raise TypeError("experiment_run must be exact FedsdaExperimentRun")
    measured_run = experiment_run.measured_run
    run_result = measured_run.run_result
    participants = measured_run.participants
    run_clients: list[FedsdaRunClient] = []
    for client_operations in participants.client_operations:
        if type(client_operations) is not FedsdaRunClient:
            raise TypeError("client operations must be exact FedsdaRunClient")
        run_clients.append(cast(FedsdaRunClient, client_operations))
    if type(participants.server_operations) is not FedsdaRunServer:
        raise TypeError("server operations must be exact FedsdaRunServer")
    run_server = cast(FedsdaRunServer, participants.server_operations)
    server_owners = run_server.owners
    record_tables: dict[str, ndarray] = {}

    # ---- 標本ごとの予測（client×処理した標本） ----
    prediction_records_by_client = [
        run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        for run_client in run_clients
    ]
    held_model_counts_by_client = [
        list(run_client.owners.held_model_count_record_store.snapshot_held_model_counts())
        for run_client in run_clients
    ]
    record_tables["sample_prediction.combined_prediction_is_correct"] = _boolean_column(
        [
            [record.combined_prediction_is_correct for record in prediction_records]
            for prediction_records in prediction_records_by_client
        ]
    )
    record_tables["sample_prediction.maximum_weight_model_id"] = _integer_column(
        [
            [record.maximum_weight_model_id for record in prediction_records]
            for prediction_records in prediction_records_by_client
        ]
    )
    record_tables["sample_prediction.held_model_count"] = _integer_column(
        held_model_counts_by_client
    )

    # ---- 概念の変更（評価用の概念列の、前の標本と概念が違う位置。処理しなかった末尾の標本を含む） ----
    concept_changes = [
        (client_id, sample_index)
        for client_id, concept_trace in enumerate(run_result.evaluation_concept_traces)
        for sample_index in range(1, len(concept_trace.concept_ids_by_sample_index))
        if concept_trace.concept_ids_by_sample_index[sample_index]
        != concept_trace.concept_ids_by_sample_index[sample_index - 1]
    ]
    record_tables["concept_change.client_id"] = _integer_column(
        [client_id for client_id, _ in concept_changes]
    )
    record_tables["concept_change.sample_index"] = _integer_column(
        [sample_index for _, sample_index in concept_changes]
    )

    # ---- 学習帰属の切替と、適応の記録 ----
    adaptation_snapshots = [
        run_client.owners.adaptation_record_store.get_state_snapshot() for run_client in run_clients
    ]
    switches = [
        (client_id, sample_index)
        for client_id, adaptation_snapshot in enumerate(adaptation_snapshots)
        for sample_index in adaptation_snapshot.training_model_switch_sample_indices
    ]
    record_tables["training_model_switch.client_id"] = _integer_column(
        [client_id for client_id, _ in switches]
    )
    record_tables["training_model_switch.sample_index"] = _integer_column(
        [sample_index for _, sample_index in switches]
    )
    adaptations = [
        (client_id, adaptation_record)
        for client_id, adaptation_snapshot in enumerate(adaptation_snapshots)
        for adaptation_record in adaptation_snapshot.adaptation_records
    ]
    for _, adaptation_record in adaptations:
        if adaptation_record.adaptation_outcome not in ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME:
            raise ValueError(
                f"adaptation outcome has no action category: {adaptation_record.adaptation_outcome!r}"
            )
    record_tables["adaptation.client_id"] = _integer_column(
        [client_id for client_id, _ in adaptations]
    )
    record_tables["adaptation.adaptation_sample_index"] = _integer_column(
        [adaptation_record.adaptation_sample_index for _, adaptation_record in adaptations]
    )
    record_tables["adaptation.adaptation_outcome"] = _text_column(
        [adaptation_record.adaptation_outcome for _, adaptation_record in adaptations]
    )
    record_tables["adaptation.adaptation_action"] = _text_column(
        [
            ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME[adaptation_record.adaptation_outcome]
            for _, adaptation_record in adaptations
        ]
    )
    record_tables["adaptation.previous_training_model_id"] = _integer_column(
        [adaptation_record.previous_training_model_id for _, adaptation_record in adaptations]
    )
    record_tables["adaptation.current_training_model_id"] = _integer_column(
        [adaptation_record.current_training_model_id for _, adaptation_record in adaptations]
    )

    # ---- 候補検証の判定（確定したものと、終端で回収した未完了のもの） ----
    decision_client_ids: list[int] = []
    decision_proposal_sample_indices: list[int] = []
    decision_is_incomplete: list[bool] = []
    decision_candidate_accepted: list[bool] = []
    decision_reasons: list[str] = []
    decision_resolution_sample_indices: list[int] = []
    for client_id, run_client in enumerate(run_clients):
        for decision_record in run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records():
            decision_client_ids.append(client_id)
            decision_proposal_sample_indices.append(decision_record.proposal_sample_index)
            if type(decision_record) is IncompletePostAlarmCandidateValidationDecisionRecord:
                incomplete_record = cast(
                    IncompletePostAlarmCandidateValidationDecisionRecord, decision_record
                )
                decision_is_incomplete.append(True)
                decision_candidate_accepted.append(False)
                decision_reasons.append("")
                decision_resolution_sample_indices.append(
                    incomplete_record.finalization_sample_index
                )
            else:
                completed_record = cast(PostAlarmCandidateValidationDecisionRecord, decision_record)
                loss_evaluation = completed_record.post_alarm_candidate_loss_evaluation
                decision_is_incomplete.append(False)
                decision_candidate_accepted.append(loss_evaluation.candidate_accepted)
                decision_reasons.append(loss_evaluation.decision_reason)
                decision_resolution_sample_indices.append(completed_record.resolution_sample_index)
    record_tables["candidate_validation_decision.client_id"] = _integer_column(decision_client_ids)
    record_tables["candidate_validation_decision.proposal_sample_index"] = _integer_column(
        decision_proposal_sample_indices
    )
    record_tables["candidate_validation_decision.is_incomplete_at_run_end"] = _boolean_column(
        decision_is_incomplete
    )
    record_tables["candidate_validation_decision.candidate_accepted"] = _boolean_column(
        decision_candidate_accepted
    )
    record_tables["candidate_validation_decision.decision_reason"] = _text_column(decision_reasons)
    record_tables["candidate_validation_decision.resolution_sample_index"] = _integer_column(
        decision_resolution_sample_indices
    )

    # ---- モデルの登録 ----
    global_model_repository = server_owners.global_model_repository
    registration_records = global_model_repository.snapshot_model_registration_records()
    final_global_model_ids = set(global_model_repository.global_model_ids)
    record_tables["model_registration.model_id"] = _integer_column(
        [registration_record.model_id for registration_record in registration_records]
    )
    record_tables["model_registration.registered_round_index"] = _integer_column(
        [
            INITIAL_MODEL_REGISTERED_ROUND_INDEX
            if registration_record.registered_round_index is None
            else registration_record.registered_round_index
            for registration_record in registration_records
        ]
    )
    record_tables["model_registration.is_final_global_model"] = _boolean_column(
        [
            registration_record.model_id in final_global_model_ids
            for registration_record in registration_records
        ]
    )

    # ---- クラスタリングの観測（モデルごと、対ごと） ----
    model_observations = (
        server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
    )
    record_tables["model_clustering.round_index"] = _integer_column(
        [model_observation.round_index for model_observation in model_observations]
    )
    record_tables["model_clustering.model_id"] = _integer_column(
        [model_observation.model_id for model_observation in model_observations]
    )
    record_tables["model_clustering.representative_model_id"] = _integer_column(
        [model_observation.representative_model_id for model_observation in model_observations]
    )
    record_tables["model_clustering.merged_with_other_models"] = _boolean_column(
        [model_observation.merged_with_other_models for model_observation in model_observations]
    )
    record_tables["model_clustering.absorbed_into_representative"] = _boolean_column(
        [model_observation.absorbed_into_representative for model_observation in model_observations]
    )
    pair_observations = (
        server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
    )
    record_tables["model_pair_clustering.round_index"] = _integer_column(
        [pair_observation.round_index for pair_observation in pair_observations]
    )
    record_tables["model_pair_clustering.lower_model_id"] = _integer_column(
        [pair_observation.lower_model_id for pair_observation in pair_observations]
    )
    record_tables["model_pair_clustering.higher_model_id"] = _integer_column(
        [pair_observation.higher_model_id for pair_observation in pair_observations]
    )
    record_tables["model_pair_clustering.assigned_to_same_cluster"] = _boolean_column(
        [pair_observation.assigned_to_same_cluster for pair_observation in pair_observations]
    )

    # ---- ラウンドごとのモデル数（同期の配布の後の値） ----
    # 同期の最後の配布で、各clientが受け取った後の保有モデルのIDから数える。グローバルモデルは、
    # 正式ID（0以上）のモデルで、全clientが同じものを受け取る。clientは、そのほかに、まだ登録していない
    # 一時ID（負）のモデルを持つことがある。
    synchronizations = run_server.snapshot_server_round_synchronizations()
    global_model_counts: list[int] = []
    client_held_model_counts: list[list[int]] = []
    for synchronization in synchronizations:
        held_model_ids_by_client = [
            distribution_application.held_model_ids
            for distribution_application in synchronization.distribution_applications
        ]
        if len(held_model_ids_by_client) != len(run_clients):
            raise ValueError("every client must have a distribution application in each round")
        distributed_global_model_ids_by_client = {
            tuple(model_id for model_id in held_model_ids if model_id >= 0)
            for held_model_ids in held_model_ids_by_client
        }
        if len(distributed_global_model_ids_by_client) != 1:
            raise ValueError("all clients must hold the same global models after a distribution")
        global_model_counts.append(len(next(iter(distributed_global_model_ids_by_client))))
        client_held_model_counts.append(
            [len(held_model_ids) for held_model_ids in held_model_ids_by_client]
        )
    record_tables["round.global_model_count"] = _integer_column(global_model_counts)
    record_tables["round.client_held_model_count"] = asarray(
        client_held_model_counts, dtype=int64
    ).reshape(len(synchronizations), len(run_clients))

    # ---- ラウンドごとのモデルの計算（ローカルの処理と、同期の別） ----
    record_tables.update(
        _build_round_computation_table(
            table_name="round_local_processing_computation",
            round_computation_counts=[
                round_counts.local_processing_model_computation_counts
                for round_counts in measured_run.round_model_computation_counts
            ],
        )
    )
    record_tables.update(
        _build_round_computation_table(
            table_name="round_synchronization_computation",
            round_computation_counts=[
                round_counts.synchronization_model_computation_counts
                for round_counts in measured_run.round_model_computation_counts
            ],
        )
    )
    return record_tables
