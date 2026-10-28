"""1 runの結果を、表ごと・列ごとの配列（保存する形）にする。

キーは`表の名前.列の名前`。列の名前は、新実装の記録のfieldの名前。事象の表（概念の変更、適応ほか）は、
行ごとに`client_id`を持つ1次元の列で、client順・起きた順に並ぶ。標本ごとの表は、client×標本の2次元。
サーバの同期のラウンドごとの表`server_round`は、ラウンド（×client）。整数は`int64`、真偽は`bool`、文字列はNumPyのUnicode。
読取りだけを行い、参加者の状態と乱数を変えない。
"""

from dataclasses import fields
from math import nan
from typing import cast

from numpy import asarray, bool_, float64, int64, ndarray, str_

from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)
from federated_learning_experiments.runtime.fedsda_change_response_metric_derivation import (
    ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import FedsdaExperimentRun
from federated_learning_experiments.runtime.fedsda_model_diagnostic_metric_derivation import (
    collect_client_held_model_training_amounts,
    count_models_after_server_round_synchronizations,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.fedsda_run_server import FedsdaRunServer

# 初期モデル（登録のラウンドがない）の、登録ラウンドの値。
INITIAL_MODEL_REGISTERED_ROUND_INDEX = -1
# 整数の列で、値がないこと（推定した変化点がない、比較した参照モデルがない、登録したclientがない、
# 最も近い相手のモデルがない、真の概念の一致が分からない、クロス評価で正誤を比べなかった、
# 標本の真の概念が分からない）を表す値。
MISSING_INTEGER_VALUE = -1


def _integer_column(values: list[int] | list[list[int]]) -> ndarray:
    return asarray(values, dtype=int64)


def _boolean_column(values: list[bool] | list[list[bool]]) -> ndarray:
    return asarray(values, dtype=bool_)


def _text_column(values: list[str]) -> ndarray:
    return asarray(values, dtype=str_)


def _float_column(values: list[float] | list[list[float]]) -> ndarray:
    return asarray(values, dtype=float64)


def _build_server_round_computation_columns(
    *, phase_name: str, round_computation_counts: list[ModelComputationCounts]
) -> dict[str, ndarray]:
    """表`server_round`の、ラウンドの中の1つの段（ローカルの処理か、同期）の、モデルの計算の列。"""
    return {
        f"server_round.{phase_name}_{count_field.name}": _integer_column(
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
    record_tables["sample_prediction.held_model_count_at_prediction"] = _integer_column(
        held_model_counts_by_client
    )
    # 観測したクラスと、観測した真の概念（分からなければ、−1）。
    record_tables["sample_prediction.observed_class_id"] = _integer_column(
        [
            [record.observed_class_id for record in prediction_records]
            for prediction_records in prediction_records_by_client
        ]
    )
    record_tables["sample_prediction.observed_concept_id"] = _integer_column(
        [
            [
                MISSING_INTEGER_VALUE
                if record.observed_concept_id is None
                else record.observed_concept_id
                for record in prediction_records
            ]
            for prediction_records in prediction_records_by_client
        ]
    )
    # Fixed-Shareの重みでの、最大重みのモデル（Fixed-Shareの重みで予測する設定では、上の列と同じ値）。
    record_tables["sample_prediction.fixed_share_maximum_weight_model_id"] = _integer_column(
        [
            [record.fixed_share_maximum_weight_model_id for record in prediction_records]
            for prediction_records in prediction_records_by_client
        ]
    )
    # 予測重みの最大と、実効モデル数（予測重みの二乗和の逆数）。実効モデル数は、Fixed-Shareの重みでの値も持つ。
    for column_name in (
        "maximum_prediction_weight",
        "effective_model_count",
        "fixed_share_effective_model_count",
    ):
        record_tables[f"sample_prediction.{column_name}"] = _float_column(
            [
                [getattr(record, column_name) for record in prediction_records]
                for prediction_records in prediction_records_by_client
            ]
        )
    # 比較用の予測の正否（予測と学習の判断には、使われていない）。
    for column_name in (
        "any_model_or_combined_prediction_is_correct",
        "maximum_weight_model_prediction_is_correct",
        "highest_confidence_model_prediction_is_correct",
        "adahedge_weighted_prediction_is_correct",
        "fixed_share_weighted_prediction_is_correct",
    ):
        record_tables[f"sample_prediction.{column_name}"] = _boolean_column(
            [
                [getattr(record, column_name) for record in prediction_records]
                for prediction_records in prediction_records_by_client
            ]
        )
    # 真の概念別の診断重みでの結合の正否: 正解1、不正解0、真の概念が分からなければ−1。
    record_tables["sample_prediction.true_concept_diagnostic_prediction_is_correct"] = (
        _integer_column(
            [
                [
                    MISSING_INTEGER_VALUE
                    if record.true_concept_diagnostic_prediction_is_correct is None
                    else int(record.true_concept_diagnostic_prediction_is_correct)
                    for record in prediction_records
                ]
                for prediction_records in prediction_records_by_client
            ]
        )
    )

    # ---- 概念の変更（評価用の概念列で、前の標本と概念が違う標本——新しい概念の最初の標本——の位置。
    # 処理しなかった末尾の標本を含む） ----
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
    record_tables["concept_change.first_new_concept_sample_index"] = _integer_column(
        [sample_index for _, sample_index in concept_changes]
    )

    # ---- 損失の変化の警報（警報ごとの、位置・検出器が推定した変化点・検出器の候補の開始位置）と、
    # 標本ごとの、損失の監視の値（log e。client×処理した標本） ----
    alarm_snapshots = [
        run_client.owners.loss_change_alarm_record_store.get_state_snapshot()
        for run_client in run_clients
    ]
    record_tables["loss_change_alarm.client_id"] = _integer_column(
        [
            client_id
            for client_id, alarm_snapshot in enumerate(alarm_snapshots)
            for _ in alarm_snapshot.alarm_sample_indices
        ]
    )
    for alarm_column_name, alarm_snapshot_field_name in (
        ("alarm_sample_index", "alarm_sample_indices"),
        ("estimated_change_point_sample_index", "estimated_change_point_sample_indices"),
        ("detector_candidate_start_sample_index", "detector_candidate_start_sample_indices"),
    ):
        record_tables[f"loss_change_alarm.{alarm_column_name}"] = _integer_column(
            [
                sample_index
                for alarm_snapshot in alarm_snapshots
                for sample_index in getattr(alarm_snapshot, alarm_snapshot_field_name)
            ]
        )
    record_tables["loss_monitoring_baseline_lowering.client_id"] = _integer_column(
        [
            client_id
            for client_id, snapshot in enumerate(alarm_snapshots)
            for _ in snapshot.baseline_lowering_sample_indices
        ]
    )
    for column_name, field_name, make_column in (
        ("sample_index", "baseline_lowering_sample_indices", _integer_column),
        ("previous_baseline_loss_mean", "previous_baseline_loss_means", _float_column),
        ("lowered_baseline_loss_mean", "lowered_baseline_loss_means", _float_column),
    ):
        record_tables[f"loss_monitoring_baseline_lowering.{column_name}"] = make_column(
            [value for snapshot in alarm_snapshots for value in getattr(snapshot, field_name)]
        )
    # ---- 警報の後の、最小記述長の原理による選択（今の扱いでは、空の表） ----
    selection_record_lists = [
        run_client.owners.alarm_interval_assignment_selection_record_store.snapshot_selection_records()
        for run_client in run_clients
    ]
    record_tables["alarm_interval_assignment_selection.client_id"] = _integer_column(
        [
            client_id
            for client_id, selection_records in enumerate(selection_record_lists)
            for _ in selection_records
        ]
    )
    selection_records = [record for records in selection_record_lists for record in records]
    for column_name in (
        "alarm_sample_index",
        "change_interval_sample_count",
        "destination_model_id",
    ):
        record_tables[f"alarm_interval_assignment_selection.{column_name}"] = _integer_column(
            [getattr(record, column_name) for record in selection_records]
        )
    record_tables["alarm_interval_assignment_selection.selected_hypothesis"] = _text_column(
        [record.selected_hypothesis for record in selection_records]
    )
    record_tables["alarm_interval_assignment_selection.description_length"] = _float_column(
        [record.description_length for record in selection_records]
    )
    record_tables["alarm_interval_assignment_selection.alarm_raised"] = _boolean_column(
        [record.alarm_raised for record in selection_records]
    )
    record_tables["alarm_interval_assignment_selection.new_model_learner_name"] = _text_column(
        [record.new_model_learner_name for record in selection_records]
    )
    record_tables["sample_loss_monitoring.monitored_log_e_value"] = _float_column(
        [list(alarm_snapshot.monitored_log_e_values) for alarm_snapshot in alarm_snapshots]
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
    record_tables["training_model_switch.switch_sample_index"] = _integer_column(
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
    record_tables["adaptation.detector_name"] = _text_column(
        [adaptation_record.detector_name for _, adaptation_record in adaptations]
    )
    record_tables["adaptation.estimated_change_point_sample_index"] = _integer_column(
        [
            MISSING_INTEGER_VALUE
            if adaptation_record.estimated_change_point_sample_index is None
            else adaptation_record.estimated_change_point_sample_index
            for _, adaptation_record in adaptations
        ]
    )
    record_tables["adaptation.previous_training_model_id"] = _integer_column(
        [adaptation_record.previous_training_model_id for _, adaptation_record in adaptations]
    )
    record_tables["adaptation.current_training_model_id"] = _integer_column(
        [adaptation_record.current_training_model_id for _, adaptation_record in adaptations]
    )

    # ---- モデルの登録 ----
    global_model_repository = server_owners.global_model_repository
    registration_records = global_model_repository.snapshot_model_registration_records()
    final_global_model_ids = set(global_model_repository.global_model_ids)
    record_tables["model_registration.model_id"] = _integer_column(
        [registration_record.model_id for registration_record in registration_records]
    )
    record_tables["model_registration.registered_server_round_index"] = _integer_column(
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
    # 登録したclient（初期モデルには、ない）。
    record_tables["model_registration.registering_client_id"] = _integer_column(
        [
            MISSING_INTEGER_VALUE
            if registration_record.registering_client_id is None
            else registration_record.registering_client_id
            for registration_record in registration_records
        ]
    )
    # 全clientの、処理した標本のうち、そのモデルが、予測で最大重みだった標本の数。
    maximum_weight_prediction_sample_counts: dict[int, int] = {}
    for prediction_records in prediction_records_by_client:
        for prediction_record in prediction_records:
            maximum_weight_prediction_sample_counts[prediction_record.maximum_weight_model_id] = (
                maximum_weight_prediction_sample_counts.get(
                    prediction_record.maximum_weight_model_id, 0
                )
                + 1
            )
    record_tables["model_registration.maximum_weight_prediction_sample_count"] = _integer_column(
        [
            maximum_weight_prediction_sample_counts.get(registration_record.model_id, 0)
            for registration_record in registration_records
        ]
    )

    # ---- クラスタリングの観測（モデルごと、対ごと） ----
    model_observations = (
        server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
    )
    record_tables["model_clustering.server_round_index"] = _integer_column(
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
    # 損失の距離が最も小さい相手と、その距離（距離を持つ対がなければ、−1とNaN）。
    record_tables["model_clustering.nearest_model_id"] = _integer_column(
        [
            MISSING_INTEGER_VALUE
            if model_observation.nearest_model_id is None
            else model_observation.nearest_model_id
            for model_observation in model_observations
        ]
    )
    record_tables["model_clustering.nearest_loss_increase_distance"] = _float_column(
        [
            nan
            if model_observation.nearest_loss_increase_distance is None
            else model_observation.nearest_loss_increase_distance
            for model_observation in model_observations
        ]
    )
    record_tables["model_clustering.cluster_model_count"] = _integer_column(
        [model_observation.cluster_model_count for model_observation in model_observations]
    )
    # 自分のクラスタの中の対の、損失の距離の最大（距離を持つ対がなければ、NaN）。
    record_tables["model_clustering.maximum_within_cluster_distance"] = _float_column(
        [
            nan
            if model_observation.maximum_within_cluster_distance is None
            else model_observation.maximum_within_cluster_distance
            for model_observation in model_observations
        ]
    )
    record_tables["model_clustering.evaluated_within_cluster_pair_count"] = _integer_column(
        [
            model_observation.evaluated_within_cluster_pair_count
            for model_observation in model_observations
        ]
    )
    record_tables["model_clustering.possible_within_cluster_pair_count"] = _integer_column(
        [
            model_observation.possible_within_cluster_pair_count
            for model_observation in model_observations
        ]
    )
    pair_observations = (
        server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
    )
    record_tables["model_pair_clustering.server_round_index"] = _integer_column(
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
    record_tables["model_pair_clustering.loss_increase_distance"] = _float_column(
        [pair_observation.loss_increase_distance for pair_observation in pair_observations]
    )
    record_tables["model_pair_clustering.decision_score"] = _float_column(
        [pair_observation.decision_score for pair_observation in pair_observations]
    )
    # 真の概念の一致（診断値）: 一致1、不一致0、分からなければ−1。
    record_tables["model_pair_clustering.true_concepts_match"] = _integer_column(
        [
            MISSING_INTEGER_VALUE
            if pair_observation.true_concepts_match is None
            else int(pair_observation.true_concepts_match)
            for pair_observation in pair_observations
        ]
    )
    record_tables["model_pair_clustering.concept_specific_parameter_distance"] = _float_column(
        [
            pair_observation.concept_specific_parameter_distance
            for pair_observation in pair_observations
        ]
    )

    # ---- クロス評価の、clientの評価ごとの記録（評価した順）と、そのクラスごとの記録（クラスの昇順） ----
    # 候補（candidate）は評価する側、対象（target）は標本を持つ側のモデル。正誤の4つの数は、
    # 標本ごとの正誤を比べなかった評価では、4つとも−1。
    cross_evaluation_records = (
        server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    )
    correctness_column_names = (
        "candidate_only_correct_sample_count",
        "target_only_correct_sample_count",
        "both_correct_sample_count",
        "both_wrong_sample_count",
    )
    for column_name, record_field_name in (
        ("server_round_index", "round_index"),
        ("client_id", "client_id"),
        ("candidate_model_id", "candidate_model_id"),
        ("target_model_id", "target_model_id"),
        ("evaluated_sample_count", "evaluated_sample_count"),
    ):
        record_tables[f"client_cross_evaluation.{column_name}"] = _integer_column(
            [
                getattr(cross_evaluation_record, record_field_name)
                for cross_evaluation_record in cross_evaluation_records
            ]
        )
    for column_name in ("bounded_loss_sum", "squared_bounded_loss_sum"):
        record_tables[f"client_cross_evaluation.{column_name}"] = _float_column(
            [
                getattr(cross_evaluation_record, column_name)
                for cross_evaluation_record in cross_evaluation_records
            ]
        )
    for count_position, column_name in enumerate(correctness_column_names):
        record_tables[f"client_cross_evaluation.{column_name}"] = _integer_column(
            [
                MISSING_INTEGER_VALUE
                if cross_evaluation_record.correctness_counts is None
                else cross_evaluation_record.correctness_counts[count_position]
                for cross_evaluation_record in cross_evaluation_records
            ]
        )
    class_rows = [
        (
            cross_evaluation_record.round_index,
            cross_evaluation_record.client_id,
            cross_evaluation_record.candidate_model_id,
            cross_evaluation_record.target_model_id,
            *class_counts,
        )
        for cross_evaluation_record in cross_evaluation_records
        for class_counts in cross_evaluation_record.class_correctness_counts
    ]
    for row_position, column_name in enumerate(
        (
            "server_round_index",
            "client_id",
            "candidate_model_id",
            "target_model_id",
            "class_id",
            "class_evaluated_sample_count",
            *correctness_column_names,
        )
    ):
        record_tables[f"client_cross_evaluation_class.{column_name}"] = _integer_column(
            [class_row[row_position] for class_row in class_rows]
        )

    # ---- clientごとの、重みの更新に関わる出来事の回数（実行の終わりの値） ----
    # AdaHedgeの重みと、Fixed-Shareの重みの別。
    record_tables["client_weight_update_counts.client_id"] = _integer_column(
        [run_client.client_id for run_client in run_clients]
    )
    for column_name, evidence_property_name in (
        ("adahedge_weight_model_pool_reset_count", "model_pool_reset_count"),
        (
            "adahedge_weight_concept_operation_restart_count",
            "concept_operation_restart_count",
        ),
        (
            "adahedge_weight_aggregation_recalibration_count",
            "aggregation_recalibration_count",
        ),
        (
            "adahedge_weight_aggregation_recalibration_replayed_sample_count",
            "aggregation_recalibration_sample_count",
        ),
    ):
        record_tables[f"client_weight_update_counts.{column_name}"] = _integer_column(
            [
                getattr(
                    run_client.owners.all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence,
                    evidence_property_name,
                )
                for run_client in run_clients
            ]
        )
    for column_name, controller_property_name in (
        ("fixed_share_weight_model_pool_reset_count", "model_pool_reset_count"),
        ("fixed_share_weight_leader_switch_count", "leader_switch_count"),
        (
            "fixed_share_weight_aggregation_recalibration_count",
            "aggregation_recalibration_count",
        ),
        (
            "fixed_share_weight_aggregation_recalibration_replayed_sample_count",
            "aggregation_recalibration_sample_count",
        ),
    ):
        record_tables[f"client_weight_update_counts.{column_name}"] = _integer_column(
            [
                getattr(
                    run_client.owners.fixed_share_weight_controller,
                    controller_property_name,
                )
                for run_client in run_clients
            ]
        )

    # ---- clientの保有モデルごとの、実行の終わりの学習量（clientの順、モデルIDの昇順） ----
    held_model_training_amounts = sorted(
        collect_client_held_model_training_amounts(run_clients=tuple(run_clients)),
        key=lambda training_amount: (training_amount.client_id, training_amount.model_id),
    )
    for column_name in (
        "client_id",
        "model_id",
        "stored_training_sample_count",
        "trained_sample_count",
        "parameter_update_step_count",
    ):
        record_tables[f"client_held_model.{column_name}"] = _integer_column(
            [
                getattr(training_amount, column_name)
                for training_amount in held_model_training_amounts
            ]
        )

    # ---- サーバの同期のラウンドごとの、モデル数（同期の最後の、配布の後の値） ----
    server_round_model_counts = count_models_after_server_round_synchronizations(
        run_clients=tuple(run_clients), run_server=run_server
    )
    record_tables["server_round.global_model_count_after_synchronization"] = _integer_column(
        list(server_round_model_counts.global_model_counts)
    )
    record_tables["server_round.client_held_model_count_after_synchronization"] = asarray(
        server_round_model_counts.client_held_model_counts, dtype=int64
    ).reshape(len(server_round_model_counts.global_model_counts), len(run_clients))

    # ---- サーバの同期のラウンドごとの、モデルの計算（標本の処理と区間末の学習＝ローカルの処理、と、同期の間の別） ----
    record_tables.update(
        _build_server_round_computation_columns(
            phase_name="local_processing",
            round_computation_counts=[
                round_counts.local_processing_model_computation_counts
                for round_counts in measured_run.round_model_computation_counts
            ],
        )
    )
    record_tables.update(
        _build_server_round_computation_columns(
            phase_name="synchronization",
            round_computation_counts=[
                round_counts.synchronization_model_computation_counts
                for round_counts in measured_run.round_model_computation_counts
            ],
        )
    )
    return record_tables
