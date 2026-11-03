"""FedDriftのrun記録を名前付き配列へ変換する。"""

from dataclasses import fields
from math import nan

from numpy import asarray, bool_, float64, int64, ndarray, str_

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord
from federated_learning_experiments.evaluation.run_metric_calculations import (
    extract_concept_change_sample_indices,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)
from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
    FeddriftDetectionBatchTransition,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import FeddriftExperimentRun
from federated_learning_experiments.runtime.feddrift_measured_run_execution import (
    FeddriftMeasuredRun,
)
from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
    FeddriftRunMetrics,
    derive_feddrift_run_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings

MISSING_INTEGER_VALUE = -1


def _integer_column(values: object) -> ndarray:
    return asarray(values, dtype=int64)


def _boolean_column(values: object) -> ndarray:
    return asarray(values, dtype=bool_)


def _float_column(values: object) -> ndarray:
    return asarray(values, dtype=float64)


def build_feddrift_run_record_tables(
    *, experiment_run: FeddriftExperimentRun
) -> dict[str, ndarray]:
    """状態を変更せず記録表を返す。"""
    if type(experiment_run) is not FeddriftExperimentRun:
        raise TypeError("exact FeddriftExperimentRunが必要です。")
    if type(experiment_run.run_settings) is not FeddriftRunSettings:
        raise TypeError("exact FeddriftRunSettingsが必要です。")
    experiment_run.run_settings.__post_init__()
    measured_run = experiment_run.measured_run
    if (
        type(measured_run) is not FeddriftMeasuredRun
        or type(experiment_run.run_metrics) is not FeddriftRunMetrics
    ):
        raise TypeError("exact計測結果と指標が必要です。")
    run_result = measured_run.run_result
    participants = measured_run.participants
    metrics = derive_feddrift_run_metrics(
        run_result=run_result,
        participants=participants,
        run_metric_settings=experiment_run.run_settings.run_metric_settings,
        model_computation_counts=measured_run.model_computation_counts,
    )
    # 可変の参加者を含む結果なので、古い要約と現在の記録の不一致を保存前に拒否する。
    if convert_settings_to_plain_mapping(
        metrics, non_finite_float_as_none=True
    ) != convert_settings_to_plain_mapping(
        experiment_run.run_metrics, non_finite_float_as_none=True
    ):
        raise ValueError("run指標と現在の参加者記録が対応していません。")
    run_clients = participants.run_clients
    run_server = participants.run_server
    record_tables: dict[str, ndarray] = {}
    if run_server._cfl_state is not None:
        rows = run_server._cfl_records
        if len(rows) != len(run_server._cfl_record_round_indices):
            raise ValueError("CFLの分割判定とround番号の件数が対応していません。")
        for column, values in (
            ("round_index", run_server._cfl_record_round_indices),
            ("timestep_index", [t for t, _ in rows]),
            ("model_id", [r[0] for _, r in rows]),
            ("new_model_id", [r[9] for _, r in rows]),
        ):
            record_tables[f"cfl_split_check.{column}"] = _integer_column(values)
        for column, index in (("mean_norm", 2), ("max_norm", 3), ("alpha_cross", 7)):
            record_tables[f"cfl_split_check.{column}"] = _float_column([r[index] for _, r in rows])
        for suffix, index in (("before", 4), ("after", 5)):
            for offset, name in enumerate(("cfl_norm", "eps1", "eps2")):
                record_tables[f"cfl_split_check.{name}_{suffix}"] = _float_column(
                    [r[index][offset] for _, r in rows]
                )
        for column, index in (("checked_bipartition", 6), ("did_split", 8)):
            record_tables[f"cfl_split_check.{column}"] = _boolean_column(
                [r[index] for _, r in rows]
            )
        for column, index in (("client_ids", 1), ("cl1_client_ids", 10), ("cl2_client_ids", 11)):
            record_tables[f"cfl_split_check.{column}"] = asarray(
                [",".join(map(str, r[index])) for _, r in rows], dtype=str_
            )
        assignments = sorted(
            (t, c.client_id, first, mid)
            for c in run_clients
            for t, first, mid in c.stream_processor._cfl_assignments
        )
        for index, column in enumerate(
            ("timestep_index", "client_id", "first_sample_index", "model_id")
        ):
            record_tables[f"cfl_assignment.{column}"] = _integer_column(
                [r[index] for r in assignments]
            )
    if run_server._driftsurf_state is not None:
        records = run_server._driftsurf_records
        if len(records) != run_result.synchronization_interval_count:
            raise ValueError("DriftSurfの記録と同期件数が対応していません。")
        for column in records[0] if records else ():
            values = [record[column] for record in records]
            dtype = (
                float64
                if column.startswith("accuracy_")
                else bool_
                if column in ("entered_reactive", "promoted")
                else str_
                if column in ("state_before", "state_after", "training_roles", "prediction_role")
                or column.endswith("_window_timesteps")
                else int64
            )
            record_tables[f"driftsurf.{column}"] = asarray(values, dtype=dtype)
    adaptive_state = run_server._adaptive_learning_rate_state
    if adaptive_state is not None:
        records = adaptive_state.snapshot_records()
        if len(records) != run_result.synchronization_interval_count:
            raise ValueError("学習率の記録と同期件数が対応していません。")
        for column in (
            "timestep_index",
            "aggregation_count",
            "variance_ratio",
            "smoothed_ratio",
            "learning_rate_for_next_timestep",
        ):
            values = [getattr(record, column) for record in records]
            record_tables[f"adaptive_learning_rate.{column}"] = (
                _integer_column(values)
                if column in ("timestep_index", "aggregation_count")
                else _float_column(values)
            )
    client_count = len(run_clients)
    processed = run_result.processed_sample_count_per_client
    predictions = [client.stream_processor.sample_predictions for client in run_clients]
    if run_clients[0]._settings.model_assignment_policy == "minimum_loss_cluster_assignment":
        selections = [
            (client.client_id, index, row)
            for client in run_clients
            for index, row in enumerate(client.detection_batch_processor._cluster_selections)
        ]
        for column, values in (
            ("client_id", [c for c, _, _ in selections]),
            ("timestep_index", [t for _, t, _ in selections]),
            ("first_sample_index", [r[0] for _, _, r in selections]),
            ("last_sample_index", [r[1] for _, _, r in selections]),
            ("selected_model_id", [r[2] for _, _, r in selections]),
        ):
            record_tables[f"cluster_selection.{column}"] = _integer_column(values)
        model_ids = run_server.global_model_repository.global_model_ids
        record_tables["cluster_selection.model_ids"] = asarray(
            [model_ids for _ in selections], dtype=int64
        ).reshape(len(selections), len(model_ids))
        record_tables["cluster_selection.model_mean_bounded_losses"] = asarray(
            [[dict(row[3]).get(mid, nan) for mid in model_ids] for _, _, row in selections],
            dtype=float64,
        ).reshape(len(selections), len(model_ids))
    for column in (
        "sample_index",
        "prediction_model_id",
        "predicted_class_id",
        "observed_class_id",
        "observed_concept_id",
        "prediction_is_correct",
    ):
        values = [
            [
                MISSING_INTEGER_VALUE
                if getattr(record, column) is None
                else getattr(record, column)
                for record in rows
            ]
            for rows in predictions
        ]
        record_tables[f"sample_prediction.{column}"] = asarray(
            values, dtype=bool_ if column == "prediction_is_correct" else int64
        ).reshape(client_count, processed)
    if (
        experiment_run.run_settings.run_participant_settings.run_client_settings.model_assignment_policy
        == "true_concept_assignment"
    ):
        record_tables["sample_prediction.training_model_id"] = asarray(
            [[record.training_model_id for record in rows] for rows in predictions], dtype=int64
        ).reshape(client_count, processed)
    record_tables["sample_prediction.held_model_count_at_prediction"] = asarray(
        [
            client.stream_processor.held_model_count_record_store.snapshot_held_model_counts()
            for client in run_clients
        ],
        dtype=int64,
    ).reshape(client_count, processed)
    changes = [
        (trace.client_id, index)
        for trace in run_result.evaluation_concept_traces
        for index in extract_concept_change_sample_indices(
            concept_ids_by_sample_index=trace.concept_ids_by_sample_index
        )
    ]
    switches = [
        (client.client_id, transition.last_sample_index)
        for client in run_clients
        for transition in client.stream_processor.completed_detection_batch_transitions
        if transition.adaptation_kind != "current_model_retained"
    ]
    for table, rows, position_name in (
        ("concept_change", changes, "first_new_concept_sample_index"),
        ("training_model_switch", switches, "switch_sample_index"),
    ):
        record_tables[f"{table}.client_id"] = _integer_column([row[0] for row in rows])
        record_tables[f"{table}.{position_name}"] = _integer_column([row[1] for row in rows])
    transitions = [
        (client.client_id, record)
        for client in run_clients
        for record in client.stream_processor.completed_detection_batch_transitions
    ]
    adaptations = [
        (client.client_id, record)
        for client in run_clients
        for record in client.owners.adaptation_record_store.get_state_snapshot().adaptation_records
    ]
    for table, rows, kind in (
        ("detection_batch", transitions, FeddriftDetectionBatchTransition),
        ("adaptation", adaptations, AdaptationRecord),
    ):
        record_tables[f"{table}.client_id"] = _integer_column([client_id for client_id, _ in rows])
        for field in fields(kind):
            values = [
                MISSING_INTEGER_VALUE
                if getattr(record, field.name) is None
                else getattr(record, field.name)
                for _, record in rows
            ]
            dtype = (
                str_
                if field.name in ("adaptation_kind", "adaptation_outcome", "detector_name")
                else float64
                if field.name == "next_previous_minimum_mean_bounded_loss"
                else int64
            )
            record_tables[f"{table}.{field.name}"] = asarray(values, dtype=dtype)
    # ---- モデルの登録 ----
    global_model_repository = run_server.global_model_repository
    registration_records = global_model_repository.snapshot_model_registration_records()
    final_global_model_ids = set(global_model_repository.global_model_ids)
    record_tables["model_registration.model_id"] = _integer_column(
        [registration_record.model_id for registration_record in registration_records]
    )
    record_tables["model_registration.registered_server_timestep_index"] = _integer_column(
        [
            MISSING_INTEGER_VALUE
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
    # ---- クラスタリングの観測（モデルごと、対ごと） ----
    model_observations = (
        run_server.model_clustering_record_store.snapshot_model_clustering_observations()
    )
    record_tables["model_clustering.server_timestep_index"] = _integer_column(
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
        run_server.model_clustering_record_store.snapshot_pair_clustering_observations()
    )
    record_tables["model_pair_clustering.server_timestep_index"] = _integer_column(
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
        run_server.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    )
    correctness_column_names = (
        "candidate_only_correct_sample_count",
        "target_only_correct_sample_count",
        "both_correct_sample_count",
        "both_wrong_sample_count",
    )
    for column_name, record_field_name in (
        ("server_timestep_index", "round_index"),
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
            "server_timestep_index",
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

    diagnostic = metrics.model_diagnostic_metrics
    for column in (
        "client_id",
        "model_id",
        "stored_training_sample_count",
        "trained_sample_count",
        "parameter_update_step_count",
    ):
        record_tables[f"client_held_model.{column}"] = _integer_column(
            [getattr(record, column) for record in diagnostic.client_held_model_training_amounts]
        )
    counts = diagnostic.synchronized_timestep_model_counts
    record_tables["server_timestep.timestep_index"] = _integer_column(
        [record.timestep_index for record in counts]
    )
    record_tables["server_timestep.global_model_count_after_synchronization"] = _integer_column(
        [record.global_model_count for record in counts]
    )
    record_tables["server_timestep.client_held_model_count_after_synchronization"] = asarray(
        [record.client_held_model_counts for record in counts], dtype=int64
    ).reshape(len(counts), client_count)
    if len(measured_run.timestep_model_computation_counts) != len(counts):
        raise ValueError("時刻の計数とモデル数の件数を一致させてください。")
    for index, record in enumerate(measured_run.timestep_model_computation_counts):
        if type(record.timestep_index) is not int or record.timestep_index != index:
            raise ValueError("時刻の計数は0から昇順が必要です。")
    for phase, attribute in (
        ("sample_processing", "sample_processing_model_computation_counts"),
        ("synchronization", "synchronization_model_computation_counts"),
    ):
        for field in fields(ModelComputationCounts):
            record_tables[f"server_timestep.{phase}_{field.name}"] = _integer_column(
                [
                    getattr(getattr(record, attribute), field.name)
                    for record in measured_run.timestep_model_computation_counts
                ]
            )
    for phase, phase_counts in (
        ("preparation", measured_run.preparation_model_computation_counts),
        ("finalization", measured_run.finalization_model_computation_counts),
    ):
        for field in fields(ModelComputationCounts):
            record_tables[f"{phase}_model_computation.{field.name}"] = _integer_column(
                [getattr(phase_counts, field.name)]
            )
    if run_clients[0]._settings.model_assignment_policy == "fedem_responsibility_assignment":
        for table, columns, attribute in (
            (
                "fedem_client_mixture",
                ("timestep_index", "client_id", "model_id", "mixture_weight"),
                "mixture_records",
            ),
            (
                "fedem_sample_responsibility",
                (
                    "timestep_index",
                    "client_id",
                    "sample_index",
                    "max_responsibility_model_id",
                    "max_responsibility",
                ),
                "responsibility_records",
            ),
        ):
            rows = [
                (row[0], client.client_id, *row[1:])
                for client in run_clients
                if client.stream_processor._fedem is not None
                for row in getattr(client.stream_processor._fedem, attribute)
            ]
            for index, column in enumerate(columns):
                values = [row[index] for row in rows]
                record_tables[f"{table}.{column}"] = (
                    _float_column(values) if index == len(columns) - 1 else _integer_column(values)
                )
    return record_tables
