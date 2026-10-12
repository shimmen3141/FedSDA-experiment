"""FedAvgの全体runを、モデルの計算の計測つきで実行する。"""

from dataclasses import dataclass
from typing import cast

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.observed_sample_generation import (
    ObservedSampleGenerator,
)
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.evaluation.computation_cost_summary import (
    ComputationCostSummary,
    ServerComputationCounts,
    summarize_computation_cost,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    DetectionMetrics,
    calculate_detection_metrics,
    calculate_prediction_accuracy,
    calculate_stable_period_prediction_accuracy,
    extract_concept_change_sample_indices,
)
from federated_learning_experiments.execution.run_execution_records import StreamProtocolRunResult
from federated_learning_experiments.execution.run_participant_contracts import (
    RunParticipants,
    RunServerOperations,
)
from federated_learning_experiments.execution.run_random_sources import RunRandomSources
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
    ModelComputationMeter,
    measure_model_computation,
    subtract_model_computation_counts,
)
from federated_learning_experiments.methods.fedavg.fedavg_client import (
    FedavgClient,
    FedavgSamplePrediction,
)
from federated_learning_experiments.runtime.fedavg_run_participant_factory import (
    FedavgRunParticipantFactory,
)
from federated_learning_experiments.runtime.fedavg_run_server import FedavgRunServer
from federated_learning_experiments.runtime.fedavg_run_settings import (
    FedavgRunParticipantSettings,
    FedavgRunSettings,
)
from federated_learning_experiments.runtime.single_run_execution import (
    execute_stream_protocol_run,
)


@dataclass(frozen=True, kw_only=True)
class FedavgRoundComputationCounts:
    """ラウンド1つの間の、モデルの計算。

    `local_processing`は、標本の処理と、区間末の学習。`synchronization`は、サーバの同期の間に、
    clientが行うモデル演算。FedAvgの同期にはモデルforwardがない。
    """

    round_index: int
    local_processing_model_computation_counts: ModelComputationCounts
    synchronization_model_computation_counts: ModelComputationCounts


@dataclass(frozen=True, kw_only=True)
class _FedavgMeasuredRun:
    """計測つきの全体runの結果。

    `model_computation_counts`は、参加者の初期準備が終わった時点から、全体runの終わりまでの、
    モデルの計算。`preparation_model_computation_counts`は、初期準備の間（事前学習ほか）の計算で、
    指標には含めない。`round_model_computation_counts`（ラウンドの順）と、
    `finalization_model_computation_counts`（最後の同期の後の、終端の処理）の合計が、
    `model_computation_counts`と一致する。
    """

    run_result: StreamProtocolRunResult
    participants: RunParticipants
    model_computation_counts: ModelComputationCounts
    preparation_model_computation_counts: ModelComputationCounts
    round_model_computation_counts: tuple[FedavgRoundComputationCounts, ...]
    finalization_model_computation_counts: ModelComputationCounts


class _RoundMeasuringServerOperations:
    """サーバの操作を包み、ラウンドの中の2つの時点の計数を控える。操作は、そのまま渡す。

    実行の枠は、ラウンドごとに、標本の処理→区間末の学習→同期の前の状態の報告→登録可能の照会→
    同期→送信待ちのno-op、の順に呼ぶ。状態の報告の入口が「ローカルの処理の終わり」、同期の出口が
    「同期の終わり」になる（照会と、送信待ちの進行は、モデルの計算を行わない）。
    """

    def __init__(
        self,
        *,
        server_operations: RunServerOperations,
        model_computation_meter: ModelComputationMeter,
    ) -> None:
        self._server_operations = server_operations
        self._model_computation_meter = model_computation_meter
        self.counts_at_local_processing_end: list[ModelComputationCounts] = []
        self.counts_at_synchronization_end: list[ModelComputationCounts] = []

    def record_client_states_before_synchronization(self, *, round_index: int) -> None:
        self.counts_at_local_processing_end.append(
            self._model_computation_meter.get_model_computation_counts()
        )
        self._server_operations.record_client_states_before_synchronization(round_index=round_index)

    def synchronize_models(
        self, *, round_index: int, new_model_registration_available: bool
    ) -> None:
        self._server_operations.synchronize_models(
            round_index=round_index,
            new_model_registration_available=new_model_registration_available,
        )
        self.counts_at_synchronization_end.append(
            self._model_computation_meter.get_model_computation_counts()
        )

    def finalize_started_communications(self, *, completed_round_count: int) -> None:
        self._server_operations.finalize_started_communications(
            completed_round_count=completed_round_count
        )


class _PreparationMeasuringParticipantFactory:
    """参加者のfactoryを包み、初期準備が終わった時点の計数を控え、サーバの操作を、計数を控える包みにする。"""

    def __init__(
        self,
        *,
        participant_factory: FedavgRunParticipantFactory,
        model_computation_meter: ModelComputationMeter,
    ) -> None:
        self._participant_factory = participant_factory
        self._model_computation_meter = model_computation_meter
        self.preparation_model_computation_counts: ModelComputationCounts | None = None
        self.round_measuring_server_operations: _RoundMeasuringServerOperations | None = None

    def validate_configuration(self) -> None:
        self._participant_factory.validate_configuration()

    def prepare_run(
        self,
        *,
        experiment_run_conditions: ExperimentRunConditions,
        run_random_sources: RunRandomSources,
        sample_generator: ObservedSampleGenerator,
    ) -> RunParticipants:
        participants = self._participant_factory.prepare_run(
            experiment_run_conditions=experiment_run_conditions,
            run_random_sources=run_random_sources,
            sample_generator=sample_generator,
        )
        self.preparation_model_computation_counts = (
            self._model_computation_meter.get_model_computation_counts()
        )
        self.round_measuring_server_operations = _RoundMeasuringServerOperations(
            server_operations=participants.server_operations,
            model_computation_meter=self._model_computation_meter,
        )
        # 実行の枠へは、サーバの操作だけを包んだ参加者を渡す（clientの操作は、そのまま）。
        return RunParticipants(
            client_operations=participants.client_operations,
            server_operations=self.round_measuring_server_operations,
        )


def _execute_measured_run(
    *,
    execution_settings: StreamProtocolExecutionSettings,
    run_participant_settings: FedavgRunParticipantSettings,
) -> _FedavgMeasuredRun:
    """単一モデルのFedAvgの全体runを実行し、結果・参加者・モデルの計算の計数を返す。

    計測は、この呼出しの間だけ行う。実行の枠の例外は、計測の登録を外してから、そのまま伝わる。
    """
    participant_factory = FedavgRunParticipantFactory(
        run_participant_settings=run_participant_settings
    )
    with measure_model_computation() as model_computation_meter:
        measuring_participant_factory = _PreparationMeasuringParticipantFactory(
            participant_factory=participant_factory,
            model_computation_meter=model_computation_meter,
        )
        run_result = execute_stream_protocol_run(
            execution_settings=execution_settings,
            participant_factory=measuring_participant_factory,
        )
    participants = participant_factory.prepared_run_participants
    preparation_model_computation_counts = (
        measuring_participant_factory.preparation_model_computation_counts
    )
    round_measuring_server_operations = (
        measuring_participant_factory.round_measuring_server_operations
    )
    if (
        participants is None
        or preparation_model_computation_counts is None
        or round_measuring_server_operations is None
    ):
        raise RuntimeError("the run finished without preparing its participants")
    final_model_computation_counts = model_computation_meter.get_model_computation_counts()
    round_model_computation_counts: list[FedavgRoundComputationCounts] = []
    previous_counts = preparation_model_computation_counts
    for round_index, (counts_at_local_processing_end, counts_at_synchronization_end) in enumerate(
        zip(
            round_measuring_server_operations.counts_at_local_processing_end,
            round_measuring_server_operations.counts_at_synchronization_end,
            strict=True,
        )
    ):
        round_model_computation_counts.append(
            FedavgRoundComputationCounts(
                round_index=round_index,
                local_processing_model_computation_counts=subtract_model_computation_counts(
                    later_counts=counts_at_local_processing_end, earlier_counts=previous_counts
                ),
                synchronization_model_computation_counts=subtract_model_computation_counts(
                    later_counts=counts_at_synchronization_end,
                    earlier_counts=counts_at_local_processing_end,
                ),
            )
        )
        previous_counts = counts_at_synchronization_end
    return _FedavgMeasuredRun(
        run_result=run_result,
        participants=participants,
        model_computation_counts=subtract_model_computation_counts(
            later_counts=final_model_computation_counts,
            earlier_counts=preparation_model_computation_counts,
        ),
        preparation_model_computation_counts=preparation_model_computation_counts,
        round_model_computation_counts=tuple(round_model_computation_counts),
        finalization_model_computation_counts=subtract_model_computation_counts(
            later_counts=final_model_computation_counts, earlier_counts=previous_counts
        ),
    )


@dataclass(frozen=True, kw_only=True)
class FedavgRunMetrics:
    """他手法と同一定義の基本指標と演算・通信量。"""

    prediction_accuracy: float
    stable_period_prediction_accuracy: float
    training_model_switch_detection_metrics: DetectionMetrics
    final_global_model_count: int
    communication_volume: CommunicationVolumeSnapshot
    final_parameter_value_count: int
    final_parameter_byte_count: int
    model_computation_counts: ModelComputationCounts
    server_computation_counts: ServerComputationCounts
    processed_sample_count: int
    held_model_sample_count: int
    client_consolidation_parameter_multiply_accumulate_count: int
    computation_cost_summary: ComputationCostSummary


@dataclass(frozen=True, kw_only=True, eq=False)
class FedavgExperimentRun:
    """完全条件、計測された参加者/進行と導出した指標。"""

    run_settings: FedavgRunSettings
    measured_run: _FedavgMeasuredRun
    run_metrics: FedavgRunMetrics


def derive_fedavg_run_metrics(
    *,
    run_settings: FedavgRunSettings,
    run_result: StreamProtocolRunResult,
    participants: RunParticipants,
    model_computation_counts: ModelComputationCounts,
) -> FedavgRunMetrics:
    """対応する位置と件数を検査してから共通定義へ接続する。"""
    for value, expected in (
        (run_settings, FedavgRunSettings),
        (run_result, StreamProtocolRunResult),
        (participants, RunParticipants),
        (model_computation_counts, ModelComputationCounts),
    ):
        if type(value) is not expected:
            raise TypeError(f"exact {expected.__name__}が必要です。")
    run_settings.__post_init__()
    run_result.__post_init__()
    participants.__post_init__()
    conditions = run_settings.execution_settings.experiment_run_conditions
    generated = conditions.per_client_sample_count
    chunk = conditions.server_aggregation_interval_per_client_samples
    processed = generated // chunk * chunk
    if (
        run_result.generated_sample_count_per_client,
        run_result.processed_sample_count_per_client,
        run_result.unprocessed_tail_sample_count_per_client,
        run_result.synchronization_interval_count,
    ) != (generated, processed, generated % chunk, generated // chunk):
        raise ValueError("run生成/処理/末尾/同期件数が条件と不一致です。")
    if (
        type(participants.server_operations) is not FedavgRunServer
        or len(participants.client_operations) != conditions.client_count
    ):
        raise ValueError("FedAvg参加者と条件を一致させてください。")
    server = cast(FedavgRunServer, participants.server_operations)
    clients = cast(tuple[FedavgClient, ...], participants.client_operations)
    if (
        server.run_clients is not clients
        or len(run_result.observed_client_streams) != len(clients)
        or len(run_result.evaluation_concept_traces) != len(clients)
    ):
        raise ValueError("server/client/観測/概念の対応が不正です。")
    if (
        len(server.synchronized_communication_volumes) != generated // chunk
        or len(server.synchronized_aggregation_parameter_counts) != generated // chunk
    ):
        raise ValueError("同期記録の件数が不一致です。")
    correctness = []
    changes = []
    for i, (client, stream, trace) in enumerate(
        zip(
            clients,
            run_result.observed_client_streams,
            run_result.evaluation_concept_traces,
            strict=True,
        )
    ):
        if (
            type(client) is not FedavgClient
            or type(client.client_id) is not int
            or client.client_id != i
        ):
            raise ValueError("FedAvg client順が不正です。")
        stream.__post_init__()
        trace.__post_init__()
        if (
            stream.client_id != i
            or trace.client_id != i
            or len(stream.observed_samples) != generated
            or len(trace.concept_ids_by_sample_index) != generated
            or len(client.sample_predictions) != processed
            or len(client.training_samples) != processed
        ):
            raise ValueError("生成/予測/保持標本の対応が不正です。")
        for j, prediction in enumerate(client.sample_predictions):
            if (
                type(prediction) is not FedavgSamplePrediction
                or type(prediction.sample_index) is not int
                or prediction.sample_index != j
                or type(prediction.prediction_model_id) is not int
                or prediction.prediction_model_id != 0
                or type(prediction.prediction_is_correct) is not bool
            ):
                raise ValueError("予測記録の位置/ID/正誤が不正です。")
            if (
                prediction.observed_class_id != stream.observed_samples[j].class_label
                or prediction.observed_concept_id != trace.concept_ids_by_sample_index[j]
                or prediction.prediction_is_correct
                != (prediction.predicted_class_id == prediction.observed_class_id)
            ):
                raise ValueError("予測記録と観測の対応が不正です。")
        correctness.append(tuple(p.prediction_is_correct for p in client.sample_predictions))
        changes.append(
            extract_concept_change_sample_indices(
                concept_ids_by_sample_index=trace.concept_ids_by_sample_index
            )
        )
    count = processed * len(clients)
    server_counts = ServerComputationCounts(
        aggregation_parameter_multiply_accumulate_count=server.aggregation_parameter_multiply_accumulate_count,
        consolidation_parameter_multiply_accumulate_count=0,
        diagnostic_parameter_distance_multiply_accumulate_count=0,
    )
    return FedavgRunMetrics(
        prediction_accuracy=calculate_prediction_accuracy(
            prediction_correctness_by_client=tuple(correctness)
        ),
        stable_period_prediction_accuracy=calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=tuple(correctness),
            concept_change_sample_indices_by_client=tuple(changes),
            recovery_window_sample_count=run_settings.run_metric_settings.post_change_recovery_window_sample_count,
        ),
        training_model_switch_detection_metrics=calculate_detection_metrics(
            concept_change_sample_indices_by_client=tuple(changes),
            detection_sample_indices_by_client=tuple(() for _ in clients),
            maximum_delay_sample_count=run_settings.run_metric_settings.maximum_detection_delay_sample_count,
        ),
        final_global_model_count=1,
        communication_volume=server.communication_volume_record_store.get_state_snapshot(),
        final_parameter_value_count=sum(v.numel() for v in server.global_parameters.values()),
        final_parameter_byte_count=sum(
            v.numel() * v.element_size() for v in server.global_parameters.values()
        ),
        model_computation_counts=model_computation_counts,
        server_computation_counts=server_counts,
        processed_sample_count=count,
        held_model_sample_count=count,
        client_consolidation_parameter_multiply_accumulate_count=0,
        computation_cost_summary=summarize_computation_cost(
            model_computation_counts=model_computation_counts,
            server_computation_counts=server_counts,
            processed_sample_count=count,
            held_model_sample_count=count,
        ),
    )


def execute_fedavg_experiment_run(*, run_settings: FedavgRunSettings) -> FedavgExperimentRun:
    """完全条件を検査して計測・実行・共通評価を行う。"""
    if type(run_settings) is not FedavgRunSettings:
        raise TypeError("exact FedavgRunSettingsが必要です。")
    run_settings.__post_init__()
    measured = _execute_measured_run(
        execution_settings=run_settings.execution_settings,
        run_participant_settings=run_settings.run_participant_settings,
    )
    metrics = derive_fedavg_run_metrics(
        run_settings=run_settings,
        run_result=measured.run_result,
        participants=measured.participants,
        model_computation_counts=measured.model_computation_counts,
    )
    return FedavgExperimentRun(
        run_settings=run_settings, measured_run=measured, run_metrics=metrics
    )
