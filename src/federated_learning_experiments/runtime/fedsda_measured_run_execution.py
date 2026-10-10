"""FedSDAの全体runを、モデルの計算の計測つきで実行する。"""

from dataclasses import dataclass

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
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
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantFactory,
    FedsdaRunParticipantSettings,
)
from federated_learning_experiments.runtime.single_run_execution import (
    execute_stream_protocol_run,
)


@dataclass(frozen=True, kw_only=True)
class RoundModelComputationCounts:
    """ラウンド1つの間の、モデルの計算。

    `local_processing`は、標本の処理と、区間末の学習。`synchronization`は、サーバの同期の間に、
    clientが行う評価（クロス評価）と再較正。
    """

    round_index: int
    local_processing_model_computation_counts: ModelComputationCounts
    synchronization_model_computation_counts: ModelComputationCounts


@dataclass(frozen=True, kw_only=True)
class FedsdaMeasuredRun:
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
    round_model_computation_counts: tuple[RoundModelComputationCounts, ...]
    finalization_model_computation_counts: ModelComputationCounts


class _RoundMeasuringServerOperations:
    """サーバの操作を包み、ラウンドの中の2つの時点の計数を控える。操作は、そのまま渡す。

    実行の枠は、ラウンドごとに、標本の処理→区間末の学習→同期の前の状態の報告→登録可能の照会→
    同期→送信待ちの進行、の順に呼ぶ。状態の報告の入口が「ローカルの処理の終わり」、同期の出口が
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
        participant_factory: FedsdaRunParticipantFactory,
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
        sample_generator: SineSampleGenerator,
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


def execute_fedsda_stream_protocol_run_with_computation_measurement(
    *,
    execution_settings: StreamProtocolExecutionSettings,
    run_participant_settings: FedsdaRunParticipantSettings,
) -> FedsdaMeasuredRun:
    """最終構成のFedSDAの全体runを実行し、結果・参加者・モデルの計算の計数を返す。

    計測は、この呼出しの間だけ行う。実行の枠の例外は、計測の登録を外してから、そのまま伝わる。
    """
    participant_factory = FedsdaRunParticipantFactory(
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
    round_model_computation_counts: list[RoundModelComputationCounts] = []
    previous_counts = preparation_model_computation_counts
    for round_index, (counts_at_local_processing_end, counts_at_synchronization_end) in enumerate(
        zip(
            round_measuring_server_operations.counts_at_local_processing_end,
            round_measuring_server_operations.counts_at_synchronization_end,
            strict=True,
        )
    ):
        round_model_computation_counts.append(
            RoundModelComputationCounts(
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
    return FedsdaMeasuredRun(
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
