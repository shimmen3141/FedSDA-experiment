"""FedSDAの全体runを、モデルの計算の計測つきで実行する。"""

from dataclasses import dataclass

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.execution.run_execution_records import StreamProtocolRunResult
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
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
class FedsdaMeasuredRun:
    """計測つきの全体runの結果。

    `model_computation_counts`は、参加者の初期準備が終わった時点から、全体runの終わりまでの、
    モデルの計算。`preparation_model_computation_counts`は、初期準備の間（事前学習ほか）の計算で、
    指標には含めない。
    """

    run_result: StreamProtocolRunResult
    participants: RunParticipants
    model_computation_counts: ModelComputationCounts
    preparation_model_computation_counts: ModelComputationCounts


class _PreparationMeasuringParticipantFactory:
    """参加者のfactoryを包み、初期準備が終わった時点の計数を控える。"""

    def __init__(
        self,
        *,
        participant_factory: FedsdaRunParticipantFactory,
        model_computation_meter: ModelComputationMeter,
    ) -> None:
        self._participant_factory = participant_factory
        self._model_computation_meter = model_computation_meter
        self.preparation_model_computation_counts: ModelComputationCounts | None = None

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
        return participants


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
    if participants is None or preparation_model_computation_counts is None:
        raise RuntimeError("the run finished without preparing its participants")
    return FedsdaMeasuredRun(
        run_result=run_result,
        participants=participants,
        model_computation_counts=subtract_model_computation_counts(
            later_counts=model_computation_meter.get_model_computation_counts(),
            earlier_counts=preparation_model_computation_counts,
        ),
        preparation_model_computation_counts=preparation_model_computation_counts,
    )
