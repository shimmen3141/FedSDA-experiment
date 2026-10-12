"""FedDrift全体runのモデル計算を計測する。"""

from dataclasses import dataclass

from federated_learning_experiments.execution.run_execution_records import StreamProtocolRunResult
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
    measure_model_computation,
    subtract_model_computation_counts,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
    FeddriftRunParticipants,
    FeddriftRunParticipantSettings,
)
from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
    FeddriftStreamProtocolExecutionSettings,
    _execute_feddrift_stream_protocol_run,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftTimestepModelComputationCounts:
    """1時刻の標本処理と同期。同期にはR回のclient学習を含む。"""

    timestep_index: int
    sample_processing_model_computation_counts: ModelComputationCounts
    synchronization_model_computation_counts: ModelComputationCounts


@dataclass(frozen=True, kw_only=True)
class FeddriftMeasuredRun:
    """実runとモデル計数。初期準備は本実行の合計へ含めない。"""

    run_result: StreamProtocolRunResult
    participants: FeddriftRunParticipants
    model_computation_counts: ModelComputationCounts
    preparation_model_computation_counts: ModelComputationCounts
    timestep_model_computation_counts: tuple[FeddriftTimestepModelComputationCounts, ...]
    finalization_model_computation_counts: ModelComputationCounts


def execute_feddrift_stream_protocol_run_with_computation_measurement(
    *,
    execution_settings: FeddriftStreamProtocolExecutionSettings,
    run_participant_settings: FeddriftRunParticipantSettings,
) -> FeddriftMeasuredRun:
    """共通hookを呼出しの間だけ登録し、準備を分けて全体runを計測する。"""
    factory = FeddriftRunParticipantFactory(run_participant_settings=run_participant_settings)
    checkpoints: dict[str, list[ModelComputationCounts]] = {}
    with measure_model_computation() as meter:

        def record_checkpoint(stage: str) -> None:
            checkpoints.setdefault(stage, []).append(meter.get_model_computation_counts())

        result = _execute_feddrift_stream_protocol_run(
            execution_settings=execution_settings,
            participant_factory=factory,
            computation_checkpoint=record_checkpoint,
        )
    participants = factory.prepared_run_participants
    if participants is None:
        raise RuntimeError("run成功時には準備済み参加者が必要です。")
    preparation = checkpoints["preparation"][0]
    final = checkpoints["finalization"][0]
    total = subtract_model_computation_counts(later_counts=final, earlier_counts=preparation)
    previous = preparation
    timesteps: list[FeddriftTimestepModelComputationCounts] = []
    for index, (local, sync) in enumerate(
        zip(
            checkpoints.get("sample_processing", ()),
            checkpoints.get("synchronization", ()),
            strict=True,
        )
    ):
        timesteps.append(
            FeddriftTimestepModelComputationCounts(
                timestep_index=index,
                sample_processing_model_computation_counts=subtract_model_computation_counts(
                    later_counts=local, earlier_counts=previous
                ),
                synchronization_model_computation_counts=subtract_model_computation_counts(
                    later_counts=sync, earlier_counts=local
                ),
            )
        )
        previous = sync
    return FeddriftMeasuredRun(
        run_result=result,
        participants=participants,
        model_computation_counts=total,
        preparation_model_computation_counts=preparation,
        timestep_model_computation_counts=tuple(timesteps),
        finalization_model_computation_counts=subtract_model_computation_counts(
            later_counts=final, earlier_counts=previous
        ),
    )
