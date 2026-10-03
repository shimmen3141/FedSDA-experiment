"""データ供給・処理範囲・実行順を表す不変の成功記録。"""

from dataclasses import dataclass

from federated_learning_experiments.data.observed_streams import ClientConceptTrace, ClientObservedStream


RUN_EXECUTION_STAGE_NAMES = (
    "configuration_validation", "initial_preparation", "participant_validation",
    "concept_trace_generation", "observed_stream_generation", "sample_processing",
    "pending_update_flush", "pre_sync_recording", "registration_readiness_check",
    "server_synchronization", "upload_wait_advance",
    "incomplete_candidate_validation_finalization", "started_communication_finalization",
)


def validate_run_execution_stage_and_positions(
    *, stage_name: str, client_id: int | None, sample_index: int | None, round_index: int | None,
) -> None:
    """成功イベントと実行例外で正式stage・位置の表現を統一する。"""
    if type(stage_name) is not str:
        raise TypeError("stage_nameには正式な実行段階の文字列を指定してください。")
    if stage_name not in RUN_EXECUTION_STAGE_NAMES:
        raise ValueError(f"stage_nameには正式な実行段階を指定してください: {stage_name!r}")
    for position_parameter_name, position_parameter_value in (
        ("client_id", client_id), ("sample_index", sample_index), ("round_index", round_index),
    ):
        if position_parameter_value is None:
            continue
        if type(position_parameter_value) is not int:
            raise TypeError(f"{position_parameter_name}には非負整数またはNoneを指定してください。")
        if position_parameter_value < 0:
            raise ValueError(f"{position_parameter_name}には0以上の整数を指定してください。")


@dataclass(frozen=True, kw_only=True)
class RunExecutionEvent:
    """成功した段階と、対象がある場合の0始まりの位置。"""

    stage_name: str
    client_id: int | None = None
    sample_index: int | None = None
    round_index: int | None = None

    def __post_init__(self) -> None:
        validate_run_execution_stage_and_positions(
            stage_name=self.stage_name, client_id=self.client_id,
            sample_index=self.sample_index, round_index=self.round_index,
        )


@dataclass(frozen=True, kw_only=True)
class StreamProtocolRunResult:
    """観測・評価情報・件数・順序を保持し、参加者や研究指標を持たない。"""

    observed_client_streams: tuple[ClientObservedStream, ...]
    evaluation_concept_traces: tuple[ClientConceptTrace, ...]
    generated_sample_count_per_client: int
    processed_sample_count_per_client: int
    unprocessed_tail_sample_count_per_client: int
    synchronization_interval_count: int
    execution_events: tuple[RunExecutionEvent, ...]

    def __post_init__(self) -> None:
        if type(self.observed_client_streams) is not tuple:
            raise TypeError("observed_client_streamsにはClientObservedStreamのtupleを指定してください。")
        for observed_client_stream in self.observed_client_streams:
            if type(observed_client_stream) is not ClientObservedStream:
                raise TypeError("observed_client_streamsの各要素にはClientObservedStreamを指定してください。")
        if type(self.evaluation_concept_traces) is not tuple:
            raise TypeError("evaluation_concept_tracesにはClientConceptTraceのtupleを指定してください。")
        for evaluation_concept_trace in self.evaluation_concept_traces:
            if type(evaluation_concept_trace) is not ClientConceptTrace:
                raise TypeError("evaluation_concept_tracesの各要素にはClientConceptTraceを指定してください。")
        if type(self.execution_events) is not tuple:
            raise TypeError("execution_eventsにはRunExecutionEventのtupleを指定してください。")
        for execution_event in self.execution_events:
            if type(execution_event) is not RunExecutionEvent:
                raise TypeError("execution_eventsの各要素にはRunExecutionEventを指定してください。")
        for sample_count_parameter_name, sample_count_parameter_value in (
            ("generated_sample_count_per_client", self.generated_sample_count_per_client),
            ("processed_sample_count_per_client", self.processed_sample_count_per_client),
            ("unprocessed_tail_sample_count_per_client", self.unprocessed_tail_sample_count_per_client),
            ("synchronization_interval_count", self.synchronization_interval_count),
        ):
            if type(sample_count_parameter_value) is not int:
                raise TypeError(f"{sample_count_parameter_name}には非負整数を指定してください。")
            if sample_count_parameter_value < 0:
                raise ValueError(f"{sample_count_parameter_name}には0以上の整数を指定してください。")
