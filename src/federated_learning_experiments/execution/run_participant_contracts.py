"""単一runの初期準備・観測処理・同期を接続する契約。"""

from dataclasses import dataclass
from typing import Protocol

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.observed_sample_generation import (
    ObservedSampleGenerator,
)
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.execution.run_random_sources import RunRandomSources


class RunClientOperations(Protocol):
    """clientが所有する状態へ観測処理・区間末・候補終端を依頼する。"""

    client_id: int

    def process_observed_sample(
        self, *, observed_sample: ObservedSample, sample_index: int, evaluation_concept_id: int
    ) -> None:
        """標本を1件処理する。

        `evaluation_concept_id`は、その標本の真の概念ID（評価用の真値）で、診断専用である。
        clientは、予測・検出・学習・モデルの選択に使ってはならない。
        """
        ...

    def flush_pending_local_updates(self, *, round_index: int) -> None: ...

    def has_model_ready_for_server_registration(self) -> bool: ...

    def advance_new_model_upload_wait_after_synchronization(self, *, round_index: int) -> None: ...

    def finalize_incomplete_candidate_validation(self) -> None: ...


class RunServerOperations(Protocol):
    """serverへ同期前記録・同期・開始済み通信の終端を依頼する。"""

    def record_client_states_before_synchronization(self, *, round_index: int) -> None: ...

    def synchronize_models(
        self, *, round_index: int, new_model_registration_available: bool
    ) -> None: ...

    def finalize_started_communications(self, *, completed_round_count: int) -> None: ...


@dataclass(frozen=True, kw_only=True)
class RunParticipants:
    """接続先が所有する可変状態への参照を固定した参加者集合。"""

    client_operations: tuple[RunClientOperations, ...]
    server_operations: RunServerOperations

    def __post_init__(self) -> None:
        if type(self.client_operations) is not tuple:
            raise TypeError("client_operationsにはclient操作のtupleを指定してください。")


class RunParticipantFactory(Protocol):
    """自身の条件検証とrunごとの新しい参加者の初期準備を提供する。"""

    def validate_configuration(self) -> None: ...

    def prepare_run(
        self,
        *,
        experiment_run_conditions: ExperimentRunConditions,
        run_random_sources: RunRandomSources,
        sample_generator: ObservedSampleGenerator,
    ) -> RunParticipants: ...
