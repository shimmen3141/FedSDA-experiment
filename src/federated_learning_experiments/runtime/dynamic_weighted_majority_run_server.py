"""パラメータを集約・配布しないDWMのサーバ。各clientは独立にDWMを動かす。"""

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_client import (
    DynamicWeightedMajorityClient,
)


class DynamicWeightedMajorityRunServer:
    """実行の枠の3つの操作を持ち、どれもパラメータを動かさない。同期ごとに通信量0のsnapshotを記録する。"""

    def __init__(self, *, run_clients: tuple[DynamicWeightedMajorityClient, ...]) -> None:
        if (
            type(run_clients) is not tuple
            or not run_clients
            or any(
                type(c) is not DynamicWeightedMajorityClient or c.client_id != i
                for i, c in enumerate(run_clients)
            )
        ):
            raise ValueError("client ID順の非空exact DWM client tupleが必要です。")
        self.run_clients = run_clients
        self.communication_volume_record_store = CommunicationVolumeRecordStore()
        self.aggregation_parameter_multiply_accumulate_count = 0
        self.synchronized_communication_volumes: list[CommunicationVolumeSnapshot] = []
        self.synchronized_aggregation_parameter_counts: list[int] = []
        self.synchronized_client_expert_counts: list[tuple[int, ...]] = []

    def record_client_states_before_synchronization(self, *, round_index: int) -> None:
        pass

    def synchronize_models(
        self, *, round_index: int, new_model_registration_available: bool
    ) -> None:
        if type(round_index) is not int or round_index != len(
            self.synchronized_communication_volumes
        ):
            raise ValueError("同期は連続するround位置が必要です。")
        if type(new_model_registration_available) is not bool or new_model_registration_available:
            raise ValueError("DWMに新モデル登録はありません。")
        self.synchronized_communication_volumes.append(
            self.communication_volume_record_store.get_state_snapshot()
        )
        self.synchronized_aggregation_parameter_counts.append(
            self.aggregation_parameter_multiply_accumulate_count
        )
        self.synchronized_client_expert_counts.append(
            tuple(len(client.experts) for client in self.run_clients)
        )

    def finalize_started_communications(self, *, completed_round_count: int) -> None:
        pass
