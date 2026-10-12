"""累積保持件数を重みに単一global modelを集約・配布する。"""

from copy import deepcopy

from torch import Tensor

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.methods.fedavg.fedavg_client import FedavgClient


class FedavgRunServer:
    """単一global parameter、集約統計と通信/集約の記録を所有する。"""

    def __init__(
        self,
        *,
        run_clients: tuple[FedavgClient, ...],
        initial_parameter_snapshot: dict[str, Tensor],
        initial_loss_statistics: ModelAndClassLossStatistics,
    ) -> None:
        if (
            type(run_clients) is not tuple
            or not run_clients
            or any(
                type(c) is not FedavgClient or c.client_id != i for i, c in enumerate(run_clients)
            )
        ):
            raise ValueError("client ID順の非空exact FedAvg client tupleが必要です。")
        initial_loss_statistics.__post_init__()
        # 各clientに一致するpayloadであることを乱数消費なしに検査する。
        for client in run_clients:
            reference = snapshot_classifier_parameters(classifier=client.classifier)
            if reference.keys() != initial_parameter_snapshot.keys() or any(
                not value.equal(initial_parameter_snapshot[key]) for key, value in reference.items()
            ):
                raise ValueError("初期server parameterとclientを一致させてください。")
        self.run_clients = run_clients
        self.global_parameters = deepcopy(initial_parameter_snapshot)
        self.global_loss_statistics = deepcopy(initial_loss_statistics)
        self.communication_volume_record_store = CommunicationVolumeRecordStore()
        self.aggregation_parameter_multiply_accumulate_count = 0
        self.synchronized_communication_volumes: list[CommunicationVolumeSnapshot] = []
        self.synchronized_aggregation_parameter_counts: list[int] = []

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
            raise ValueError("FedAvgに新モデル登録はありません。")
        uploads = [
            (
                len(client.training_samples),
                snapshot_classifier_parameters(classifier=client.classifier),
                client.loss_statistics_store.get_model_loss_statistics(model_id=0),
            )
            for client in self.run_clients
            if client.training_samples
        ]
        for _, snapshot, statistics in uploads:
            if snapshot.keys() != self.global_parameters.keys() or any(
                snapshot[key].shape != value.shape for key, value in self.global_parameters.items()
            ):
                raise ValueError("集約parameterの構造が不一致です。")
            if statistics is None:
                raise ValueError("client損失統計が必要です。")
        total = 0
        result = None
        stat_count = 0
        weighted_mean = 0.0
        for weight, snapshot, statistics in uploads:
            if result is None:
                result = {key: value * weight for key, value in snapshot.items()}
            else:
                for key in result:
                    result[key] = result[key] + snapshot[key] * weight
            total += weight
            assert statistics is not None  # 全uploadの検査で確定した不変統計。
            moments = statistics.overall_loss_moments
            weighted_mean += moments.mean_loss * moments.observed_loss_count
            stat_count += moments.observed_loss_count
            self.communication_volume_record_store.record_model_transfers(
                transfer_direction="upload", model_count=1
            )
            self.communication_volume_record_store.record_parameter_transfer(
                transfer_direction="upload", parameter_snapshot=snapshot
            )
            self.aggregation_parameter_multiply_accumulate_count += sum(
                v.numel() for v in snapshot.values()
            )
        if result is not None and total > 0:
            self.global_parameters = {key: value / total for key, value in result.items()}
        if stat_count > 0:
            self.global_loss_statistics = ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=stat_count,
                    mean_loss=weighted_mean / stat_count,
                    sum_squared_loss_deviations=0.0,
                )
            )
        self.communication_volume_record_store.record_model_transfers(
            transfer_direction="download", model_count=len(self.run_clients)
        )
        self.communication_volume_record_store.record_parameter_transfer(
            transfer_direction="download",
            parameter_snapshot=self.global_parameters,
            transfer_count=len(self.run_clients),
        )
        for client in self.run_clients:
            client.apply_global_parameters(parameter_snapshot=self.global_parameters)
        self.synchronized_communication_volumes.append(
            self.communication_volume_record_store.get_state_snapshot()
        )
        self.synchronized_aggregation_parameter_counts.append(
            self.aggregation_parameter_multiply_accumulate_count
        )

    def finalize_started_communications(self, *, completed_round_count: int) -> None:
        pass
