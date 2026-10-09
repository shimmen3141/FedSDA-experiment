"""サーバの1ラウンドの後半: 全グローバルモデルと損失統計を、全clientへ配布する。"""

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.global_model_distribution_application import (
    GlobalModelDistributionApplication,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    split_shared_and_concept_specific_parameters,
)


def _validate_distribution_inputs(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
    model_id_mapping: dict[int, int],
) -> None:
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be builtin tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    client_ids = [run_client.client_id for run_client in run_clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("run_clients must not hold duplicate client IDs")
    if type(global_model_repository) is not GlobalModelRepository:
        raise TypeError("global_model_repository must be exact GlobalModelRepository")
    if type(communication_volume_record_store) is not CommunicationVolumeRecordStore:
        raise TypeError(
            "communication_volume_record_store must be exact CommunicationVolumeRecordStore"
        )
    if type(model_id_mapping) is not dict:
        raise TypeError("model_id_mapping must be builtin dict")
    for original_model_id, mapped_model_id in model_id_mapping.items():
        if type(original_model_id) is not int or type(mapped_model_id) is not int:
            raise TypeError("model_id_mapping keys and values must be builtin int")


def distribute_global_models_to_clients(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
    model_id_mapping: dict[int, int],
) -> tuple[GlobalModelDistributionApplication, ...]:
    """下りの通信量を記録してから、全clientへ、渡された順に、ID対応と全グローバルモデルを受け取らせる。

    通信量は、共有部1個（最初のグローバルモデルのもの）と、全モデルの概念固有部を、clientの数だけ数える。
    グローバルモデルの状態は変えない。あるclientの受取りが失敗したら、後のclientへ進まない
    （通信量と、済んだclientの受取りは残る）。
    """
    _validate_distribution_inputs(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
        model_id_mapping=model_id_mapping,
    )
    distributed_parameter_snapshots = tuple(
        (model_id, global_model_repository.get_global_model_parameters(model_id=model_id))
        for model_id in global_model_repository.global_model_ids
    )
    distributed_loss_statistics = global_model_repository.snapshot_global_model_loss_statistics()
    # 通信量を記録する前に、全モデルのパラメータを共有部と概念固有部へ分けておく（分けられなければ何も変えない）。
    split_parameter_snapshots = tuple(
        split_shared_and_concept_specific_parameters(parameter_snapshot=parameter_snapshot)
        for _, parameter_snapshot in distributed_parameter_snapshots
    )
    client_count = len(run_clients)
    if split_parameter_snapshots:
        communication_volume_record_store.record_parameter_transfer(
            transfer_direction="download",
            parameter_snapshot=split_parameter_snapshots[0][0],
            transfer_count=client_count,
        )
        for _, concept_specific_parameters in split_parameter_snapshots:
            communication_volume_record_store.record_parameter_transfer(
                transfer_direction="download",
                parameter_snapshot=concept_specific_parameters,
                transfer_count=client_count,
            )
    communication_volume_record_store.record_model_transfers(
        transfer_direction="download",
        model_count=len(distributed_parameter_snapshots) * client_count,
    )
    if model_id_mapping:
        communication_volume_record_store.record_messages(
            transfer_direction="download", message_count=client_count
        )
    return tuple(
        run_client.apply_global_model_distribution(
            model_id_mapping=model_id_mapping,
            distributed_parameter_snapshots=distributed_parameter_snapshots,
            distributed_loss_statistics=distributed_loss_statistics,
        )
        for run_client in run_clients
    )
