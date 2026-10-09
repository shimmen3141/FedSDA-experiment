"""サーバの1ラウンドの前半: 送信できる新規モデルの登録と、clientのモデルの集約。"""

from dataclasses import dataclass

from torch import Tensor

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.loss_statistics.server_loss_mean_aggregation import (
    aggregate_participating_client_loss_means,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    confirm_held_model_registration,
)

# 分類器のパラメータ名のうち、共有部（共有特徴抽出部）を表す接頭辞。それ以外は概念固有部。
_SHARED_PARAMETER_NAME_PREFIX = "feature_extractor."


@dataclass(frozen=True, kw_only=True)
class RegisteredClientModel:
    """登録1件の結果。clientの現在の学習帰属が非負のIDへ戻っていた場合、学習帰属は変わらない（None）。"""

    client_id: int
    registered_global_model_id: int
    training_assignment_change: TrainingModelAssignmentChange | None


@dataclass(frozen=True, kw_only=True)
class ClientModelAggregation:
    """集約1回の結果。件数は、対象のモデルIDごとの、参加した学習データの総件数（参加がなければ0）。"""

    aggregated_global_model_ids: tuple[int, ...]
    aggregated_training_sample_counts_by_model_id: dict[int, int]


def _validate_run_clients(*, run_clients: tuple[FedsdaRunClient, ...]) -> None:
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be builtin tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    client_ids = [run_client.client_id for run_client in run_clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("run_clients must not hold duplicate client IDs")


def _validate_global_model_repository(*, global_model_repository: GlobalModelRepository) -> None:
    if type(global_model_repository) is not GlobalModelRepository:
        raise TypeError("global_model_repository must be exact GlobalModelRepository")


def register_ready_client_models(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    round_index: int,
) -> tuple[RegisteredClientModel, ...]:
    """送信できるモデルを持つclientへ、渡された順に、正式IDを採番し、来歴を記録し、clientへ確認させる。

    パラメータは送らない（集約のときに1回だけ送る）ので、通信量もグローバルモデルの値も変えない。
    clientの現在の学習帰属が非負のIDへ戻っていた場合も、採番と来歴の記録は行う（旧の挙動の維持）。
    """
    _validate_run_clients(run_clients=run_clients)
    _validate_global_model_repository(global_model_repository=global_model_repository)
    if type(round_index) is not int:
        raise TypeError("round_index must be builtin int")
    if round_index < 0:
        raise ValueError("round_index must be nonnegative")
    registered_client_models: list[RegisteredClientModel] = []
    for run_client in run_clients:
        if not run_client.has_model_ready_for_server_registration():
            continue
        registered_global_model_id = global_model_repository.allocate_global_model_id()
        global_model_repository.record_model_registration(
            model_id=registered_global_model_id,
            registered_round_index=round_index,
            registering_client_id=run_client.client_id,
        )
        owners = run_client.owners
        # 学習帰属の変更は、診断へ通知しない（旧も、切替の通知を呼ばずに現行モデルを置き換える）。
        training_assignment_change = confirm_held_model_registration(
            registered_global_model_id=registered_global_model_id,
            held_model_training_state_registry=owners.held_model_training_state_registry,
            loss_statistics_store=owners.loss_statistics_store,
            training_sample_store=owners.training_sample_store,
            evaluation_sample_store=owners.model_evaluation_sample_store,
            model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
            current_training_model_assignment=owners.current_training_model_assignment,
            pending_model_upload_state=owners.pending_model_upload_state,
        )
        registered_client_models.append(
            RegisteredClientModel(
                client_id=run_client.client_id,
                registered_global_model_id=registered_global_model_id,
                training_assignment_change=training_assignment_change,
            )
        )
    return tuple(registered_client_models)


def _split_shared_and_concept_specific_parameters(
    *, parameter_snapshot: dict[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """完全なパラメータを、名前の順を保ったまま、共有部と概念固有部へ分ける。"""
    shared_parameters = {
        parameter_name: parameter_values
        for parameter_name, parameter_values in parameter_snapshot.items()
        if parameter_name.startswith(_SHARED_PARAMETER_NAME_PREFIX)
    }
    concept_specific_parameters = {
        parameter_name: parameter_values
        for parameter_name, parameter_values in parameter_snapshot.items()
        if not parameter_name.startswith(_SHARED_PARAMETER_NAME_PREFIX)
    }
    if not shared_parameters or not concept_specific_parameters:
        raise ValueError("parameter snapshot must hold both shared and concept-specific parameters")
    return shared_parameters, concept_specific_parameters


def _add_weighted_parameters(
    *,
    weighted_parameter_sum: dict[str, Tensor] | None,
    parameters: dict[str, Tensor],
    sample_count: int,
) -> dict[str, Tensor]:
    """重み付きの和へ足す。最初は「値×重み」、以後は「和＋値×重み」（旧と同じ式の形）。"""
    if weighted_parameter_sum is None:
        return {
            parameter_name: parameter_values * sample_count
            for parameter_name, parameter_values in parameters.items()
        }
    if weighted_parameter_sum.keys() != parameters.keys():
        raise ValueError("client model parameters must have the same names")
    for parameter_name, parameter_values in parameters.items():
        weighted_parameter_sum[parameter_name] = (
            weighted_parameter_sum[parameter_name] + parameter_values * sample_count
        )
    return weighted_parameter_sum


def _divide_parameters(
    *, weighted_parameter_sum: dict[str, Tensor], total_sample_count: int
) -> dict[str, Tensor]:
    return {
        parameter_name: parameter_values / total_sample_count
        for parameter_name, parameter_values in weighted_parameter_sum.items()
    }


def aggregate_client_models_into_global_models(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
) -> ClientModelAggregation:
    """clientのモデルを、学習データの件数で重み付けて平均し、グローバルモデルと損失統計を置き換える。

    共有部は、clientごとに1回、そのclientの参加モデルの学習データの総件数で重み付ける。概念固有部と
    損失統計の平均は、モデルIDごとに重み付ける。全部の計算の後で、通信量→パラメータ→統計の順に反映する。
    clientの状態は読むだけ。
    """
    _validate_run_clients(run_clients=run_clients)
    _validate_global_model_repository(global_model_repository=global_model_repository)
    if type(communication_volume_record_store) is not CommunicationVolumeRecordStore:
        raise TypeError(
            "communication_volume_record_store must be exact CommunicationVolumeRecordStore"
        )
    # (1) 対象: いずれかのclientが保有する、非負のIDのモデル。
    aggregated_global_model_ids = tuple(
        sorted(
            {
                held_model_training_state.model_id
                for run_client in run_clients
                for held_model_training_state in run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
                if held_model_training_state.model_id >= 0
            }
        )
    )
    # (2) clientごとの参加モデルを、重み付きの和へ足す。
    aggregated_sample_counts_by_model_id = dict.fromkeys(aggregated_global_model_ids, 0)
    concept_specific_parameter_sums_by_model_id: dict[int, dict[str, Tensor] | None] = (
        dict.fromkeys(aggregated_global_model_ids)
    )
    participating_loss_moments_by_model_id: dict[int, list[BoundedLossMoments]] = {
        model_id: [] for model_id in aggregated_global_model_ids
    }
    shared_parameter_sum: dict[str, Tensor] | None = None
    shared_sample_count = 0
    uploaded_model_count = 0
    uploaded_parameter_snapshots: list[dict[str, Tensor]] = []
    for run_client in run_clients:
        owners = run_client.owners
        classifiers_by_model_id = {
            held_model_training_state.model_id: held_model_training_state.classifier
            for held_model_training_state in owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        }
        training_sample_counts_by_model_id = {
            model_training_samples.model_id: len(model_training_samples.training_samples)
            for model_training_samples in owners.training_sample_store.snapshot_ordered_model_training_samples()
        }
        participating_models = [
            (model_id, training_sample_counts_by_model_id.get(model_id, 0))
            for model_id in aggregated_global_model_ids
            if model_id in classifiers_by_model_id
            and training_sample_counts_by_model_id.get(model_id, 0) > 0
        ]
        if not participating_models:
            continue
        uploaded_model_count += len(participating_models)
        client_shared_parameters, _ = _split_shared_and_concept_specific_parameters(
            parameter_snapshot=snapshot_classifier_parameters(
                classifier=classifiers_by_model_id[participating_models[0][0]]
            )
        )
        client_sample_count = sum(sample_count for _, sample_count in participating_models)
        uploaded_parameter_snapshots.append(client_shared_parameters)
        shared_parameter_sum = _add_weighted_parameters(
            weighted_parameter_sum=shared_parameter_sum,
            parameters=client_shared_parameters,
            sample_count=client_sample_count,
        )
        shared_sample_count += client_sample_count
        for model_id, sample_count in participating_models:
            _, client_concept_specific_parameters = _split_shared_and_concept_specific_parameters(
                parameter_snapshot=snapshot_classifier_parameters(
                    classifier=classifiers_by_model_id[model_id]
                )
            )
            uploaded_parameter_snapshots.append(client_concept_specific_parameters)
            concept_specific_parameter_sums_by_model_id[model_id] = _add_weighted_parameters(
                weighted_parameter_sum=concept_specific_parameter_sums_by_model_id[model_id],
                parameters=client_concept_specific_parameters,
                sample_count=sample_count,
            )
            aggregated_sample_counts_by_model_id[model_id] += sample_count
            client_loss_statistics = owners.loss_statistics_store.get_model_loss_statistics(
                model_id=model_id
            )
            if client_loss_statistics is not None:
                participating_loss_moments_by_model_id[model_id].append(
                    client_loss_statistics.overall_loss_moments
                )
    # (3) 共有部の平均。どのclientも参加していなければ、既存の最初のグローバルモデルの共有部。
    existing_global_model_ids = global_model_repository.global_model_ids
    updated_parameter_snapshots_by_model_id: dict[int, dict[str, Tensor]] = {}
    updated_loss_statistics_by_model_id: dict[int, ModelAndClassLossStatistics] = {}
    if shared_parameter_sum is not None and shared_sample_count > 0:
        global_shared_parameters = _divide_parameters(
            weighted_parameter_sum=shared_parameter_sum, total_sample_count=shared_sample_count
        )
    elif existing_global_model_ids:
        global_shared_parameters, _ = _split_shared_and_concept_specific_parameters(
            parameter_snapshot=global_model_repository.get_global_model_parameters(
                model_id=existing_global_model_ids[0]
            )
        )
    else:
        global_shared_parameters = None
    # (4) 対象のモデルごとに、置くパラメータと統計を決める。
    if global_shared_parameters is not None:
        for model_id in aggregated_global_model_ids:
            concept_specific_parameter_sum = concept_specific_parameter_sums_by_model_id[model_id]
            if (
                concept_specific_parameter_sum is not None
                and aggregated_sample_counts_by_model_id[model_id] > 0
            ):
                global_concept_specific_parameters = _divide_parameters(
                    weighted_parameter_sum=concept_specific_parameter_sum,
                    total_sample_count=aggregated_sample_counts_by_model_id[model_id],
                )
            elif model_id in existing_global_model_ids:
                _, global_concept_specific_parameters = (
                    _split_shared_and_concept_specific_parameters(
                        parameter_snapshot=global_model_repository.get_global_model_parameters(
                            model_id=model_id
                        )
                    )
                )
            else:
                continue
            updated_parameter_snapshots_by_model_id[model_id] = {
                **global_shared_parameters,
                **global_concept_specific_parameters,
            }
            aggregated_loss_moments = aggregate_participating_client_loss_means(
                participating_client_loss_moments=tuple(
                    participating_loss_moments_by_model_id[model_id]
                )
            )
            if aggregated_loss_moments is not None:
                updated_loss_statistics_by_model_id[model_id] = ModelAndClassLossStatistics(
                    overall_loss_moments=aggregated_loss_moments
                )
    # (5) 反映: 通信量→パラメータ→統計。
    communication_volume_record_store.record_model_transfers(
        transfer_direction="upload", model_count=uploaded_model_count
    )
    for uploaded_parameter_snapshot in uploaded_parameter_snapshots:
        communication_volume_record_store.record_parameter_transfer(
            transfer_direction="upload", parameter_snapshot=uploaded_parameter_snapshot
        )
    for model_id, parameter_snapshot in updated_parameter_snapshots_by_model_id.items():
        global_model_repository.set_global_model_parameters(
            model_id=model_id, parameter_snapshot=parameter_snapshot
        )
    for model_id, loss_statistics in updated_loss_statistics_by_model_id.items():
        global_model_repository.set_global_model_loss_statistics(
            model_id=model_id, loss_statistics=loss_statistics
        )
    return ClientModelAggregation(
        aggregated_global_model_ids=aggregated_global_model_ids,
        aggregated_training_sample_counts_by_model_id=aggregated_sample_counts_by_model_id,
    )
