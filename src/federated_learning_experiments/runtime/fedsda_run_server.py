"""最終構成のFedSDAのサーバ1つを組み立て、単一runの実行の枠が求めるサーバの操作を提供する。"""

from dataclasses import dataclass
from random import Random

from torch import Tensor

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.server_round_synchronization import (
    ServerRoundSynchronization,
    synchronize_models_in_server_round,
)


@dataclass(frozen=True, kw_only=True)
class FedsdaRunServerOwners:
    """サーバが持つ全ownerへの参照。状態はowner自身が持つ。"""

    global_model_repository: GlobalModelRepository
    communication_volume_record_store: CommunicationVolumeRecordStore
    cross_evaluation_record_store: CrossEvaluationRecordStore
    model_clustering_record_store: ModelClusteringRecordStore


def _validate_nonnegative_round_count(*, round_count: int, round_count_name: str) -> None:
    if type(round_count) is not int:
        raise TypeError(f"{round_count_name} must be builtin int")
    if round_count < 0:
        raise ValueError(f"{round_count_name} must be nonnegative")


class FedsdaRunServer:
    """最終構成のFedSDAのサーバ。ownerと設定を持ち、操作を、1ラウンドの同期の関数と通信量のownerへ渡す。

    生成は`assemble_fedsda_run_server`が行う。`__init__`は、受け取ったものを検査せずに持つ。
    """

    def __init__(
        self,
        *,
        run_clients: tuple[FedsdaRunClient, ...],
        owners: FedsdaRunServerOwners,
        model_clustering_criteria: ModelClusteringCriteria,
        maximum_evaluating_client_count_per_model: int,
        python_random_generator: Random,
    ) -> None:
        self._run_clients = run_clients
        self._owners = owners
        self._model_clustering_criteria = model_clustering_criteria
        self._maximum_evaluating_client_count_per_model = maximum_evaluating_client_count_per_model
        self._python_random_generator = python_random_generator
        self._server_round_synchronizations: list[ServerRoundSynchronization] = []

    @property
    def owners(self) -> FedsdaRunServerOwners:
        return self._owners

    def record_client_states_before_synchronization(self, *, round_index: int) -> None:
        """同期の前の、全clientからの状態の報告（軽量メッセージ）を、通信量へ数える。"""
        _validate_nonnegative_round_count(round_count=round_index, round_count_name="round_index")
        self._owners.communication_volume_record_store.record_messages(
            transfer_direction="upload", message_count=len(self._run_clients)
        )

    def synchronize_models(
        self, *, round_index: int, new_model_registration_available: bool
    ) -> None:
        """サーバの1ラウンドの同期を行う。クラスタリングは、新規モデルの登録が可能なラウンドだけ有効にする。"""
        owners = self._owners
        self._server_round_synchronizations.append(
            synchronize_models_in_server_round(
                run_clients=self._run_clients,
                global_model_repository=owners.global_model_repository,
                communication_volume_record_store=owners.communication_volume_record_store,
                cross_evaluation_record_store=owners.cross_evaluation_record_store,
                model_clustering_record_store=owners.model_clustering_record_store,
                model_clustering_criteria=self._model_clustering_criteria,
                maximum_evaluating_client_count_per_model=(
                    self._maximum_evaluating_client_count_per_model
                ),
                python_random_generator=self._python_random_generator,
                round_index=round_index,
                model_clustering_enabled=new_model_registration_available,
            )
        )

    def finalize_started_communications(self, *, completed_round_count: int) -> None:
        """終端の処理。最終構成のサーバには、終端まで持ち越す通信がないので、何も変えない。"""
        _validate_nonnegative_round_count(
            round_count=completed_round_count, round_count_name="completed_round_count"
        )

    def snapshot_server_round_synchronizations(self) -> tuple[ServerRoundSynchronization, ...]:
        """ラウンドの順の、同期の結果の写し。"""
        return tuple(self._server_round_synchronizations)


def assemble_fedsda_run_server(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    initial_model_id: int,
    initial_parameter_snapshot: dict[str, Tensor],
    initial_loss_statistics: ModelAndClassLossStatistics,
    model_clustering_criteria: ModelClusteringCriteria,
    maximum_evaluating_client_count_per_model: int,
    python_random_generator: Random,
) -> FedsdaRunServer:
    """引数の検査の後、初期モデルを持つグローバルモデルのownerと、空の記録のownerを作って、サーバを返す。

    clientの列と乱数生成器は借りる（写さない）。乱数は使わない。
    """
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be builtin tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    client_ids = [run_client.client_id for run_client in run_clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("run_clients must not hold duplicate client IDs")
    if type(model_clustering_criteria) is not ModelClusteringCriteria:
        raise TypeError("model_clustering_criteria must be exact ModelClusteringCriteria")
    # frozenを回避して組み立てた値も拒否できるよう、基準の検査をもう一度行う。
    model_clustering_criteria.__post_init__()
    if type(maximum_evaluating_client_count_per_model) is not int:
        raise TypeError("maximum_evaluating_client_count_per_model must be builtin int")
    if maximum_evaluating_client_count_per_model < 1:
        raise ValueError("maximum_evaluating_client_count_per_model must be positive")
    if type(python_random_generator) is not Random:
        raise TypeError("python_random_generator must be exact random.Random")
    return FedsdaRunServer(
        run_clients=run_clients,
        owners=FedsdaRunServerOwners(
            # 初期モデルID・パラメータ・統計の検査は、グローバルモデルのownerが行う。
            global_model_repository=GlobalModelRepository(
                initial_model_id=initial_model_id,
                initial_parameter_snapshot=initial_parameter_snapshot,
                initial_loss_statistics=initial_loss_statistics,
            ),
            communication_volume_record_store=CommunicationVolumeRecordStore(),
            cross_evaluation_record_store=CrossEvaluationRecordStore(),
            model_clustering_record_store=ModelClusteringRecordStore(),
        ),
        model_clustering_criteria=model_clustering_criteria,
        maximum_evaluating_client_count_per_model=maximum_evaluating_client_count_per_model,
        python_random_generator=python_random_generator,
    )
