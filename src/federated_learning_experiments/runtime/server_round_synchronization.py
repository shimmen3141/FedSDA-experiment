"""サーバの1ラウンドの同期: 登録→集約→（クロス評価→クラスタリングと統合）→配布→全clientの再較正。"""

from dataclasses import dataclass
from random import Random

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.global_model_distribution import (
    distribute_global_models_to_clients,
)
from federated_learning_experiments.runtime.global_model_distribution_application import (
    GlobalModelDistributionApplication,
)
from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
    ModelConsolidation,
    cluster_and_consolidate_global_models,
)
from federated_learning_experiments.runtime.model_cross_evaluation import (
    ModelCrossEvaluation,
    cross_evaluate_global_models,
)
from federated_learning_experiments.runtime.post_aggregation_prediction_recalibration import (
    PostAggregationPredictionRecalibration,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    ClientModelAggregation,
    RegisteredClientModel,
    aggregate_client_models_into_global_models,
    register_ready_client_models,
)


@dataclass(frozen=True, kw_only=True)
class ServerRoundSynchronization:
    """同期1回の、各段の結果。クラスタリングを行わなかったラウンドでは、クロス評価と統合の結果はNone。"""

    registered_client_models: tuple[RegisteredClientModel, ...]
    client_model_aggregation: ClientModelAggregation
    model_cross_evaluation: ModelCrossEvaluation | None
    model_consolidation: ModelConsolidation | None
    distribution_applications: tuple[GlobalModelDistributionApplication, ...]
    prediction_recalibrations: tuple[PostAggregationPredictionRecalibration, ...]


def synchronize_models_in_server_round(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
    cross_evaluation_record_store: CrossEvaluationRecordStore,
    model_clustering_record_store: ModelClusteringRecordStore,
    model_clustering_criteria: ModelClusteringCriteria,
    maximum_evaluating_client_count_per_model: int,
    python_random_generator: Random,
    round_index: int,
    model_clustering_enabled: bool,
) -> ServerRoundSynchronization:
    """ラウンド境界で、サーバと全clientのモデルを同期させる。

    送信できる新規モデルを登録し、clientのモデルを集約する。クラスタリングが有効で、集約の対象のモデルが
    2つ以上あれば、クロス評価の結果からクラスタリングして、同じクラスタのモデルを統合する。その後、
    全グローバルモデルを（統合のID対応とともに）配布し、全clientの予測の重みと診断証拠を再較正する。
    途中の段が失敗したら、後の段へ進まない（済んだ段は残る）。
    """
    if type(model_clustering_enabled) is not bool:
        raise TypeError("model_clustering_enabled must be builtin bool")
    # 判定の基準は、クロス評価（通信量と乱数を進める）の後の段で使う。不正なら、どの段より前に拒否する。
    if type(model_clustering_criteria) is not ModelClusteringCriteria:
        raise TypeError("model_clustering_criteria must be exact ModelClusteringCriteria")
    model_clustering_criteria.__post_init__()
    registered_client_models = register_ready_client_models(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        round_index=round_index,
    )
    client_model_aggregation = aggregate_client_models_into_global_models(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
    )
    model_cross_evaluation = None
    model_consolidation = None
    model_id_mapping: dict[int, int] = {}
    if model_clustering_enabled and len(client_model_aggregation.aggregated_global_model_ids) > 1:
        model_cross_evaluation = cross_evaluate_global_models(
            run_clients=run_clients,
            global_model_repository=global_model_repository,
            communication_volume_record_store=communication_volume_record_store,
            cross_evaluation_record_store=cross_evaluation_record_store,
            cross_evaluated_model_ids=client_model_aggregation.aggregated_global_model_ids,
            round_index=round_index,
            maximum_evaluating_client_count_per_model=maximum_evaluating_client_count_per_model,
            python_random_generator=python_random_generator,
        )
        model_consolidation = cluster_and_consolidate_global_models(
            run_clients=run_clients,
            model_cross_evaluation=model_cross_evaluation,
            aggregated_training_sample_counts_by_model_id=(
                client_model_aggregation.aggregated_training_sample_counts_by_model_id
            ),
            global_model_repository=global_model_repository,
            model_clustering_record_store=model_clustering_record_store,
            model_clustering_criteria=model_clustering_criteria,
            round_index=round_index,
        )
        model_id_mapping = model_consolidation.model_id_mapping
    distribution_applications = distribute_global_models_to_clients(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
        model_id_mapping=model_id_mapping,
    )
    prediction_recalibrations = tuple(
        run_client.recalibrate_prediction_state_after_aggregation() for run_client in run_clients
    )
    return ServerRoundSynchronization(
        registered_client_models=registered_client_models,
        client_model_aggregation=client_model_aggregation,
        model_cross_evaluation=model_cross_evaluation,
        model_consolidation=model_consolidation,
        distribution_applications=distribution_applications,
        prediction_recalibrations=prediction_recalibrations,
    )
