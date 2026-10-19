"""サーバの1ラウンドの同期: 登録→集約→（クロス評価→クラスタリングと統合）→配布→全clientの再較正。"""

from dataclasses import dataclass
from random import Random

from torch import Tensor
from torch import equal as torch_equal

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
    record_global_model_distribution_volume,
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
    split_shared_and_concept_specific_parameters,
)

# クロス評価のモデルの送り方（`FedsdaRunParticipantSettings.cross_evaluation_model_transfer_policy`の値）。
SEND_CANDIDATE_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME = (
    "send_candidate_models_for_cross_evaluation"
)
REUSE_DISTRIBUTED_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME = (
    "reuse_distributed_models_for_cross_evaluation"
)
CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES = (
    SEND_CANDIDATE_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME,
    REUSE_DISTRIBUTED_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME,
)


@dataclass(frozen=True, kw_only=True)
class ServerRoundSynchronization:
    """同期1回の、各段の結果。クラスタリングを行わなかったラウンドでは、クロス評価と統合の結果はNone。

    `prediction_recalibrations`は、clientごとの、集約後の再較正の結果（再較正しない方針のclientでは、None）。
    """

    registered_client_models: tuple[RegisteredClientModel, ...]
    client_model_aggregation: ClientModelAggregation
    model_cross_evaluation: ModelCrossEvaluation | None
    model_consolidation: ModelConsolidation | None
    distribution_applications: tuple[GlobalModelDistributionApplication, ...]
    prediction_recalibrations: tuple[PostAggregationPredictionRecalibration | None, ...]


def validate_cross_evaluation_model_transfer_policy(
    *, cross_evaluation_model_transfer_policy: str
) -> None:
    """クロス評価のモデルの送り方が、builtin strの、2つの値のどちらかであること。"""
    if type(cross_evaluation_model_transfer_policy) is not str:
        raise TypeError("cross_evaluation_model_transfer_policy must be builtin str")
    if cross_evaluation_model_transfer_policy not in CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES:
        raise ValueError(
            "cross_evaluation_model_transfer_policy must be one of "
            f"{CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES}"
        )


def _record_consolidated_model_resend_volume(
    *,
    communication_volume_record_store: CommunicationVolumeRecordStore,
    global_model_repository: GlobalModelRepository,
    model_id_mapping: dict[int, int],
    early_distributed_shared_parameters: dict[str, Tensor],
    client_count: int,
) -> None:
    """統合の後の送り直しの下りの通信量（クロス評価で配布するモデルを使う扱い）。状態は変えない。

    ID対応が空（統合が無い）なら、何も記録しない。統合が起きたら、ID対応のメッセージ（client数）と、値が変わった
    代表のモデル（元と行き先が違う対の、行き先）の概念固有部×client数とモデルの転送回数を数える。統合は共有部も
    平均するので、配布する共有部（最初のグローバルモデルのもの）が、先に送った値とビット単位で違えば、共有部×
    client数も数える。損失統計のbyteは、配布と同じく数えない。
    """
    if not model_id_mapping:
        return
    representative_model_ids = sorted(
        {
            representative_model_id
            for model_id, representative_model_id in model_id_mapping.items()
            if model_id != representative_model_id
        }
    )
    representative_concept_specific_parameters = tuple(
        split_shared_and_concept_specific_parameters(
            parameter_snapshot=global_model_repository.get_global_model_parameters(
                model_id=model_id
            )
        )[1]
        for model_id in representative_model_ids
    )
    distributed_shared_parameters = split_shared_and_concept_specific_parameters(
        parameter_snapshot=global_model_repository.get_global_model_parameters(
            model_id=global_model_repository.global_model_ids[0]
        )
    )[0]
    shared_parameters_changed = distributed_shared_parameters.keys() != (
        early_distributed_shared_parameters.keys()
    ) or any(
        not torch_equal(parameter_values, early_distributed_shared_parameters[parameter_name])
        for parameter_name, parameter_values in distributed_shared_parameters.items()
    )
    communication_volume_record_store.record_messages(
        transfer_direction="download", message_count=client_count
    )
    for concept_specific_parameters in representative_concept_specific_parameters:
        communication_volume_record_store.record_parameter_transfer(
            transfer_direction="download",
            parameter_snapshot=concept_specific_parameters,
            transfer_count=client_count,
        )
    communication_volume_record_store.record_model_transfers(
        transfer_direction="download", model_count=len(representative_model_ids) * client_count
    )
    if shared_parameters_changed:
        communication_volume_record_store.record_parameter_transfer(
            transfer_direction="download",
            parameter_snapshot=distributed_shared_parameters,
            transfer_count=client_count,
        )


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
    cross_evaluation_model_transfer_policy: str = SEND_CANDIDATE_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME,
) -> ServerRoundSynchronization:
    """ラウンド境界で、サーバと全clientのモデルを同期させる。

    送信できる新規モデルを登録し、clientのモデルを集約する。クラスタリングが有効で、集約の対象のモデルが
    2つ以上あれば、クロス評価の結果からクラスタリングして、同じクラスタのモデルを統合する。その後、
    全グローバルモデルを（統合のID対応とともに）配布し、全clientの予測の重みとAdaHedgeの証拠を再較正する。
    途中の段が失敗したら、後の段へ進まない（済んだ段は残る）。

    クロス評価で配布するモデルを使う扱いでは、状態の更新の順は同じまま、クロス評価をするラウンドの通信量を、
    クロス評価の前の全モデルの配布、クロス評価のメッセージ、統合の後の送り直し、として記録する。
    """
    if type(model_clustering_enabled) is not bool:
        raise TypeError("model_clustering_enabled must be builtin bool")
    validate_cross_evaluation_model_transfer_policy(
        cross_evaluation_model_transfer_policy=cross_evaluation_model_transfer_policy
    )
    reuses_distributed_models = (
        cross_evaluation_model_transfer_policy
        == REUSE_DISTRIBUTED_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME
    )
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
    early_distributed_shared_parameters: dict[str, Tensor] | None = None
    if model_clustering_enabled and len(client_model_aggregation.aggregated_global_model_ids) > 1:
        if reuses_distributed_models:
            # クロス評価の前に、その時点の全グローバルモデルを全clientへ配布する（clientは受け取ったものを保持し、
            # 手元の保有モデルを、統合の後まで置き換えない）。送った共有部を、統合の後と比べるために残す。
            record_global_model_distribution_volume(
                communication_volume_record_store=communication_volume_record_store,
                global_model_repository=global_model_repository,
                client_count=len(run_clients),
            )
            early_distributed_shared_parameters = {
                parameter_name: parameter_values.clone()
                for parameter_name, parameter_values in split_shared_and_concept_specific_parameters(
                    parameter_snapshot=global_model_repository.get_global_model_parameters(
                        model_id=global_model_repository.global_model_ids[0]
                    )
                )[0].items()
            }
        model_cross_evaluation = cross_evaluate_global_models(
            run_clients=run_clients,
            global_model_repository=global_model_repository,
            communication_volume_record_store=communication_volume_record_store,
            cross_evaluation_record_store=cross_evaluation_record_store,
            cross_evaluated_model_ids=client_model_aggregation.aggregated_global_model_ids,
            round_index=round_index,
            maximum_evaluating_client_count_per_model=maximum_evaluating_client_count_per_model,
            python_random_generator=python_random_generator,
            record_model_transfers=not reuses_distributed_models,
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
    if early_distributed_shared_parameters is not None:
        _record_consolidated_model_resend_volume(
            communication_volume_record_store=communication_volume_record_store,
            global_model_repository=global_model_repository,
            model_id_mapping=model_id_mapping,
            early_distributed_shared_parameters=early_distributed_shared_parameters,
            client_count=len(run_clients),
        )
    distribution_applications = distribute_global_models_to_clients(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
        model_id_mapping=model_id_mapping,
        record_communication_volume=early_distributed_shared_parameters is None,
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
