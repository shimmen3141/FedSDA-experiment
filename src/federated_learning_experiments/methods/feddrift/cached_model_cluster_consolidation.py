"""FedDriftの保存済みモデル群の加重統合を計算する。"""

from dataclasses import dataclass

from torch import Tensor, float32, isfinite, no_grad, strided

from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
    _validate_loss_moments,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftCachedClusterConsolidation:
    """serverの計算結果。client自身のparameter平均とは区別する。"""

    model_parameters_by_representative_id: dict[int, dict[str, Tensor]]
    loss_moments_by_representative_id: dict[int, BoundedLossMoments]
    model_weights_by_representative_id: dict[int, int]
    isolation_until_by_representative_id: dict[int, int]
    model_id_mapping: dict[int, int]
    cluster_weights_by_representative_id: dict[int, dict[int, int]]


def consolidate_feddrift_cached_model_clusters(
    *,
    timestep_index: int,
    model_clusters: tuple[tuple[int, ...], ...],
    model_parameters_by_model_id: dict[int, dict[str, Tensor]],
    historical_model_weights_by_model_id: dict[int, int],
    loss_moments_by_model_id: dict[int, BoundedLossMoments],
    isolation_until_by_model_id: dict[int, int],
) -> FeddriftCachedClusterConsolidation:
    """serverの保存済みparameter・統計・重み・隔離期限を計算する。"""
    if type(timestep_index) is not int:
        raise TypeError("時刻はbuiltin intが必要です。")
    if timestep_index < 0:
        raise ValueError("時刻は非負が必要です。")
    _validate_clusters_and_parameters(
        model_clusters=model_clusters, model_parameters_by_model_id=model_parameters_by_model_id
    )
    for values in (
        historical_model_weights_by_model_id,
        loss_moments_by_model_id,
        isolation_until_by_model_id,
    ):
        if type(values) is not dict:
            raise TypeError("モデル別の値はdictが必要です。")
        for model_id in values:
            _validate_model_id(model_id)
            if model_id not in model_parameters_by_model_id:
                raise ValueError("モデル別の値はsnapshot IDに属することが必要です。")
    for weight in historical_model_weights_by_model_id.values():
        if type(weight) is not int:
            raise TypeError("過去重みはbuiltin intが必要です。")
    for expiry in isolation_until_by_model_id.values():
        if type(expiry) is not int:
            raise TypeError("期限はbuiltin intが必要です。")
        if expiry < 0:
            raise ValueError("期限は非負が必要です。")
    for moments in loss_moments_by_model_id.values():
        _validate_loss_moments(loss_moments=moments)
    cluster_weights: dict[int, dict[int, int]] = {}
    loss_moments: dict[int, BoundedLossMoments] = {}
    model_weights: dict[int, int] = {}
    isolation_until: dict[int, int] = {}
    model_id_mapping: dict[int, int] = {}
    for cluster in model_clusters:
        representative = min(cluster)
        weights = {
            model_id: max(historical_model_weights_by_model_id.get(model_id, 0), 0)
            for model_id in cluster
        }
        if sum(weights.values()) <= 0:
            weights = {model_id: 1 for model_id in cluster}
        cluster_weights[representative] = weights
        model_weights[representative] = sum(weights.values())
        members = [model_id for model_id in cluster if model_id in loss_moments_by_model_id]
        total_count = sum(loss_moments_by_model_id[mid].observed_loss_count for mid in members)
        mean = (
            sum(
                loss_moments_by_model_id[mid].mean_loss
                * loss_moments_by_model_id[mid].observed_loss_count
                for mid in members
            )
            / total_count
            if total_count
            else 0.0
        )
        loss_moments[representative] = BoundedLossMoments(
            observed_loss_count=total_count, mean_loss=mean, sum_squared_loss_deviations=0.0
        )
        expiry = max(isolation_until_by_model_id.get(mid, 0) for mid in cluster)
        if expiry > timestep_index:
            isolation_until[representative] = expiry
        for model_id in cluster:
            model_id_mapping[model_id] = representative
    parameters = average_feddrift_cached_cluster_parameters(
        model_clusters=model_clusters,
        model_parameters_by_model_id=model_parameters_by_model_id,
        cluster_weights_by_representative_id=cluster_weights,
    )
    return FeddriftCachedClusterConsolidation(
        model_parameters_by_representative_id=parameters,
        loss_moments_by_representative_id=loss_moments,
        model_weights_by_representative_id=model_weights,
        isolation_until_by_representative_id=isolation_until,
        model_id_mapping=model_id_mapping,
        cluster_weights_by_representative_id=cluster_weights,
    )


def _validate_model_id(model_id: int) -> None:
    if type(model_id) is not int:
        raise TypeError("model IDはbuiltin intが必要です。")
    if model_id < 0:
        raise ValueError("model IDは非負が必要です。")


def _validate_clusters_and_parameters(
    *,
    model_clusters: tuple[tuple[int, ...], ...],
    model_parameters_by_model_id: dict[int, dict[str, Tensor]],
) -> None:
    if type(model_clusters) is not tuple or type(model_parameters_by_model_id) is not dict:
        raise TypeError("分割はtuple、snapshotはdictが必要です。")
    for model_id, parameters in model_parameters_by_model_id.items():
        _validate_model_id(model_id)
        if type(parameters) is not dict:
            raise TypeError("parameter snapshotはdictが必要です。")
        if not parameters:
            raise ValueError("parameter snapshotは空にできません。")
        for name, value in parameters.items():
            if type(name) is not str or not name:
                raise ValueError("parameter名は空でないstrが必要です。")
            if type(value) is not Tensor:
                raise TypeError("parameter値はexact Tensorが必要です。")
            if (
                value.device.type != "cpu"
                or value.dtype != float32
                or value.layout != strided
                or value.is_nested
                or value.numel() == 0
            ):
                raise ValueError("parameter値は非空CPU float32 stridedが必要です。")
            if not isfinite(value).all().item():
                raise ValueError("parameter値は有限値が必要です。")
    members: list[int] = []
    for cluster in model_clusters:
        if type(cluster) is not tuple:
            raise TypeError("各群はtupleが必要です。")
        if not cluster:
            raise ValueError("各群は空にできません。")
        for model_id in cluster:
            _validate_model_id(model_id)
            if model_id not in model_parameters_by_model_id or model_id in members:
                raise ValueError("群内IDは存在し、分割内で重複しないことが必要です。")
            members.append(model_id)
        first = model_parameters_by_model_id[cluster[0]]
        for model_id in cluster[1:]:
            parameters = model_parameters_by_model_id[model_id]
            if set(parameters) != set(first):
                raise ValueError("群内parameter名が一致することが必要です。")
            if any(parameters[name].shape != first[name].shape for name in first):
                raise ValueError("群内parameter shapeが一致することが必要です。")
    if set(members) != set(model_parameters_by_model_id):
        raise ValueError("分割は全snapshotを含むことが必要です。")


@no_grad()
def average_feddrift_cached_cluster_parameters(
    *,
    model_clusters: tuple[tuple[int, ...], ...],
    model_parameters_by_model_id: dict[int, dict[str, Tensor]],
    cluster_weights_by_representative_id: dict[int, dict[int, int]],
) -> dict[int, dict[str, Tensor]]:
    """決定済み重みを呼出し側自身のparameterへ適用する。"""
    _validate_clusters_and_parameters(
        model_clusters=model_clusters, model_parameters_by_model_id=model_parameters_by_model_id
    )
    if type(cluster_weights_by_representative_id) is not dict:
        raise TypeError("代表ごとの重みはdictが必要です。")
    for representative, weights in cluster_weights_by_representative_id.items():
        _validate_model_id(representative)
        if type(weights) is not dict:
            raise TypeError("群内重みはdictが必要です。")
        for model_id, weight in weights.items():
            _validate_model_id(model_id)
            if type(weight) is not int:
                raise TypeError("重みはbuiltin intが必要です。")
            if weight < 0:
                raise ValueError("群内重みは非負が必要です。")
    if set(cluster_weights_by_representative_id) != {min(c) for c in model_clusters}:
        raise ValueError("重みの代表IDは群の最小IDと一致することが必要です。")
    for cluster in model_clusters:
        if set(cluster_weights_by_representative_id[min(cluster)]) != set(cluster):
            raise ValueError("群内重みは全memberへ対応することが必要です。")
    result: dict[int, dict[str, Tensor]] = {}
    for cluster in model_clusters:
        weights = cluster_weights_by_representative_id[min(cluster)]
        total = sum(weights.values())
        if total <= 0:
            weights = {model_id: 1 for model_id in cluster}
            total = len(cluster)
        averaged: dict[str, Tensor] | None = None
        for model_id in cluster:
            ratio = weights[model_id] / total
            parameters = model_parameters_by_model_id[model_id]
            if averaged is None:
                averaged = {name: value * ratio for name, value in parameters.items()}
            else:
                for name, value in parameters.items():
                    averaged[name] = averaged[name] + value * ratio
        # 非空群を検査済みなので必ずdictになる。
        assert averaged is not None
        if any(not isfinite(value).all().item() for value in averaged.values()):
            raise ValueError("統合parameterは有限値が必要です。")
        result[min(cluster)] = averaged
    return result
