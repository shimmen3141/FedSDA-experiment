"""共有バックボーンと概念別ヘッドを集約するFedSDAサーバ。"""

import copy

from .. import config
from ..models import SharedBackboneMLP, parameter_payload_size
from .fedsda import FedSDANoCachedServer


def _add_weighted(accumulator, params, weight):
    if accumulator is None:
        return {name: value * weight for name, value in params.items()}
    for name, value in params.items():
        accumulator[name] = accumulator[name] + value * weight
    return accumulator


def _divide_params(params, denominator):
    return {name: value / denominator for name, value in params.items()}


class SharedBackboneFedSDANoCachedServer(FedSDANoCachedServer):
    """共有部をクライアントごとに1回、概念別ヘッドをIDごとに集約する。"""

    def run_round(self, t, clustering_enabled=True):
        """集約・配布後、必要なら更新前のルーティング証拠を破棄する。"""
        super().run_round(t, clustering_enabled=clustering_enabled)
        if config.SHARED_BACKBONE_ROUTING_RECALIBRATION != "none":
            for client in self.clients:
                client.recalibrate_routing_after_aggregation()

    def _begin_cross_evaluation_model_transfers(self):
        """共有部と概念別ヘッドの同一クライアントへの重複送信を避ける。"""
        self._cross_evaluation_backbone_recipients = set()
        self._cross_evaluation_head_recipients = set()

    def _record_cross_evaluation_model_transfer(
        self, model_id, params, target_clients
    ):
        """共有バックボーン1回と各概念ヘッド1回として通信量を数える。"""
        backbone, head = SharedBackboneMLP.split_params(params)
        for client in target_clients:
            client_key = id(client)
            if client_key not in self._cross_evaluation_backbone_recipients:
                self.record_parameter_transfer("down", backbone)
                self._cross_evaluation_backbone_recipients.add(client_key)
            head_key = (int(model_id), client_key)
            if head_key in self._cross_evaluation_head_recipients:
                continue
            self.comm_models_down += 1
            self.record_parameter_transfer("down", head)
            self._cross_evaluation_head_recipients.add(head_key)

    def update_global_models(self, active_ids):
        """共有バックボーン1個と各概念ヘッドを別々の重みでFedAvgする。"""
        agg_weights = {model_id: 0 for model_id in active_ids}
        head_sums = {model_id: None for model_id in active_ids}
        stat_weighted_sums = {model_id: 0.0 for model_id in active_ids}
        stat_counts = {model_id: 0 for model_id in active_ids}
        backbone_sum = None
        backbone_weight = 0

        for client in self.clients:
            assigned = []
            for model_id in active_ids:
                if model_id not in client.models:
                    continue
                sample_count = len(client.train_data_store.get(model_id, []))
                if sample_count > 0:
                    assigned.append((model_id, sample_count))
            if not assigned:
                continue

            # 従来の論理モデル転送回数は比較用に維持する。
            self.comm_models_up += len(assigned)
            representative = client.models[assigned[0][0]]
            backbone_params, _ = SharedBackboneMLP.split_params(
                representative.get_params()
            )
            client_weight = sum(sample_count for _, sample_count in assigned)
            self.record_parameter_transfer("up", backbone_params)
            backbone_sum = _add_weighted(
                backbone_sum, backbone_params, client_weight
            )
            backbone_weight += client_weight

            for model_id, sample_count in assigned:
                _, head_params = SharedBackboneMLP.split_params(
                    client.models[model_id].get_params()
                )
                self.record_parameter_transfer("up", head_params)
                head_sums[model_id] = _add_weighted(
                    head_sums[model_id], head_params, sample_count
                )
                agg_weights[model_id] += sample_count
                stats = client.model_stats.get(model_id)
                if stats is not None:
                    stat_weighted_sums[model_id] += stats['mean'] * stats['n']
                    stat_counts[model_id] += stats['n']

        if backbone_weight > 0:
            global_backbone = _divide_params(backbone_sum, backbone_weight)
        elif self.global_models:
            global_backbone, _ = SharedBackboneMLP.split_params(
                next(iter(self.global_models.values()))
            )
        else:
            return agg_weights

        for model_id in active_ids:
            if agg_weights[model_id] > 0:
                global_head = _divide_params(
                    head_sums[model_id], agg_weights[model_id]
                )
            elif model_id in self.global_models:
                _, global_head = SharedBackboneMLP.split_params(
                    self.global_models[model_id]
                )
            else:
                continue
            self.global_models[model_id] = SharedBackboneMLP.combine_params(
                global_backbone, global_head
            )
            if stat_counts[model_id] > 0:
                self.global_stats[model_id] = {
                    'n': stat_counts[model_id],
                    'mean': stat_weighted_sums[model_id] / stat_counts[model_id],
                    'M2': 0.0,
                }
        return agg_weights

    def broadcast_models(self, id_mapping=None):
        """共有部1個と全概念ヘッドを配布し、論理モデル数も従来どおり記録する。"""
        if self.global_models:
            backbone, _ = SharedBackboneMLP.split_params(
                next(iter(self.global_models.values()))
            )
            self.record_parameter_transfer(
                "down", backbone, count=len(self.clients)
            )
            for params in self.global_models.values():
                _, head = SharedBackboneMLP.split_params(params)
                self.record_parameter_transfer(
                    "down", head, count=len(self.clients)
                )
        self.comm_models_down += len(self.global_models) * len(self.clients)
        if id_mapping:
            self.comm_messages_down += len(self.clients)
        for client in self.clients:
            client.apply_server_mapping(
                id_mapping or {}, self.global_models, self.global_stats
            )

    def final_parameter_footprint(self):
        """共有部は1個、概念別ヘッドはモデル数分として最終容量を数える。"""
        if not self.global_models:
            return 0, 0
        backbone, _ = SharedBackboneMLP.split_params(
            next(iter(self.global_models.values()))
        )
        values, byte_count = parameter_payload_size(backbone)
        for params in self.global_models.values():
            _, head = SharedBackboneMLP.split_params(params)
            head_values, head_bytes = parameter_payload_size(head)
            values += head_values
            byte_count += head_bytes
        return values, byte_count
