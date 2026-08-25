"""共有バックボーンと概念別ヘッドを集約するFedSDAサーバ。"""

import copy
import itertools
import math

from .. import config
from ..clustering import paired_mean_upper_bound
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

    def _record_distillation_teacher_transfer(
        self, teacher_params_by_model, clients,
    ):
        """蒸留クライアントへ共有部を1回、teacher個別部を各1回送る。"""
        if not clients:
            return
        first_params = next(iter(teacher_params_by_model.values()))
        backbone, _ = SharedBackboneMLP.split_params(first_params)
        self.record_parameter_transfer("down", backbone, count=len(clients))
        for params in teacher_params_by_model.values():
            _, personalized = SharedBackboneMLP.split_params(params)
            self.record_parameter_transfer(
                "down", personalized, count=len(clients)
            )
        self.comm_models_down += len(teacher_params_by_model) * len(clients)

    @staticmethod
    def _aggregate_distillation_updates(updates):
        """ローカルstudentの概念別パラメータを学習標本数でFedAvgする。"""
        total = sum(update.sample_count for _, update in updates)
        if total <= 0:
            return None
        accumulator = None
        for _, update in updates:
            accumulator = _add_weighted(
                accumulator, update.personalized_params, update.sample_count
            )
        return _divide_params(accumulator, total)

    def _select_distillation_participants(self, cluster):
        """全概念の検証標本を最もよく被覆するクライアント集合を選ぶ。"""
        candidates = []
        for client in self.clients:
            counts = client.distillation_split_sample_counts(cluster)
            if sum(training for training, _ in counts.values()) < 2:
                continue
            candidates.append((client, counts))

        max_clients = min(config.CROSS_EVAL_MAX_CLIENTS, len(candidates))
        minimum = max(config.CLUSTER_MIN_EVAL_N, 2)
        best = None
        for size in range(1, max_clients + 1):
            for selected in itertools.combinations(candidates, size):
                validation_counts = {
                    model_id: sum(
                        counts[model_id][1] for _, counts in selected
                    )
                    for model_id in cluster
                }
                supported = sum(
                    count >= minimum for count in validation_counts.values()
                )
                score = (
                    supported == len(cluster),
                    supported,
                    min(validation_counts.values(), default=0),
                    sum(validation_counts.values()),
                    -size,
                )
                if best is None or score > best[0]:
                    best = (score, selected, validation_counts)

        if best is None:
            return [], {model_id: 0 for model_id in cluster}
        return [client for client, _ in best[1]], best[2]

    def _distill_and_validate_clusters(self, t, clusters, stats_matrix):
        """teacher混合をadapter/headへ蒸留し、集約後studentだけを非劣性採択する。"""
        validated_clusters = []
        consolidation_params = {}
        for cluster in clusters:
            cluster = sorted(cluster)
            if len(cluster) <= 1:
                validated_clusters.append(cluster)
                continue

            candidate_id = self._select_minimax_representative(
                cluster, stats_matrix
            )
            teacher_params = {
                model_id: copy.deepcopy(self.global_models[model_id])
                for model_id in cluster
            }
            participants, validation_counts = (
                self._select_distillation_participants(cluster)
            )
            job_id = (int(t), int(min(cluster)), tuple(cluster))
            before_values = (
                self.comm_parameter_values_up + self.comm_parameter_values_down
            )
            before_bytes = self.comm_bytes_up + self.comm_bytes_down

            minimum = max(config.CLUSTER_MIN_EVAL_N, 2)
            if any(
                validation_counts[model_id] < minimum
                for model_id in cluster
            ):
                validated_clusters.extend([[model_id] for model_id in cluster])
                self._record_distillation_result(
                    t=t, cluster=cluster, candidate_id=candidate_id,
                    updates=[], target_stats={}, accepted=False,
                    before_values=before_values, before_bytes=before_bytes,
                    precheck_rejected=True,
                    precheck_min_validation_sample_count=min(
                        validation_counts.values(), default=0
                    ),
                )
                continue

            self.comm_messages_down += len(participants)
            self._record_distillation_teacher_transfer(
                teacher_params, participants
            )
            updates = []
            for client in participants:
                update = client.prepare_distillation_update(
                    job_id, teacher_params, candidate_id
                )
                if update is None:
                    continue
                updates.append((client, update))
                self.comm_messages_up += 1
                self.comm_models_up += 1
                self.record_parameter_transfer(
                    "up", update.personalized_params
                )

            personalized = self._aggregate_distillation_updates(updates)
            if personalized is None:
                for client in participants:
                    client.discard_distillation_job(job_id)
                validated_clusters.extend([[model_id] for model_id in cluster])
                self._record_distillation_result(
                    t=t, cluster=cluster, candidate_id=candidate_id,
                    updates=updates, target_stats={}, accepted=False,
                    before_values=before_values, before_bytes=before_bytes,
                    precheck_rejected=False,
                    precheck_min_validation_sample_count=min(
                        validation_counts.values(), default=0
                    ),
                )
                continue

            backbone, _ = SharedBackboneMLP.split_params(
                self.global_models[candidate_id]
            )
            student_params = SharedBackboneMLP.combine_params(
                backbone, personalized
            )
            target_stats = {
                model_id: [0, 0.0, 0.0] for model_id in cluster
            }
            for client, _ in updates:
                self.comm_messages_down += 1
                self.comm_messages_up += 1
                self.comm_models_down += 1
                self.record_parameter_transfer("down", personalized)
                result = client.evaluate_distilled_student(
                    job_id, teacher_params, student_params
                )
                for model_id, stats in result.by_target_model.items():
                    aggregate = target_stats[model_id]
                    aggregate[0] += stats[0]
                    aggregate[1] += stats[1]
                    aggregate[2] += stats[2]

            accepted = True
            target_records = {}
            for model_id, stats in target_stats.items():
                upper_bound = math.inf
                if stats[0] >= max(config.CLUSTER_MIN_EVAL_N, 2):
                    upper_bound = paired_mean_upper_bound(
                        tuple(stats), confidence=self.clustering_confidence
                    )
                target_records[model_id] = {
                    "n": int(stats[0]),
                    "mean_difference": (
                        float(stats[1] / stats[0])
                        if stats[0] else float("nan")
                    ),
                    "upper_bound": float(upper_bound),
                }
                accepted = accepted and (
                    upper_bound <= self.merge_noninferiority_margin
                )

            if accepted:
                validated_clusters.append(cluster)
                consolidation_params[min(cluster)] = student_params
            else:
                validated_clusters.extend([[model_id] for model_id in cluster])
            self._record_distillation_result(
                t=t, cluster=cluster, candidate_id=candidate_id,
                updates=updates, target_stats=target_records,
                accepted=accepted, before_values=before_values,
                before_bytes=before_bytes,
                precheck_rejected=False,
                precheck_min_validation_sample_count=min(
                    validation_counts.values(), default=0
                ),
            )

        return (
            sorted(validated_clusters, key=lambda members: min(members)),
            consolidation_params,
        )

    def _record_distillation_result(
        self, *, t, cluster, candidate_id, updates, target_stats, accepted,
        before_values, before_bytes, precheck_rejected,
        precheck_min_validation_sample_count,
    ):
        """採否と追加通信の損益分岐を一候補一レコードで保存する。"""
        extra_values = (
            self.comm_parameter_values_up + self.comm_parameter_values_down
            - before_values
        )
        extra_bytes = self.comm_bytes_up + self.comm_bytes_down - before_bytes
        _, personalized = SharedBackboneMLP.split_params(
            self.global_models[candidate_id]
        )
        values_per_head, bytes_per_head = parameter_payload_size(personalized)
        values_saved_per_round = (
            values_per_head * (len(cluster) - 1) * len(self.clients)
        )
        bytes_saved_per_round = (
            bytes_per_head * (len(cluster) - 1) * len(self.clients)
        )
        self.clustering_distillation_diagnostics.append({
            "round_index": int(t),
            "cluster": tuple(int(model_id) for model_id in cluster),
            "candidate_model_id": int(candidate_id),
            "cluster_size": int(len(cluster)),
            "local_update_count": int(len(updates)),
            "training_sample_count": int(sum(
                update.sample_count for _, update in updates
            )),
            "validation_sample_count": int(sum(
                record["n"] for record in target_stats.values()
            )),
            "max_upper_bound": float(max(
                (record["upper_bound"] for record in target_stats.values()),
                default=math.inf,
            )),
            "accepted": bool(accepted),
            "precheck_rejected": bool(precheck_rejected),
            "precheck_min_validation_sample_count": int(
                precheck_min_validation_sample_count
            ),
            "extra_parameter_values": int(extra_values),
            "extra_bytes": int(extra_bytes),
            "values_saved_per_round": int(values_saved_per_round),
            "bytes_saved_per_round": int(bytes_saved_per_round),
            "break_even_rounds": float(
                extra_values / values_saved_per_round
                if values_saved_per_round > 0 else math.inf
            ),
        })

    def distillation_summary(self):
        """蒸留候補の採択率・標本数・追加通信量を集約する。"""
        records = self.clustering_distillation_diagnostics
        accepted = [record for record in records if record["accepted"]]
        finite_break_even = [
            record["break_even_rounds"] for record in accepted
            if math.isfinite(record["break_even_rounds"])
        ]
        return {
            "clustering_distillation_candidate_count": len(records),
            "clustering_distillation_accepted_count": len(accepted),
            "clustering_distillation_rejected_count": len(records) - len(accepted),
            "clustering_distillation_precheck_rejected_count": sum(
                record["precheck_rejected"] for record in records
            ),
            "clustering_distillation_acceptance_rate": (
                len(accepted) / len(records) if records else 0.0
            ),
            "clustering_distillation_local_update_count": sum(
                record["local_update_count"] for record in records
            ),
            "clustering_distillation_training_sample_count": sum(
                record["training_sample_count"] for record in records
            ),
            "clustering_distillation_validation_sample_count": sum(
                record["validation_sample_count"] for record in records
            ),
            "clustering_distillation_extra_parameter_values": sum(
                record["extra_parameter_values"] for record in records
            ),
            "clustering_distillation_extra_bytes": sum(
                record["extra_bytes"] for record in records
            ),
            "clustering_distillation_break_even_rounds_mean": (
                sum(finite_break_even) / len(finite_break_even)
                if finite_break_even else 0.0
            ),
        }

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
