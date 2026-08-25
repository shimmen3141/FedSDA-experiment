"""FedSDA固有のサーバ実装。"""

import copy
import random
from collections import defaultdict

from .. import config
from ..clustering import paired_mean_upper_bound
from .clustering import CrossEvaluationClusteringServer


class FedSDANoCachedServer(CrossEvaluationClusteringServer):
    """現ラウンドのFedAvg済みモデルをクロス評価するFedSDA NoCachedサーバ。

    ラウンド内の処理順序は、回収 → FedAvg → クロス評価/クラスタリング → 配布。

    距離評価には同じラウンドの学習を反映し、マージはFedAvg済みパラメータの
    データ量加重平均で行う。新規モデルはIDだけを先に採番し、パラメータ送信を
    FedAvgのアップロードへ集約する。配布はラウンド末の1回だけ行う。
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("distance_threshold") is None:
            kwargs["distance_threshold"] = config.FEDSDA_DISTANCE_THRESHOLD
        kwargs.setdefault("clustering_decision", config.FEDSDA_CLUSTERING_DECISION)
        kwargs.setdefault("linkage", config.FEDSDA_CLUSTER_LINKAGE)
        kwargs.setdefault(
            "clustering_confidence", config.FEDSDA_CLUSTERING_CONFIDENCE
        )
        kwargs.setdefault("collect_pair_diagnostics", True)
        super().__init__(*args, **kwargs)
        self.clustering_consolidation = config.FEDSDA_CLUSTERING_CONSOLIDATION
        self.merge_noninferiority_margin = (
            config.FEDSDA_MERGE_NONINFERIORITY_MARGIN
        )
        if self.merge_noninferiority_margin < 0.0:
            raise ValueError("非劣性許容幅は0以上にしてください")
        self.clustering_noninferiority_diagnostics = []
        self.clustering_distillation_diagnostics = []
        if (
            self.clustering_consolidation
            not in config.FEDSDA_CLUSTERING_CONSOLIDATIONS
        ):
            choices = ", ".join(config.FEDSDA_CLUSTERING_CONSOLIDATIONS)
            raise ValueError(
                "未対応のクラスタリング後処理: "
                f"{self.clustering_consolidation!r}。選択肢: {choices}"
            )

    def run_round(self, t, clustering_enabled=True):
        """新規登録 → FedAvg → (任意でクラスタリング) → 配布を実行する。

        新規モデルは回収でグローバルID を採番するだけにし、パラメータ送信は次の FedAvg に
        1回に集約する。
        """
        self._register_new_models(t)

        # 全クライアントが保持するグローバルモデルID(既存 + 今ラウンド採番の新規)
        active_ids = sorted({mid for c in self.clients for mid in c.models if mid >= 0})

        # FedAvg: パラメータ送信はここ1回のみ。今ラウンドのローカル学習が反映される
        agg_weights = self.update_global_models(active_ids)

        id_mapping = {}
        if clustering_enabled:
            id_mapping = self._cluster_and_consolidate(
                t, active_ids, agg_weights
            )

        # 配布は1回のみ。マージの ID 付け替えも同時に適用する
        self.broadcast_models(id_mapping)

    def _register_new_models(self, t):
        """pending の新規モデルにグローバルID を採番する(パラメータ送信なし)。

        パラメータは後段の update_global_models(FedAvg)で1回だけ送るため、回収時に
        パラメータを送る _collect_pending_models は用いない。採番順は
        _collect_pending_models と同一(クライアント走査順)なので ID の付き方は変わらない。
        """
        n_new = 0
        for c in self.clients:
            if c.has_pending_model():
                model_id = self.request_new_model_id()
                self.record_model_registration(model_id, t, c)
                c.confirm_model_registration(model_id)
                n_new += 1
        if n_new > 0 and self.verbose:
            print(f"Server [t={t}]: Registered {n_new} new models (params sent once in FedAvg).")

    def _cluster_and_consolidate(self, t, active_ids, agg_weights):
        """FedAvg済みモデルをクラスタリングし、設定された後処理を適用する。

        ``merge`` はIDを統合し、``parameter_share`` は各IDを保ったまま
        クラスタ内の加重平均パラメータを共有する。再配布はrun_round末尾に
        一度だけ行う。
        """
        M = len(active_ids)
        if M <= 1:
            return {}

        stats_matrix = self._cross_evaluate(
            active_ids,
            round_index=t,
            collect_paired_loss_differences=(
                self.clustering_consolidation == "noninferiority_merge"
            ),
        )
        clusters = self.perform_hierarchical_clustering(active_ids, stats_matrix)
        consolidation_params = {}
        if self.clustering_consolidation == "noninferiority_merge":
            clusters, consolidation_params = self._validate_noninferiority_clusters(
                t, clusters, stats_matrix
            )
        elif self.clustering_consolidation == "distillation_merge":
            clusters, consolidation_params = self._distill_and_validate_clusters(
                t, clusters, stats_matrix
            )
        self.record_clustering_diagnostics(t, active_ids, clusters)
        if len(clusters) >= M:
            return {}

        if self.clustering_consolidation == "parameter_share":
            self._share_cluster_parameters(clusters, agg_weights)
            if self.verbose:
                print(
                    f"\nServer [t={t}]: PARAMETER SHARING EXECUTED "
                    "(IDs preserved)"
                )
                print(f"  - IDs: {active_ids}")
                print(f"  - Clusters: {clusters}\n")
            return {}

        return self._merge_clusters(
            active_ids, clusters, agg_weights, t,
            consolidation_params=consolidation_params,
        )

    def _distill_and_validate_clusters(self, t, clusters, stats_matrix):
        """共有表現サーバだけが実装する蒸留後処理のフック。"""
        raise ValueError(
            "distillation_mergeは共有バックボーン＋概念別adapter構成でのみ利用できます"
        )

    def _validate_noninferiority_clusters(self, t, clusters, stats_matrix):
        """各元モデルに対する予測性能を保つ統合候補だけを残す。

        初段のクラスタリングは候補生成に限定する。minimax代表モデルと各元モデルを、
        その元モデルを保有するクライアントの同一標本で比較し、損失差の片側上限が
        許容幅以下であることを全メンバーについて要求する。
        """
        validated = []
        consolidation_params = {}
        for cluster in clusters:
            if len(cluster) <= 1:
                validated.append(cluster)
                continue
            remaining = sorted(cluster)
            while len(remaining) > 1:
                candidate_model_id = self._select_minimax_representative(
                    remaining, stats_matrix
                )
                candidate = self.global_models[candidate_model_id]
                records = [
                    self._evaluate_noninferiority(
                        t, remaining, target_model_id, candidate,
                        candidate_model_id=candidate_model_id,
                    )
                    for target_model_id in remaining
                ]
                accepted_members = sorted(
                    record["target_model_id"]
                    for record in records if record["accepted"]
                )
                if candidate_model_id not in accepted_members:
                    accepted_members = [candidate_model_id]
                merged = len(accepted_members) > 1
                for record in records:
                    record["cluster_accepted"] = merged
                    record["target_in_accepted_cluster"] = (
                        record["target_model_id"] in accepted_members
                    )
                    self.clustering_noninferiority_diagnostics.append(record)

                validated.append(accepted_members)
                if merged:
                    consolidation_params[min(accepted_members)] = (
                        copy.deepcopy(candidate)
                    )
                accepted_set = set(accepted_members)
                remaining = [
                    model_id for model_id in remaining
                    if model_id not in accepted_set
                ]
            validated.extend([[model_id] for model_id in remaining])
        return (
            sorted(validated, key=lambda cluster: min(cluster)),
            consolidation_params,
        )

    @staticmethod
    def _select_minimax_representative(cluster, stats_matrix):
        """全元概念に対する最大平均損失増加が最小の既存モデルを選ぶ。"""
        scored = []
        for candidate_model_id in cluster:
            worst_increase = float("inf")
            increases = []
            for target_model_id in cluster:
                candidate_stats = stats_matrix[candidate_model_id][
                    target_model_id
                ]
                reference_stats = stats_matrix[target_model_id][target_model_id]
                if candidate_stats[0] <= 0 or reference_stats[0] <= 0:
                    increases = []
                    break
                increases.append(
                    candidate_stats[1] / candidate_stats[0]
                    - reference_stats[1] / reference_stats[0]
                )
            if increases:
                worst_increase = max(increases)
            scored.append((worst_increase, candidate_model_id))
        return min(scored)[1]

    def _evaluate_noninferiority(
        self, t, cluster, target_model_id, candidate_params,
        candidate_model_id,
    ):
        target_clients = [
            client for client in self.clients
            if target_model_id in client.get_held_model_ids()
        ]
        if len(target_clients) > config.CROSS_EVAL_MAX_CLIENTS:
            target_clients = random.sample(
                target_clients, config.CROSS_EVAL_MAX_CLIENTS
            )

        cached_stats = self._last_paired_loss_difference_stats.get(
            (candidate_model_id, target_model_id)
        )
        if cached_stats is not None:
            total_n, total_sum, total_sum_sq = cached_stats
        else:
            # 十分統計を収集できなかった対象だけ、従来の個別評価へ戻す。
            self.comm_messages_down += len(target_clients)
            self.comm_messages_up += len(target_clients)
            self.record_model_transfer(
                "down", candidate_params, count=len(target_clients)
            )
            total_n, total_sum, total_sum_sq = 0, 0.0, 0.0
            reference_params = self.global_models[target_model_id]
            for client in target_clients:
                n, difference_sum, difference_sum_sq = (
                    client.evaluate_model_loss_difference(
                        candidate_params, reference_params, target_model_id
                    )
                )
                total_n += n
                total_sum += difference_sum
                total_sum_sq += difference_sum_sq

        upper_bound = float("inf")
        if total_n >= max(config.CLUSTER_MIN_EVAL_N, 2):
            upper_bound = paired_mean_upper_bound(
                (total_n, total_sum, total_sum_sq),
                confidence=self.clustering_confidence,
            )
        return {
            "round_index": int(t),
            "representative_model_id": int(min(cluster)),
            "candidate_model_id": int(candidate_model_id),
            "target_model_id": int(target_model_id),
            "cluster_size": int(len(cluster)),
            "n": int(total_n),
            "mean_difference": (
                float(total_sum / total_n) if total_n else float("nan")
            ),
            "upper_bound": float(upper_bound),
            "margin": float(self.merge_noninferiority_margin),
            "accepted": bool(
                upper_bound <= self.merge_noninferiority_margin
            ),
        }

    def noninferiority_summary(self):
        """非劣性マージの候補数・採択率・評価標本数を返す。"""
        records = self.clustering_noninferiority_diagnostics
        representatives = {
            (record["round_index"], record["candidate_model_id"])
            for record in records
        }
        accepted_representatives = {
            (record["round_index"], record["candidate_model_id"])
            for record in records if record["cluster_accepted"]
        }
        candidate_count = len(representatives)
        accepted_count = len(accepted_representatives)
        return {
            "clustering_noninferiority_candidate_count": candidate_count,
            "clustering_noninferiority_accepted_count": accepted_count,
            "clustering_noninferiority_rejected_count": (
                candidate_count - accepted_count
            ),
            "clustering_noninferiority_comparison_count": len(records),
            "clustering_noninferiority_sample_count": sum(
                record["n"] for record in records
            ),
            "clustering_noninferiority_acceptance_rate": (
                accepted_count / candidate_count if candidate_count else 0.0
            ),
        }

    def _merge_clusters(
        self, active_ids, clusters, agg_weights, t,
        consolidation_params=None,
    ):
        """クラスタを指定パラメータまたは加重平均で代表IDへ統合する。"""
        if self.verbose:
            print(
                f"\nServer [t={t}]: MERGE EXECUTED "
                "(NoCached: weighted average)"
            )
            print(f"  - Before: {active_ids}")
            print(f"  - Clusters: {clusters}")

        id_mapping = {}
        consolidation_params = consolidation_params or {}
        for cluster in clusters:
            rep_id = min(cluster)
            for old_id in cluster:
                id_mapping[old_id] = rep_id
            if len(cluster) > 1:
                params = consolidation_params.get(rep_id)
                if params is None:
                    params = self._weighted_average_params(cluster, agg_weights)
                self.global_models[rep_id] = copy.deepcopy(params)
                self._merge_stats(rep_id, cluster)

        # 非代表IDのグローバル状態を削除(クライアント側の付け替えは broadcast で行う)
        for old_id in active_ids:
            if id_mapping.get(old_id, old_id) != old_id:
                if old_id in self.global_models:
                    del self.global_models[old_id]
                if old_id in self.global_stats:
                    del self.global_stats[old_id]

        if self.verbose:
            print(f"  - After IDs: {sorted(list(self.global_models.keys()))}\n")
        return id_mapping

    def _share_cluster_parameters(self, clusters, agg_weights):
        """モデルID・統計・データストアを保ち、クラスタ内パラメータだけ共有する。

        同一クラスタの各IDへ別々のtensor辞書を配置し、次ラウンド以降の
        ローカル学習で概念ごとに再び特殊化できるようにする。
        """
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            shared_params = self._weighted_average_params(
                cluster, agg_weights
            )
            for model_id in cluster:
                self.global_models[model_id] = copy.deepcopy(shared_params)

    def _weighted_average_params(self, cluster, agg_weights):
        """クラスタメンバーの FedAvg 済みパラメータをデータ量で加重平均する。

        加重平均の結合則により「統合クラスタの全データでの加重平均」と同値になる。
        重みが全て 0 の場合は代表(最小ID)のパラメータを維持する。
        """
        weights = {m: max(agg_weights.get(m, 0), 0) for m in cluster}
        total = sum(weights.values())
        if total <= 0:
            return self.global_models[min(cluster)]

        avg = None
        for m in cluster:
            w = weights[m]
            if w == 0:
                continue
            params = self.global_models[m]
            if avg is None:
                avg = {k: v * w for k, v in params.items()}
            else:
                for k in avg:
                    avg[k] = avg[k] + params[k] * w
        for k in avg:
            avg[k] = avg[k] / total
        return avg

    def _merge_stats(self, rep_id, cluster):
        """クラスタメンバーの損失統計を n 加重平均で統合する(update_global_models と同じ簡易形)。"""
        members = [m for m in cluster if m in self.global_stats]
        total_n = sum(self.global_stats[m]['n'] for m in members)
        if total_n > 0:
            mean = sum(self.global_stats[m]['mean'] * self.global_stats[m]['n']
                       for m in members) / total_n
            self.global_stats[rep_id] = {'n': total_n, 'mean': mean, 'M2': 0.0}


class FedSDACachedServer(FedSDANoCachedServer):
    """配布済みモデルのキャッシュでクロス評価するFedSDA Cachedサーバ。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.clustering_consolidation != "merge":
            raise ValueError(
                "非標準のクラスタリング後処理はFedSDA NoCachedでのみ実装されています"
            )
        self.clustering_policy = config.FEDSDA_CLUSTERING_POLICY
        if self.clustering_policy not in config.FEDSDA_CLUSTERING_POLICIES:
            choices = ", ".join(config.FEDSDA_CLUSTERING_POLICIES)
            raise ValueError(
                f"Unknown Cached clustering policy: {self.clustering_policy!r}. "
                f"Choose one of: {choices}."
            )
        self.model_weights = {}
        # 初回配布を終え、次ラウンドのクラスタリングを待つ新規モデル。
        self.models_pending_clustering = set()

    def register_model_params(self, model_id, params):
        super().register_model_params(model_id, params)
        self.model_weights.setdefault(model_id, config.PRETRAIN_SAMPLES)

    def run_round(self, t):
        """キャッシュ評価・新規登録・FedAvg・通常配布をこの順で1回ずつ行う。"""
        self._cluster_distributed_models(t)
        new_model_ids = self._register_new_models(t)

        active_ids = sorted({
            model_id for client in self.clients
            for model_id in client.models if model_id >= 0
        })
        round_weights = self.update_global_models(active_ids)
        for model_id, weight in round_weights.items():
            if weight > 0:
                self.model_weights[model_id] = weight

        self.broadcast_models()
        # この配布によって初めて全クライアントのキャッシュに入る。
        self.models_pending_clustering.update(new_model_ids)

    def finalize_protocol(self, t):
        """初回配布済みで評価待ちのモデルだけを、追加学習なしでクラスタリングする。

        ローカルで未送信のpendingモデルは回収しない。これにより全方式の
        final_model_countを「実行済み通信に対応するプロトコルを確定した後」で統一する。
        キャッシュ評価の依頼・統計返送は実通信なので、軽量メッセージとして通常どおり数える。
        """
        self._cluster_distributed_models(t)

    def _register_new_models(self, t):
        """送信可能な新規モデルへIDを割り当て、初回FedAvgの対象にする。"""
        new_model_ids = []
        for client in self.clients:
            if not client.has_pending_model():
                continue
            model_id = self.request_new_model_id()
            self.record_model_registration(model_id, t, client)
            client.confirm_model_registration(model_id)
            self.comm_messages_down += 1
            new_model_ids.append(model_id)

        if new_model_ids and self.verbose:
            print(f"Server [t={t}]: Registered {len(new_model_ids)} new cached models.")
        return new_model_ids

    def _cluster_distributed_models(self, t):
        """設定された実行方針に従い、配布済みキャッシュで距離評価する。"""
        if not self._clustering_is_due():
            return

        model_ids = sorted(self.global_models)
        self.models_pending_clustering.clear()
        if len(model_ids) <= 1:
            return

        stats_matrix = self._cross_evaluate(
            model_ids,
            send_model_params=False,
            use_client_cache=True,
            round_index=t,
        )
        clusters = self.perform_hierarchical_clustering(model_ids, stats_matrix)
        self.record_clustering_diagnostics(t, model_ids, clusters)
        if len(clusters) < len(model_ids):
            self._merge_cached_clusters(t, clusters)

    def _clustering_is_due(self):
        """現在の集約ラウンドでクラスタリングを実行するかを返す。"""
        if self.clustering_policy == 'disabled':
            return False
        if self.clustering_policy == 'every_round':
            return True
        return bool(self.models_pending_clustering)

    def _merge_cached_clusters(self, t, clusters):
        """配布済みモデルを累積重みで統合し、クライアントの学習状態にも対応を適用する。"""
        if self.verbose:
            print(f"\nServer [t={t}]: MERGE EXECUTED (FedSDA Cached)")
            print(f"  - Clusters: {clusters}")

        cluster_weights = {}
        new_models = {}
        new_stats = {}
        new_model_weights = {}

        for cluster in clusters:
            representative = min(cluster)
            weights = {
                model_id: max(self.model_weights.get(model_id, 0), 0)
                for model_id in cluster
            }
            if sum(weights.values()) <= 0:
                weights = {model_id: 1 for model_id in cluster}

            cluster_weights[representative] = weights
            new_models[representative] = self._weighted_average_params(cluster, weights)
            new_stats[representative] = self._combined_stats(cluster)
            new_model_weights[representative] = sum(weights.values())

        for client in self.clients:
            client.apply_cached_merge(clusters, cluster_weights, new_stats)
        self.comm_messages_down += len(self.clients)

        self.global_models = new_models
        self.global_stats = defaultdict(
            lambda: {'n': 0, 'mean': 0.0, 'M2': 0.0}, new_stats
        )
        self.model_weights = new_model_weights

    def _combined_stats(self, cluster):
        members = [model_id for model_id in cluster if model_id in self.global_stats]
        total_n = sum(self.global_stats[model_id]['n'] for model_id in members)
        if total_n == 0:
            return {'n': 0, 'mean': 0.0, 'M2': 0.0}
        mean = sum(
            self.global_stats[model_id]['mean'] * self.global_stats[model_id]['n']
            for model_id in members
        ) / total_n
        return {'n': total_n, 'mean': mean, 'M2': 0.0}
