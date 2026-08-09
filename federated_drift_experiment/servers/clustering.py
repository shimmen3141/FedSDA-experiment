"""クロス評価と階層クラスタリングを提供する共通サーバ。"""

import random
from collections import defaultdict
from statistics import NormalDist

from .. import config
from ..clustering import (
    SUPPORTED_CLUSTERING_DECISIONS,
    cluster_models,
    mean_loss,
    standardized_mean_increase,
)
from .base import BaseServer


class CrossEvaluationClusteringServer(BaseServer):
    """クロス評価と、その統計に基づく階層クラスタリングを共有する基底サーバ。"""

    def __init__(
        self,
        *args,
        linkage="connected",
        clustering_decision="distance",
        clustering_confidence=0.95,
        collect_pair_diagnostics=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.linkage = linkage
        if clustering_decision not in SUPPORTED_CLUSTERING_DECISIONS:
            choices = ", ".join(sorted(SUPPORTED_CLUSTERING_DECISIONS))
            raise ValueError(
                f"未対応のクラスタリング判定: {clustering_decision!r}。"
                f"選択肢: {choices}"
            )
        if not 0.5 < clustering_confidence < 1.0:
            raise ValueError("クラスタリング信頼水準は0.5より大きく1未満にしてください")
        self.clustering_decision = clustering_decision
        self.clustering_confidence = clustering_confidence
        self.collect_pair_diagnostics = collect_pair_diagnostics
        self.pair_prediction_diagnostics = []
        self._last_pair_prediction_diagnostics = []
        self._last_pair_distances = {}
        self._last_pair_decision_scores = {}

    def _cross_evaluate(self, model_ids, send_model_params=True, use_client_cache=False):
        """モデル対をクライアントで評価し、集約統計を返す。

        use_client_cache=True は、事前に全対象モデルが配布済みであるプロトコル専用。
        この場合はモデル本体を再送せず、クライアントの不変キャッシュを評価する。
        """
        self._last_pair_prediction_diagnostics = []
        holders = defaultdict(list)
        for c in self.clients:
            held_ids = c.get_held_model_ids()
            for mid in held_ids:
                holders[mid].append(c)

        stats_matrix = defaultdict(dict)

        for id_i in model_ids:
            params_i = self.global_models[id_i]
            for id_j in model_ids:
                target_clients = holders.get(id_j, [])
                if len(target_clients) > config.CROSS_EVAL_MAX_CLIENTS:
                    target_clients = random.sample(target_clients, config.CROSS_EVAL_MAX_CLIENTS)

                # 評価依頼と評価統計は、モデル転送とは別の軽量メッセージとして全方式で数える。
                self.comm_messages_down += len(target_clients)
                self.comm_messages_up += len(target_clients)
                if send_model_params:
                    # キャッシュを使わない評価ではモデルを各対象クライアントへ送る。
                    self.record_model_transfer(
                        "down", params_i, count=len(target_clients)
                    )

                total_n, total_S, total_SS = 0, 0.0, 0.0
                for c in target_clients:
                    if self.collect_pair_diagnostics and id_i != id_j:
                        if use_client_cache:
                            stats, diagnostic = c.evaluate_cached_model_diagnostics(
                                id_i, target_model_id=id_j
                            )
                        else:
                            stats, diagnostic = c.evaluate_model_diagnostics(
                                params_i, target_model_id=id_j
                            )
                        n, S, SS = stats
                        if diagnostic is not None:
                            record = {
                                "candidate_model_id": id_i,
                                "target_model_id": id_j,
                                **diagnostic,
                            }
                            self.pair_prediction_diagnostics.append(record)
                            self._last_pair_prediction_diagnostics.append(record)
                    elif use_client_cache:
                        n, S, SS = c.evaluate_cached_model(id_i, target_model_id=id_j)
                    else:
                        n, S, SS = c.evaluate_model(params_i, target_model_id=id_j)
                    total_n += n; total_S += S; total_SS += SS

                stats_matrix[id_i][id_j] = (total_n, total_S, total_SS)
        return stats_matrix

    def pair_diagnostic_summary(self):
        """全クロス評価で観測したモデル対の正誤相補性を集約する。"""
        records = self.pair_prediction_diagnostics
        total = sum(item["n"] for item in records)
        candidate_only = sum(item["candidate_only_correct"] for item in records)
        target_only = sum(item["target_only_correct"] for item in records)
        both_correct = sum(item["both_correct"] for item in records)
        oracle_gain = sum(
            min(item["candidate_only_correct"], item["target_only_correct"])
            for item in records
        )
        if total == 0:
            return {
                "model_pair_evaluation_count": 0,
                "model_pair_sample_count": 0,
                "model_pair_correctness_disagreement_rate": 0.0,
                "model_pair_oracle_gain_rate": 0.0,
                "model_pair_both_correct_rate": 0.0,
            }
        return {
            "model_pair_evaluation_count": len(records),
            "model_pair_sample_count": total,
            "model_pair_correctness_disagreement_rate": (
                candidate_only + target_only
            ) / total,
            # 2モデルのうち良い方に対し、標本ごとのoracle選択で得られる上限改善。
            "model_pair_oracle_gain_rate": oracle_gain / total,
            "model_pair_both_correct_rate": both_correct / total,
        }

    def perform_hierarchical_clustering(self, model_ids, stats_matrix):
        """損失ベースの距離が閾値以下のモデル対を辺とみなし、連結成分をクラスタとして返す。

        距離 dist(i,j) = max(「モデルiをjのデータで評価した際の損失悪化量」, その逆向き)。
        評価サンプル数が CLUSTER_MIN_EVAL_N 未満の対は判定しない。
        """
        cutoff = self._clustering_cutoff()
        if self.verbose:
            print(
                "Server: Clustering models "
                f"(decision={self.clustering_decision}, cutoff={cutoff:.3f})..."
            )

        pair_distances = {}
        pair_decision_scores = {}
        M = len(model_ids)

        for i in range(M):
            for j in range(i + 1, M):
                id_i, id_j = model_ids[i], model_ids[j]

                stats_ii = stats_matrix[id_i].get(id_i, (0, 0, 0))
                stats_ij = stats_matrix[id_i].get(id_j, (0, 0, 0))
                stats_jj = stats_matrix[id_j].get(id_j, (0, 0, 0))
                stats_ji = stats_matrix[id_j].get(id_i, (0, 0, 0))

                min_n = config.CLUSTER_MIN_EVAL_N
                if stats_ii[0] < min_n or stats_ij[0] < min_n or stats_jj[0] < min_n or stats_ji[0] < min_n:
                    continue

                diff_i_to_j = mean_loss(stats_ij) - mean_loss(stats_ii)
                diff_j_to_i = mean_loss(stats_ji) - mean_loss(stats_jj)
                dist = max(diff_i_to_j, diff_j_to_i)
                pair_distances[(id_i, id_j)] = dist

                if self.clustering_decision in {"confidence", "confidence_margin"}:
                    margin = (
                        self.distance_threshold
                        if self.clustering_decision == "confidence_margin"
                        else 0.0
                    )
                    score = max(
                        standardized_mean_increase(stats_ij, stats_ii, margin),
                        standardized_mean_increase(stats_ji, stats_jj, margin),
                    )
                else:
                    score = dist
                pair_decision_scores[(id_i, id_j)] = score

                if score <= cutoff:
                    if self.verbose and random.random() < 0.1:
                        print(
                            f"  MERGE candidate: {id_i}-{id_j} "
                            f"(distance={dist:.3f}, score={score:.3f})"
                        )

        self._last_pair_distances = pair_distances
        self._last_pair_decision_scores = pair_decision_scores
        return cluster_models(
            model_ids,
            pair_decision_scores,
            cutoff,
            self.linkage,
        )

    def _clustering_cutoff(self):
        """選択中の判定尺度に対応するクラスタ分割閾値を返す。"""
        if self.clustering_decision in {"confidence", "confidence_margin"}:
            return NormalDist().inv_cdf(self.clustering_confidence)
        return self.distance_threshold

    def record_clustering_diagnostics(self, t, model_ids, clusters):
        """直前の距離計算とクラスタ割当を診断履歴へ保存する。"""
        self.model_lineage.record_clustering(
            t,
            model_ids,
            self._last_pair_distances,
            clusters,
        )
