"""FedSDA・FedDriftのサーバで共有するモデルクラスタリング戦略。"""

import math
from collections import deque
from dataclasses import dataclass
from statistics import NormalDist


SUPPORTED_LINKAGES = frozenset({"connected", "complete"})
SUPPORTED_CLUSTERING_DECISIONS = frozenset(
    {
        "distance", "confidence", "confidence_margin", "functional",
        "oracle_concept",
    }
)


@dataclass
class FunctionalPairStats:
    """同一標本上で観測したモデル対の固有正解数を集約する。"""

    sample_count: int = 0
    left_only_correct: int = 0
    right_only_correct: int = 0

    def add(self, sample_count, left_only_correct, right_only_correct):
        self.sample_count += int(sample_count)
        self.left_only_correct += int(left_only_correct)
        self.right_only_correct += int(right_only_correct)

    def complementarity_distance(self):
        """片方だけが正解する割合の大きい側を機能的距離として返す。"""
        if self.sample_count <= 0:
            raise ValueError("機能的距離の計算には1件以上の評価が必要です")
        return max(
            self.left_only_correct,
            self.right_only_correct,
        ) / self.sample_count


def mean_loss(stats):
    """``(n, sum, sum_sq)`` 形式の評価統計から平均損失を返す。"""
    n, total, _ = stats
    if n <= 0:
        raise ValueError("平均損失の計算には1件以上の評価が必要です")
    return total / n


def standardized_mean_increase(target_stats, reference_stats, margin=0.0):
    """2つの独立標本間における平均損失増加の標準化量を返す。

    クロス評価では同じモデルを異なるクライアント集合で評価するため、対応のない
    Welch型の標準誤差を使う。分散が0の場合も決定的に扱う。
    """
    target_n, target_sum, target_sum_sq = target_stats
    reference_n, reference_sum, reference_sum_sq = reference_stats
    if target_n < 2 or reference_n < 2:
        raise ValueError("標準化には各標本2件以上の評価が必要です")

    target_mean = target_sum / target_n
    reference_mean = reference_sum / reference_n
    increase = target_mean - reference_mean - margin

    target_var = max(
        (target_sum_sq - target_sum * target_sum / target_n) / (target_n - 1),
        0.0,
    )
    reference_var = max(
        (reference_sum_sq - reference_sum * reference_sum / reference_n)
        / (reference_n - 1),
        0.0,
    )
    standard_error = math.sqrt(
        target_var / target_n + reference_var / reference_n
    )
    if math.isclose(standard_error, 0.0, abs_tol=1e-12):
        return math.inf if increase > 0.0 else 0.0
    return increase / standard_error


def paired_mean_upper_bound(difference_stats, confidence=0.95):
    """対応のある損失差の平均に対する片側正規近似上限を返す。

    ``difference_stats`` は ``candidate_loss - reference_loss`` の
    ``(n, sum, sum_sq)`` である。正の値ほど候補モデルが悪い。
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidenceは0.5より大きく1未満にしてください")
    n, total, total_sq = difference_stats
    if n < 2:
        raise ValueError("非劣性上限の計算には2件以上の対応標本が必要です")
    mean = total / n
    variance = max((total_sq - total * total / n) / (n - 1), 0.0)
    standard_error = math.sqrt(variance / n)
    return mean + NormalDist().inv_cdf(confidence) * standard_error


def cluster_models(model_ids, pair_distances, threshold, linkage):
    """``threshold`` で切った決定的なクラスタを返す。

    ``connected`` は従来実装を維持し、閾値以下の辺から連結成分を作る
    （single-linkageを閾値で切ることと同等）。``complete`` はFedDriftの
    max-linkage規則を実装する。
    """
    if linkage not in SUPPORTED_LINKAGES:
        choices = ", ".join(sorted(SUPPORTED_LINKAGES))
        raise ValueError(f"未対応のクラスタリング戦略: {linkage!r}。選択肢: {choices}")

    ids = sorted(model_ids)
    if linkage == "connected":
        return _connected_components(ids, pair_distances, threshold)
    return _complete_linkage(ids, pair_distances, threshold)


def _connected_components(model_ids, pair_distances, threshold):
    adjacency = {mid: set() for mid in model_ids}
    for pos, left in enumerate(model_ids):
        for right in model_ids[pos + 1:]:
            distance = pair_distances.get((left, right))
            if distance is not None and distance <= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited = set()
    clusters = []
    for start in model_ids:
        if start in visited:
            continue
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(sorted(component))
    return clusters


def _complete_linkage(model_ids, pair_distances, threshold):
    clusters = [(mid,) for mid in model_ids]

    while True:
        best = None
        for left_pos, left in enumerate(clusters):
            for right_pos in range(left_pos + 1, len(clusters)):
                right = clusters[right_pos]
                distances = [
                    pair_distances.get(tuple(sorted((a, b))))
                    for a in left for b in right
                ]
                if any(distance is None for distance in distances):
                    continue
                cluster_distance = max(distances)
                candidate = (cluster_distance, left, right, left_pos, right_pos)
                if best is None or candidate < best:
                    best = candidate

        if best is None or best[0] > threshold:
            break

        _, left, right, left_pos, right_pos = best
        merged = tuple(sorted(left + right))
        clusters = [
            cluster for pos, cluster in enumerate(clusters)
            if pos not in (left_pos, right_pos)
        ]
        clusters.append(merged)
        clusters.sort()

    return [list(cluster) for cluster in clusters]
