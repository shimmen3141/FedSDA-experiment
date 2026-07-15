"""FedSDA・FedDriftのサーバで共有するモデルクラスタリング戦略。"""

from collections import deque


SUPPORTED_LINKAGES = frozenset({"connected", "complete"})


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
