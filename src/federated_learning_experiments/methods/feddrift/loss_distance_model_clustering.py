"""FedDriftの損失距離によるモデル分割。"""

from math import isfinite


def _loss_increase_distances(
    *,
    model_ids: tuple[int, ...],
    loss_sums_by_model_pair: dict[tuple[int, int], tuple[int, float, float]],
    minimum_evaluated_sample_count: int,
) -> dict[tuple[int, int], float]:
    if type(model_ids) is not tuple or type(loss_sums_by_model_pair) is not dict:
        raise TypeError("ID列はtuple、評価統計はdictが必要です。")
    if (
        any(type(mid) is not int for mid in model_ids)
        or type(minimum_evaluated_sample_count) is not int
    ):
        raise TypeError("IDと最低件数はbuiltin intが必要です。")
    if (
        any(mid < 0 for mid in model_ids)
        or len(set(model_ids)) != len(model_ids)
        or minimum_evaluated_sample_count < 1
    ):
        raise ValueError("非負で一意なIDと正の最低件数が必要です。")
    for pair, statistics in loss_sums_by_model_pair.items():
        if type(pair) is not tuple or len(pair) != 2 or any(type(mid) is not int for mid in pair):
            raise TypeError("評価対は2個のbuiltin intのtupleが必要です。")
        if any(mid not in model_ids for mid in pair):
            raise ValueError("評価対は対象IDに属することが必要です。")
        if type(statistics) is not tuple or len(statistics) != 3:
            raise TypeError("損失統計は件数/和/二乗和のtupleが必要です。")
        count, total, squares = statistics
        if type(count) is not int or type(total) is not float or type(squares) is not float:
            raise TypeError("損失統計の型が不正です。")
        if count < 0 or not isfinite(total) or not isfinite(squares) or total < 0 or squares < 0:
            raise ValueError("損失統計は非負有限値が必要です。")
    result = {}
    ids = sorted(model_ids)
    for pos, left in enumerate(ids):
        for right in ids[pos + 1 :]:
            ll, lr, rr, rl = tuple(
                loss_sums_by_model_pair.get(pair, (0, 0.0, 0.0))
                for pair in ((left, left), (left, right), (right, right), (right, left))
            )
            if min(ll[0], lr[0], rr[0], rl[0]) < minimum_evaluated_sample_count:
                continue
            result[(left, right)] = max(
                lr[1] / lr[0] - ll[1] / ll[0], rl[1] / rl[0] - rr[1] / rr[0]
            )
    return result


def cluster_feddrift_model_ids(
    *,
    model_ids: tuple[int, ...],
    loss_sums_by_model_pair: dict[tuple[int, int], tuple[int, float, float]],
    minimum_evaluated_sample_count: int,
    maximum_same_cluster_loss_increase: float,
    cluster_linkage: str,
) -> tuple[tuple[int, ...], ...]:
    """件数条件を満たす4損失から距離を作り、指定linkageで分割する。"""
    if type(maximum_same_cluster_loss_increase) is not float or type(cluster_linkage) is not str:
        raise TypeError("距離閾値はfloat、linkageはstrが必要です。")
    if not isfinite(maximum_same_cluster_loss_increase) or cluster_linkage not in (
        "connected",
        "average",
        "complete",
    ):
        raise ValueError("距離閾値またはlinkageが不正です。")
    distances = _loss_increase_distances(
        model_ids=model_ids,
        loss_sums_by_model_pair=loss_sums_by_model_pair,
        minimum_evaluated_sample_count=minimum_evaluated_sample_count,
    )
    ids = sorted(model_ids)
    if cluster_linkage == "connected":
        remaining = set(ids)
        result = []
        for start in ids:
            if start not in remaining:
                continue
            remaining.remove(start)
            component, queue = [start], [start]
            while queue:
                current = queue.pop(0)
                for neighbor in sorted(remaining):
                    distance = distances.get((min(current, neighbor), max(current, neighbor)))
                    if distance is not None and distance <= maximum_same_cluster_loss_increase:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            result.append(tuple(sorted(component)))
        return tuple(result)
    clusters: list[tuple[int, ...]] = [(mid,) for mid in ids]
    while True:
        best = None
        for left_pos, left in enumerate(clusters):
            for right_pos in range(left_pos + 1, len(clusters)):
                right = clusters[right_pos]
                values = [distances.get((min(a, b), max(a, b))) for a in left for b in right]
                if any(value is None for value in values):
                    continue
                present = [value for value in values if value is not None]
                score = (
                    sum(present) / len(present) if cluster_linkage == "average" else max(present)
                )
                candidate = (score, left, right, left_pos, right_pos)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] > maximum_same_cluster_loss_increase:
            break
        _, left, right, left_pos, right_pos = best
        clusters = [c for pos, c in enumerate(clusters) if pos not in (left_pos, right_pos)]
        clusters.append(tuple(sorted(left + right)))
        clusters.sort()
    return tuple(clusters)
