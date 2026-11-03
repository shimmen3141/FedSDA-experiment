"""CFLの完全連結2分割と、run全体で共有する閾値の更新。"""

from dataclasses import dataclass
from heapq import heappush, heappushpop

import numpy as np


def bipartition_complete_linkage(similarities: np.ndarray) -> np.ndarray:
    """SciPyのcompleteの近隣鎖・安定ソートとsklearnの木の切り方を再現する。"""
    n = len(similarities)
    if n < 2 or similarities.shape != (n, n) or not np.isfinite(similarities).all():
        raise ValueError("有限の正方行列と2件以上のclientが必要です。")
    distances = -np.array(similarities, dtype=np.float64, copy=True)
    # scipy.linkageのcondensed入力は上三角だけを読む。
    distances = np.triu(distances, 1) + np.triu(distances, 1).T
    sizes = np.ones(n, dtype=int)
    chain: list[int] = []
    merges = []
    for _ in range(n - 1):
        if not chain:
            chain.append(next(i for i in range(n) if sizes[i]))
        while True:
            x = chain[-1]
            y = chain[-2] if len(chain) > 1 else -1
            current = distances[x, y] if y >= 0 else float("inf")
            for i in range(n):
                if sizes[i] and i != x and distances[x, i] < current:
                    current, y = distances[x, i], i
            if len(chain) > 1 and y == chain[-2]:
                break
            chain.append(y)
        del chain[-2:]
        x, y = sorted((x, y))
        merges.append((x, y, current))
        sizes[y] += sizes[x]
        sizes[x] = 0
        for i in range(n):
            if sizes[i] and i != y:
                distances[i, y] = distances[y, i] = max(distances[i, x], distances[i, y])
    # 安定な距離順に並べ、葉の代表を内部節点IDに置き換える。
    parents = list(range(2 * n - 1))
    children: list[tuple[int, int]] = []

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    for index, (x, y, _) in enumerate(sorted(merges, key=lambda row: row[2])):
        left, right = sorted((root(x), root(y)))
        children.append((left, right))
        parents[left] = parents[right] = n + index
    # sklearn._hc_cut: 最大IDの節点をheapから展開し、heapの順にラベルを付ける。
    nodes = [-(2 * n - 2)]
    left, right = children[-nodes[0] - n]
    heappush(nodes, -left)
    heappushpop(nodes, -right)
    labels = np.empty(n, dtype=np.int64)
    for label, node in enumerate(nodes):
        pending = [-node]
        while pending:
            i = pending.pop()
            if i < n:
                labels[i] = label
            else:
                pending.extend(children[i - n])
    return labels


@dataclass
class CflClusterSplittingState:
    """モデルID順に判定し、観測最大ノルムと次のIDをrun全体で保持する。"""

    cfl_norm: float = 0.0
    eps1: float = 0.0
    eps2: float = 10000.0
    next_model_id: int = 1

    def check(
        self,
        *,
        updates_by_model: dict[int, dict[int, np.ndarray]],
        gamma: float,
        maximum_model_count: int,
    ) -> list[tuple]:
        """標本があるclientだけを受け取り、判定前後とラベルを返す。"""
        records = []
        for mid, reports in sorted(updates_by_model.items()):
            if not reports:
                continue
            clients = tuple(reports)
            updates = list(reports.values())
            maximum = float(np.max([np.linalg.norm(u) for u in updates]))
            mean_value = np.linalg.norm(np.mean(np.stack(updates), axis=0))
            mean = float(mean_value)
            before = (self.cfl_norm, self.eps1, self.eps2)
            checked, split, alpha, new_id, one, two = False, False, float("nan"), -1, (), ()
            if mean > self.cfl_norm:
                # 公式と同じく、更新のNumPy dtypeで割り算・掛け算してから保存する。
                eps1_value = mean_value / 10.0
                self.cfl_norm, self.eps1, self.eps2 = mean, float(eps1_value), float(6 * eps1_value)
            elif mean < self.eps1 and maximum > self.eps2:
                checked = True
                similarities = np.zeros((len(updates), len(updates)))
                for i, u in enumerate(updates):
                    for j, v in enumerate(updates):
                        similarities[i, j] = np.dot(u, v) / (
                            np.linalg.norm(u) * np.linalg.norm(v) + 1e-12
                        )
                labels = bipartition_complete_linkage(similarities)
                one = tuple(clients[i] for i in np.flatnonzero(labels == 0))
                two = tuple(clients[i] for i in np.flatnonzero(labels == 1))
                alpha = float(
                    max(
                        similarities[i, j]
                        for i in np.flatnonzero(labels == 0)
                        for j in np.flatnonzero(labels == 1)
                    )
                )
                if ((1 - alpha) / 2.0) ** 0.5 > gamma and self.next_model_id < maximum_model_count:
                    split, new_id = True, self.next_model_id
                    self.next_model_id += 1
            records.append(
                (
                    mid,
                    clients,
                    mean,
                    maximum,
                    before,
                    (self.cfl_norm, self.eps1, self.eps2),
                    checked,
                    alpha,
                    split,
                    new_id,
                    one,
                    two,
                )
            )
        return records
