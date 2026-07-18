"""コンセプト系列の生成戦略。"""
import random

import numpy as np


# FedDrift元実装 data/changepoints/A.cp, B.cp の先頭10時点。
# 行が時点、列がクライアントを表す。末尾の11行目は最終評価用なので除外する。
FEDDRIFT_2CONCEPT_PATTERN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
], dtype=np.int64)

FEDDRIFT_4CONCEPT_PATTERN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 2, 2, 2, 0, 0, 0, 0],
    [2, 2, 1, 2, 2, 2, 2, 1, 0, 0],
    [2, 2, 2, 1, 2, 2, 2, 1, 3, 0],
    [2, 3, 2, 1, 2, 3, 3, 1, 3, 0],
    [3, 3, 2, 1, 1, 3, 3, 2, 1, 0],
    [3, 0, 3, 3, 1, 3, 3, 2, 1, 3],
    [0, 0, 3, 3, 3, 1, 3, 2, 2, 3],
    [0, 0, 0, 3, 3, 1, 1, 3, 2, 2],
], dtype=np.int64)


def make_random_schedules(n_clients, total_data_points, num_concepts,
                          min_stable_period, drift_prob):
    schedules = []
    for _ in range(n_clients):
        schedule = []
        curr = 0
        last_drift = 0
        for data_idx in range(total_data_points):
            if (data_idx - last_drift > min_stable_period
                    and random.random() < drift_prob):
                curr = random.choice([cid for cid in range(num_concepts) if cid != curr])
                last_drift = data_idx
            schedule.append(curr)
        schedules.append(schedule)
    return schedules


def expand_feddrift_pattern(pattern, n_clients, total_data_points):
    """論文の10時点パターンをサンプル単位の系列へ展開する。"""
    if not 1 <= n_clients <= pattern.shape[1]:
        raise ValueError(
            f"FedDrift fixed schedule supports 1..{pattern.shape[1]} clients, got {n_clients}"
        )
    if total_data_points < 1:
        raise ValueError("total_data_points must be positive")

    # 5000サンプルでは論文どおり各時点500サンプル。端数は前方の時点へ配る。
    block_lengths = [len(block) for block in np.array_split(
        np.arange(total_data_points), pattern.shape[0]
    )]
    schedules = []
    for client_id in range(n_clients):
        schedule = []
        for time_id, block_length in enumerate(block_lengths):
            schedule.extend([int(pattern[time_id, client_id])] * block_length)
        schedules.append(schedule)
    return schedules


def make_feddrift_fixed_schedules(n_clients, total_data_points, num_concepts):
    """概念数に対応するFedDrift固定パターンをサンプル系列へ展開する。"""
    patterns = {
        2: FEDDRIFT_2CONCEPT_PATTERN,
        4: FEDDRIFT_4CONCEPT_PATTERN,
    }
    try:
        pattern = patterns[num_concepts]
    except KeyError as exc:
        raise ValueError(
            "feddrift_fixed schedule supports datasets with 2 or 4 concepts, "
            f"got {num_concepts}"
        ) from exc
    return expand_feddrift_pattern(pattern, n_clients, total_data_points)
