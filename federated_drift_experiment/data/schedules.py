"""ランダム系列とFedDrift論文固定系列の生成。"""
import random

import numpy as np

from .. import config


# FedDrift元実装 data/changepoints/A.cp, B.cp の先頭10時点。
# 行が時点、列がクライアント。11行目は最終評価用なので除外する。
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
        current = 0
        last_drift = 0
        for data_idx in range(total_data_points):
            if (data_idx - last_drift > min_stable_period
                    and random.random() < drift_prob):
                current = random.choice(
                    [concept_id for concept_id in range(num_concepts)
                     if concept_id != current]
                )
                last_drift = data_idx
            schedule.append(current)
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
    """概念数に対応するFedDrift固定パターンを展開する。"""
    patterns = {2: FEDDRIFT_2CONCEPT_PATTERN, 4: FEDDRIFT_4CONCEPT_PATTERN}
    try:
        pattern = patterns[num_concepts]
    except KeyError as exc:
        raise ValueError(
            "feddrift_fixed schedule supports datasets with 2 or 4 concepts, "
            f"got {num_concepts}"
        ) from exc
    return expand_feddrift_pattern(pattern, n_clients, total_data_points)


def make_concept_schedules(n_clients, total_data_points,
                           min_stable_period=None, drift_prob=None,
                           schedule_type=None, dataset=None):
    """設定に応じてクライアントごとの概念系列を生成する。"""
    min_stable_period = (config.MIN_STABLE_PERIOD if min_stable_period is None
                         else min_stable_period)
    drift_prob = config.DRIFT_PROB if drift_prob is None else drift_prob
    schedule_type = config.CONCEPT_SCHEDULE if schedule_type is None else schedule_type
    if schedule_type not in config.CONCEPT_SCHEDULES:
        raise ValueError(f"Unknown concept schedule: {schedule_type!r}")
    num_concepts = config.num_concepts(dataset)
    if schedule_type == "feddrift_fixed":
        return make_feddrift_fixed_schedules(
            n_clients, total_data_points, num_concepts
        )
    return make_random_schedules(
        n_clients, total_data_points, num_concepts, min_stable_period, drift_prob
    )


def extract_true_drift_events(schedules):
    """概念系列からクライアントごとの真のドリフト位置を抽出する。"""
    events = {client_id: [] for client_id in range(len(schedules))}
    for client_id, schedule in enumerate(schedules):
        for index in range(1, len(schedule)):
            if schedule[index] != schedule[index - 1]:
                events[client_id].append(index)
    return events
