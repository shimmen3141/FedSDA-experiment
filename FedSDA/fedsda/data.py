"""合成データ生成とコンセプトドリフトのスケジュール生成。

コンセプト定義(2クラス分類・2次元特徴):
- concept 0 / 2: 2つのガウス塊(ラベルと中心の対応が 0 と 2 で反転)
- concept 1 / 3: 同心円(内側/外側とラベルの対応が 1 と 3 で反転)
"""
import random

import numpy as np
import torch

from . import config


def generate_data(concept_id, n_samples=1):
    """指定コンセプトからデータを生成する。

    n_samples == 1 のとき (x: FloatTensor(2,), y: FloatTensor(1,)) を返し、
    それ以外は (X: FloatTensor(n,2), Y: FloatTensor(n,1)) を返す。
    """
    x_list = []
    y_list = []

    if concept_id in [0, 2]:
        sigma = 0.6
        if concept_id == 0:
            centers = [(-2, -2), (2, 2)]
        else:
            centers = [(2, 2), (-2, -2)]

        for _ in range(n_samples):
            label = 0.0 if np.random.rand() < 0.5 else 1.0
            center = centers[int(label)]
            x = np.random.randn(2) * sigma + np.array(center)
            x_list.append(x)
            y_list.append(label)

    elif concept_id in [1, 3]:
        for _ in range(n_samples):
            label = 0.0 if np.random.rand() < 0.5 else 1.0
            is_inner = False
            if concept_id == 1:
                if label == 0.0:
                    is_inner = True
            else:
                if label == 1.0:
                    is_inner = True

            if is_inner:
                r = np.random.normal(loc=1.5, scale=0.4)
            else:
                r = np.random.normal(loc=4.5, scale=0.5)

            theta = np.random.uniform(0, 2 * np.pi)
            x = np.array([r * np.cos(theta), r * np.sin(theta)])
            x_list.append(x)
            y_list.append(label)

    if n_samples == 1:
        return torch.FloatTensor(x_list[0]), torch.FloatTensor([y_list[0]])
    else:
        return torch.FloatTensor(np.array(x_list)), torch.FloatTensor(np.array(y_list)).unsqueeze(1)


def make_concept_schedules(n_clients, total_data_points,
                           min_stable_period=None, drift_prob=None):
    """クライアントごとのコンセプト系列(長さ total_data_points)を生成する。

    直近のドリフトから min_stable_period サンプル経過後、毎サンプル確率
    drift_prob で別コンセプトへ遷移する。
    """
    if min_stable_period is None:
        min_stable_period = config.MIN_STABLE_PERIOD
    if drift_prob is None:
        drift_prob = config.DRIFT_PROB

    schedules = []
    for _ in range(n_clients):
        schedule = []
        curr = 0
        last_drift = 0
        for data_idx in range(total_data_points):
            if (data_idx - last_drift > min_stable_period) and (random.random() < drift_prob):
                candidates = [cid for cid in range(config.NUM_CONCEPTS) if cid != curr]
                curr = random.choice(candidates)
                last_drift = data_idx
            schedule.append(curr)
        schedules.append(schedule)
    return schedules


def extract_true_drift_events(schedules):
    """スケジュールからクライアントごとの真のドリフト位置(サンプルindex)を抽出する。"""
    true_drift_events = {i: [] for i in range(len(schedules))}
    for i, sched in enumerate(schedules):
        for idx in range(1, len(sched)):
            if sched[idx] != sched[idx - 1]:
                true_drift_events[i].append(idx)
    return true_drift_events


def build_data_streams(schedules):
    """スケジュールに従って全クライアントのデータストリームを事前生成する。"""
    all_client_data = []
    for schedule in schedules:
        stream = [generate_data(cid) for cid in schedule]
        all_client_data.append(stream)
    return all_client_data
