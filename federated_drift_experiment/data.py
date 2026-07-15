"""合成データ生成とコンセプトドリフトのスケジュール生成。

データセットは config.DATASET で切り替える。

blobs(2クラス分類・2次元特徴):
- concept 0 / 2: 2つのガウス塊(ラベルと中心の対応が 0 と 2 で反転)
- concept 1 / 3: 同心円(内側/外側とラベルの対応が 1 と 3 で反転)

sea(FedDrift SEA-4・2クラス分類・3次元特徴):
- 特徴 x1,x2,x3 ~ U[0,10]、x3 はノイズ特徴、label = 1 iff (x1+x2) <= 閾値
- 閾値・ノイズ率は config.SEA_THRESHOLDS / config.SEA_LABEL_NOISE

circle(FedDrift CIRCLE-2・2クラス分類・2次元特徴):
- 特徴 x1,x2 ~ U[0,1]^2、概念別の円の外側を label=1(config.CIRCLE_PARAMS)

sine(FedDrift SINE-2・2クラス分類・2次元特徴):
- 特徴 x1,x2 ~ U[0,1]^2、概念0: x2<=sin(x1) を label=1、概念1: 反転
"""
import random

import numpy as np
import torch

from . import config


def generate_data(concept_id, n_samples=1, dataset=None):
    """指定コンセプトからデータを生成する。

    dataset を省略すると config.DATASET を用いる。特徴次元 d は
    config.input_dim(dataset)。n_samples == 1 のとき (x: FloatTensor(d,),
    y: FloatTensor(1,)) を、それ以外は (X: FloatTensor(n,d), Y: FloatTensor(n,1))
    を返す。
    """
    if dataset is None:
        dataset = config.DATASET
    if concept_id not in range(config.num_concepts(dataset)):
        raise ValueError(f"Unknown concept_id {concept_id} for dataset {dataset!r}")

    if dataset == 'blobs':
        x_list, y_list = _generate_blobs(concept_id, n_samples)
    elif dataset == 'sea':
        x_list, y_list = _generate_sea(concept_id, n_samples)
    elif dataset == 'circle':
        x_list, y_list = _generate_circle(concept_id, n_samples)
    elif dataset == 'sine':
        x_list, y_list = _generate_sine(concept_id, n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if n_samples == 1:
        return torch.FloatTensor(x_list[0]), torch.FloatTensor([y_list[0]])
    else:
        return torch.FloatTensor(np.array(x_list)), torch.FloatTensor(np.array(y_list)).unsqueeze(1)


def _generate_blobs(concept_id, n_samples):
    """2次元合成データ(ガウス塊 / 同心円)。(x_list, y_list) を返す。"""
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

    return x_list, y_list


def _generate_sea(concept_id, n_samples):
    """FedDrift SEA-4 の3次元合成データ。(x_list, y_list) を返す。

    x1,x2,x3 ~ U[0,10](x3 はノイズ特徴)、label = 1 iff (x1+x2) <= 閾値
    (FedDrift 論文 appendix の定義)。確率 config.SEA_LABEL_NOISE でラベルを反転する。
    """
    theta = config.SEA_THRESHOLDS[concept_id]
    noise_prob = config.SEA_LABEL_NOISE

    x_list = []
    y_list = []
    for _ in range(n_samples):
        f = np.random.uniform(0.0, 10.0, size=3)
        label = 1.0 if (f[0] + f[1]) <= theta else 0.0
        if np.random.rand() < noise_prob:
            label = 1.0 - label
        x_list.append(f)
        y_list.append(label)

    return x_list, y_list


def _generate_circle(concept_id, n_samples):
    """FedDrift CIRCLE-2 の2次元合成データ。(x_list, y_list) を返す。

    x1,x2 ~ U[0,1]^2、概念別の円 (cx,cy,r) の外側を label=1 とする。
    """
    cx, cy, r = config.CIRCLE_PARAMS[concept_id]

    x_list = []
    y_list = []
    for _ in range(n_samples):
        f = np.random.uniform(0.0, 1.0, size=2)
        z = (f[0] - cx) ** 2 + (f[1] - cy) ** 2 - r ** 2
        label = 1.0 if z > 0 else 0.0
        x_list.append(f)
        y_list.append(label)

    return x_list, y_list


def _generate_sine(concept_id, n_samples):
    """FedDrift SINE-2 の2次元合成データ。(x_list, y_list) を返す。

    x1,x2 ~ U[0,1]^2、概念0: label = 1 iff x2 <= sin(x1)、概念1: そのラベルを反転。
    """
    x_list = []
    y_list = []
    for _ in range(n_samples):
        f = np.random.uniform(0.0, 1.0, size=2)
        below = f[1] <= np.sin(f[0])
        if concept_id == 0:
            label = 1.0 if below else 0.0
        else:
            label = 0.0 if below else 1.0
        x_list.append(f)
        y_list.append(label)

    return x_list, y_list


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
                candidates = [cid for cid in range(config.num_concepts()) if cid != curr]
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
