"""合成データセットの概念別サンプル生成器。"""
import numpy as np

from .. import config


def generate_blobs(concept_id, n_samples):
    """2次元合成データ（ガウス塊・同心円）を返す。"""
    x_list = []
    y_list = []
    if concept_id in (0, 2):
        sigma = 0.6
        centers = [(-2, -2), (2, 2)] if concept_id == 0 else [(2, 2), (-2, -2)]
        for _ in range(n_samples):
            label = 0.0 if np.random.rand() < 0.5 else 1.0
            x_list.append(np.random.randn(2) * sigma + np.array(centers[int(label)]))
            y_list.append(label)
    else:
        for _ in range(n_samples):
            label = 0.0 if np.random.rand() < 0.5 else 1.0
            is_inner = (concept_id == 1 and label == 0.0) or (
                concept_id == 3 and label == 1.0
            )
            if is_inner:
                radius = np.random.normal(loc=1.5, scale=0.4)
            else:
                radius = np.random.normal(loc=4.5, scale=0.5)
            theta = np.random.uniform(0, 2 * np.pi)
            x_list.append(np.array([radius * np.cos(theta), radius * np.sin(theta)]))
            y_list.append(label)
    return x_list, y_list


def generate_sea(concept_id, n_samples):
    """FedDrift SEAの3次元合成データを返す。"""
    theta = config.SEA_THRESHOLDS[concept_id]
    x_list = []
    y_list = []
    for _ in range(n_samples):
        features = np.random.uniform(0.0, 10.0, size=3)
        label = 1.0 if (features[0] + features[1]) <= theta else 0.0
        if np.random.rand() < config.SEA_LABEL_NOISE:
            label = 1.0 - label
        x_list.append(features)
        y_list.append(label)
    return x_list, y_list


def generate_circle(concept_id, n_samples):
    """FedDrift CIRCLE-2の2次元合成データを返す。"""
    cx, cy, radius = config.CIRCLE_PARAMS[concept_id]
    x_list = []
    y_list = []
    for _ in range(n_samples):
        features = np.random.uniform(0.0, 1.0, size=2)
        distance = (features[0] - cx) ** 2 + (features[1] - cy) ** 2 - radius ** 2
        x_list.append(features)
        y_list.append(1.0 if distance > 0 else 0.0)
    return x_list, y_list


def generate_sine(concept_id, n_samples):
    """FedDrift SINE-2の2次元合成データを返す。"""
    x_list = []
    y_list = []
    for _ in range(n_samples):
        features = np.random.uniform(0.0, 1.0, size=2)
        below = features[1] <= np.sin(features[0])
        if concept_id == 0:
            label = 1.0 if below else 0.0
        else:
            label = 0.0 if below else 1.0
        x_list.append(features)
        y_list.append(label)
    return x_list, y_list
