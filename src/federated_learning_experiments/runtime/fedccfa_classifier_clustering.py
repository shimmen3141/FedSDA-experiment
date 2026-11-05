"""FedCCFAの個人の分類層、診断、クラスごとの組とアンカーの整合。"""

from copy import deepcopy
from math import ceil
from random import Random

import numpy as np
import torch
from torch.nn.functional import cosine_similarity, cross_entropy

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)


def _convert_binary_classifier(classifier: ModelArchitectureClassifier) -> None:
    if classifier.class_count != 2 or classifier.binary_softmax_output:
        return
    layer = classifier.classification_layer
    layer.weight = torch.nn.Parameter(torch.cat((-layer.weight / 2, layer.weight / 2)))
    layer.bias = torch.nn.Parameter(torch.cat((-layer.bias / 2, layer.bias / 2)))
    layer.out_features = 2
    classifier.binary_softmax_output = True


def _madd(vectors: np.ndarray) -> np.ndarray:
    n = len(vectors)
    result = np.zeros((n, n))
    # 欠けたクラスによって参加者が2人以下にもなる。そのとき比較相手が無く、距離は0。
    if n <= 2:
        return result
    norms = np.linalg.norm(vectors, axis=1)
    similarities = np.divide(
        vectors @ vectors.T,
        norms[:, None] * norms[None, :],
        out=np.zeros((n, n)),
        where=(norms[:, None] * norms[None, :]) > 0,
    )
    for i in range(n):
        for j in range(i + 1, n):
            mask = np.ones(n, dtype=bool)
            mask[[i, j]] = False
            result[i, j] = result[j, i] = np.abs(
                similarities[i, mask] - similarities[j, mask]
            ).sum() / (n - 2)
    return result


def _dbscan_min_samples_one_labels(distances: np.ndarray, eps: float) -> np.ndarray:
    """距離行列を与えたDBSCAN（`min_samples=1`）のラベル。公式は`sklearn.cluster.DBSCAN`（`entities/FedCCFA.py:370-371`）。

    `min_samples=1`ではどの点も核になるので、組は「距離がeps以下の点どうしをつないだ連結成分」になる。
    sklearnと同じく、番号の小さい点から順に走査し、まだ組の無い点から新しい組の番号を付けて広げる
    （sklearnは実行時の依存に入れていないので、ここで同じ規則を書く。一致はtestでsklearnと照合する）。
    """
    count = len(distances)
    labels = np.full(count, -1, dtype=np.int64)
    next_label = 0
    for start in range(count):
        if labels[start] != -1:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            point = stack.pop()
            for neighbor in np.flatnonzero(distances[point] <= eps):
                if labels[neighbor] == -1:
                    labels[neighbor] = next_label
                    stack.append(int(neighbor))
        next_label += 1
    return labels


def _label_groups(vectors: np.ndarray, eps: float) -> tuple[tuple[int, ...], ...]:
    if len(vectors) < 2:
        return ()
    labels = _dbscan_min_samples_one_labels(_madd(vectors), eps)
    return tuple(
        tuple(int(i) for i in np.flatnonzero(labels == label))
        for label in sorted(set(labels))
        if np.sum(labels == label) > 1
    )


def _weighted_mean(values: list[torch.Tensor], counts: list[int]) -> torch.Tensor:
    result = torch.zeros_like(values[0])
    for value, count in zip(values, counts):
        result += value * count
    return result / sum(counts)


def _entropy_weight(counts: list[int], gamma: float) -> float:
    p = np.asarray([count for count in counts if count], dtype=float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum() / gamma)


def _balanced_indices(labels: torch.Tensor, rng: Random) -> list[int]:
    present = labels.unique(sorted=True).tolist()
    populations = [torch.where(labels == c)[0].tolist() for c in present]
    size = min(5, min(map(len, populations)))
    return [i for population in populations for i in rng.sample(population, size)]


def _alignment_start_round(policy: str, samples: int, interval: int, rounds: int) -> int:
    return 20 if policy == "fixed_round_20" else ceil(samples * rounds / (10 * interval))


def _alignment_loss(
    features: torch.Tensor,
    anchors: dict[int, torch.Tensor],
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    classes = sorted(anchors)
    valid = torch.tensor([int(c) in anchors for c in labels], dtype=torch.bool)
    if not valid.any():
        return features.sum() * 0
    features, labels = features[valid], labels[valid]
    # 未観測でアンカーの無いクラスはロジットから除く。ラベルを残った列に対応付ける。
    targets = torch.tensor([classes.index(int(c)) for c in labels], dtype=torch.long)
    logits = cosine_similarity(
        features[:, None, :], torch.stack([anchors[c] for c in classes])[None, :, :], dim=2
    )
    return cross_entropy(logits / temperature, targets)


class FedccfaClientState:
    """clientの窓・診断用の分類層・個人の分類層とアンカーを所有する。"""

    def __init__(self, *, classifier: ModelArchitectureClassifier, recent_window: bool) -> None:
        self.classifier = classifier
        self.initial_classifier_parameters = deepcopy(classifier.classification_layer.state_dict())
        self.diagnostic_parameters = deepcopy(self.initial_classifier_parameters)
        self.anchors: dict[int, torch.Tensor] = {}
        self.local_anchors: dict[int, torch.Tensor] = {}
        self.samples: list[ObservedTrainingSample] = []
        self.sample_indices: list[int] = []
        self.class_counts = [0] * classifier.class_count
        self.balanced_sample_indices: list[int] = []
        self.recent_window = recent_window
        self.round_records: list[tuple] = []
        self.cluster_records: list[tuple] = []
        self.round_index = 0
        self.alignment_start_round = 20
        self.stage_trace: list[str] = []
        self.completed_stage_updates = (0, 0, 0)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def append_timestep_samples(
        self,
        samples: tuple[ObservedTrainingSample, ...],
        end: int,
        timestep_sample_count: int | None = None,
    ) -> None:
        interval = timestep_sample_count or len(samples)
        if not self.recent_window and (end - len(samples)) % interval == 0:
            self.samples.clear()
            self.sample_indices.clear()
        capacity = max(interval, 200) if self.recent_window else interval
        self.samples.extend(samples)
        self.sample_indices.extend(range(end - len(samples), end))
        self.samples[:] = self.samples[-capacity:]
        self.sample_indices[:] = self.sample_indices[-capacity:]

    def train_round(
        self,
        *,
        settings,
        rng: Random,
        timestep_index: int,
        round_index: int,
        official_steps: tuple[int, int, int] | None = None,
        balanced_indices: list[int] | None = None,
    ) -> tuple[float, ...]:
        """通常は共通Adam予算。official_stepsは照合専用でrun設定からは指定できない。"""
        if not self.samples:
            return ()
        x = torch.cat([s.input_features for s in self.samples])
        y = torch.cat([s.observed_class_labels for s in self.samples]).view(-1).long()
        self.stage_trace = ["label_distribution"]
        self.class_counts = torch.bincount(y, minlength=self.classifier.class_count).tolist()
        weight = _entropy_weight(self.class_counts, settings.fedccfa_alignment_gamma)
        indices = _balanced_indices(y, rng) if balanced_indices is None else balanced_indices
        self.balanced_sample_indices = indices
        budget = settings.updates_per_sample * settings.detection_batch_sample_count
        eligible = len(self.samples) >= settings.local_training_batch_sample_count
        diagnostic_steps, personal_steps, representation_steps = (
            official_steps
            if official_steps is not None
            else (5, ceil(budget / 5) if eligible else 0, budget if eligible else 0)
        )
        active = (
            bool(self.anchors)
            and self.round_index > 0
            and self.round_index >= self.alignment_start_round
            and weight > 0
            and any(self.class_counts[c] > 0 for c in self.anchors)
        )
        classifier = self.classifier
        head = classifier.classification_layer
        personal = deepcopy(head.state_dict())
        losses: list[float] = []
        lr = settings.new_model_optimizer_settings.learning_rate
        representation = tuple(
            p
            for name, p in classifier.named_parameters()
            if not name.startswith("classification_layer.")
        )

        def optimize(stage: str, steps: int, representation_stage: bool) -> None:
            self.stage_trace.append(stage)
            for p in representation:
                p.requires_grad_(representation_stage)
            for p in head.parameters():
                p.requires_grad_(not representation_stage)
            parameters = representation if representation_stage else tuple(head.parameters())
            rate = (
                (0.01 if representation_stage else 0.1)
                if official_steps is not None
                else lr * (1 if representation_stage else 10)
            )
            optimizer = (
                torch.optim.SGD(parameters, lr=rate, momentum=0.9, weight_decay=1e-5)
                if official_steps is not None
                else torch.optim.Adam(parameters, lr=rate, amsgrad=True, weight_decay=0.001)
            )
            for _ in range(steps):
                batch = (
                    indices
                    if stage == "diagnostic"
                    else (
                        list(range(len(y)))
                        if official_steps is not None
                        else rng.sample(range(len(y)), settings.local_training_batch_sample_count)
                    )
                )
                optimizer.zero_grad()
                if representation_stage:
                    features = classifier.model_owned_feature_transform(
                        classifier.feature_extractor(x[batch])
                    )
                else:
                    with torch.no_grad():
                        features = classifier.model_owned_feature_transform(
                            classifier.feature_extractor(x[batch])
                        )
                loss = cross_entropy(head(features), y[batch])
                if representation_stage and active and weight != 0:
                    loss = loss + weight * _alignment_loss(
                        features, self.anchors, y[batch], settings.fedccfa_alignment_temperature
                    )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))

        try:
            head.load_state_dict(self.initial_classifier_parameters)
            optimize("diagnostic", diagnostic_steps, False)
            self.diagnostic_parameters = deepcopy(head.state_dict())
            # 公式は最初のroundだけ空のp_clf_paramsを復元するため、診断後の分類層から始まる。
            # 評価runは要求1.2の通り事前学習の個人の分類層を初回から使う。
            if official_steps is None or self.round_index != 0:
                head.load_state_dict(personal)
            optimize("personal_classifier", personal_steps, False)
            optimize("representation", representation_steps, True)
            self.stage_trace.append("local_anchors")
            with torch.no_grad():
                features = classifier.model_owned_feature_transform(classifier.feature_extractor(x))
                self.local_anchors = {int(c): features[y == c].mean(0) for c in y.unique()}
                self.anchors.update({c: p.clone() for c, p in self.local_anchors.items()})
        finally:
            for p in classifier.parameters():
                p.requires_grad_(True)
                p.grad = None
        self.completed_stage_updates = (diagnostic_steps, personal_steps, representation_steps)
        self.round_records.append(
            (timestep_index, round_index, int(active), weight, len(y), self.class_counts.count(0))
        )
        self.round_index += 1
        return tuple(losses)
