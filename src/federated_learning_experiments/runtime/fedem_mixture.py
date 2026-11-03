"""FedEMのclient内のE-stepと、時刻をまたぐ予測の混合。"""

import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy

from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    convert_model_outputs_to_prediction_probabilities,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)


def _fedem_responsibilities(losses: torch.Tensor, mixture_weights: torch.Tensor) -> torch.Tensor:
    """公式MixtureClient.update_sample_weights。行が標本、列が成分。"""
    return torch.softmax(mixture_weights.log() - losses, dim=1)


def _fedem_weighted_loss(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """公式Learner.fit_batch。重みの和で割らず、ミニバッチ件数で割る。"""
    return (losses.view(-1) @ weights.view(-1)) / losses.numel()


def _fedem_per_sample_loss(
    outputs: torch.Tensor, labels: torch.Tensor, class_count: int
) -> torch.Tensor:
    return (
        binary_cross_entropy(outputs, labels, reduction="none").view(-1)
        if class_count == 2
        else cross_entropy(outputs, labels.view(-1).long(), reduction="none")
    )


class FedemClientMixture:
    """π・qと最後のroundの記録を保持する。モデルと学習標本のownerは借用する。"""

    def __init__(
        self,
        *,
        registry: HeldModelTrainingStateRegistry,
        training_samples: ModelTrainingSampleStore,
        current_timestep_window: bool,
    ) -> None:
        self._registry = registry
        self._training_samples = training_samples
        self._current_timestep_window = current_timestep_window
        self.model_ids = tuple(
            s.model_id for s in registry.snapshot_ordered_held_model_training_states()
        )
        self.mixture_weights = torch.full((len(self.model_ids),), 1 / len(self.model_ids))
        self.responsibilities = torch.empty((0, len(self.model_ids)))
        self.sample_indices: list[int] = []
        self._samples: list[ObservedTrainingSample] = []
        self._current_start = 0
        self._timestep_index = -1
        self.mixture_records: list[tuple[int, int, float]] = []
        self.responsibility_records: list[tuple[int, int, int, float]] = []

    @torch.no_grad()
    def predict(self, input_features: torch.Tensor) -> torch.Tensor:
        """ラベルを参照せず、直前のπで各成分の確率を混ぜる。"""
        states = self._registry.snapshot_ordered_held_model_training_states()
        probabilities = convert_model_outputs_to_prediction_probabilities(
            model_outputs_by_model_id={s.model_id: s.classifier(input_features) for s in states},
            class_count=states[0].classifier.class_count,
        )
        return (
            torch.stack([probabilities[mid] for mid in self.model_ids])
            .mul(self.mixture_weights[:, None, None])
            .sum(0)
            .clamp(0, 1)
        )

    def append_timestep_samples(
        self,
        *,
        samples: tuple[ObservedTrainingSample, ...],
        processed_sample_count: int,
    ) -> None:
        """ラベル観測後の標本を全成分へ渡す。historyでは以前の標本を残す。"""
        self._timestep_index += 1
        if self._current_timestep_window:
            self._samples.clear()
            self.sample_indices.clear()
            self._training_samples.clear_model_training_samples()
        self._current_start = len(self._samples)
        self._samples.extend(samples)
        self.sample_indices.extend(
            range(processed_sample_count - len(samples), processed_sample_count)
        )
        for mid in self.model_ids:
            self._training_samples.append_model_training_samples(
                model_id=mid, training_samples=samples
            )

    @torch.no_grad()
    def expectation_step(self) -> dict[int, torch.Tensor]:
        """配布済みモデルで全学習標本のqを作り、今の時刻のqの平均でπを更新する。"""
        if not self._samples:
            return {}
        states = {
            s.model_id: s for s in self._registry.snapshot_ordered_held_model_training_states()
        }
        # historyでも一度に巨大なforwardを作らない。全標本をちょうど一度評価する。
        chunks = []
        for start in range(0, len(self._samples), 256):
            samples = self._samples[start : start + 256]
            features = torch.cat([s.input_features for s in samples])
            labels = torch.cat([s.observed_class_labels for s in samples])
            losses = torch.stack(
                [
                    _fedem_per_sample_loss(
                        states[mid].classifier(features), labels, states[mid].classifier.class_count
                    )
                    for mid in self.model_ids
                ],
                dim=1,
            )
            chunks.append(_fedem_responsibilities(losses, self.mixture_weights))
        self.responsibilities = torch.cat(chunks)
        current = self.responsibilities[self._current_start :]
        self.mixture_weights = current.mean(0)
        # 同じ時刻の複数roundでは最後のE-stepだけを保存する。
        t = self._timestep_index
        self.mixture_records[:] = [r for r in self.mixture_records if r[0] != t]
        self.mixture_records.extend(
            (t, mid, float(w)) for mid, w in zip(self.model_ids, self.mixture_weights)
        )
        self.responsibility_records[:] = [r for r in self.responsibility_records if r[0] != t]
        maxima, columns = current.max(1)
        self.responsibility_records.extend(
            (t, index, self.model_ids[int(column)], float(value))
            for index, column, value in zip(
                self.sample_indices[self._current_start :], columns, maxima
            )
        )
        return {mid: self.responsibilities[:, column] for column, mid in enumerate(self.model_ids)}
