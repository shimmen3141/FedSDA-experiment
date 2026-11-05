"""放す時点に固定した責任度、学習の標本とコンセプト別の量を所有する。"""

from math import exp, fsum, isfinite, log
from random import Random

from torch import cat
from torch import isfinite as tensor_isfinite

from .model_training_sample_records import ObservedTrainingSample, SampledModelTrainingBatch


def calculate_training_responsibilities(
    *,
    prediction_weights: dict[int, float],
    true_class_probabilities: dict[int, float],
    fallback_model_id: int,
) -> dict[int, float]:
    """重み×真のラベルの確率を正規化する。小さい分母でも積のunderflowを避ける。"""
    if not prediction_weights or prediction_weights.keys() != true_class_probabilities.keys():
        raise ValueError("重みと確率のモデル集合は同じ非空集合が必要です。")
    if type(fallback_model_id) is not int or fallback_model_id not in prediction_weights:
        raise ValueError("分母0時のモデルは保有モデルでなければなりません。")
    for model_id in prediction_weights:
        if type(model_id) is not int:
            raise TypeError("モデルIDはbuiltin intが必要です。")
        for value in (prediction_weights[model_id], true_class_probabilities[model_id]):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError("重みと確率は非負の有限値が必要です。")
        if true_class_probabilities[model_id] > 1:
            raise ValueError("確率は1以下が必要です。")
    log_products = {
        m: log(w) + log(true_class_probabilities[m])
        for m, w in prediction_weights.items()
        if w > 0 and true_class_probabilities[m] > 0
    }
    if not log_products:
        return {m: float(m == fallback_model_id) for m in prediction_weights}
    maximum = max(log_products.values())
    products = {
        m: exp(log_products[m] - maximum) if m in log_products else 0.0 for m in prediction_weights
    }
    total = fsum(products.values())
    return {m: value / total for m, value in products.items()}


def derive_training_mass_purity(
    *, mass_by_model_and_concept: dict[int, dict[int, float]]
) -> tuple[float | None, float | None]:
    """clientの純度と、全部を1つのモデルへ入れた場合の基準。分母0は欠損。"""
    concept_totals: dict[int, float] = {}
    numerator = 0.0
    for masses in mass_by_model_and_concept.values():
        if any(not isfinite(v) or v < 0 for v in masses.values()):
            raise ValueError("学習の量は非負の有限値が必要です。")
        numerator += max(masses.values(), default=0.0)
        for concept, mass in masses.items():
            concept_totals[concept] = concept_totals.get(concept, 0.0) + mass
    total = fsum(concept_totals.values())
    if total == 0:
        return None, None
    return numerator / total, max(concept_totals.values()) / total


class ResponsibilityWeightedTrainingStore:
    """上限のない共有ストア。列の移動・削除で正規化し直さず、既存標本の新列は0。"""

    def __init__(self) -> None:
        self._samples: list[tuple[ObservedTrainingSample, dict[int, float], int | None]] = []
        self._masses: dict[int, float] = {}
        self._support: dict[int, int] = {}
        self._concept_masses: dict[int, dict[int, float]] = {}
        self.effective_model_counts: list[float] = []
        self.duplicate_sample_fractions: list[float] = []
        self.participating_model_counts: list[int] = []

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def training_masses_by_model_id(self) -> dict[int, float]:
        return dict(self._masses)

    @property
    def support_counts_by_model_id(self) -> dict[int, int]:
        return dict(self._support)

    @property
    def mass_by_model_and_concept(self) -> dict[int, dict[int, float]]:
        return {m: dict(v) for m, v in self._concept_masses.items()}

    def append(
        self,
        *,
        training_sample: ObservedTrainingSample,
        responsibilities: dict[int, float],
        observed_concept_id: int | None,
    ) -> None:
        """検証した責任度をコピーして固定し、既知の概念だけ純度の量へ加える。"""
        if type(training_sample) is not ObservedTrainingSample:
            raise TypeError("標本はObservedTrainingSampleが必要です。")
        if observed_concept_id is not None and type(observed_concept_id) is not int:
            raise TypeError("概念はbuiltin intまたはNoneが必要です。")
        if not responsibilities or any(
            type(m) is not int or type(q) not in (int, float) or not isfinite(q) or q < 0
            for m, q in responsibilities.items()
        ):
            raise ValueError("責任度は整数IDと非負の有限値が必要です。")
        if abs(fsum(responsibilities.values()) - 1.0) > 1e-12:
            raise ValueError("追加時の責任度の和は1が必要です。")
        q = dict(responsibilities)
        self._samples.append((training_sample, q, observed_concept_id))
        self.effective_model_counts.append(1.0 / fsum(v * v for v in q.values()))
        self._add_masses(q, observed_concept_id)

    def _add_masses(self, q: dict[int, float], concept: int | None) -> None:
        for m, v in q.items():
            self._masses[m] = self._masses.get(m, 0.0) + v
            self._support[m] = self._support.get(m, 0) + int(v > 0)
            concepts = self._concept_masses.setdefault(m, {})
            if concept is not None:
                concepts[concept] = concepts.get(concept, 0.0) + v

    def remap(self, *, model_id_mapping: dict[int, int]) -> None:
        """一回対応で列を足し合わせ、標本順を保つ。IDの登録にも使う。"""
        if type(model_id_mapping) is not dict or any(
            type(k) is not int or type(v) is not int for k, v in model_id_mapping.items()
        ):
            raise TypeError("対応表はbuiltin int間のdictが必要です。")
        samples = []
        for sample, q, concept in self._samples:
            mapped: dict[int, float] = {}
            for m, value in q.items():
                destination = model_id_mapping.get(m, m)
                mapped[destination] = mapped.get(destination, 0.0) + value
            samples.append((sample, mapped, concept))
        self._samples = samples
        self._rebuild_masses()

    def discard(self, *, model_id: int) -> None:
        """列を捨て、正の列が残らない標本を除く。残る列は正規化しない。"""
        if type(model_id) is not int:
            raise TypeError("モデルIDはbuiltin intが必要です。")
        samples = []
        for sample, q, concept in self._samples:
            remaining = {m: v for m, v in q.items() if m != model_id}
            if any(v > 0 for v in remaining.values()):
                samples.append((sample, remaining, concept))
        self._samples = samples
        self._rebuild_masses()

    def _rebuild_masses(self) -> None:
        self._masses = {}
        self._support = {}
        self._concept_masses = {}
        for _, q, concept in self._samples:
            self._add_masses(q, concept)

    def sample_batches(
        self,
        *,
        held_model_ids: frozenset[int],
        batch_sample_count: int,
        python_random_generator: Random,
    ) -> tuple[SampledModelTrainingBatch, ...]:
        """正の責任度の標本数で参加を決め、借用Randomで重み付きの復元抽出をする。"""
        if type(batch_sample_count) is not int or batch_sample_count < 1:
            raise ValueError("batchの件数は正のbuiltin intが必要です。")
        if (
            type(python_random_generator) is not Random
            or type(held_model_ids) is not frozenset
            or any(type(m) is not int for m in held_model_ids)
        ):
            raise TypeError("保有IDのfrozensetとrandom.Randomが必要です。")
        state = python_random_generator.getstate()
        batches = []
        duplicates = []
        try:
            for m in self._masses:
                if m not in held_model_ids or self._support[m] < batch_sample_count:
                    continue
                weights = [q.get(m, 0.0) for _, q, _ in self._samples]
                indices = python_random_generator.choices(
                    range(len(self._samples)), weights=weights, k=batch_sample_count
                )
                samples = [self._samples[i][0] for i in indices]
                batch = SampledModelTrainingBatch(
                    model_id=m,
                    input_features=cat([s.input_features for s in samples]),
                    observed_class_labels=cat([s.observed_class_labels for s in samples]),
                )
                if (
                    not tensor_isfinite(batch.input_features).all()
                    or not tensor_isfinite(batch.observed_class_labels).all()
                ):
                    raise ValueError("抽出した標本は有限値が必要です。")
                batches.append(batch)
                duplicates.append(1 - len(set(indices)) / batch_sample_count)
        except BaseException:
            python_random_generator.setstate(state)
            raise
        self.duplicate_sample_fractions.extend(duplicates)
        if batches:
            self.participating_model_counts.append(len(batches))
        return tuple(batches)


def derive_run_training_mass_purity(
    *, client_masses: tuple[dict[int, dict[int, float]], ...]
) -> tuple[float | None, float | None, int]:
    """分母0のclientを除いた等しい重みの平均と、除いたclient数。全欠損ならrunも欠損。"""
    purities = [
        derive_training_mass_purity(mass_by_model_and_concept=masses) for masses in client_masses
    ]
    valid = [
        (purity, baseline)
        for purity, baseline in purities
        if purity is not None and baseline is not None
    ]
    if not valid:
        return None, None, len(purities)
    return (
        fsum(p for p, _ in valid) / len(valid),
        fsum(b for _, b in valid) / len(valid),
        len(purities) - len(valid),
    )
