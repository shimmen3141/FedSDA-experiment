"""1標本の処理（製品の`process_observed_sample`の木の版。同期・集約を持たない）。

順は製品と同じ: 予測→重みの更新→監視→保留へ追加→警報の判定→割り当てと学習。
- 予測: 保有モデル（作った順）の平滑化した確率を、Fixed-Shareのラベル観測前の重みで混ぜる。
- 監視: 最初のW件（ウォームアップ）は監視の部品を呼ばない。W件目の処理の最後に、現在のモデルの損失の統計から
  選んだ基準値で監視の部品を作る。
- 警報: 検出器の合図があれば、保留の列でMDLの選択をし、min(n, k)が最小の変化区間より短ければ待つ（検出器を
  始め直さない）。警報なら、新しいモデル（選ばれた学習者と同じ作り方）→前のn−k件を現在のモデルへ→後のk件を
  帰属先へ→現在のモデルを替え、保留を空にし、帰属先の統計の基準値で監視を始め直す。
- 割り当て: そのモデルの学習前の監視用の損失を統計へ記録してから、その標本で1回学習する。
"""

from dataclasses import dataclass
from math import nan

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_loss_monitoring_baseline_mean_loss,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)
from single_stream_hard_assignment_soft_prediction.alarm_interval_assignment import (
    build_selected_new_concept_model,
    select_alarm_interval_assignment,
)
from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
    HoeffdingTreeConceptModel,
)
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
    SingleStreamHardAssignmentSoftPredictionSettings,
)

# 適応の結果: 現在のモデルのまま、保有モデルへ、新しいモデルへ、変化区間が短く警報としない。
ADAPTATION_OUTCOMES = ("stay", "reuse", "new", "too_short")
_HYPOTHESIS_TO_OUTCOME = {"current_model": "stay", "held_model": "reuse", "new_model": "new"}


@dataclass(frozen=True, kw_only=True)
class SampleProcessingRecord:
    """標本ごとの記録（予測は学習・監視より前）。"""

    sample_index: int
    predicted_class_id: int
    prediction_is_correct: bool
    observed_class_id: int
    observed_concept_id: int
    current_model_id: int
    held_model_count: int
    monitored_log_e_value: float


@dataclass(frozen=True, kw_only=True)
class AlarmRecord:
    """検出器の合図ごとの記録。帰属先は、そのまま・新しいモデルなら選んだ後の現在のモデルのID。"""

    sample_index: int
    selected_hypothesis: str
    change_interval_sample_count: int
    window_sample_count: int
    destination_model_id: int
    new_model_learner_name: str
    description_length: float
    alarm_raised: bool
    adaptation_outcome: str
    window_tail_concept_ids: tuple[int, ...]


class SingleStreamHardAssignmentSoftPredictionProcessor:
    """保有モデル・重み・監視・保留・損失の統計・記録を所有し、標本を1件ずつ処理する。"""

    def __init__(
        self, *, settings: SingleStreamHardAssignmentSoftPredictionSettings, class_count: int
    ) -> None:
        if type(settings) is not SingleStreamHardAssignmentSoftPredictionSettings:
            raise TypeError(
                "settingsはSingleStreamHardAssignmentSoftPredictionSettingsが必要です。"
            )
        if type(class_count) is not int or class_count < 2:
            raise ValueError("class_countは2以上のintが必要です。")
        self.settings = settings
        self.class_count = class_count
        initial_model = HoeffdingTreeConceptModel.create_empty(
            class_count=class_count,
            probability_smoothing_weight=settings.probability_smoothing_weight,
        )
        self.held_models: list[tuple[int, HoeffdingTreeConceptModel]] = [(0, initial_model)]
        self.current_model_id = 0
        self._next_model_id = 1
        self.model_created_sample_index_by_model_id: dict[int, int] = {0: -1}
        self.fixed_share_weight_controller = FixedShareWeightController(
            prediction_combination_settings=settings.prediction_combination_settings
        )
        self.loss_statistics_store = ModelAndClassLossStatisticsStore()
        self.loss_change_monitor: OverallAndTrueClassLossMonitor | None = None
        # 保留: (標本の番号, 特徴, 観測クラス, 真のコンセプト)。真のコンセプトは記録にだけ使う。
        self._pending: list[tuple[int, dict[int, float], int, int]] = []
        self.assigned_model_id_by_sample_index: dict[int, int] = {}
        self.alarm_records: list[AlarmRecord] = []
        self.processed_sample_count = 0
        self._last_sample_index: int | None = None

    @property
    def pending_sample_indices(self) -> tuple[int, ...]:
        return tuple(sample_index for sample_index, _, _, _ in self._pending)

    def _model(self, model_id: int) -> HoeffdingTreeConceptModel:
        for held_id, model in self.held_models:
            if held_id == model_id:
                return model
        raise KeyError(model_id)

    def _baseline(self, model_id: int) -> float:
        statistics = self.loss_statistics_store.get_model_loss_statistics(model_id=model_id)
        return select_loss_monitoring_baseline_mean_loss(
            loss_moments=None if statistics is None else statistics.overall_loss_moments
        )

    def _assign(
        self,
        *,
        model_id: int,
        sample_index: int,
        features: dict[int, float],
        observed_class_id: int,
    ) -> None:
        """学習前の監視用の損失を統計へ記録してから、その標本で1回学習する。"""
        if sample_index in self.assigned_model_id_by_sample_index:
            raise RuntimeError(f"標本{sample_index}は割り当て済みです。")
        model = self._model(model_id)
        self.loss_statistics_store.record_assigned_loss(
            model_id=model_id,
            observed_loss=model.monitoring_loss(
                features=features, observed_class_id=observed_class_id
            ),
            observed_class_id=observed_class_id,
        )
        model.learn_one(features=features, observed_class_id=observed_class_id)
        self.assigned_model_id_by_sample_index[sample_index] = model_id

    def _validate_sample(
        self,
        *,
        sample_index: object,
        features: object,
        observed_class_id: object,
        observed_concept_id: object,
    ) -> None:
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError("sample_indexは0以上のintが必要です。")
        if self._last_sample_index is not None and sample_index <= self._last_sample_index:
            raise ValueError("sample_indexは増える順が必要です。")
        if type(features) is not dict:
            raise TypeError("featuresは辞書が必要です。")
        if type(observed_class_id) is not int or not 0 <= observed_class_id < self.class_count:
            raise ValueError("observed_class_idは0以上class_count未満のintが必要です。")
        if type(observed_concept_id) is not int:
            raise TypeError("observed_concept_idはintが必要です。")

    def process_sample(
        self,
        *,
        sample_index: int,
        features: dict[int, float],
        observed_class_id: int,
        observed_concept_id: int,
    ) -> SampleProcessingRecord:
        """標本1件を、予測→重みの更新→監視→保留→警報の判定→割り当て・学習の順で処理する。"""
        self._validate_sample(
            sample_index=sample_index,
            features=features,
            observed_class_id=observed_class_id,
            observed_concept_id=observed_concept_id,
        )
        settings = self.settings
        # (1) 予測。
        model_ids = tuple(model_id for model_id, _ in self.held_models)
        probabilities_by_model_id = {
            model_id: model.predict_class_probabilities(features=features)
            for model_id, model in self.held_models
        }
        weights = self.fixed_share_weight_controller.get_weights_before_label_observation(
            model_ids=model_ids
        )
        mixture = [
            sum(
                weights[model_id] * probabilities_by_model_id[model_id][class_id]
                for model_id in model_ids
            )
            for class_id in range(self.class_count)
        ]
        predicted_class_id = max(
            range(self.class_count), key=lambda class_id: (mixture[class_id], -class_id)
        )
        current_model_id_at_prediction = self.current_model_id
        held_model_count_at_prediction = len(model_ids)
        # (2) 重みの更新（予測に使ったラベル観測前の重みを渡す）。
        losses_by_model_id = {
            model_id: 1.0 - probabilities_by_model_id[model_id][observed_class_id]
            for model_id in model_ids
        }
        self.fixed_share_weight_controller.update_weights_after_loss_observation(
            observed_losses_by_model_id=losses_by_model_id,
            prediction_weights_by_model_id=dict(weights),
        )
        # (3) 監視（ウォームアップの後だけ）。
        log_e_value = nan
        drift_detected = False
        if self.loss_change_monitor is not None:
            observation = self.loss_change_monitor.observe_loss_after_label_observation(
                observed_loss=losses_by_model_id[self.current_model_id],
                observed_class_id=observed_class_id,
                sample_index=sample_index,
                current_model_baseline_loss_mean=self._baseline(self.current_model_id),
            )
            log_e_value = observation.log_e_value
            drift_detected = observation.drift_detected
        # (4) 保留へ追加。
        self._pending.append((sample_index, features, observed_class_id, observed_concept_id))
        # (5) 警報の判定と、(6)(7) 割り当て・学習。
        alarm_raised = False
        if drift_detected:
            alarm_raised = self._handle_detector_signal(sample_index=sample_index)
        if (
            not alarm_raised
            and len(self._pending) > settings.pending_assignment_buffer_capacity_samples
        ):
            released_index, released_features, released_class_id, _ = self._pending.pop(0)
            self._assign(
                model_id=self.current_model_id,
                sample_index=released_index,
                features=released_features,
                observed_class_id=released_class_id,
            )
        self.processed_sample_count += 1
        self._last_sample_index = sample_index
        # (8) ウォームアップの終わりに、監視の部品を作る。
        if (
            self.loss_change_monitor is None
            and self.processed_sample_count == settings.warm_up_sample_count
        ):
            self.loss_change_monitor = OverallAndTrueClassLossMonitor(
                loss_change_detection_settings=settings.loss_change_detection_settings,
                class_count=self.class_count,
                initial_baseline_loss_mean=self._baseline(self.current_model_id),
                maximum_retained_candidate_count=settings.loss_monitor_maximum_retained_candidate_count,
                betting_fractions=settings.loss_monitor_betting_fractions,
            )
        return SampleProcessingRecord(
            sample_index=sample_index,
            predicted_class_id=predicted_class_id,
            prediction_is_correct=predicted_class_id == observed_class_id,
            observed_class_id=observed_class_id,
            observed_concept_id=observed_concept_id,
            current_model_id=current_model_id_at_prediction,
            held_model_count=held_model_count_at_prediction,
            monitored_log_e_value=log_e_value,
        )

    def _handle_detector_signal(self, *, sample_index: int) -> bool:
        """保留の列でMDLの選択をし、警報なら割り当てて始め直す。警報としたかを返す。"""
        settings = self.settings
        window = tuple((features, class_id) for _, features, class_id, _ in self._pending)
        window_length = len(window)
        selection = select_alarm_interval_assignment(
            pending_samples=window,
            held_models=tuple(self.held_models),
            current_model_id=self.current_model_id,
            model_ids_with_loss_statistics=frozenset(
                model_id
                for model_id, _ in self.held_models
                if self.loss_statistics_store.get_model_loss_statistics(model_id=model_id)
                is not None
            ),
            switching_probability=settings.switching_probability,
            minimum_change_interval_sample_count=settings.minimum_change_interval_sample_count,
        )
        k = selection.change_interval_sample_count
        tail_concept_ids = tuple(
            concept_id for _, _, _, concept_id in self._pending[window_length - k :]
        )
        if min(window_length, k) < settings.minimum_change_interval_sample_count:
            self.alarm_records.append(
                AlarmRecord(
                    sample_index=sample_index,
                    selected_hypothesis=selection.selected_hypothesis,
                    change_interval_sample_count=k,
                    window_sample_count=window_length,
                    destination_model_id=self.current_model_id,
                    new_model_learner_name=selection.new_model_learner_name,
                    description_length=selection.description_length,
                    alarm_raised=False,
                    adaptation_outcome="too_short",
                    window_tail_concept_ids=tail_concept_ids,
                )
            )
            return False
        previous_model_id = self.current_model_id
        before = self._pending[: window_length - k]
        after = self._pending[window_length - k :]
        if selection.selected_hypothesis == "new_model":
            # (a) 新しいモデルを、測った学習者と同じ作り方で作り、後のk件を新しい順に「統計へ記録→学習」する。
            new_model, recorded_losses = build_selected_new_concept_model(
                learner_name=selection.new_model_learner_name,
                current_model=self._model(previous_model_id),
                pending_samples=window,
                change_interval_sample_count=k,
            )
            destination_model_id = self._next_model_id
            self._next_model_id += 1
            for (pending_index, _, class_id, _), loss in zip(
                reversed(after), recorded_losses, strict=True
            ):
                self.loss_statistics_store.record_assigned_loss(
                    model_id=destination_model_id, observed_loss=loss, observed_class_id=class_id
                )
                self.assigned_model_id_by_sample_index[pending_index] = destination_model_id
            self.held_models.append((destination_model_id, new_model))
            self.model_created_sample_index_by_model_id[destination_model_id] = sample_index
        elif selection.selected_hypothesis == "held_model":
            destination_model_id = int(selection.destination_model_id)
        else:
            destination_model_id = previous_model_id
        # (b) 前のn−k件を、古い順に現在のモデルへ。
        for pending_index, features, class_id, _ in before:
            self._assign(
                model_id=previous_model_id,
                sample_index=pending_index,
                features=features,
                observed_class_id=class_id,
            )
        # (c) 後のk件を、古い順に帰属先へ（新しいモデルは(a)で学習済み）。
        if selection.selected_hypothesis != "new_model":
            for pending_index, features, class_id, _ in after:
                self._assign(
                    model_id=destination_model_id,
                    sample_index=pending_index,
                    features=features,
                    observed_class_id=class_id,
                )
        # (d) 現在のモデルを替え、保留を空にし、帰属先の統計の基準値で監視を始め直す。
        self.current_model_id = destination_model_id
        self._pending = []
        assert self.loss_change_monitor is not None
        self.loss_change_monitor.reset(baseline_loss_mean=self._baseline(destination_model_id))
        self.alarm_records.append(
            AlarmRecord(
                sample_index=sample_index,
                selected_hypothesis=selection.selected_hypothesis,
                change_interval_sample_count=k,
                window_sample_count=window_length,
                destination_model_id=destination_model_id,
                new_model_learner_name=selection.new_model_learner_name,
                description_length=selection.description_length,
                alarm_raised=True,
                adaptation_outcome=_HYPOTHESIS_TO_OUTCOME[selection.selected_hypothesis],
                window_tail_concept_ids=tail_concept_ids,
            )
        )
        return True
