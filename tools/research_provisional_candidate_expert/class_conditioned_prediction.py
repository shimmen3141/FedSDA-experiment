"""E2の研究介入。既定の予測を残し、予測クラス別の専門家と混ぜる。"""

from contextlib import contextmanager
from dataclasses import replace
from math import fsum
from unittest.mock import patch


def choose_class_conditioned_prediction(
    *,
    baseline_probabilities,
    probabilities_by_model_id,
    source_weights,
    class_controllers,
    source_mixture_weights,
    expert_mixture,
    class_count,
):
    """ラベルを受け取らず、ルーティングと最終の予測を決める。"""
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        combine_model_prediction_probabilities,
        normalize_model_prediction_weights,
        predict_class_labels_from_prediction_scores,
    )

    route = int(
        predict_class_labels_from_prediction_scores(
            prediction_scores=baseline_probabilities,
            class_count=class_count,
        ).item()
    )
    conditional_weights = normalize_model_prediction_weights(
        prediction_weights_by_model_id=class_controllers[
            route
        ].get_weights_before_label_observation(
            model_ids=source_weights,
        )
    )
    conditional_sources = [
        combine_model_prediction_probabilities(
            prediction_probabilities_by_model_id=probabilities,
            prediction_weights_by_model_id=conditional_weights,
            class_count=class_count,
        )
        for probabilities in probabilities_by_model_id
    ]
    conditional = (
        conditional_sources[0] * source_mixture_weights[0]
        + conditional_sources[1] * source_mixture_weights[1]
    )
    combined = expert_mixture.mix_prediction_probabilities(
        synchronized_probabilities=baseline_probabilities,
        local_probabilities=conditional,
    )
    baseline_weight, conditional_weight = expert_mixture.snapshot_weights()
    effective_weights = {
        mid: baseline_weight * source_weights[mid]
        + conditional_weight * conditional_weights[mid]
        for mid in source_weights
    }
    # 製品の正規化は範囲検査を先に行うため、1個のモデルで1+εとなる丸めを先に除く。
    total = fsum(effective_weights.values())
    effective_weights = normalize_model_prediction_weights(
        prediction_weights_by_model_id={
            mid: w / total for mid, w in effective_weights.items()
        }
    )
    return route, conditional_weights, conditional, combined, effective_weights


@contextmanager
def class_conditioned_prediction_policy(*, diagnostics, output_baseline=False):
    """有効経路を通る対照も用意し、モデルの再評価なしでE2を予測へ接続する。"""
    import federated_learning_experiments.runtime.observed_sample_prediction as prediction
    import federated_learning_experiments.runtime.observed_sample_processing as processing
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        predict_class_labels_from_prediction_scores,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (
        SynchronizedAndLocalPredictionMixture,
        true_class_loss_of_prediction_probabilities,
    )
    from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

    original_predict = processing.predict_observed_sample_and_update_prediction_weights
    original_convert = prediction.convert_model_outputs_to_prediction_probabilities
    original_reset = FixedShareWeightController.reset_weights_after_aggregation
    original_replay = (
        FixedShareWeightController.replay_observed_losses_after_aggregation
    )
    original_recalibrate = (
        FedsdaRunClient.recalibrate_prediction_state_after_aggregation
    )
    # controllerを強参照する辞書。id整数を鍵にしないため、破棄後のid再利用は起きない。
    states = {}
    replay_routes = {}
    diagnostics.update(
        predictions=0,
        conditional_updates=0,
        replay_updates=0,
        changed_predictions=0,
        baseline_correct=0,
        conditional_correct=0,
        combined_correct=0,
        class_route_counts={},
        mean_conditional_weight=0.0,
    )

    def reset(self):
        result = original_reset(self)
        if self in states:
            for controller in states[self]["controllers"]:
                original_reset(controller)
        return result

    def replay(self, *, observed_loss_sequence):
        sequence = tuple(observed_loss_sequence)
        result = original_replay(self, observed_loss_sequence=sequence)
        if self in states and sequence:
            routes = replay_routes[self]
            if len(sequence) != len(routes):
                raise ValueError("FIFOの損失と予測時のルートが対応していません")
            controllers = states[self]["controllers"]
            for route, controller in enumerate(controllers):
                original_reset(controller)
                selected = tuple(
                    losses
                    for losses, c in zip(sequence, routes, strict=True)
                    if c == route
                )
                original_replay(controller, observed_loss_sequence=selected)
                diagnostics["replay_updates"] += len(selected)
        return result

    def recalibrate(self):
        controller = self.owners.fixed_share_weight_controller
        state = states.get(controller)
        if state is None:
            return original_recalibrate(self)
        pending = self.owners.pending_sample_observation_store.snapshot_pending_sample_observations()
        replay_routes[controller] = tuple(
            state["routes"][o.sample_index] for o in pending
        )
        try:
            return original_recalibrate(self)
        finally:
            replay_routes.pop(controller, None)
            # 再較正後に必要なのは、まだFIFOに残っている標本のルートだけ。
            state["routes"] = {
                o.sample_index: state["routes"][o.sample_index] for o in pending
            }

    def predict(**arguments):
        controller = arguments["fixed_share_weight_controller"]
        index = arguments["indexed_observation"].sample_index
        if controller in states and index in states[controller]["routes"]:
            raise ValueError("同じ標本のルートを上書きできません")
        mixture = arguments["synchronized_and_local_prediction_mixture"]
        source_mixture_weights = mixture.snapshot_weights()
        captured = []

        def convert(**kw):
            probabilities = original_convert(**kw)
            captured.append(probabilities)
            return probabilities

        # 既定の全状態は製品のまま更新する。2回の推論（同期・局所）の確率を再利用する。
        with patch.object(
            prediction, "convert_model_outputs_to_prediction_probabilities", convert
        ):
            result = original_predict(**arguments)
        if len(captured) != 2:
            raise ValueError("E2は同期と局所を混ぜる製品の既定だけに対応します")
        held_states = arguments[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states()
        class_count = held_states[0].classifier.class_count
        expected_width = 1 if class_count == 2 else class_count
        if any(s.classifier.class_count != class_count for s in held_states) or any(
            p.shape != (1, expected_width)
            for source in captured
            for p in source.values()
        ):
            raise ValueError("E2のクラス数とモデル別確率が対応していません")
        if controller not in states:
            states[controller] = dict(
                controllers=[
                    FixedShareWeightController(
                        prediction_combination_settings=controller._prediction_combination_settings,
                    )
                    for _ in range(class_count)
                ],
                mixture=SynchronizedAndLocalPredictionMixture(
                    share_rate=1.0
                    / controller._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
                ),
                routes={},
            )
        state = states[controller]
        # 未選択クラスも集合の変更時に初期化し、古いIDの証拠を残さない。
        for class_controller in state["controllers"]:
            class_controller.get_weights_before_label_observation(
                model_ids=result.prediction_weights_by_model_id
            )
        route, conditional_weights, conditional, combined, effective_weights = (
            choose_class_conditioned_prediction(
                baseline_probabilities=result.combined_prediction_probabilities,
                probabilities_by_model_id=captured,
                source_weights=result.prediction_weights_by_model_id,
                class_controllers=state["controllers"],
                source_mixture_weights=source_mixture_weights,
                expert_mixture=state["mixture"],
                class_count=class_count,
            )
        )
        # ここまでの予測にはラベルも真の概念IDも渡していない。以下で初めて損失を使う。
        label = result.sample_prediction_record.observed_class_id
        predicted = predict_class_labels_from_prediction_scores(
            prediction_scores=combined, class_count=class_count
        )
        correct = bool(int(predicted.item()) == label)
        old_record = result.sample_prediction_record
        diagnostics["predictions"] += 1
        diagnostics["baseline_correct"] += int(
            old_record.combined_prediction_is_correct
        )
        diagnostics["conditional_correct"] += int(
            predict_class_labels_from_prediction_scores(
                prediction_scores=conditional, class_count=class_count
            ).item()
            == label
        )
        diagnostics["combined_correct"] += int(correct)
        diagnostics["changed_predictions"] += int(
            not predicted.equal(result.predicted_class_labels)
        )
        route_counts = diagnostics["class_route_counts"]
        route_counts[str(route)] = route_counts.get(str(route), 0) + 1
        n = diagnostics["predictions"]
        diagnostics["mean_conditional_weight"] += (
            state["mixture"].snapshot_weights()[1]
            - diagnostics["mean_conditional_weight"]
        ) / n
        state["routes"][old_record.sample_index] = route
        state["controllers"][route].update_weights_after_loss_observation(
            observed_losses_by_model_id=result.observed_losses_by_model_id,
            prediction_weights_by_model_id=conditional_weights,
        )
        diagnostics["conditional_updates"] += 1
        state["mixture"].update_weights_after_loss_observation(
            synchronized_loss=true_class_loss_of_prediction_probabilities(
                prediction_probabilities=result.combined_prediction_probabilities,
                observed_class_id=label,
                class_count=class_count,
            ),
            local_loss=true_class_loss_of_prediction_probabilities(
                prediction_probabilities=conditional,
                observed_class_id=label,
                class_count=class_count,
            ),
        )
        maximum_id = controller.select_maximum_weight_model_id(
            prediction_weights_by_model_id=effective_weights,
            preferred_model_id=arguments[
                "current_training_model_assignment"
            ].current_training_model_id,
        )
        model_correct = {
            mid: int(
                predict_class_labels_from_prediction_scores(
                    prediction_scores=p, class_count=class_count
                ).item()
            )
            == label
            for mid, p in captured[0].items()
        }
        record = replace(
            old_record,
            combined_prediction_is_correct=correct,
            any_model_or_combined_prediction_is_correct=correct
            or any(model_correct.values()),
            maximum_weight_model_id=maximum_id,
            maximum_prediction_weight=max(effective_weights.values()),
            effective_model_count=1.0 / sum(w * w for w in effective_weights.values()),
            maximum_weight_model_prediction_is_correct=model_correct[maximum_id],
        )
        if output_baseline:
            return result
        store = arguments["sample_prediction_record_store"]
        if store._sample_prediction_records[-1] is not old_record:
            raise ValueError("予測の記録の末尾が一致しません")
        store._sample_prediction_records[-1] = record
        return replace(
            result,
            sample_prediction_record=record,
            predicted_class_labels=predicted,
            combined_prediction_probabilities=combined,
            prediction_weights_by_model_id=effective_weights,
        )

    with (
        patch.object(
            processing, "predict_observed_sample_and_update_prediction_weights", predict
        ),
        patch.object(
            FixedShareWeightController, "reset_weights_after_aggregation", reset
        ),
        patch.object(
            FixedShareWeightController,
            "replay_observed_losses_after_aggregation",
            replay,
        ),
        patch.object(
            FedsdaRunClient,
            "recalibrate_prediction_state_after_aggregation",
            recalibrate,
        ),
    ):
        yield states
