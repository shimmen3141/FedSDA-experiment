"""E1aの研究介入。製品の設定と再較正の入口は変更しない。"""

import math
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def mixing_past_posteriors_policy(*, memory_fraction, diagnostics):
    """過去の事後のEMAへ配り直す。再生では記憶を更新しない。"""
    if not 0.0 <= memory_fraction <= 1.0:
        raise ValueError("記憶の混合率は0以上1以下が必要です")
    if memory_fraction == 0.0:
        yield {}
        return

    import federated_learning_experiments.runtime.observed_sample_processing as processing
    import federated_learning_experiments.runtime.server_model_registration_and_aggregation as registration
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

    # オブジェクト自体を鍵にし、破棄後のid再利用で別clientへ記憶が渡ることを避ける。
    memories = {}
    replay_depth = {}
    controllers_by_registry = {}
    update_original = FixedShareWeightController.update_weights_after_loss_observation
    replay_original = FixedShareWeightController.replay_observed_losses
    confirm_original = registration.confirm_held_model_registration
    distribution_original = FedsdaRunClient.apply_global_model_distribution
    predict_original = processing.predict_observed_sample_and_update_prediction_weights
    synchronize_original = FixedShareWeightController._synchronize_model_pool
    diagnostics.update(real_observation_updates=0, replay_updates=0, remapped_ids=0)

    def align(controller, ids):
        previous = memories.get(controller, {})
        values = {mid: previous.get(mid, 0.0) for mid in ids}
        total = sum(values.values())
        return {mid: values[mid] / total if total else 1.0 / len(ids) for mid in ids}

    def remap(controller, mapping):
        if controller not in memories or not mapping:
            return
        values = {}
        for mid, value in memories[controller].items():
            new_id = mapping.get(mid, mid)
            values[new_id] = values.get(new_id, 0.0) + value
            diagnostics["remapped_ids"] += int(new_id != mid)
        memories[controller] = values

    def synchronize(self, *, validated_model_ids):
        result = synchronize_original(self, validated_model_ids=validated_model_ids)
        if self in memories and set(memories[self]) != set(validated_model_ids):
            memories[self] = align(self, validated_model_ids)
        return result

    def update(self, *, observed_losses_by_model_id, prediction_weights_by_model_id):
        observed_losses_by_model_id = self._validated_observed_losses(
            observed_losses_by_model_id=observed_losses_by_model_id
        )
        prediction_weights_by_model_id = self._validated_prediction_weights(
            prediction_weights_by_model_id=prediction_weights_by_model_id
        )
        previous_switch_count = self._leader_switch_count
        previous_leader = self.select_maximum_weight_model_id(
            prediction_weights_by_model_id=prediction_weights_by_model_id
        )
        # 入力検査と累積分散の更新を製品に任せ、その同じ分散から事後を計算する。
        result = update_original(
            self,
            observed_losses_by_model_id=observed_losses_by_model_id,
            prediction_weights_by_model_id=prediction_weights_by_model_id,
        )
        ids = tuple(sorted(observed_losses_by_model_id))
        alpha = (
            1.0
            / self._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        )
        if len(ids) == 1:
            posterior = {ids[0]: 1.0}
        else:
            eta = min(
                1.0,
                math.sqrt(
                    2.0
                    * math.log(len(ids))
                    / max(self._cumulative_observed_loss_variance, 1e-12)
                ),
            )
            values = {
                mid: prediction_weights_by_model_id[mid]
                * math.exp(-eta * min(1.0, max(0.0, observed_losses_by_model_id[mid])))
                for mid in ids
            }
            total = sum(values.values())
            posterior = {mid: values[mid] / total for mid in ids}
        memory = align(self, ids)
        if not replay_depth.get(self, 0):
            memory = {
                mid: (1.0 - alpha) * memory[mid] + alpha * posterior[mid] for mid in ids
            }
            memories[self] = memory
            diagnostics["real_observation_updates"] += 1
        else:
            diagnostics["replay_updates"] += 1
        self._weights_by_model_id = {
            mid: (1.0 - alpha) * posterior[mid]
            + alpha
            * (memory_fraction * memory[mid] + (1.0 - memory_fraction) / len(ids))
            for mid in ids
        }
        final_leader = self.select_maximum_weight_model_id(
            prediction_weights_by_model_id=self._weights_by_model_id
        )
        self._leader_switch_count = previous_switch_count + int(
            final_leader != previous_leader
        )
        return result

    def replay(self, **arguments):
        replay_depth[self] = replay_depth.get(self, 0) + 1
        try:
            return replay_original(self, **arguments)
        finally:
            replay_depth[self] -= 1

    def confirm(**arguments):
        registry = arguments["held_model_training_state_registry"]
        before = {
            s.model_id for s in registry.snapshot_ordered_held_model_training_states()
        }
        result = confirm_original(**arguments)
        after = {
            s.model_id for s in registry.snapshot_ordered_held_model_training_states()
        }
        removed, added = before - after, after - before
        if len(removed) == len(added) == 1:
            mapping = {next(iter(removed)): next(iter(added))}
            # 仮IDは別client間で重なるため、登録したclientの記憶だけを付け替える。
            entry = controllers_by_registry.get(id(registry))
            if entry is not None and entry[0] is registry:
                remap(entry[1], mapping)
        return result

    def predict(**arguments):
        registry = arguments["held_model_training_state_registry"]
        controllers_by_registry[id(registry)] = (
            registry,
            arguments["fixed_share_weight_controller"],
        )
        return predict_original(**arguments)

    def distribute(self, *, model_id_mapping, **arguments):
        controller = self.owners.fixed_share_weight_controller
        previous = memories.get(controller)
        # 配布の中のFIFO再生より先にIDを付け替える。
        remap(controller, model_id_mapping)
        try:
            return distribution_original(
                self, model_id_mapping=model_id_mapping, **arguments
            )
        except Exception:
            if previous is None:
                memories.pop(controller, None)
            else:
                memories[controller] = previous
            raise

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                FixedShareWeightController, "_synchronize_model_pool", synchronize
            )
        )
        stack.enter_context(
            patch.object(
                FixedShareWeightController,
                "update_weights_after_loss_observation",
                update,
            )
        )
        stack.enter_context(
            patch.object(FixedShareWeightController, "replay_observed_losses", replay)
        )
        stack.enter_context(
            patch.object(registration, "confirm_held_model_registration", confirm)
        )
        stack.enter_context(
            patch.object(FedsdaRunClient, "apply_global_model_distribution", distribute)
        )
        stack.enter_context(
            patch.object(
                processing,
                "predict_observed_sample_and_update_prediction_weights",
                predict,
            )
        )
        yield memories
