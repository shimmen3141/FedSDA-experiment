"""基準版16の判断を使う時点だけを、研究runの寿命の間変更する。"""

import hashlib
import json
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import wraps
from inspect import signature
from unittest.mock import patch

AGGREGATION_TIME_VARIANTS = (
    "aggregation-time-prediction",
    "aggregation-time-assignment",
    "aggregation-time-decisions",
)


def surviving_snapshot_weights(weights, model_ids):
    """写しにあって今も保有するモデルだけで正規化する。"""
    retained = {key: value for key, value in weights.items() if key in model_ids}
    if not retained or sum(retained.values()) <= 0:
        raise RuntimeError("予測の写しにあるモデルがすべて保有から外れました")
    # 集合が同じときは、余分な丸めを加えない。
    if len(retained) == len(weights):
        return retained
    total = sum(retained.values())
    return {key: value / total for key, value in retained.items()}


def combine_snapshot_predictions(*, synchronized, local, weights, beta, class_count):
    """各モデルの現在の確率を、写した集合・重み・βだけで混ぜる。"""
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        combine_model_prediction_probabilities,
    )

    def combine(probabilities):
        return combine_model_prediction_probabilities(
            prediction_probabilities_by_model_id={key: probabilities[key] for key in weights},
            prediction_weights_by_model_id=weights,
            class_count=class_count,
        )

    return combine(synchronized) * beta[0] + combine(local) * beta[1]


def _call(function, arguments):
    return function(**{name: arguments[name] for name in signature(function).parameters})


def resolve_pending_alarm(arguments):
    """集約直前のFIFOだけを、既存のMDL選択・応答へ渡す。

    応答のAPIは最後の観測位置を要求するので、応答位置は集約区間の末尾。
    最初の警報位置は呼出側の診断に独立に残す。
    """
    from federated_learning_experiments.runtime import (
        observed_sample_processing as processing,
    )

    values = dict(arguments)
    observations = values["pending_sample_observation_store"].snapshot_pending_sample_observations()
    values["pending_sample_observations"] = observations
    if not observations:
        model_id = values["current_training_model_assignment"].current_training_model_id
        statistics = values["loss_statistics_store"].get_model_loss_statistics(model_id=model_id)
        values["loss_change_monitor"].reset(
            baseline_loss_mean=processing.select_loss_monitoring_baseline_mean_loss(
                loss_moments=None if statistics is None else statistics.overall_loss_moments
            )
        )
        return None
    if (
        len(observations) < values["minimum_change_interval_sample_count"]
        and values["short_change_interval_alarm_policy"]
        == "wait_until_change_interval_reaches_minimum"
    ):
        # 製品の「待つ」方針では応答もresetも行わない。次の集約まで保留する。
        return "wait_for_minimum"
    last = (
        values["pending_training_assignment_buffer"].get_state_snapshot().last_observed_sample_index
    )
    values.update(
        alarm_sample_index=last,
        change_interval_grid_sample_count=values["minimum_change_interval_sample_count"],
    )
    selection = None
    if len(observations) >= values["minimum_change_interval_sample_count"]:
        selection = _call(processing.select_alarm_interval_assignment, values)
        values["alarm_interval_assignment_selection_record_store"].append_selection_record(
            alarm_sample_index=last, selection=selection, alarm_raised=True
        )
    span = len(observations) if selection is None else selection.change_interval_sample_count
    values.update(
        estimated_change_span_sample_count=span,
        estimated_change_point_sample_index=last - span + 1,
        detection_episode_id=None,
        alarm_interval_assignment_selection=selection,
        architecture_reference_classifier=values["held_model_training_state_registry"]
        .get_held_model_training_state(
            model_id=values["current_training_model_assignment"].current_training_model_id
        )
        .classifier,
    )
    result = _call(processing.handle_alarm_occurrence, values)
    values["pending_sample_observation_store"].retain_latest_pending_sample_observations(
        retained_sample_indices=values["pending_training_assignment_buffer"]
        .get_state_snapshot()
        .pending_sample_indices
    )
    return result


@contextmanager
def aggregation_time_policy(*, variant, diagnostics, sample_diagnostics=False):
    """予測・警報・集約境界を包む。標本単位の記録は検査時だけ残す。"""
    from federated_learning_experiments.evaluation.sample_prediction_record_store import (
        SamplePredictionRecordStore,
    )
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        ResponsibilityWeightedTrainingStore,
        derive_run_training_mass_purity,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (
        SynchronizedAndLocalPredictionMixture,
    )
    from federated_learning_experiments.runtime import (
        alarm_interval_candidate_training as candidate_training,
    )
    from federated_learning_experiments.runtime import (
        fedsda_run_client as client_module,
    )
    from federated_learning_experiments.runtime import (
        fedsda_run_server as server_module,
    )
    from federated_learning_experiments.runtime import (
        held_model_training_request_handling as training_requests,
    )
    from federated_learning_experiments.runtime import (
        observed_sample_prediction as prediction,
    )
    from federated_learning_experiments.runtime import (
        observed_sample_processing as processing,
    )
    from federated_learning_experiments.runtime import (
        server_round_synchronization as synchronization,
    )

    if variant not in ("product-default", *AGGREGATION_TIME_VARIANTS):
        raise ValueError("集約時判断の未定義の変種です")
    freezes = variant in ("aggregation-time-prediction", "aggregation-time-decisions")
    delays = variant in ("aggregation-time-assignment", "aggregation-time-decisions")
    diagnostics.update(
        schema_version=2,
        sample_diagnostics_enabled=sample_diagnostics,
        alarms=[],
        selections=[],
        synchronizations=[],
        distributions=[],
        creations=[],
    )
    if sample_diagnostics:
        diagnostics.update(samples=[], detector_waits=[], snapshots=[])
    diagnostics["training_attempt_count_by_client"] = {}
    diagnostics["detector_wait_sample_count_by_client"] = {}
    clients = {}
    snapshots = {}
    pending = {}
    last_arguments = {}
    active = {}
    cls = client_module.FedsdaRunClient
    initialize = cls.__init__

    def initialize_client(self, **kwargs):
        initialize(self, **kwargs)
        # 診断だけを有効化する。責任度providerは付けず、既定の学習・集約を保つ。
        store = self.owners.training_sample_store
        store.record_training_mass_by_concept = True
        store.responsibility_training_store = ResponsibilityWeightedTrainingStore()

    process = client_module.process_observed_sample
    predict = processing.predict_observed_sample_and_update_prediction_weights
    observe = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation
    flush = cls.flush_pending_local_updates
    recalibrate = cls.recalibrate_prediction_state_after_aggregation
    distribute = cls.apply_global_model_distribution
    synchronize = synchronization.synchronize_models_in_server_round
    select = processing.select_alarm_interval_assignment
    train_candidate = candidate_training.train_candidate_classifier_epochs
    train_iterations = training_requests.perform_held_model_joint_training_iterations
    diagnostics["candidate_training"] = []

    def train_local_iterations(**kwargs):
        counts = diagnostics["training_attempt_count_by_client"]
        client_id = active["client"]
        counts[client_id] = (
            counts.get(client_id, 0) + kwargs["requested_joint_update_iteration_count"]
        )
        active["training_attempts"] = (
            active.get("training_attempts", 0) + kwargs["requested_joint_update_iteration_count"]
        )
        return train_iterations(**kwargs)

    def state_digest(monitor):
        return hashlib.sha256(
            json.dumps(asdict(monitor.get_state_snapshot()), sort_keys=True).encode()
        ).hexdigest()

    def train_candidate_epochs(**kwargs):
        result = train_candidate(**kwargs)
        diagnostics["candidate_training"].append(
            dict(
                client=active["client"],
                sample=last_arguments[active["client"]]["indexed_observation"].sample_index,
                result=asdict(result),
            )
        )
        return result

    validate = processing._validate_observed_sample_processing_inputs

    def validate_sample(**kwargs):
        if delays and active.get("client") in pending:
            monitor = kwargs["loss_change_monitor"]
            last = monitor.last_observation
            # 入口の連続性検査だけに直前位置を提示する。実状態と検出器は触らない。
            with patch.object(
                OverallAndTrueClassLossMonitor,
                "last_observation",
                property(
                    lambda owner: (
                        replace(
                            last,
                            sample_index=kwargs["indexed_observation"].sample_index - 1,
                        )
                        if owner is monitor and last is not None
                        else None
                    )
                ),
            ):
                return validate(**kwargs)
        return validate(**kwargs)

    def ids(registry):
        return tuple(s.model_id for s in registry.snapshot_ordered_held_model_training_states())

    def capture(client, source):
        owners = client.owners
        model_ids = ids(owners.held_model_training_state_registry)
        weights = owners.fixed_share_weight_controller.weights_by_model_id
        # 初回の予測より前は、製品も初回予測の入口で一様に初期化する。
        if set(weights) != set(model_ids):
            weights = {key: 1.0 / len(model_ids) for key in model_ids}
        beta = client.synchronized_and_local_prediction_mixture.snapshot_weights()
        snapshots[client.client_id] = (weights, beta)
        if sample_diagnostics:
            diagnostics["snapshots"].append(
                dict(
                    client=client.client_id,
                    after_sample=owners.sample_prediction_record_store.last_recorded_sample_index,
                    source=source,
                    weights=weights,
                    beta=beta,
                )
            )

    # client自体の入口は、初回の写しと配布イベントのclientの特定にも使う。
    outer_process = cls.process_observed_sample

    def process_client(self, **kwargs):
        clients[self.client_id] = self
        if self.client_id not in snapshots:
            capture(self, "initial")
        return outer_process(self, **kwargs)

    def predict_sample(**kwargs):
        client_id = active["client"]
        model_ids = ids(kwargs["held_model_training_state_registry"])
        beta = kwargs["synchronized_and_local_prediction_mixture"].snapshot_weights()
        if not freezes:
            result = predict(**kwargs)
            used = result.prediction_weights_by_model_id
        else:
            # 内部のFixed-Share・β更新には、既定と同じ損失と重みを渡す。
            # 既定の計算済みの確率を借り、外に出す予測と記録だけを作り直す。
            local_probabilities = {}
            last_combination = {}
            combine = prediction.combine_model_prediction_probabilities
            append = SamplePredictionRecordStore.append_sample_prediction_record
            mix = SynchronizedAndLocalPredictionMixture.mix_prediction_probabilities

            def combine_and_observe(**values):
                last_combination.clear()
                last_combination.update(values["prediction_probabilities_by_model_id"])
                return combine(**values)

            def mix_and_observe(self, **values):
                # 混合の直前の結合は局所側。後続の診断用の結合とは区別する。
                local_probabilities.update(last_combination)
                return mix(self, **values)

            with (
                patch.object(
                    prediction,
                    "combine_model_prediction_probabilities",
                    combine_and_observe,
                ),
                patch.object(
                    SynchronizedAndLocalPredictionMixture,
                    "mix_prediction_probabilities",
                    mix_and_observe,
                ),
                patch.object(
                    SamplePredictionRecordStore,
                    "append_sample_prediction_record",
                    lambda *a, **k: None,
                ),
            ):
                result = predict(**kwargs)
            weights, beta = snapshots[client_id]
            used = surviving_snapshot_weights(weights, model_ids)
            count = (
                kwargs["held_model_training_state_registry"]
                .get_held_model_training_state(model_id=model_ids[0])
                .classifier.class_count
            )

            probabilities = combine_snapshot_predictions(
                synchronized=result.prediction_probabilities_by_model_id,
                local=local_probabilities,
                weights=used,
                beta=beta,
                class_count=count,
            )
            labels = prediction.predict_class_labels_from_prediction_scores(
                prediction_scores=probabilities, class_count=count
            )
            correct = bool(
                labels.view(-1)[0].item()
                == kwargs["indexed_observation"]
                .training_sample.observed_class_labels.view(-1)[0]
                .item()
            )
            observed_label = (
                kwargs["indexed_observation"]
                .training_sample.observed_class_labels.view(-1)[0]
                .item()
            )
            correct_by_model = {
                key: bool(
                    prediction.predict_class_labels_from_prediction_scores(
                        prediction_scores=value, class_count=count
                    )
                    .view(-1)[0]
                    .item()
                    == observed_label
                )
                for key, value in result.prediction_probabilities_by_model_id.items()
            }
            maximum = kwargs["fixed_share_weight_controller"].select_maximum_weight_model_id(
                prediction_weights_by_model_id=used,
                preferred_model_id=kwargs[
                    "current_training_model_assignment"
                ].current_training_model_id,
            )
            record = replace(
                result.sample_prediction_record,
                combined_prediction_is_correct=correct,
                maximum_weight_model_id=maximum,
                maximum_prediction_weight=max(used.values()),
                effective_model_count=1.0 / sum(w * w for w in used.values()),
                maximum_weight_model_prediction_is_correct=correct_by_model[maximum],
                any_model_or_combined_prediction_is_correct=correct
                or any(correct_by_model.values()),
            )
            append(
                kwargs["sample_prediction_record_store"],
                sample_prediction_record=record,
            )
            result = replace(
                result,
                sample_prediction_record=record,
                predicted_class_labels=labels,
                combined_prediction_probabilities=probabilities,
                prediction_weights_by_model_id=used,
            )
        if sample_diagnostics:
            active["prediction"] = dict(
                weights=used,
                beta=beta,
                internal_weights_after=kwargs["fixed_share_weight_controller"].weights_by_model_id,
                internal_beta_after=kwargs[
                    "synchronized_and_local_prediction_mixture"
                ].snapshot_weights(),
            )
        return result

    def observe_loss(self, **kwargs):
        client_id = active["client"]
        if delays and client_id in pending:
            before = self.get_state_snapshot()
            # 観測結果だけを作る。検出器へ損失を渡さず、状態を変えない。
            result = replace(
                pending[client_id][1],
                sample_index=kwargs["sample_index"],
                drift_detected=False,
                component_update_count=0,
                evaluated_candidate_bet_count=0,
                decrease_detector_component_update_count=0,
                decrease_detector_evaluated_candidate_bet_count=0,
                baseline_lowered=False,
                previous_baseline_loss_mean=None,
                lowered_baseline_loss_mean=None,
            )
            assert before == self.get_state_snapshot()
            counts = diagnostics["detector_wait_sample_count_by_client"]
            counts[client_id] = counts.get(client_id, 0) + 1
            if sample_diagnostics:
                diagnostics["detector_waits"].append(
                    dict(
                        client=client_id,
                        sample=kwargs["sample_index"],
                        alarm=pending[client_id][0],
                        state_digest=state_digest(self),
                    )
                )
            return result
        result = observe(self, **kwargs)
        if result.drift_detected:
            diagnostics["alarms"].append(
                dict(
                    client=client_id,
                    sample=kwargs["sample_index"],
                    state_digest=state_digest(self),
                )
            )
            if delays:
                pending[client_id] = (kwargs["sample_index"], result)
                # 製品側にも実際の警報位置を残す。MDLによる変化点は選択時の別記録。
                last_arguments[client_id]["loss_change_alarm_record_store"].append_alarm_record(
                    alarm_sample_index=result.sample_index,
                    estimated_change_point_sample_index=result.detector_candidate_start_sample_index,
                    detector_candidate_start_sample_index=result.detector_candidate_start_sample_index,
                )
                return replace(result, drift_detected=False)
        return result

    @wraps(select)
    def select_assignment(**kwargs):
        result = select(**kwargs)
        diagnostics["selections"].append(
            dict(
                client=kwargs["client_id"],
                sample=kwargs["alarm_sample_index"],
                pending_alarm=pending.get(kwargs["client_id"], (None,))[0],
                window=[x.sample_index for x in kwargs["pending_sample_observations"]],
                selection=asdict(result),
                before_aggregation=active.get("flushing", False),
            )
        )
        return result

    def process_sample(**kwargs):
        client_id = kwargs["client_id"]
        active.update(client=client_id, flushing=False, training_attempts=0)
        last_arguments[client_id] = kwargs
        before_ids = ids(kwargs["held_model_training_state_registry"])
        result = process(**kwargs)
        after_ids = ids(kwargs["held_model_training_state_registry"])
        sample = kwargs["indexed_observation"].sample_index
        diagnostics["creations"].extend(
            dict(client=client_id, sample=sample, model=key)
            for key in after_ids
            if key not in before_ids
        )
        if sample_diagnostics:
            diagnostics["samples"].append(
                dict(
                    client=client_id,
                    sample=sample,
                    **active["prediction"],
                    pending_alarm=pending.get(client_id, (None,))[0],
                    current=kwargs["current_training_model_assignment"].current_training_model_id,
                    released=[x.sample_index for x in result.released_sample_observations],
                    stores={
                        s.model_id: len(s.training_samples)
                        for s in kwargs[
                            "training_sample_store"
                        ].snapshot_ordered_model_training_samples()
                    },
                    training_counts=asdict(
                        kwargs[
                            "model_training_and_assignment_counts_store"
                        ].snapshot_model_training_and_assignment_counts()
                    ),
                    joint_updates=len(result.completed_joint_update_losses),
                    training_attempts=active["training_attempts"],
                    candidate_training_count=len(diagnostics["candidate_training"]),
                )
            )
        return result

    def flush_client(self, **kwargs):
        client_id = self.client_id
        active.update(client=client_id, flushing=True)
        if delays and client_id in pending:
            before_ids = ids(self.owners.held_model_training_state_registry)
            arguments = last_arguments[client_id]
            result = resolve_pending_alarm(arguments)
            diagnostics.setdefault("responses", []).append(
                dict(
                    client=client_id,
                    sample=arguments["indexed_observation"].sample_index,
                    alarm=pending[client_id][0],
                    outcome=(
                        result
                        if isinstance(result, str)
                        else None
                        if result is None
                        else result.adaptation_record.adaptation_outcome
                    ),
                )
            )
            after_ids = ids(self.owners.held_model_training_state_registry)
            diagnostics["creations"].extend(
                dict(
                    client=client_id,
                    sample=arguments["indexed_observation"].sample_index,
                    model=key,
                )
                for key in after_ids
                if key not in before_ids
            )
            if result != "wait_for_minimum":
                del pending[client_id]
        return flush(self, **kwargs)

    def apply_distribution(self, **kwargs):
        result = distribute(self, **kwargs)
        diagnostics["distributions"].append(
            dict(
                client=self.client_id,
                after_sample=self.owners.sample_prediction_record_store.last_recorded_sample_index,
                models=ids(self.owners.held_model_training_state_registry),
            )
        )
        if (
            self._run_client_settings.prediction_combination_settings.prediction_weight_recalibration_after_aggregation_policy
            == "keep_weights_without_recalibration"
        ):
            capture(self, "distribution_without_recalibration")
        return result

    def after_recalibration(self, **kwargs):
        result = recalibrate(self, **kwargs)
        if (
            self._run_client_settings.prediction_combination_settings.prediction_weight_recalibration_after_aggregation_policy
            != "keep_weights_without_recalibration"
        ):
            capture(self, "recalibration")
        return result

    def synchronize_round(**kwargs):
        result = synchronize(**kwargs)
        diagnostics["synchronizations"].append(
            dict(
                round=kwargs["round_index"],
                after_sample=kwargs["run_clients"][
                    0
                ].owners.sample_prediction_record_store.last_recorded_sample_index,
                registered=[asdict(record) for record in result.registered_client_models],
                communication=asdict(
                    kwargs["communication_volume_record_store"].get_state_snapshot()
                ),
            )
        )
        return result

    with ExitStack() as stack:
        for target, name, replacement in (
            (cls, "__init__", initialize_client),
            (cls, "process_observed_sample", process_client),
            (client_module, "process_observed_sample", process_sample),
            (
                processing,
                "_validate_observed_sample_processing_inputs",
                validate_sample,
            ),
            (
                processing,
                "predict_observed_sample_and_update_prediction_weights",
                predict_sample,
            ),
            (
                OverallAndTrueClassLossMonitor,
                "observe_loss_after_label_observation",
                observe_loss,
            ),
            (processing, "select_alarm_interval_assignment", select_assignment),
            (cls, "flush_pending_local_updates", flush_client),
            (cls, "apply_global_model_distribution", apply_distribution),
            (
                cls,
                "recalibrate_prediction_state_after_aggregation",
                after_recalibration,
            ),
            (server_module, "synchronize_models_in_server_round", synchronize_round),
            (
                candidate_training,
                "train_candidate_classifier_epochs",
                train_candidate_epochs,
            ),
            (
                training_requests,
                "perform_held_model_joint_training_iterations",
                train_local_iterations,
            ),
        ):
            stack.enter_context(patch.object(target, name, replacement))
        yield
    diagnostics["final_held_model_counts"] = {
        key: len(ids(client.owners.held_model_training_state_registry))
        for key, client in clients.items()
    }
    diagnostics["alarm_count"] = len(diagnostics["alarms"])
    diagnostics["selection_count"] = len(diagnostics["selections"])
    diagnostics["selection_outcomes"] = {
        name: sum(s["selection"]["selected_hypothesis"] == name for s in diagnostics["selections"])
        for name in ("current_model", "held_model", "new_model")
    }
    diagnostics["new_model_adoption_count"] = len(diagnostics["creations"])
    masses = [
        client.owners.training_sample_store.responsibility_training_store.mass_by_model_and_concept
        for client in clients.values()
    ]
    diagnostics["training_mass_by_client_model_concept"] = {
        key: client.owners.training_sample_store.responsibility_training_store.mass_by_model_and_concept
        for key, client in clients.items()
    }
    diagnostics["training_mass_purity"] = derive_run_training_mass_purity(
        client_masses=tuple(masses)
    )
    diagnostics["final_training_counts"] = {
        key: asdict(
            client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        )
        for key, client in clients.items()
    }
