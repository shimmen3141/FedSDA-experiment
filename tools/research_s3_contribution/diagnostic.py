"""採用・登録・統合と、統合前の実混合に対する候補除去差を観測する。"""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from unittest.mock import patch

from tools.research_adopted_candidate_removal import RemovalTracker, combine_branches
from tools.research_current_mixture_benefit import probability_statistics


def parameter_digest(classifier):
    h = hashlib.sha256()
    for name, tensor in classifier.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().numpy().tobytes())
    return h.hexdigest()


def remove_models(sync, local, weights, branches, classes, removed):
    """複数の候補を同時に除去し、候補対の利得の重複加算を避ける。"""
    import math

    if (
        set(sync) != set(local)
        or set(weights) != set(sync)
        or not set(removed) <= set(weights)
    ):
        raise ValueError("除去対象と枝のモデル集合が一致しません")
    remaining = {m: w for m, w in weights.items() if m not in removed}
    mass = math.fsum(remaining.values())
    if not remaining or mass <= 1e-12:
        return None
    return combine_branches(
        {m: sync[m] for m in remaining},
        {m: local[m] for m in remaining},
        {m: w / mass for m, w in remaining.items()},
        branches,
        classes,
    )


class ContributionTracker(RemovalTracker):
    """全新候補を追跡し、warm枝の同時除去と全標本・全MDL判断を保存する。"""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.predictions = []
        self.decisions = []
        self.joint_removals = []
        self.selection_context = None
        self.resolution_context = None

    def predicted(self, client, sample, result, branch_weights):
        # 親が単体除去と、実予測を再構成した値の完全一致を検査する。
        super().predicted(client, sample, result, branch_weights)
        removed = [
            m
            for m in result.prediction_weights_by_model_id
            if any(
                self.records[t]["learner"] == "warm_started_learner"
                for t in self.direct_tokens(client, m)
            )
        ]
        if not removed:
            return
        local, weights, branches = self.captured
        sync = result.prediction_probabilities_by_model_id
        frame = self.frames[-1]
        classes = (
            frame["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=next(iter(weights)))
            .classifier.class_count
        )
        observation = frame["indexed_observation"]
        label = int(observation.training_sample.observed_class_labels.item())
        actual = result.combined_prediction_probabilities.detach()
        # 全対象の重みが0なら再正規化の丸めも入れず、同一予測とする。
        without = (
            actual.clone()
            if all(weights[m] == 0 for m in removed)
            else remove_models(sync, local, weights, branches, classes, removed)
        )
        a = probability_statistics(actual, label=label, classes=classes)
        b = (
            None
            if without is None
            else probability_statistics(without, label=label, classes=classes)
        )
        self.joint_removals.append(
            dict(
                client_id=client,
                sample_index=sample,
                concept=observation.observed_concept_id,
                removed_model_ids=removed,
                status="undefined" if b is None else "defined",
                actual=a,
                without=b,
                actual_probabilities=actual.reshape(-1).tolist(),
                without_probabilities=None
                if without is None
                else without.reshape(-1).tolist(),
                correctness_benefit=None
                if b is None
                else int(a["correct"]) - int(b["correct"]),
                log_loss_benefit=None if b is None else b["log_loss"] - a["log_loss"],
            )
        )

    @contextmanager
    def activated(self):
        import federated_learning_experiments.runtime.observed_sample_processing as processing
        import federated_learning_experiments.runtime.alarm_buffer_response as response
        import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution
        import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as selection
        import federated_learning_experiments.runtime.alarm_interval_candidate_training as candidate_training

        original_select = processing.select_alarm_interval_assignment
        original_mdl = selection.select_assignment_by_minimum_description_length
        original_warm = selection.compute_warm_started_learner_prequential_code_length
        original_fresh = selection.compute_reinitialized_learner_prequential_code_length
        original_resolve = response.resolve_alarm_change_interval
        original_adopt = resolution.adopt_candidate_as_current_training_model
        original_create = candidate_training.create_independent_candidate_training_state

        def select(**kw):
            registry = kw["held_model_training_state_registry"]
            record = dict(
                client_id=kw["client_id"],
                sample_index=kw["alarm_sample_index"],
                prediction_order=len(self.predictions) - 1,
                current_model_id=kw[
                    "current_training_model_assignment"
                ].current_training_model_id,
                held_parameter_hashes={
                    str(s.model_id): parameter_digest(s.classifier)
                    for s in registry.snapshot_ordered_held_model_training_states()
                },
                observation_indices=[
                    o.sample_index for o in kw["pending_sample_observations"]
                ],
                learners=[],
                resolved=False,
            )
            self.selection_context = record
            try:
                result = original_select(**kw)
                record["selection"] = asdict(result)
                self.decisions.append(record)
                return result
            finally:
                self.selection_context = None

        def mdl(**kw):
            if self.selection_context is not None:
                self.selection_context["mdl_inputs"] = kw
            return original_mdl(**kw)

        def score(original, learner, kw):
            value = original(**kw)
            if self.selection_context is not None:
                self.selection_context["learners"].append(
                    dict(
                        learner=learner,
                        samples=len(kw["input_features"]),
                        code_length=value,
                    )
                )
            return value

        def resolve(**kw):
            chosen = kw.get("alarm_interval_assignment_selection")
            context = dict(
                selection=None if chosen is None else asdict(chosen),
                sample_index=kw["proposal_sample_index"],
                detection_episode_id=kw["detection_episode_id"],
            )
            self.resolution_context = context
            try:
                result = original_resolve(**kw)
                if self.decisions:
                    last = self.decisions[-1]
                    frame = self.frames[-1]
                    if (last["client_id"], last["sample_index"]) != (
                        frame["client_id"],
                        kw["proposal_sample_index"],
                    ):
                        raise ValueError("MDLの判断と警報の対応が不明です")
                    last.update(
                        resolved=True,
                        resolution_outcome=result.resolution_outcome,
                        assigned_model_id=result.assigned_model_id,
                    )
                return result
            finally:
                self.resolution_context = None

        def adopt(**kw):
            context = self.resolution_context
            if context is None:
                raise ValueError("採用の警報文脈がありません")
            before_hash = parameter_digest(kw["adopted_candidate_classifier"])
            result = original_adopt(**kw)
            frame = self.frames[-1]
            client = frame["client_id"]
            token = len(self.records)
            # S1に依存するのは既存trackerの入力形式だけ。実際の採用は製品APIを上で呼ぶ。
            self.adopted(
                dict(
                    episode=dict(
                        adopted_model_id=result.current_model_id,
                        episode_id=token,
                        adoption_sample_index=context["sample_index"],
                    )
                ),
                dict(
                    client_id=client,
                    held_model_training_state_registry=kw[
                        "held_model_training_state_registry"
                    ],
                    pending_model_upload_state=kw["pending_model_upload_state"],
                ),
            )
            self.records[-1].update(
                learner=context["selection"]["new_model_learner_name"],
                selection=context["selection"],
                detection_episode_id=context["detection_episode_id"],
                is_head_only=self.variant == "output-layer-matched"
                and context["selection"]["new_model_learner_name"]
                == "warm_started_learner",
                prediction_order=len(self.predictions) - 1,
                pre_adoption_parameter_hash=before_hash,
                post_adoption_parameter_hash=parameter_digest(
                    kw["adopted_candidate_classifier"]
                ),
                trained_samples=kw["candidate_trained_sample_count"],
                update_steps=kw["candidate_parameter_update_step_count"],
                initial_classifier_hash=context.get("initial_classifier_hash"),
            )
            return result

        def create(**kw):
            state = original_create(**kw)
            if self.resolution_context is not None:
                self.resolution_context["initial_classifier_hash"] = parameter_digest(
                    state.candidate_classifier
                )
            return state

        with super().activated():
            original_predict = (
                processing.predict_observed_sample_and_update_prediction_weights
            )

            def predict(**kw):
                result = original_predict(**kw)
                frame = self.frames[-1]
                o = kw["indexed_observation"]
                self.predictions.append(
                    dict(
                        client_id=frame["client_id"],
                        sample_index=o.sample_index,
                        concept=o.observed_concept_id,
                        label=int(o.training_sample.observed_class_labels.item()),
                        input_hash=hashlib.sha256(
                            o.training_sample.input_features.numpy().tobytes()
                        ).hexdigest(),
                        probabilities=result.combined_prediction_probabilities.detach()
                        .reshape(-1)
                        .tolist(),
                        correct=bool(
                            result.sample_prediction_record.combined_prediction_is_correct
                        ),
                        current_model_id=kw[
                            "current_training_model_assignment"
                        ].current_training_model_id,
                    )
                )
                return result

            with (
                patch.object(
                    processing,
                    "predict_observed_sample_and_update_prediction_weights",
                    predict,
                ),
                patch.object(processing, "select_alarm_interval_assignment", select),
                patch.object(
                    selection, "select_assignment_by_minimum_description_length", mdl
                ),
                patch.object(
                    selection,
                    "compute_warm_started_learner_prequential_code_length",
                    lambda **kw: score(original_warm, "warm", kw),
                ),
                patch.object(
                    selection,
                    "compute_reinitialized_learner_prequential_code_length",
                    lambda **kw: score(original_fresh, "reinitialized", kw),
                ),
                patch.object(response, "resolve_alarm_change_interval", resolve),
                patch.object(
                    candidate_training,
                    "create_independent_candidate_training_state",
                    create,
                ),
                patch.object(
                    resolution, "adopt_candidate_as_current_training_model", adopt
                ),
            ):
                yield
