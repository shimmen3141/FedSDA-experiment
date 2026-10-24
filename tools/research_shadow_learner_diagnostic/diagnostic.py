"""S1の成立性の診断。製品の判定は変えず、選ばれなかった学習者を継続する。"""

from contextlib import contextmanager
from dataclasses import asdict


def best_new_model_hypothesis(arguments):
    """製品と同じ加算順・同点規則で、最良の新モデルの仮説を返す。"""
    losses = arguments["current_model_log_losses"]
    candidates = arguments["new_model_code_lengths_by_change_interval_sample_count"]
    if not candidates:
        return None
    span = min(
        sorted(candidates),
        key=lambda k: (
            (sum(losses[: len(losses) - k]) + arguments["switching_code_length"])
            + candidates[k]
        ),
    )
    code_length = (
        sum(losses[: len(losses) - span]) + arguments["switching_code_length"]
    ) + candidates[span]
    return (
        span,
        arguments["new_model_learner_names_by_change_interval_sample_count"][span],
        sum(losses),
        code_length,
    )


def continue_shadow_observation(
    *, learner, optimizer, current_classifier, input_features, observed_class_labels
):
    """双方の学習前の損失を測り、影の共有部以外を1回学ぶ。optimizerの状態を継続する。"""
    import torch
    from torch.nn import BCELoss, CrossEntropyLoss
    from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
        evaluate_classifier_per_sample_log_losses,
    )

    with torch.random.fork_rng(devices=[]):
        current_loss = float(
            evaluate_classifier_per_sample_log_losses(
                classifier=current_classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            ).sum()
        )
        shadow_loss = float(
            evaluate_classifier_per_sample_log_losses(
                classifier=learner,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            ).sum()
        )
        with torch.no_grad():
            shared = learner.extract_shared_features(input_features)
        optimizer.zero_grad()
        predictions = learner.forward_from_shared_features(shared)
        loss = (
            BCELoss()(predictions, observed_class_labels)
            if learner.class_count == 2
            else CrossEntropyLoss()(
                predictions, observed_class_labels.reshape(-1).long()
            )
        )
        loss.backward()
        optimizer.step()
    return current_loss, shadow_loss


@contextmanager
def shadow_learner_diagnostic_policy(
    *,
    diagnostics,
    trace,
    horizon_samples=240,
    online_current_control=False,
    adoption_handler=None,
):
    """1 clientに影1つ。currentの警報で作り直し、held/newの警報で終了。採用しない。"""
    from unittest.mock import patch
    import federated_learning_experiments.learning.training.prequential_code_length as prequential
    import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as mdl
    import federated_learning_experiments.runtime.fedsda_run_client as client_module
    import federated_learning_experiments.runtime.observed_sample_processing as processing
    from federated_learning_experiments.learning.models.model_computation_measurement import (
        measure_model_computation,
    )

    if type(horizon_samples) is not int or horizon_samples < 1:
        raise ValueError("影の観測上限は正の整数が必要です")
    diagnostics.update(
        episodes=[], extra_model_computation_counts={}, horizon_samples=horizon_samples
    )
    active = {}
    process_stack = []
    selection_stack = []
    accumulation_stack = []
    original_process = client_module.process_observed_sample
    original_predict = processing.predict_observed_sample_and_update_prediction_weights
    original_select = processing.select_alarm_interval_assignment
    original_compare = mdl.select_assignment_by_minimum_description_length
    original_accumulate = prequential._accumulate_prequential_code_length
    original_optimizer = prequential.create_parameter_optimizer
    original_reinitialized = mdl.compute_reinitialized_learner_prequential_code_length
    original_warm = mdl.compute_warm_started_learner_prequential_code_length

    def close(client_id, reason, sample_index):
        state = active.pop(client_id, None)
        if state is not None:
            state["episode"].update(end_reason=reason, end_sample_index=sample_index)

    def optimizer(**arguments):
        result = original_optimizer(**arguments)
        if accumulation_stack:
            accumulation_stack[-1]["optimizer"] = result
        return result

    def accumulate(**arguments):
        if not selection_stack:
            return original_accumulate(**arguments)
        frame = selection_stack[-1]
        entry = {"learner": arguments["learner"]}
        accumulation_stack.append(entry)
        try:
            code_length = original_accumulate(**arguments)
        finally:
            accumulation_stack.pop()
        entry["code_length"] = code_length
        frame["learners"][
            (arguments["input_features"].shape[0], frame["learner_name"])
        ] = entry
        return code_length

    def reinitialized(**arguments):
        selection_stack[-1]["learner_name"] = "reinitialized_learner"
        return original_reinitialized(**arguments)

    def warm(**arguments):
        selection_stack[-1]["learner_name"] = "warm_started_learner"
        return original_warm(**arguments)

    def compare(**arguments):
        result = original_compare(**arguments)
        if result.selected_hypothesis == "current_model":
            selection_stack[-1]["best_new"] = best_new_model_hypothesis(arguments)
        return result

    def select(**arguments):
        frame = {"learners": {}, "arguments": arguments}
        selection_stack.append(frame)
        try:
            result = original_select(**arguments)
        finally:
            selection_stack.pop()
        frame["selection"] = result
        process_stack[-1]["selection"] = frame
        return result

    def predict(**arguments):
        result = original_predict(**arguments)
        client_id = process_stack[-1]["arguments"]["client_id"]
        state = active.get(client_id)
        if state is None:
            return result
        observation = arguments["indexed_observation"]
        episode = state["episode"]
        if observation.sample_index <= episode["alarm_sample_index"]:
            raise ValueError("警報の後の標本だけを影の継続に使います")
        current_id = arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        current = (
            arguments["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        with measure_model_computation() as meter:
            current_loss, shadow_loss = continue_shadow_observation(
                learner=state["learner"],
                optimizer=state["optimizer"],
                current_classifier=current,
                input_features=observation.training_sample.input_features,
                observed_class_labels=observation.training_sample.observed_class_labels,
            )
            control_values = {}
            if online_current_control:
                from tools.research_shadow_learner_diagnostic.controls import (
                    observe_online_current_control,
                )

                control_values = observe_online_current_control(
                    state=state,
                    current_classifier=current,
                    input_features=observation.training_sample.input_features,
                    observed_class_labels=observation.training_sample.observed_class_labels,
                    current_loss=current_loss,
                    shadow_loss=shadow_loss,
                    age=episode["observed_sample_count"] + 1,
                    same_concept=(
                        state["same_concept"]
                        and observation.observed_concept_id
                        == episode["alarm_concept_id"]
                    ),
                )
        counts = diagnostics["extra_model_computation_counts"]
        for key, value in asdict(meter.get_model_computation_counts()).items():
            counts[key] = counts.get(key, 0) + value
        age = episode["observed_sample_count"] + 1
        score = episode["last_score"] + (current_loss - shadow_loss)
        # 真の概念は診断列だけ。影の生成・更新・打切りには使わない。
        same_concept = (
            state["same_concept"]
            and observation.observed_concept_id == episode["alarm_concept_id"]
        )
        state["same_concept"] = same_concept
        trace.append(
            dict(
                episode_id=episode["episode_id"],
                client_id=client_id,
                sample_index=observation.sample_index,
                age=age,
                observed_concept_id=observation.observed_concept_id,
                current_model_id=current_id,
                current_log_loss=current_loss,
                shadow_log_loss=shadow_loss,
                score=score,
                same_alarm_concept_so_far=same_concept,
                **control_values,
            )
        )
        episode.update(
            observed_sample_count=age,
            last_score=score,
            maximum_score=max(episode["maximum_score"], score),
            last_sample_index=observation.sample_index,
        )
        if adoption_handler is not None:
            state["observations"].append(observation)
        if score > 0.0 and episode["first_crossing_age"] is None:
            episode.update(
                first_crossing_age=age,
                first_crossing_sample_index=observation.sample_index,
                first_crossing_same_alarm_concept=same_concept,
            )
        if age >= horizon_samples and not (adoption_handler is not None and score > 0):
            close(client_id, "horizon", observation.sample_index)
        return result

    def process(**arguments):
        frame = {"arguments": arguments}
        process_stack.append(frame)
        try:
            result = original_process(**arguments)
        finally:
            process_stack.pop()
        client_id = arguments["client_id"]
        state = active.get(client_id)
        if (
            adoption_handler is not None
            and state is not None
            and result.alarm_occurrence_handling is None
            and state["episode"]["last_score"] > 0
        ):
            adoption_handler(state=state, arguments=arguments)
            close(
                client_id,
                "shadow_adopted",
                arguments["indexed_observation"].sample_index,
            )
            return result
        if (
            adoption_handler is not None
            and state is not None
            and state["episode"]["observed_sample_count"] >= horizon_samples
        ):
            close(client_id, "horizon", arguments["indexed_observation"].sample_index)
        if "selection" not in frame or result.alarm_occurrence_handling is None:
            return result
        selection_frame = frame["selection"]
        choice = selection_frame["selection"]
        observation = arguments["indexed_observation"]
        client_id = arguments["client_id"]
        if (
            choice.selected_hypothesis == "current_model"
            and selection_frame["best_new"] is None
        ):
            return result
        close(
            client_id, "alarm_" + choice.selected_hypothesis, observation.sample_index
        )
        if choice.selected_hypothesis != "current_model":
            return result
        span, learner_name, current_code, new_code = selection_frame["best_new"]
        entry = selection_frame["learners"][(span, learner_name)]
        score = current_code - new_code
        if score > 0.0:
            raise ValueError("currentが選ばれた判定でnewの符号長が短くなっています")
        pending = selection_frame["arguments"]["pending_sample_observations"]
        episode = dict(
            episode_id=len(diagnostics["episodes"]),
            client_id=client_id,
            alarm_sample_index=observation.sample_index,
            alarm_concept_id=observation.observed_concept_id,
            pending_first_concept_id=pending[0].observed_concept_id,
            pending_is_single_concept=all(
                o.observed_concept_id == observation.observed_concept_id
                for o in pending
            ),
            new_interval_first_concept_id=pending[-span].observed_concept_id,
            learner_name=learner_name,
            change_interval_sample_count=span,
            current_code_length=current_code,
            new_code_length=new_code,
            initial_score=score,
            last_score=score,
            maximum_score=score,
            observed_sample_count=0,
            last_sample_index=observation.sample_index,
            first_crossing_age=None,
            first_crossing_sample_index=None,
            first_crossing_same_alarm_concept=None,
            end_reason=None,
            end_sample_index=None,
        )
        diagnostics["episodes"].append(episode)
        active[client_id] = dict(
            learner=entry["learner"],
            optimizer=entry["optimizer"],
            episode=episode,
            same_concept=True,
        )
        if adoption_handler is not None:
            active[client_id]["observations"] = list(pending[-span:])
        if online_current_control:
            from tools.research_shadow_learner_diagnostic.controls import (
                initialize_online_current_control,
            )

            initialize_online_current_control(
                state=active[client_id], arguments=selection_frame["arguments"]
            )
        return result

    with (
        patch.object(client_module, "process_observed_sample", process),
        patch.object(
            processing, "predict_observed_sample_and_update_prediction_weights", predict
        ),
        patch.object(processing, "select_alarm_interval_assignment", select),
        patch.object(mdl, "select_assignment_by_minimum_description_length", compare),
        patch.object(
            mdl, "compute_reinitialized_learner_prequential_code_length", reinitialized
        ),
        patch.object(mdl, "compute_warm_started_learner_prequential_code_length", warm),
        patch.object(prequential, "_accumulate_prequential_code_length", accumulate),
        patch.object(prequential, "create_parameter_optimizer", optimizer),
    ):
        try:
            yield
        finally:
            for client_id in tuple(active):
                close(
                    client_id,
                    "run_end",
                    active[client_id]["episode"]["last_sample_index"],
                )
