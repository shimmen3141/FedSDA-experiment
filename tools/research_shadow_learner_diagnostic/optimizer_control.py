"""①: 現在の同じ写しについてoptimizerを継続する場合と初期化する場合を同時に追う。"""

from contextlib import contextmanager
from copy import deepcopy


@contextmanager
def optimizer_reset_control():
    from unittest.mock import patch
    import tools.research_shadow_learner_diagnostic.controls as controls
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        continue_shadow_observation,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
        create_parameter_optimizer,
    )

    original_start, original_observe = (
        controls.initialize_online_current_control,
        controls.observe_online_current_control,
    )

    def start(*, state, arguments):
        original_start(state=state, arguments=arguments)
        learner = deepcopy(state["online_control"]["classifier"])
        state["reset_control"] = dict(
            classifier=learner,
            optimizer=create_parameter_optimizer(
                parameters=tuple(
                    p
                    for n, p in learner.named_parameters()
                    if not n.startswith("feature_extractor.")
                ),
                optimizer_settings=arguments["parameter_optimizer_settings"],
            ),
        )
        state["episode"].update(
            reset_control_score=state["episode"]["initial_score"],
            reset_control_first_crossing_age=None,
            reset_control_first_crossing_same_alarm_concept=None,
        )

    def observe(**kwargs):
        result = original_observe(**kwargs)
        state = kwargs["state"]
        _, loss = continue_shadow_observation(
            learner=state["reset_control"]["classifier"],
            optimizer=state["reset_control"]["optimizer"],
            **{
                k: kwargs[k]
                for k in (
                    "current_classifier",
                    "input_features",
                    "observed_class_labels",
                )
            },
        )
        e = state["episode"]
        e["reset_control_score"] += loss - kwargs["shadow_loss"]
        if (
            e["reset_control_score"] > 0
            and e["reset_control_first_crossing_age"] is None
        ):
            e["reset_control_first_crossing_age"] = kwargs["age"]
            e["reset_control_first_crossing_same_alarm_concept"] = kwargs[
                "same_concept"
            ]
        result.update(
            reset_current_log_loss=loss, reset_control_score=e["reset_control_score"]
        )
        return result

    with (
        patch.object(controls, "initialize_online_current_control", start),
        patch.object(controls, "observe_online_current_control", observe),
    ):
        yield
