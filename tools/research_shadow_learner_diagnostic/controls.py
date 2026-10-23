"""無変化の入力と、現在のモデルのオンライン継続を対照にする研究用部品。"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace


def initialize_online_current_control(*, state, arguments):
    """警報の処理後の現在のモデルとoptimizerを、参照対応を保って一緒に写す。"""
    model_id = arguments["current_training_model_assignment"].current_training_model_id
    held = arguments[
        "held_model_training_state_registry"
    ].get_held_model_training_state(model_id=model_id)
    classifier, optimizer = deepcopy(
        (
            held.classifier,
            held.concept_specific_parameter_optimizer_state.parameter_optimizer,
        )
    )
    state["online_control"] = dict(classifier=classifier, optimizer=optimizer)
    state["episode"].update(
        control_score=state["episode"]["initial_score"],
        current_control_advantage=0.0,
        control_first_crossing_age=None,
        control_first_crossing_same_alarm_concept=None,
    )


def observe_online_current_control(
    *,
    state,
    current_classifier,
    input_features,
    observed_class_labels,
    current_loss,
    shadow_loss,
    age,
    same_concept,
):
    """現在の写しも予測後に1回学び、影の優位と学習手順の差を分けて記録する。"""
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        continue_shadow_observation,
    )

    control = state["online_control"]
    _, control_loss = continue_shadow_observation(
        learner=control["classifier"],
        optimizer=control["optimizer"],
        current_classifier=current_classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    episode = state["episode"]
    episode["control_score"] += control_loss - shadow_loss
    episode["current_control_advantage"] += current_loss - control_loss
    if episode["control_score"] > 0 and episode["control_first_crossing_age"] is None:
        episode.update(
            control_first_crossing_age=age,
            control_first_crossing_same_alarm_concept=same_concept,
        )
    return dict(
        online_current_log_loss=control_loss,
        control_score=episode["control_score"],
        current_control_advantage=episode["current_control_advantage"],
    )


@contextmanager
def stationary_concept_policy(*, dataset=None, concept_id=None):
    """元の概念列の生成で乱数を消費してから、列だけを固定概念へ置換する。"""
    from unittest.mock import patch
    import federated_learning_experiments.runtime.single_run_execution as execution
    from federated_learning_experiments.data.sea.sea_sample_generation import (
        SeaSampleGenerator,
    )
    from federated_learning_experiments.data.sine.sine_sample_generation import (
        SineSampleGenerator,
    )

    if concept_id is None:
        yield
        return
    valid = {
        "sea4": (SeaSampleGenerator, range(4)),
        "sine2": (SineSampleGenerator, range(2)),
    }
    if dataset not in valid or concept_id not in valid[dataset][1]:
        raise ValueError("固定概念はSEA4の0〜3またはSINE2の0〜1が必要です")
    generator_type = valid[dataset][0]
    original_sample = generator_type.generate_sample
    original = execution.generate_random_client_concept_traces

    def generate(**arguments):
        traces = original(**arguments)
        return tuple(
            replace(
                t,
                concept_ids_by_sample_index=(concept_id,)
                * len(t.concept_ids_by_sample_index),
            )
            for t in traces
        )

    def generate_sample(self, *, concept_id):
        # 事前学習も同じ概念にする。評価開始時に人工的な変化を作らない。
        return original_sample(self, concept_id=fixed_concept)

    fixed_concept = concept_id
    with (
        patch.object(execution, "generate_random_client_concept_traces", generate),
        patch.object(generator_type, "generate_sample", generate_sample),
    ):
        yield
