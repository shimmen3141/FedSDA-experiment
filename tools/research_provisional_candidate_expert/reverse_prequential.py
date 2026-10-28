"""IMPROVE-032。製品の更新処理を使い、逆順の符号長の途中和を再利用する研究介入。"""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from hashlib import sha256
from math import log
from unittest.mock import patch

from torch import cat

from federated_learning_experiments.learning.training import prequential_code_length as training
from federated_learning_experiments.methods.fedsda.training_data_assignment import (
    minimum_description_length_assignment_selection as selection,
)
from federated_learning_experiments.runtime import (
    alarm_interval_minimum_description_length_selection as runtime,
)
from federated_learning_experiments.runtime import (
    observed_sample_processing as processing,
)

REVERSE_PREQUENTIAL_VARIANTS = (
    "default-reverse-prequential-grid5",
    "default-reverse-prequential-grid1",
    "default-reverse-prequential-grid1-position-cost",
)


def candidate_lengths(sequence_length, minimum_count, dense):
    """警報の保留に使う最小件数を維持し、候補の刻みだけを選ぶ。"""
    return tuple(range(minimum_count, sequence_length + 1, 1 if dense else minimum_count))


def reverse_prequential_prefixes(*, compute, **arguments):
    """渡された順序で製品の学習を1回実行し、各更新前の損失の途中和を返す。"""
    evaluate = training.evaluate_classifier_per_sample_log_losses
    prefixes = {0: 0.0}

    def observe(**samples):
        losses = evaluate(**samples)
        prefixes[len(prefixes)] = prefixes[len(prefixes) - 1] + float(losses.sum())
        return losses

    # optimizerの状態は標本間で維持され、初期化・共有部固定・batchは製品そのもの。
    with patch.object(training, "evaluate_classifier_per_sample_log_losses", observe):
        total = compute(**arguments)
    assert total == prefixes[len(prefixes) - 1]
    return prefixes


def select_reverse_assignment(*, position_cost, **arguments):
    """候補kだけを広げる。仮説の順序・加算順序・厳密な短さの比較は製品と同じ。"""
    current = arguments["current_model_log_losses"]
    lengths = arguments["new_model_code_lengths_by_change_interval_sample_count"]
    learners = arguments["new_model_learner_names_by_change_interval_sample_count"]
    switching = arguments["switching_code_length"] + (
        log(len(lengths)) if position_cost and lengths else 0.0
    )
    best = selection.MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="current_model",
        change_interval_sample_count=len(current),
        destination_model_id=None,
        description_length=sum(current),
        new_model_learner_name=selection.NO_NEW_MODEL_LEARNER_NAME,
    )
    for k in lengths:
        before = len(current) - k
        earlier = sum(current[:before]) + switching
        options = [
            (sum(losses[before:]), "held_model", model_id)
            for model_id, losses in arguments["held_model_log_losses_by_model_id"]
        ]
        options.append((lengths[k], "new_model", None))
        for later, hypothesis, model_id in options:
            total = earlier + later
            if total < best.description_length:
                best = replace(
                    best,
                    selected_hypothesis=hypothesis,
                    change_interval_sample_count=k,
                    destination_model_id=model_id,
                    description_length=total,
                    new_model_learner_name=learners[k]
                    if hypothesis == "new_model"
                    else selection.NO_NEW_MODEL_LEARNER_NAME,
                )
    return best


@contextmanager
def reverse_prequential_policy(*, variant, diagnostics):
    """run内だけ差し込み、既定にも測定用学習者の実際のoptimizer.step回数を記録する。"""
    original_select = processing.select_alarm_interval_assignment
    original_optimizer = training.create_parameter_optimizer
    diagnostics.update(learner_update_count=0, learner_update_counts_by_name={}, selections=[])

    def select(**arguments):
        call_counts = {}
        active_learner = [None]

        def optimizer_factory(**settings):
            optimizer = original_optimizer(**settings)
            step = optimizer.step

            def counted_step(*args, **kwargs):
                result = step(*args, **kwargs)
                name = active_learner[0]
                call_counts[name] = call_counts.get(name, 0) + 1
                diagnostics["learner_update_count"] += 1
                totals = diagnostics["learner_update_counts_by_name"]
                totals[name] = totals.get(name, 0) + 1
                return result

            optimizer.step = counted_step
            return optimizer

        originals = {
            selection.REINITIALIZED_LEARNER_NAME: runtime.compute_reinitialized_learner_prequential_code_length,
            selection.WARM_STARTED_LEARNER_NAME: runtime.compute_warm_started_learner_prequential_code_length,
        }

        def measured(name):
            def compute(**samples):
                active_learner[0] = name
                return originals[name](**samples)

            return compute

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(training, "create_parameter_optimizer", optimizer_factory)
            )
            # 列Wが最小変化区間より短いと切替の候補が無く、製品は学習者を使わずに「現在のまま」を選ぶ
            # （その後、件数不足で警報を保留する）。この場合は変種でも製品の選択をそのまま使う。
            reverse_candidates = (
                candidate_lengths(
                    len(arguments["pending_sample_observations"]),
                    arguments["change_interval_grid_sample_count"],
                    variant != REVERSE_PREQUENTIAL_VARIANTS[0],
                )
                if variant in REVERSE_PREQUENTIAL_VARIANTS
                else ()
            )
            if not reverse_candidates:
                for name, attribute in (
                    (
                        selection.REINITIALIZED_LEARNER_NAME,
                        "compute_reinitialized_learner_prequential_code_length",
                    ),
                    (
                        selection.WARM_STARTED_LEARNER_NAME,
                        "compute_warm_started_learner_prequential_code_length",
                    ),
                ):
                    stack.enter_context(patch.object(runtime, attribute, measured(name)))
                result = original_select(**arguments)
                candidates = (
                    candidate_lengths(
                        len(arguments["pending_sample_observations"]),
                        arguments["change_interval_grid_sample_count"],
                        False,
                    )
                    if arguments
                    else ()
                )
            else:
                observations = arguments["pending_sample_observations"]
                candidates = reverse_candidates
                classifier = (
                    arguments["held_model_training_state_registry"]
                    .get_held_model_training_state(
                        model_id=arguments[
                            "current_training_model_assignment"
                        ].current_training_model_id
                    )
                    .classifier
                )
                samples = dict(
                    template_classifier=classifier,
                    input_features=cat(
                        [o.training_sample.input_features for o in observations]
                    ).flip(0),
                    observed_class_labels=cat(
                        [o.training_sample.observed_class_labels for o in observations]
                    ).flip(0),
                    parameter_optimizer_settings=arguments["parameter_optimizer_settings"],
                )
                # kにも窓長にも依存しない初期化。runの乱数の消費は製品の隔離処理で維持する。
                seed_text = f"{arguments['client_id']}:{arguments['alarm_sample_index']}"
                seed = int.from_bytes(sha256(seed_text.encode("utf-8")).digest()[:8], "big")
                prefixes = {}
                for name in originals:
                    prefixes[name] = reverse_prequential_prefixes(
                        compute=measured(name),
                        **samples,
                        **(
                            {"initialization_seed": seed}
                            if name == selection.REINITIALIZED_LEARNER_NAME
                            else {}
                        ),
                    )
                chosen = {
                    k: selection.select_new_model_learner(
                        prequential_code_lengths_by_learner_name=tuple(
                            (name, values[k]) for name, values in prefixes.items()
                        )
                    )
                    for k in candidates
                }

                # 製品runtimeの保持モデルの評価と検査を使い、測定の再実行だけを止める。
                for name, attribute in (
                    (
                        selection.REINITIALIZED_LEARNER_NAME,
                        "compute_reinitialized_learner_prequential_code_length",
                    ),
                    (
                        selection.WARM_STARTED_LEARNER_NAME,
                        "compute_warm_started_learner_prequential_code_length",
                    ),
                ):

                    def cached(*, _name=name, **values):
                        return prefixes[_name][values["input_features"].shape[0]]

                    stack.enter_context(patch.object(runtime, attribute, cached))

                def compare(**values):
                    values.pop("change_interval_grid_sample_count")
                    values["new_model_code_lengths_by_change_interval_sample_count"] = {
                        k: v.description_length for k, v in chosen.items()
                    }
                    values["new_model_learner_names_by_change_interval_sample_count"] = {
                        k: v.learner_name for k, v in chosen.items()
                    }
                    return select_reverse_assignment(
                        position_cost=variant.endswith("-position-cost"), **values
                    )

                stack.enter_context(
                    patch.object(
                        runtime, "select_assignment_by_minimum_description_length", compare
                    )
                )
                result = original_select(**arguments)
        if arguments:
            diagnostics["selections"].append(
                dict(
                    client_id=arguments["client_id"],
                    alarm_sample_index=arguments["alarm_sample_index"],
                    window_sample_count=len(arguments["pending_sample_observations"]),
                    candidate_lengths=candidates,
                    learner_update_counts=call_counts,
                    selected_k=result.change_interval_sample_count,
                    selected_hypothesis=result.selected_hypothesis,
                )
            )
        return result

    with patch.object(processing, "select_alarm_interval_assignment", select):
        yield
