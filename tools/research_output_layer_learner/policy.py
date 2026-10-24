"""warm学習者の制限と、選ばれた候補の初期値・学習範囲を対照する。"""

from contextlib import contextmanager, nullcontext
from unittest.mock import patch


@contextmanager
def output_layer_training():
    """adapterを勾配なしで呼び、出力層への逆伝播だけを残す。終了後に戻す。"""
    from torch import no_grad
    from federated_learning_experiments.learning.models.model_architecture_classifier import (
        ModelArchitectureClassifier,
    )

    def forward(self, shared_features):
        with no_grad():
            features = self.model_owned_feature_transform(shared_features)
        return self.output_activation(self.classification_layer(features))

    # requires_gradを変えないので、製品のoptimizer参照検査と採用後の通常学習を維持する。
    # adapterはgrad=Noneとなり、weight decayやmomentumも進まない。
    with patch.object(
        ModelArchitectureClassifier, "forward_from_shared_features", forward
    ):
        yield


@contextmanager
def learner_policy(*, variant, events):
    """既定、warm初期値を揃えた対照、S3を同じ2学習者・ln2で比べる。"""
    import torch
    import federated_learning_experiments.runtime.alarm_buffer_response as response
    import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution
    import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as selection

    if variant == "product-default":
        yield
        return
    if variant not in ("warm-matched", "output-layer-matched"):
        raise ValueError(variant)
    head_only = variant == "output-layer-matched"
    original_score = selection.compute_warm_started_learner_prequential_code_length
    original_resolve = response.resolve_alarm_change_interval
    original_train = resolution.train_candidate_on_alarm_change_interval

    def score(**arguments):
        with output_layer_training() if head_only else nullcontext():
            value = original_score(**arguments)
        events.append(
            dict(event="warm_score", samples=len(arguments["input_features"]))
        )
        return value

    def resolve(**arguments):
        chosen = arguments.get("alarm_interval_assignment_selection")
        events.append(
            dict(
                event="assignment",
                hypothesis=None if chosen is None else chosen.selected_hypothesis,
                learner=None if chosen is None else chosen.new_model_learner_name,
            )
        )
        if chosen is None or chosen.new_model_learner_name != "warm_started_learner":
            return original_resolve(**arguments)

        def train(**training_arguments):
            registry = training_arguments["held_model_training_state_registry"]
            current_id = training_arguments[
                "current_training_model_assignment"
            ].current_training_model_id
            current = registry.get_held_model_training_state(
                model_id=current_id
            ).classifier
            initial = {k: v.detach().clone() for k, v in current.state_dict().items()}
            initial_changed = any(
                not torch.equal(
                    v, training_arguments["initial_candidate_parameter_snapshot"][k]
                )
                for k, v in initial.items()
            )
            training_arguments["initial_candidate_parameter_snapshot"] = initial
            with output_layer_training() if head_only else nullcontext():
                result = original_train(**training_arguments)
            classifier = result.candidate_training_state.candidate_classifier
            frozen_unchanged = all(
                torch.equal(v, classifier.state_dict()[k])
                for k, v in initial.items()
                if not k.startswith("classification_layer.")
            )
            if head_only and not frozen_unchanged:
                raise AssertionError("S3候補の共有部またはadapterが変化しました")
            events.append(
                dict(
                    event="matched_candidate",
                    head_only=head_only,
                    initialization_changed=initial_changed,
                    frozen_unchanged=frozen_unchanged,
                    trained_samples=result.candidate_epoch_training_result.candidate_trained_sample_count,
                )
            )
            return result

        with patch.object(
            resolution, "train_candidate_on_alarm_change_interval", train
        ):
            return original_resolve(**arguments)

    with (
        patch.object(
            selection, "compute_warm_started_learner_prequential_code_length", score
        ),
        patch.object(response, "resolve_alarm_change_interval", resolve),
    ):
        yield
