"""警報後の候補検証標本の観測を、実旧の観測処理と到達時の確定へ対照する。"""

from collections import OrderedDict

import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    convert_legacy_parameter_name,
)
from test_adopted_candidate_local_adoption import (
    build_local_adoption_oracle,
)

import federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation as observation_module
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import (
    observe_post_alarm_candidate_validation_sample,
)

# 実旧の確定処理が切替位置として記録する、最後の検証標本の位置（上流の確定testと同じ値）。
LAST_VALIDATION_SAMPLE_INDEX = 57


class IntSubclass(int):
    """位置やIDとして受理しない派生型。"""


def make_acceptance_settings(*, required_validation_sample_count):
    return CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy=(
            "current_model_first_reuse_then_two_segment_candidate_validation"
        ),
        candidate_post_alarm_validation_sample_count=required_validation_sample_count,
    )


def prepare_validation_session_in_both_implementations(
    *,
    adoption_arguments,
    legacy_client,
    class_count,
    required_validation_sample_count,
    historical_mean_case="none",
):
    """現在の保有モデルの値で、実旧の参照snapshotと同じ値の新の参照分類器・損失収集を作る。"""
    legacy_client.model_cls = lambda: ResidualAdapterMLP(input_dim=2, dataset="sine2")
    legacy_session = legacy_client._forward_validation
    legacy_session.target_count = required_validation_sample_count
    legacy_session.proposal_position = (
        LAST_VALIDATION_SAMPLE_INDEX - required_validation_sample_count
    )
    # 実旧の複製処理で、警報時点の値を持つ独立した参照モデルを作る。
    legacy_session.reference_models = legacy_client._snapshot_reference_models()
    legacy_session.candidate_losses = []
    legacy_session.reference_losses = {model_id: [] for model_id in legacy_session.reference_models}
    # 履歴平均を十分大きくすると、そのモデルは前向き損失が履歴の範囲内と評価される。
    legacy_session.reference_historical_means = {
        "none": {},
        "current": {9: 1.0},
        "other": {4: 1.0},
    }[historical_mean_case]
    reference_classifiers_by_model_id = {}
    for model_id, legacy_reference_model in legacy_session.reference_models.items():
        reference_classifier = ModelArchitectureClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=2,
            ),
            input_feature_count=2,
            hidden_layer_widths=(5, 4),
            class_count=class_count,
        )
        reference_classifier.load_state_dict(
            {
                convert_legacy_parameter_name(parameter_name): parameter_value
                for parameter_name, parameter_value in legacy_reference_model.state_dict().items()
            }
        )
        reference_classifiers_by_model_id[model_id] = reference_classifier
    observation_arguments = dict(
        candidate_classifier=adoption_arguments["adopted_candidate_classifier"],
        reference_classifiers_by_model_id=reference_classifiers_by_model_id,
        post_alarm_candidate_loss_collection=PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=make_acceptance_settings(
                required_validation_sample_count=required_validation_sample_count
            ),
            proposal_sample_index=legacy_session.proposal_position,
            reference_model_ids=tuple(reference_classifiers_by_model_id),
        ),
    )
    input_features = adoption_arguments["initial_statistics_input_features"]
    observed_class_labels = adoption_arguments["initial_statistics_observed_class_labels"]
    validation_samples = tuple(
        (
            legacy_session.proposal_position + 1 + sample_offset,
            input_features[sample_offset % len(input_features)].reshape(1, -1) * 0.5
            + 0.1 * sample_offset,
            observed_class_labels[(sample_offset * 2) % len(observed_class_labels)].reshape(1, 1),
        )
        for sample_offset in range(required_validation_sample_count)
    )
    return observation_arguments, validation_samples, legacy_session


def build_observation_oracle(
    *,
    class_count,
    monkeypatch,
    required_validation_sample_count=4,
    historical_mean_case="none",
    optimizer_variant="standard",
):
    adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
        class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
    )
    observation_arguments, validation_samples, legacy_session = (
        prepare_validation_session_in_both_implementations(
            adoption_arguments=adoption_arguments,
            legacy_client=legacy_client,
            class_count=class_count,
            required_validation_sample_count=required_validation_sample_count,
            historical_mean_case=historical_mean_case,
        )
    )
    return (
        observation_arguments,
        validation_samples,
        adoption_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_session,
    )


def assert_collected_losses_match_legacy(*, post_alarm_candidate_loss_collection, legacy_session):
    collection_state = post_alarm_candidate_loss_collection.get_state_snapshot()
    assert collection_state.candidate_losses == tuple(legacy_session.candidate_losses)
    assert tuple(
        model_id for model_id, _ in collection_state.reference_losses_by_model_id
    ) == tuple(legacy_session.reference_losses)
    for model_id, reference_losses in collection_state.reference_losses_by_model_id:
        assert reference_losses == tuple(legacy_session.reference_losses[model_id])
    assert collection_state.ready_for_acceptance_evaluation is legacy_session.ready
    assert collection_state.validation_sample_count == legacy_session.validation_count


@pytest.mark.parametrize(
    ("invalid_case", "expected_exception"),
    [
        ("collection_none", TypeError),
        ("collection_object", TypeError),
        ("collection_subclass", TypeError),
        ("reference_classifiers_list", TypeError),
        ("reference_classifiers_none", TypeError),
        ("reference_classifiers_ordered_dict", TypeError),
        ("features_two_rows", ValueError),
        ("features_one_dimension", ValueError),
        ("features_list", TypeError),
        ("features_feature_count", ValueError),
        ("features_non_finite", ValueError),
        ("labels_shape", ValueError),
        ("labels_out_of_range", ValueError),
        ("labels_list", TypeError),
        ("candidate_object", (TypeError, ValueError)),
        ("reference_classifier_object", (TypeError, ValueError)),
        ("sample_index_not_consecutive", ValueError),
        ("sample_index_repeated", ValueError),
        ("sample_index_bool", TypeError),
        ("sample_index_subclass", TypeError),
        ("reference_model_missing", ValueError),
        ("reference_model_extra", ValueError),
        ("reference_model_id_bool", TypeError),
        ("collection_already_ready", RuntimeError),
    ],
)
def test_rejected_observation_does_not_change_collection(
    invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        observation_arguments, validation_samples, _, _, _, _ = build_observation_oracle(
            class_count=4,
            required_validation_sample_count=2 if invalid_case == "collection_already_ready" else 4,
            monkeypatch=monkeypatch,
        )
        collection = observation_arguments["post_alarm_candidate_loss_collection"]
        reference_classifiers_by_model_id = observation_arguments[
            "reference_classifiers_by_model_id"
        ]
        # 1件目を正常に観測し、拒否で既存の系列が変わらないことを観測する。
        first_sample_index, first_input_features, first_observed_class_labels = validation_samples[
            0
        ]
        observe_post_alarm_candidate_validation_sample(
            sample_index=first_sample_index,
            input_features=first_input_features,
            observed_class_labels=first_observed_class_labels,
            **observation_arguments,
        )
        sample_index, input_features, observed_class_labels = validation_samples[1]
        if invalid_case == "collection_already_ready":
            observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                **observation_arguments,
            )
            sample_index += 1
        forward_call_count_before_rejection = None
        if invalid_case in ("features_two_rows", "features_one_dimension"):
            # 1標本契約の違反は、どの分類器のforwardよりも前に拒否する。
            forward_call_count_before_rejection = []
            original_evaluation = observation_module.evaluate_classifier_per_sample_bounded_losses

            def count_then_evaluate(**arguments):
                forward_call_count_before_rejection.append(1)
                return original_evaluation(**arguments)

            monkeypatch.setattr(
                observation_module,
                "evaluate_classifier_per_sample_bounded_losses",
                count_then_evaluate,
            )
        if invalid_case == "collection_none":
            observation_arguments["post_alarm_candidate_loss_collection"] = None
        elif invalid_case == "collection_object":
            observation_arguments["post_alarm_candidate_loss_collection"] = object()
        elif invalid_case == "collection_subclass":
            observation_arguments["post_alarm_candidate_loss_collection"] = type(
                "CollectionSubclass", (PostAlarmCandidateLossCollection,), {}
            )(
                candidate_model_training_and_acceptance_settings=make_acceptance_settings(
                    required_validation_sample_count=4
                ),
                proposal_sample_index=sample_index - 1,
                reference_model_ids=tuple(reference_classifiers_by_model_id),
            )
        elif invalid_case == "reference_classifiers_list":
            observation_arguments["reference_classifiers_by_model_id"] = list(
                reference_classifiers_by_model_id.items()
            )
        elif invalid_case == "reference_classifiers_none":
            observation_arguments["reference_classifiers_by_model_id"] = None
        elif invalid_case == "reference_classifiers_ordered_dict":
            observation_arguments["reference_classifiers_by_model_id"] = OrderedDict(
                reference_classifiers_by_model_id
            )
        elif invalid_case == "features_two_rows":
            input_features = torch.cat([input_features] * 2)
            observed_class_labels = torch.cat([observed_class_labels] * 2)
        elif invalid_case == "features_one_dimension":
            input_features = input_features.reshape(-1)
        elif invalid_case == "features_list":
            input_features = input_features.tolist()
        elif invalid_case == "features_feature_count":
            input_features = torch.ones((1, 3))
        elif invalid_case == "features_non_finite":
            input_features = torch.tensor([[float("nan"), 0.0]])
        elif invalid_case == "labels_shape":
            observed_class_labels = observed_class_labels.reshape(-1)
        elif invalid_case == "labels_out_of_range":
            observed_class_labels = torch.tensor([[4.0]])
        elif invalid_case == "labels_list":
            observed_class_labels = observed_class_labels.tolist()
        elif invalid_case == "candidate_object":
            observation_arguments["candidate_classifier"] = object()
        elif invalid_case == "reference_classifier_object":
            observation_arguments["reference_classifiers_by_model_id"] = {
                **reference_classifiers_by_model_id,
                9: object(),
            }
        elif invalid_case == "sample_index_not_consecutive":
            sample_index += 1
        elif invalid_case == "sample_index_repeated":
            sample_index -= 1
        elif invalid_case == "sample_index_bool":
            sample_index = True
        elif invalid_case == "sample_index_subclass":
            sample_index = IntSubclass(sample_index)
        elif invalid_case == "reference_model_missing":
            observation_arguments["reference_classifiers_by_model_id"] = {
                4: reference_classifiers_by_model_id[4]
            }
        elif invalid_case == "reference_model_extra":
            observation_arguments["reference_classifiers_by_model_id"] = {
                **reference_classifiers_by_model_id,
                12: reference_classifiers_by_model_id[4],
            }
        elif invalid_case == "reference_model_id_bool":
            observation_arguments["reference_classifiers_by_model_id"] = {
                4: reference_classifiers_by_model_id[4],
                True: reference_classifiers_by_model_id[9],
            }
        previous_collection_state = collection.get_state_snapshot()
        with pytest.raises(expected_exception):
            observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                **observation_arguments,
            )
        assert collection.get_state_snapshot() == previous_collection_state
        if forward_call_count_before_rejection is not None:
            assert forward_call_count_before_rejection == []


def test_all_losses_are_evaluated_before_single_collection_update(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        observation_arguments, validation_samples, _, _, _, _ = build_observation_oracle(
            class_count=2, monkeypatch=monkeypatch
        )
        actual_call_order = []
        original_evaluation = observation_module.evaluate_classifier_per_sample_bounded_losses

        def record_then_evaluate(**arguments):
            actual_call_order.append(arguments["classifier"])
            return original_evaluation(**arguments)

        monkeypatch.setattr(
            observation_module,
            "evaluate_classifier_per_sample_bounded_losses",
            record_then_evaluate,
        )
        collection = observation_arguments["post_alarm_candidate_loss_collection"]
        original_observation = collection.observe_losses_after_label_observation

        def record_then_observe(**arguments):
            actual_call_order.append("observe_losses_after_label_observation")
            return original_observation(**arguments)

        monkeypatch.setattr(
            collection, "observe_losses_after_label_observation", record_then_observe
        )
        sample_index, input_features, observed_class_labels = validation_samples[0]
        observe_post_alarm_candidate_validation_sample(
            sample_index=sample_index,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
            **observation_arguments,
        )
        expected_call_order = [
            observation_arguments["candidate_classifier"],
            *observation_arguments["reference_classifiers_by_model_id"].values(),
            "observe_losses_after_label_observation",
        ]
        assert len(actual_call_order) == len(expected_call_order)
        assert all(
            actual is expected for actual, expected in zip(actual_call_order, expected_call_order)
        )
