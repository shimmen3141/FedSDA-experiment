"""採用候補の一時ID登録を、実旧登録処理と旧FedSDAの待機設定へ対照する。"""

import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_classifier_bounded_loss_evaluation import assert_initial_loss_statistics_match_legacy
from test_joint_model_parameter_update import assert_nested_state_equal, run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

import federated_learning_experiments.runtime.adopted_candidate_initial_local_registration as registration_module
from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.clients.shared_backbone import (
    SharedBackboneClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.data.specs import DatasetSpec
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import (
    register_adopted_candidate_as_temporary_held_model,
)
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    confirm_held_model_registration,
)

TEMPORARY_MODEL_ID = -7
OWNER_NAMES = (
    "held_model_training_state_registry",
    "loss_statistics_store",
    "current_training_model_assignment",
    "pending_model_upload_state",
)


class IntSubclass(int):
    """IDや回数として受理しない派生型。"""


def convert_legacy_parameter_name(legacy_parameter_name):
    return (
        legacy_parameter_name.replace("backbone.net.", "feature_extractor.hidden_layers.")
        .replace("adapter.down.", "residual_adapter.feature_compression.")
        .replace("adapter.up.", "residual_adapter.feature_expansion.")
        .replace("head.", "classification_layer.")
    )


def build_initial_registration_oracle(
    *,
    class_count,
    held_model_ids,
    current_model_is_held,
    monkeypatch,
    optimizer_variant="standard",
    held_models_share_feature_extractor=True,
    existing_pending_model_id=None,
    statistics_sample_labels=None,
):
    """同じ初期値の新owner群・候補と、実旧client・旧候補を構築する。"""
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset: DatasetSpec(
            input_dim=2, num_concepts=2, num_classes=class_count, hidden_dims=(5, 4)
        ),
    )
    monkeypatch.setattr(config, "SHARED_ADAPTER_RANK", 2)
    monkeypatch.setattr(config, "OPTIMIZER", "sgd" if optimizer_variant == "sgd" else "adam")
    monkeypatch.setattr(config, "BASE_LR", 0.01)
    monkeypatch.setattr(config, "NEW_MODEL_LR", 0.01)
    monkeypatch.setattr(config, "WEIGHT_DECAY", 0.001)
    monkeypatch.setattr(config, "AMSGRAD", optimizer_variant == "amsgrad")
    optimizer_settings = (
        SgdParameterOptimizerSettings(learning_rate=0.01)
        if optimizer_variant == "sgd"
        else AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
        )
    )
    legacy_models = []
    classifiers = []
    optimizer_owners = []
    shared_optimizer_owners = []
    # 末尾が独立した共有部を持つ候補、それ以前が保有モデル。
    for model_index in range(len(held_model_ids) + 1):
        is_candidate = model_index == len(held_model_ids)
        reuse_first_feature_extractor = (
            bool(legacy_models) and not is_candidate and held_models_share_feature_extractor
        )
        legacy_model = ResidualAdapterMLP(
            input_dim=2,
            dataset="sine2",
            backbone=legacy_models[0].backbone if reuse_first_feature_extractor else None,
        )
        # ゼロ初期化adapterだけの対照にならないよう、モデルごとに異なる値を置く。
        with torch.no_grad():
            legacy_model.adapter.up.weight.fill_(0.15 + 0.02 * model_index)
            legacy_model.adapter.up.bias.fill_(0.05 - 0.01 * model_index)
        classifier = ResidualAdapterClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=2,
            ),
            input_feature_count=2,
            hidden_layer_widths=(5, 4),
            class_count=class_count,
            shared_feature_extractor=classifiers[0].feature_extractor
            if reuse_first_feature_extractor
            else None,
        )
        classifier.load_state_dict(
            {
                convert_legacy_parameter_name(parameter_name): parameter_value
                for parameter_name, parameter_value in legacy_model.state_dict().items()
            }
        )
        optimizer_owner = ParameterOptimizerState(
            parameters=tuple(classifier.residual_adapter.parameters())
            + tuple(classifier.classification_layer.parameters()),
            optimizer_settings=optimizer_settings,
        )
        shared_optimizer_owner = (
            shared_optimizer_owners[0]
            if reuse_first_feature_extractor
            else ParameterOptimizerState(
                parameters=tuple(classifier.feature_extractor.parameters()),
                optimizer_settings=optimizer_settings,
            )
        )
        legacy_models.append(legacy_model)
        classifiers.append(classifier)
        optimizer_owners.append(optimizer_owner)
        shared_optimizer_owners.append(shared_optimizer_owner)
    # 蓄積stateとgradを持たせ、resetされるoptimizerと保持されるoptimizerを区別する。
    stepped_shared_optimizer_ids = set()
    for model_index, (legacy_model, classifier) in enumerate(zip(legacy_models, classifiers)):
        for parameter in tuple(classifier.parameters()) + tuple(legacy_model.parameters()):
            parameter.grad = torch.full_like(parameter, 0.5 + 0.25 * model_index)
        optimizer_owners[model_index].parameter_optimizer.step()
        legacy_model.head_optimizer.step()
        shared_optimizer = shared_optimizer_owners[model_index].parameter_optimizer
        if id(shared_optimizer) not in stepped_shared_optimizer_ids:
            stepped_shared_optimizer_ids.add(id(shared_optimizer))
            shared_optimizer.step()
            legacy_model.backbone.optimizer.step()

    registry = HeldModelTrainingStateRegistry()
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    legacy_client = SharedBackboneClassConditionalESRFedSDAClient.__new__(
        SharedBackboneClassConditionalESRFedSDAClient
    )
    legacy_client.models = {}
    legacy_client.model_stats = {}
    for model_index, model_id in enumerate(held_model_ids):
        registry.register_held_model_training_state(
            model_id=model_id,
            classifier=classifiers[model_index],
            concept_specific_parameter_optimizer_state=optimizer_owners[model_index],
        )
        loss_statistics_store.set_model_loss_statistics(
            model_id=model_id,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=3 + model_index,
                    mean_loss=0.25 + 0.125 * model_index,
                    sum_squared_loss_deviations=0.5,
                ),
                class_loss_moments_by_class_id=(
                    (
                        1,
                        BoundedLossMoments(
                            observed_loss_count=2,
                            mean_loss=0.125,
                            sum_squared_loss_deviations=0.25,
                        ),
                    ),
                ),
            ),
        )
        legacy_client.models[model_id] = legacy_models[model_index]
        legacy_client.model_stats[model_id] = {
            "n": 3 + model_index,
            "mean": 0.25 + 0.125 * model_index,
            "M2": 0.5,
            "class_stats": {1: {"n": 2, "mean": 0.125, "M2": 0.25}},
        }
    current_model_id = held_model_ids[-1] if current_model_is_held and held_model_ids else 77
    pending_upload_state = PendingModelUploadState()
    legacy_client.current_model_id = current_model_id
    legacy_client.compute_counters = defaultdict(int)
    legacy_client.verbose = False
    legacy_client.client_id = 0
    legacy_client.pending_model_params = None
    legacy_client.pending_model_stats = None
    legacy_client.pending_model_ready = True
    legacy_client._pending_upload_rounds = 0
    if existing_pending_model_id is not None:
        pending_upload_state.queue_model_upload(
            model_id=existing_pending_model_id,
            parameter_snapshot=snapshot_classifier_parameters(classifier=classifiers[0]),
            upload_delay_round_count=3,
        )
        legacy_client.pending_model_params = legacy_models[0].get_params()
        legacy_client.pending_model_stats = legacy_client.model_stats.get(held_model_ids[0])
        legacy_client.pending_model_ready = False
        legacy_client._pending_upload_rounds = 3

    if statistics_sample_labels is None:
        statistics_sample_labels = tuple(
            sample_index % class_count for sample_index in range(class_count + 3)
        )
    statistics_sample_count = len(statistics_sample_labels)
    input_features = (
        torch.arange(statistics_sample_count * 2, dtype=torch.float32).reshape(
            statistics_sample_count, 2
        )
        - 3
    ) / 7
    observed_class_labels = torch.tensor(statistics_sample_labels, dtype=torch.float32).reshape(
        -1, 1
    )
    registration_arguments = dict(
        temporary_model_id=TEMPORARY_MODEL_ID,
        adopted_candidate_classifier=classifiers[-1],
        candidate_concept_specific_parameter_optimizer_state=optimizer_owners[-1],
        initial_statistics_input_features=input_features,
        initial_statistics_observed_class_labels=observed_class_labels,
        upload_delay_round_count=2,
        held_model_training_state_registry=registry,
        loss_statistics_store=loss_statistics_store,
        current_training_model_assignment=CurrentTrainingModelAssignment(
            initial_model_id=current_model_id
        ),
        pending_model_upload_state=pending_upload_state,
    )
    return registration_arguments, shared_optimizer_owners, legacy_client, legacy_models[-1]


def register_candidate_in_legacy_client(*, legacy_client, legacy_candidate_model, arguments):
    """旧登録と、旧FedSDAが登録直後に行う待機設定を同じ順で行う。"""
    legacy_client._register_trained_new_model(
        arguments["temporary_model_id"],
        legacy_candidate_model,
        arguments["initial_statistics_input_features"].clone(),
        arguments["initial_statistics_observed_class_labels"].clone(),
        pending_ready=False,
    )
    legacy_client._pending_upload_rounds = arguments["upload_delay_round_count"]


def assert_held_model_states_match_legacy(
    *, registry, shared_optimizer_owners, legacy_client, input_features
):
    """保有一覧の順序と、各モデルの全値・grad・optimizer state・共有参照・出力を対照する。"""
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    assert tuple(state.model_id for state in held_model_training_states) == tuple(
        legacy_client.models
    )
    shared_optimizer_owner_by_extractor_id = {
        id(tuple(owner.parameter_optimizer.param_groups[0]["params"])[0]): owner
        for owner in shared_optimizer_owners
    }
    legacy_models = tuple(legacy_client.models.values())
    for state_index, (state, legacy_model) in enumerate(
        zip(held_model_training_states, legacy_models)
    ):
        legacy_parameters = dict(legacy_model.named_parameters())
        assert tuple(
            parameter_name for parameter_name, _ in state.classifier.named_parameters()
        ) == tuple(
            convert_legacy_parameter_name(parameter_name) for parameter_name in legacy_parameters
        )
        for (_, parameter), legacy_parameter in zip(
            state.classifier.named_parameters(), legacy_parameters.values()
        ):
            assert torch.equal(parameter, legacy_parameter)
            assert_nested_state_equal(parameter.grad, legacy_parameter.grad)
        assert_nested_state_equal(
            state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
            legacy_model.head_optimizer.state_dict(),
        )
        first_shared_parameter = next(iter(state.classifier.feature_extractor.parameters()))
        assert_nested_state_equal(
            shared_optimizer_owner_by_extractor_id[
                id(first_shared_parameter)
            ].parameter_optimizer.state_dict(),
            legacy_model.backbone.optimizer.state_dict(),
        )
        # 共有参照の構造（どのモデル同士が同じ共有部か）も旧と同じにする。
        for other_state, other_legacy_model in zip(
            held_model_training_states[state_index + 1 :], legacy_models[state_index + 1 :]
        ):
            assert (
                state.classifier.feature_extractor is other_state.classifier.feature_extractor
            ) is (legacy_model.backbone is other_legacy_model.backbone)
        with torch.no_grad():
            assert torch.equal(state.classifier(input_features), legacy_model(input_features))


def assert_initial_registration_matches_legacy(
    *, registration_arguments, shared_optimizer_owners, legacy_client
):
    registry = registration_arguments["held_model_training_state_registry"]
    assert_held_model_states_match_legacy(
        registry=registry,
        shared_optimizer_owners=shared_optimizer_owners,
        legacy_client=legacy_client,
        input_features=registration_arguments["initial_statistics_input_features"],
    )
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    loss_statistics_store = registration_arguments["loss_statistics_store"]
    assert_store_statistics_match_legacy(
        loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
    )
    temporary_model_id = registration_arguments["temporary_model_id"]
    # 旧の保留統計は登録した統計dictの別名。新は対応IDから現在統計を取得する。
    assert legacy_client.pending_model_stats is legacy_client.model_stats[temporary_model_id]
    assert_initial_loss_statistics_match_legacy(
        loss_statistics_store.get_model_loss_statistics(model_id=temporary_model_id),
        legacy_client.pending_model_stats,
    )
    pending_upload_state = registration_arguments["pending_model_upload_state"]
    pending_model_upload = pending_upload_state.get_pending_model_upload()
    assert pending_model_upload.model_id == temporary_model_id
    legacy_pending_parameters = legacy_client.pending_model_params
    assert tuple(pending_model_upload.parameter_snapshot) == tuple(
        convert_legacy_parameter_name(parameter_name)
        for parameter_name in legacy_pending_parameters
    )
    registered_parameter_storage_addresses = {
        parameter.data_ptr()
        for state in held_model_training_states
        for parameter in state.classifier.parameters()
    }
    for parameter_values, legacy_parameter_values in zip(
        pending_model_upload.parameter_snapshot.values(), legacy_pending_parameters.values()
    ):
        assert torch.equal(parameter_values, legacy_parameter_values)
        assert not parameter_values.requires_grad
        assert parameter_values.data_ptr() not in registered_parameter_storage_addresses
    # 待機が満了するラウンド境界も旧と同じにする。
    for _ in range(registration_arguments["upload_delay_round_count"] + 1):
        assert pending_upload_state.has_ready_model_upload() is legacy_client.has_pending_model()
        assert pending_upload_state.remaining_upload_delay_round_count == max(
            0, legacy_client._pending_upload_rounds
        )
        pending_upload_state.advance_upload_readiness_at_round_boundary()
        legacy_client.promote_pending_to_ready()
    assert pending_upload_state.has_ready_model_upload()


def snapshot_registration_state(*, registration_arguments, shared_optimizer_owners):
    """拒否時の不変を比較する、owner構造・共有部・optimizer・乱数の観測値。"""
    registry = registration_arguments["held_model_training_state_registry"]
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    candidate = registration_arguments["adopted_candidate_classifier"]
    candidate_optimizer_owner = registration_arguments[
        "candidate_concept_specific_parameter_optimizer_state"
    ]
    classifiers = tuple(state.classifier for state in held_model_training_states) + (
        (candidate,) if isinstance(candidate, torch.nn.Module) else ()
    )
    optimizer_owners = (
        tuple(
            state.concept_specific_parameter_optimizer_state for state in held_model_training_states
        )
        + tuple(shared_optimizer_owners)
        + (
            (candidate_optimizer_owner,)
            if isinstance(candidate_optimizer_owner, ParameterOptimizerState)
            else ()
        )
    )
    pending_upload_state = registration_arguments["pending_model_upload_state"]
    current_training_model_assignment = registration_arguments["current_training_model_assignment"]
    loss_statistics_store = registration_arguments["loss_statistics_store"]
    return dict(
        held_model_training_states=held_model_training_states,
        feature_extractors=tuple(
            getattr(classifier, "feature_extractor", None) for classifier in classifiers
        ),
        parameter_snapshots=snapshot_parameter_values_and_gradients(
            parameter for classifier in classifiers for parameter in classifier.parameters()
        ),
        optimizers=tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in optimizer_owners
        ),
        loss_statistics=(
            loss_statistics_store.get_state_snapshot()
            if isinstance(loss_statistics_store, ModelAndClassLossStatisticsStore)
            else None
        ),
        pending_model_upload=(
            (
                pending_upload_state.get_pending_model_upload(),
                pending_upload_state.remaining_upload_delay_round_count,
            )
            if isinstance(pending_upload_state, PendingModelUploadState)
            else None
        ),
        current_training_model_id=(
            current_training_model_assignment.current_training_model_id
            if isinstance(current_training_model_assignment, CurrentTrainingModelAssignment)
            else None
        ),
        random_states=(torch.get_rng_state().clone(), random.getstate(), np.random.get_state()),
        classifiers=classifiers,
    )


def assert_registration_state_unchanged(
    *,
    previous_snapshot,
    registry,
    loss_statistics_store,
    pending_upload_state,
    current_training_model_assignment,
):
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    assert len(held_model_training_states) == len(previous_snapshot["held_model_training_states"])
    for state, previous_state in zip(
        held_model_training_states, previous_snapshot["held_model_training_states"]
    ):
        assert state is previous_state
    for classifier, feature_extractor in zip(
        previous_snapshot["classifiers"], previous_snapshot["feature_extractors"]
    ):
        assert getattr(classifier, "feature_extractor", None) is feature_extractor
    assert_parameter_values_and_gradients_unchanged(previous_snapshot["parameter_snapshots"])
    for owner, previous_optimizer, previous_optimizer_state in previous_snapshot["optimizers"]:
        assert owner.parameter_optimizer is previous_optimizer
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
    assert loss_statistics_store.get_state_snapshot() == previous_snapshot["loss_statistics"]
    pending_model_upload, remaining_round_count = previous_snapshot["pending_model_upload"]
    assert pending_upload_state.get_pending_model_upload() is pending_model_upload
    assert pending_upload_state.remaining_upload_delay_round_count == remaining_round_count
    assert (
        current_training_model_assignment.current_training_model_id
        == previous_snapshot["current_training_model_id"]
    )
    previous_torch_state, previous_python_state, previous_numpy_state = previous_snapshot[
        "random_states"
    ]
    assert torch.equal(torch.get_rng_state(), previous_torch_state)
    assert random.getstate() == previous_python_state
    numpy_state = np.random.get_state()
    assert numpy_state[0] == previous_numpy_state[0]
    assert np.array_equal(numpy_state[1], previous_numpy_state[1])
    assert numpy_state[2:] == previous_numpy_state[2:]


def assert_rejected_without_any_change(
    *, registration_arguments, shared_optimizer_owners, valid_owners, expected_exception
):
    previous_snapshot = snapshot_registration_state(
        registration_arguments={**registration_arguments, **valid_owners},
        shared_optimizer_owners=shared_optimizer_owners,
    )
    with pytest.raises(expected_exception):
        register_adopted_candidate_as_temporary_held_model(**registration_arguments)
    assert_registration_state_unchanged(
        previous_snapshot=previous_snapshot,
        registry=valid_owners["held_model_training_state_registry"],
        loss_statistics_store=valid_owners["loss_statistics_store"],
        pending_upload_state=valid_owners["pending_model_upload_state"],
        current_training_model_assignment=valid_owners["current_training_model_assignment"],
    )


def select_valid_owners(registration_arguments):
    return {owner_name: registration_arguments[owner_name] for owner_name in OWNER_NAMES}


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("held_model_ids", [(4,), (4, 9), (9, -3, 4)])
@pytest.mark.parametrize("current_model_is_held", [True, False])
@pytest.mark.parametrize("held_models_share_feature_extractor", [True, False])
@pytest.mark.parametrize("existing_pending_model_id", [None, 9])
def test_initial_registration_matches_old_registration_and_upload_delay(
    class_count,
    held_model_ids,
    current_model_is_held,
    held_models_share_feature_extractor,
    existing_pending_model_id,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(211)
        registration_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
            build_initial_registration_oracle(
                class_count=class_count,
                held_model_ids=held_model_ids,
                current_model_is_held=current_model_is_held,
                held_models_share_feature_extractor=held_models_share_feature_extractor,
                existing_pending_model_id=existing_pending_model_id,
                monkeypatch=monkeypatch,
            )
        )
        registry = registration_arguments["held_model_training_state_registry"]
        candidate = registration_arguments["adopted_candidate_classifier"]
        candidate_optimizer_owner = registration_arguments[
            "candidate_concept_specific_parameter_optimizer_state"
        ]
        current_model_id = registration_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        previous_states = registry.snapshot_ordered_held_model_training_states()
        expected_active_feature_extractor = (
            registry.get_held_model_training_state(model_id=current_model_id)
            if current_model_is_held
            else previous_states[0]
        ).classifier.feature_extractor
        active_parameters = tuple(expected_active_feature_extractor.parameters())
        previous_candidate_optimizer = candidate_optimizer_owner.parameter_optimizer
        assert previous_candidate_optimizer.state_dict()["state"] or isinstance(
            previous_candidate_optimizer, torch.optim.SGD
        )
        previous_optimizers = tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in tuple(shared_optimizer_owners[:-1])
            + tuple(state.concept_specific_parameter_optimizer_state for state in previous_states)
        )
        concept_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for classifier in tuple(state.classifier for state in previous_states) + (candidate,)
            for parameter in tuple(classifier.residual_adapter.parameters())
            + tuple(classifier.classification_layer.parameters())
        )
        random_states = (torch.get_rng_state().clone(), random.getstate())

        register_candidate_in_legacy_client(
            legacy_client=legacy_client,
            legacy_candidate_model=legacy_candidate_model,
            arguments=registration_arguments,
        )
        assert register_adopted_candidate_as_temporary_held_model(**registration_arguments) is None

        assert candidate.feature_extractor is expected_active_feature_extractor
        assert all(
            parameter is previous_parameter
            for parameter, previous_parameter in zip(
                expected_active_feature_extractor.parameters(), active_parameters
            )
        )
        states = registry.snapshot_ordered_held_model_training_states()
        assert states[:-1] == previous_states
        assert all(state is previous for state, previous in zip(states, previous_states))
        assert states[-1].model_id == TEMPORARY_MODEL_ID
        assert states[-1].classifier is candidate
        assert states[-1].concept_specific_parameter_optimizer_state is candidate_optimizer_owner
        assert candidate_optimizer_owner.parameter_optimizer is not previous_candidate_optimizer
        assert not candidate_optimizer_owner.parameter_optimizer.state
        for owner, previous_optimizer, previous_optimizer_state in previous_optimizers:
            assert owner.parameter_optimizer is previous_optimizer
            assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert_parameter_values_and_gradients_unchanged(concept_parameter_snapshots)
        assert (
            registration_arguments["current_training_model_assignment"].current_training_model_id
            == current_model_id
        )
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        assert_initial_registration_matches_legacy(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )


# 単一標本では実旧のtorch.varが自由度0のwarningを出す。旧はNaN分散を0.1へ置換する。
@pytest.mark.filterwarnings(r"ignore:var\(\).*degrees of freedom:UserWarning")
@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "statistics_sample_labels",
    [(1,), (0, 1, 1), (1, 1, 1, 1), (0, 0, 1, 1, 1, 0, 1)],
)
def test_initial_statistics_match_old_singleton_and_missing_class_batches(
    class_count, statistics_sample_labels, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(223)
        if class_count == 4:
            statistics_sample_labels = tuple(label * 3 for label in statistics_sample_labels)
        registration_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
            build_initial_registration_oracle(
                class_count=class_count,
                held_model_ids=(4, 9),
                current_model_is_held=True,
                statistics_sample_labels=statistics_sample_labels,
                monkeypatch=monkeypatch,
            )
        )
        register_candidate_in_legacy_client(
            legacy_client=legacy_client,
            legacy_candidate_model=legacy_candidate_model,
            arguments=registration_arguments,
        )
        register_adopted_candidate_as_temporary_held_model(**registration_arguments)
        assert_initial_registration_matches_legacy(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )


@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
def test_initial_registration_matches_old_for_each_optimizer(optimizer_variant, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(227)
        registration_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
            build_initial_registration_oracle(
                class_count=4,
                held_model_ids=(4, 9),
                current_model_is_held=True,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        register_candidate_in_legacy_client(
            legacy_client=legacy_client,
            legacy_candidate_model=legacy_candidate_model,
            arguments=registration_arguments,
        )
        register_adopted_candidate_as_temporary_held_model(**registration_arguments)
        assert_initial_registration_matches_legacy(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )


def test_candidate_already_attached_to_active_feature_extractor_is_registered(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, _, _, _ = build_initial_registration_oracle(
            class_count=2, held_model_ids=(4,), current_model_is_held=True, monkeypatch=monkeypatch
        )
        registry = registration_arguments["held_model_training_state_registry"]
        active = registry.get_held_model_training_state(model_id=4).classifier.feature_extractor
        candidate = registration_arguments["adopted_candidate_classifier"]
        candidate.attach_shared_feature_extractor(shared_feature_extractor=active)
        active_snapshot = snapshot_parameter_values_and_gradients(active.parameters())
        register_adopted_candidate_as_temporary_held_model(**registration_arguments)
        assert candidate.feature_extractor is active
        assert_parameter_values_and_gradients_unchanged(active_snapshot)
        assert (
            registry.get_held_model_training_state(model_id=TEMPORARY_MODEL_ID).classifier
            is candidate
        )


@pytest.mark.parametrize(
    ("invalid_temporary_model_id", "expected_exception"),
    [
        (0, ValueError),
        (5, ValueError),
        (10**40, ValueError),
        (True, TypeError),
        (False, TypeError),
        (-7.0, TypeError),
        ("-7", TypeError),
        (None, TypeError),
        (IntSubclass(-7), TypeError),
    ],
)
def test_invalid_temporary_model_id_is_rejected_without_any_change(
    invalid_temporary_model_id, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, shared_optimizer_owners, _, _ = build_initial_registration_oracle(
            class_count=2,
            held_model_ids=(4, 9),
            current_model_is_held=True,
            monkeypatch=monkeypatch,
        )
        registration_arguments["temporary_model_id"] = invalid_temporary_model_id
        assert_rejected_without_any_change(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            valid_owners=select_valid_owners(registration_arguments),
            expected_exception=expected_exception,
        )


@pytest.mark.parametrize("operation_name", ["registry", "statistics", "pending"])
def test_temporary_model_id_already_in_use_is_rejected_without_any_change(
    operation_name, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, shared_optimizer_owners, _, _ = build_initial_registration_oracle(
            class_count=2,
            held_model_ids=(4, TEMPORARY_MODEL_ID) if operation_name == "registry" else (4, 9),
            current_model_is_held=False,
            existing_pending_model_id=TEMPORARY_MODEL_ID if operation_name == "pending" else None,
            monkeypatch=monkeypatch,
        )
        loss_statistics_store = registration_arguments["loss_statistics_store"]
        if operation_name == "registry":
            # 一覧だけの重複を観測するため、同IDの統計は別IDへ移す。
            loss_statistics_store.reassign_model_loss_statistics_id(
                original_model_id=TEMPORARY_MODEL_ID, reassigned_model_id=30
            )
        elif operation_name == "statistics":
            loss_statistics_store.set_model_loss_statistics(
                model_id=TEMPORARY_MODEL_ID,
                loss_statistics=loss_statistics_store.get_model_loss_statistics(model_id=4),
            )
        assert_rejected_without_any_change(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            valid_owners=select_valid_owners(registration_arguments),
            expected_exception=ValueError,
        )


@pytest.mark.parametrize(
    ("invalid_upload_delay_round_count", "expected_exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (IntSubclass(1), TypeError),
    ],
)
def test_invalid_upload_delay_round_count_is_rejected_without_any_change(
    invalid_upload_delay_round_count, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, shared_optimizer_owners, _, _ = build_initial_registration_oracle(
            class_count=2,
            held_model_ids=(4, 9),
            current_model_is_held=True,
            monkeypatch=monkeypatch,
        )
        registration_arguments["upload_delay_round_count"] = invalid_upload_delay_round_count
        assert_rejected_without_any_change(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            valid_owners=select_valid_owners(registration_arguments),
            expected_exception=expected_exception,
        )


@pytest.mark.parametrize("invalid_owner_name", OWNER_NAMES)
@pytest.mark.parametrize("invalid_owner", ["none", "object", "subclass"])
def test_invalid_owner_is_rejected_without_any_change(
    invalid_owner_name, invalid_owner, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, shared_optimizer_owners, _, _ = build_initial_registration_oracle(
            class_count=2,
            held_model_ids=(4, 9),
            current_model_is_held=True,
            monkeypatch=monkeypatch,
        )
        valid_owners = select_valid_owners(registration_arguments)
        if invalid_owner == "subclass":
            owner_type = type(valid_owners[invalid_owner_name])
            owner_subclass = type("OwnerSubclass", (owner_type,), {})
            registration_arguments[invalid_owner_name] = (
                owner_subclass(initial_model_id=4)
                if owner_type is CurrentTrainingModelAssignment
                else owner_subclass()
            )
        else:
            registration_arguments[invalid_owner_name] = (
                None if invalid_owner == "none" else object()
            )
        assert_rejected_without_any_change(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            valid_owners=valid_owners,
            expected_exception=TypeError,
        )


def test_empty_held_model_list_is_rejected_without_any_change(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, shared_optimizer_owners, _, _ = build_initial_registration_oracle(
            class_count=2, held_model_ids=(), current_model_is_held=False, monkeypatch=monkeypatch
        )
        assert_rejected_without_any_change(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            valid_owners=select_valid_owners(registration_arguments),
            expected_exception=LookupError,
        )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "candidate_type",
        "optimizer_owner_type",
        "optimizer_owner_wrong_parameters",
        "candidate_feature_extractor_structure",
        "features_type",
        "features_empty",
        "features_feature_count",
        "features_non_finite",
        "features_dtype",
        "labels_type",
        "labels_shape",
        "labels_out_of_range",
        "labels_fractional",
        "candidate_parameter_non_finite",
        "candidate_shared_parameter_non_finite",
    ],
)
def test_invalid_candidate_or_samples_are_rejected_without_any_change(invalid_case, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, shared_optimizer_owners, _, _ = build_initial_registration_oracle(
            class_count=4,
            held_model_ids=(4, 9),
            current_model_is_held=True,
            monkeypatch=monkeypatch,
        )
        candidate = registration_arguments["adopted_candidate_classifier"]
        input_features = registration_arguments["initial_statistics_input_features"]
        observed_class_labels = registration_arguments["initial_statistics_observed_class_labels"]
        if invalid_case == "candidate_type":
            registration_arguments["adopted_candidate_classifier"] = object()
        elif invalid_case == "optimizer_owner_type":
            registration_arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                object()
            )
        elif invalid_case == "optimizer_owner_wrong_parameters":
            registration_arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                shared_optimizer_owners[-1]
            )
        elif invalid_case == "candidate_feature_extractor_structure":
            candidate.feature_extractor.hidden_layers[1] = torch.nn.Identity()
        elif invalid_case == "features_type":
            registration_arguments["initial_statistics_input_features"] = input_features.tolist()
        elif invalid_case == "features_empty":
            registration_arguments["initial_statistics_input_features"] = input_features[:0]
            registration_arguments["initial_statistics_observed_class_labels"] = (
                observed_class_labels[:0]
            )
        elif invalid_case == "features_feature_count":
            registration_arguments["initial_statistics_input_features"] = torch.ones(
                (len(input_features), 3)
            )
        elif invalid_case == "features_non_finite":
            input_features[1, 0] = float("nan")
        elif invalid_case == "features_dtype":
            registration_arguments["initial_statistics_input_features"] = input_features.double()
        elif invalid_case == "labels_type":
            registration_arguments["initial_statistics_observed_class_labels"] = (
                observed_class_labels.tolist()
            )
        elif invalid_case == "labels_shape":
            registration_arguments["initial_statistics_observed_class_labels"] = (
                observed_class_labels.reshape(-1)
            )
        elif invalid_case == "labels_out_of_range":
            observed_class_labels[0, 0] = 4.0
        elif invalid_case == "labels_fractional":
            observed_class_labels[0, 0] = 0.5
        elif invalid_case == "candidate_parameter_non_finite":
            with torch.no_grad():
                candidate.classification_layer.bias[0] = float("inf")
        elif invalid_case == "candidate_shared_parameter_non_finite":
            with torch.no_grad():
                # NaNは不変比較で自分自身と等しくならないため、infで非有限値を表す。
                next(iter(candidate.feature_extractor.parameters()))[0, 0] = float("-inf")
        assert_rejected_without_any_change(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            valid_owners=select_valid_owners(registration_arguments),
            expected_exception=(TypeError, ValueError),
        )


def test_all_values_are_generated_before_explicit_state_update_order(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        registration_arguments, _, _, _ = build_initial_registration_oracle(
            class_count=2,
            held_model_ids=(4, 9),
            current_model_is_held=True,
            monkeypatch=monkeypatch,
        )
        actual_call_order = []
        for operation_name in (
            "evaluate_classifier_per_sample_bounded_losses",
            "initialize_model_and_class_loss_statistics_from_batch",
            "snapshot_classifier_parameters",
            "integrate_adopted_candidate_shared_features",
        ):
            original_operation = getattr(registration_module, operation_name)

            def record_then_call(
                *, operation_name=operation_name, original_operation=original_operation, **arguments
            ):
                actual_call_order.append(operation_name)
                return original_operation(**arguments)

            monkeypatch.setattr(registration_module, operation_name, record_then_call)
        for owner_name, operation_name in (
            ("held_model_training_state_registry", "register_held_model_training_state"),
            ("loss_statistics_store", "set_model_loss_statistics"),
            ("pending_model_upload_state", "queue_model_upload"),
        ):
            original_operation = getattr(registration_arguments[owner_name], operation_name)

            def record_then_update(
                *, operation_name=operation_name, original_operation=original_operation, **arguments
            ):
                actual_call_order.append(operation_name)
                return original_operation(**arguments)

            monkeypatch.setattr(
                registration_arguments[owner_name], operation_name, record_then_update
            )
        register_adopted_candidate_as_temporary_held_model(**registration_arguments)
        expected_call_order = [
            "evaluate_classifier_per_sample_bounded_losses",
            "initialize_model_and_class_loss_statistics_from_batch",
            "snapshot_classifier_parameters",
            "integrate_adopted_candidate_shared_features",
            "register_held_model_training_state",
            "set_model_loss_statistics",
            "queue_model_upload",
        ]
        assert actual_call_order == expected_call_order


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_registered_candidate_continues_actual_joint_training_and_confirmation(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(229)
        registration_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
            build_initial_registration_oracle(
                class_count=class_count,
                held_model_ids=(4, 9),
                current_model_is_held=True,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = registration_arguments["held_model_training_state_registry"]
        candidate = registration_arguments["adopted_candidate_classifier"]
        candidate_optimizer_owner = registration_arguments[
            "candidate_concept_specific_parameter_optimizer_state"
        ]
        active = registry.get_held_model_training_state(model_id=9).classifier.feature_extractor
        input_features = registration_arguments["initial_statistics_input_features"]
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        # 実旧の共同学習が読む属性だけを与える。標本は固定batchを同じ順で渡す。
        legacy_training_batches = []
        legacy_client.updates_per_sample = 1
        legacy_client.backbone_gradient_diagnostics = defaultdict(float)
        legacy_client.model_training_examples = defaultdict(int)
        legacy_client.model_optimizer_steps = defaultdict(int)
        legacy_client.phase_seconds = defaultdict(float)
        legacy_client._sample_training_batches = lambda: legacy_training_batches
        training_batch_tensors = []
        for training_batch_index, sample_count in enumerate((3, 5, 4)):
            training_batch_tensors.append(
                (
                    (
                        torch.arange(sample_count * 2, dtype=torch.float32).reshape(sample_count, 2)
                        - training_batch_index * 2
                    )
                    / 7,
                    ((torch.arange(sample_count) + training_batch_index) % class_count)
                    .float()
                    .reshape(-1, 1),
                )
            )

        def run_joint_update_in_both_implementations():
            training_bindings = registry.snapshot_ordered_held_model_training_bindings()
            legacy_training_batches[:] = [
                (training_binding.model_id, batch_features.clone(), batch_labels.clone())
                for training_binding, (batch_features, batch_labels) in zip(
                    training_bindings, training_batch_tensors
                )
            ]
            expected_joint_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            actual_joint_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=active,
                shared_parameter_optimizer=shared_optimizer_owners[0].parameter_optimizer,
                participating_training_batches=tuple(
                    ParticipatingModelTrainingBatch(
                        classifier=training_binding.classifier,
                        concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
                        input_features=batch_features,
                        observed_class_labels=batch_labels,
                    )
                    for training_binding, (batch_features, batch_labels) in zip(
                        training_bindings, training_batch_tensors
                    )
                ),
                update_shared_features=update_shared_features,
            )
            assert actual_joint_loss == expected_joint_loss
            assert_held_model_states_match_legacy(
                registry=registry,
                shared_optimizer_owners=shared_optimizer_owners,
                legacy_client=legacy_client,
                input_features=input_features,
            )

        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        # 登録前: 保有2モデルの共同更新と、独立した共有部を持つ候補だけの学習。
        run_joint_update_in_both_implementations()
        candidate_features, candidate_labels = training_batch_tensors[2]
        legacy_candidate_model.update(candidate_features.clone(), candidate_labels.clone())
        perform_joint_model_parameter_update(
            local_training_settings=local_training_settings,
            shared_feature_extractor=candidate.feature_extractor,
            shared_parameter_optimizer=shared_optimizer_owners[-1].parameter_optimizer,
            participating_training_batches=(
                ParticipatingModelTrainingBatch(
                    classifier=candidate,
                    concept_specific_parameter_optimizer=candidate_optimizer_owner.parameter_optimizer,
                    input_features=candidate_features,
                    observed_class_labels=candidate_labels,
                ),
            ),
            update_shared_features=True,
        )
        for (parameter_name, parameter), legacy_parameter in zip(
            candidate.named_parameters(), legacy_candidate_model.parameters()
        ):
            assert torch.equal(parameter, legacy_parameter), parameter_name

        register_candidate_in_legacy_client(
            legacy_client=legacy_client,
            legacy_candidate_model=legacy_candidate_model,
            arguments=registration_arguments,
        )
        register_adopted_candidate_as_temporary_held_model(**registration_arguments)
        assert candidate.feature_extractor is active
        assert not candidate_optimizer_owner.parameter_optimizer.state
        assert_initial_registration_matches_legacy(
            registration_arguments=registration_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )
        pending_upload_state = registration_arguments["pending_model_upload_state"]
        loss_statistics_store = registration_arguments["loss_statistics_store"]
        pending_parameter_snapshot = deepcopy(
            pending_upload_state.get_pending_model_upload().parameter_snapshot
        )

        # 登録後: 一時IDの候補を含む3モデルの共同更新。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        if optimizer_variant != "sgd":
            assert candidate_optimizer_owner.parameter_optimizer.state
        registered_candidate_optimizer = candidate_optimizer_owner.parameter_optimizer
        # 学習後も送信保留の値は登録時のまま。統計も学習だけでは変わらない。
        assert_nested_state_equal(
            pending_upload_state.get_pending_model_upload().parameter_snapshot,
            pending_parameter_snapshot,
        )
        assert_store_statistics_match_legacy(
            loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
        )

        # 現在の学習帰属IDの切替えは後続specの責務。ここでは確認処理へのtest-only接続として行う。
        current_training_model_assignment = registration_arguments[
            "current_training_model_assignment"
        ]
        current_training_model_assignment.assign_model_for_training(model_id=TEMPORARY_MODEL_ID)
        legacy_client.current_model_id = TEMPORARY_MODEL_ID
        legacy_client.train_data_store = {}
        legacy_client.stored_data = {}
        legacy_client.model_concept_counts = {}
        assignment_change = confirm_held_model_registration(
            registered_global_model_id=12,
            held_model_training_state_registry=registry,
            loss_statistics_store=loss_statistics_store,
            training_sample_store=ModelTrainingSampleStore(),
            evaluation_sample_store=ModelEvaluationSampleStore(
                maximum_stored_sample_count_per_model=3, added_batch_sample_count=1
            ),
            model_training_and_assignment_counts_store=ModelTrainingAndAssignmentCountsStore(),
            current_training_model_assignment=current_training_model_assignment,
            pending_model_upload_state=pending_upload_state,
        )
        BaseClient.confirm_model_registration(legacy_client, 12)
        assert (assignment_change.previous_model_id, assignment_change.current_model_id) == (
            TEMPORARY_MODEL_ID,
            12,
        )
        assert tuple(legacy_client.models) == (4, 9, 12)
        assert registry.get_held_model_training_state(model_id=12).classifier is candidate
        assert candidate_optimizer_owner.parameter_optimizer is registered_candidate_optimizer
        assert pending_upload_state.get_pending_model_upload() is None
        assert legacy_client.pending_model_params is None
        assert_store_statistics_match_legacy(
            loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
        )

        # 正式ID確認後: 同じoptimizer stateで学習を継続する。
        run_joint_update_in_both_implementations()
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
