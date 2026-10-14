"""FedSDAの予測を、最後の同期の時点のモデルで行う設定（`parameters_at_last_synchronization`）を確認する。

設定、同期の時点の写しのowner、予測の関数と標本の処理への接続、clientの配布の後の写し、
手元の診断（実行時の差し替え）との一致、構成名を、ここで確かめる。この構成はgoldenの事例にしない
（`behavior_golden_cases.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）。
"""

from dataclasses import replace

import pytest
import torch

import federated_learning_experiments.runtime.observed_sample_processing as observed_sample_processing
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_prediction_model_snapshot import (
    SynchronizedPredictionModelSnapshot,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.sweep.experiment_configurations import (
    INDEPENDENT_MODELS_CONFIGURATION_NAME,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
    WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
    build_experiment_run_settings,
)

_ADAM = AdamParameterOptimizerSettings(
    learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
)


def _valid_prediction_combination_values():
    return dict(
        prediction_combination_strategy="fixed_share_weighted_prediction",
        prediction_mixture_activation_policy="always",
        prediction_weight_recalibration_after_aggregation_policy=(
            "recompute_buffer_losses_and_replay_weight_updates"
        ),
        prediction_state_reset_on_training_assignment_change_policy=(
            "restart_adahedge_preserve_fixed_share_prediction_state"
        ),
        prediction_model_parameter_source="current_local_parameters",
        fixed_share_weight_redistribution_time_scale_samples=2,
    )


# ---- 設定 ----


@pytest.mark.parametrize(
    "parameter_source", ["current_local_parameters", "parameters_at_last_synchronization"]
)
def test_prediction_model_parameter_source_accepts_declared_values(parameter_source):
    settings = PredictionCombinationSettings(
        **(
            _valid_prediction_combination_values()
            | {"prediction_model_parameter_source": parameter_source}
        )
    )
    assert settings.prediction_model_parameter_source == parameter_source


@pytest.mark.parametrize("parameter_source", ["distributed", "", None, 1])
def test_prediction_model_parameter_source_rejects_undeclared_values(parameter_source):
    with pytest.raises((RunSettingsValidationError, TypeError, ValueError)):
        PredictionCombinationSettings(
            **(
                _valid_prediction_combination_values()
                | {"prediction_model_parameter_source": parameter_source}
            )
        )


# ---- 写しのowner ----


def _make_classifier(
    *,
    shared_feature_extractor=None,
    model_architecture_name="shared_backbone_residual_adapter",
    seed=0,
):
    torch.manual_seed(seed)
    return ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name=model_architecture_name, residual_adapter_requested_rank=2
        ),
        input_feature_count=2,
        hidden_layer_widths=(5, 4),
        class_count=2,
        shared_feature_extractor=shared_feature_extractor,
    )


def _register(registry, model_id, classifier):
    registry.register_held_model_training_state(
        model_id=model_id,
        classifier=classifier,
        concept_specific_parameter_optimizer_state=ParameterOptimizerState(
            parameters=tuple(classifier.model_owned_feature_transform.parameters())
            + tuple(classifier.classification_layer.parameters()),
            optimizer_settings=_ADAM,
        ),
    )


def _registry_with_shared_backbone(model_ids):
    registry = HeldModelTrainingStateRegistry()
    first = _make_classifier(seed=0)
    _register(registry, model_ids[0], first)
    for position, model_id in enumerate(model_ids[1:], start=1):
        _register(
            registry,
            model_id,
            _make_classifier(shared_feature_extractor=first.feature_extractor, seed=position),
        )
    return registry


def _perturb(classifier, amount):
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.add_(amount)


def _held_classifiers(registry):
    return {
        state.model_id: state.classifier
        for state in registry.snapshot_ordered_held_model_training_states()
    }


def test_empty_snapshot_selects_current_classifiers_and_first_model_for_shared_features():
    registry = _registry_with_shared_backbone([3, -1])
    snapshot = SynchronizedPredictionModelSnapshot()
    shared_feature_classifier, classifiers = snapshot.select_prediction_classifiers(
        held_model_training_state_registry=registry
    )
    held = _held_classifiers(registry)
    assert classifiers.keys() == held.keys()
    assert all(classifiers[model_id] is held[model_id] for model_id in held)
    assert shared_feature_classifier is held[-1]


def test_snapshot_keeps_synchronized_values_while_current_classifiers_are_updated():
    registry = _registry_with_shared_backbone([0, 2])
    snapshot = SynchronizedPredictionModelSnapshot()
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    held = _held_classifiers(registry)
    inputs = torch.tensor([[0.3, -0.2], [1.0, 0.5]])
    with torch.no_grad():
        synchronized_outputs = {model_id: held[model_id](inputs) for model_id in held}
    for classifier in held.values():
        _perturb(classifier, 0.25)
    shared_feature_classifier, classifiers = snapshot.select_prediction_classifiers(
        held_model_training_state_registry=registry
    )
    with torch.no_grad():
        shared_features = shared_feature_classifier.extract_shared_features(inputs)
        for model_id, classifier in classifiers.items():
            assert classifier is not held[model_id]
            assert torch.equal(
                classifier.forward_from_shared_features(shared_features),
                synchronized_outputs[model_id],
            )
            assert not torch.equal(held[model_id](inputs), synchronized_outputs[model_id])
    # 写しの中でも、共有部は1つのオブジェクト。現在の共有部とは別。
    assert classifiers[0].feature_extractor is classifiers[2].feature_extractor
    assert classifiers[0].feature_extractor is not held[0].feature_extractor


def test_models_added_or_replaced_after_snapshot_use_current_classifier_with_synchronized_backbone():
    registry = _registry_with_shared_backbone([0, 2])
    snapshot = SynchronizedPredictionModelSnapshot()
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    held = _held_classifiers(registry)
    # 同期の後に作ったモデル（一時ID）は、同じ共有部につながる。
    added = _make_classifier(shared_feature_extractor=held[0].feature_extractor, seed=9)
    _register(registry, -4, added)
    replaced = _make_classifier(shared_feature_extractor=held[0].feature_extractor, seed=8)
    _register(registry, 2, replaced)
    shared_feature_classifier, classifiers = snapshot.select_prediction_classifiers(
        held_model_training_state_registry=registry
    )
    assert classifiers[-4] is added
    assert classifiers[2] is replaced
    assert classifiers[0] is not held[0]
    # 共有特徴は、写した分類器（モデル0）の共有部で計算する。
    assert shared_feature_classifier is classifiers[0]


def test_capture_replaces_previous_snapshot():
    registry = _registry_with_shared_backbone([0])
    snapshot = SynchronizedPredictionModelSnapshot()
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    first_copy = snapshot.select_prediction_classifiers(
        held_model_training_state_registry=registry
    )[1][0]
    _perturb(_held_classifiers(registry)[0], 0.5)
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    second_copy = snapshot.select_prediction_classifiers(
        held_model_training_state_registry=registry
    )[1][0]
    assert second_copy is not first_copy
    for current, copied in zip(
        _held_classifiers(registry)[0].parameters(), second_copy.parameters(), strict=True
    ):
        assert torch.equal(current, copied)


def test_owner_rejects_non_registry_inputs():
    snapshot = SynchronizedPredictionModelSnapshot()
    with pytest.raises(TypeError):
        snapshot.capture_held_models(held_model_training_state_registry=object())
    with pytest.raises(TypeError):
        snapshot.select_prediction_classifiers(held_model_training_state_registry=None)
    with pytest.raises(ValueError):
        snapshot.select_prediction_classifiers(
            held_model_training_state_registry=HeldModelTrainingStateRegistry()
        )


def test_snapshot_of_independent_models_keeps_each_model_separate():
    registry = HeldModelTrainingStateRegistry()
    for model_id, seed in ((0, 0), (1, 1)):
        _register(
            registry,
            model_id,
            _make_classifier(
                model_architecture_name="independent_models_without_shared_backbone", seed=seed
            ),
        )
    snapshot = SynchronizedPredictionModelSnapshot()
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    held = _held_classifiers(registry)
    inputs = torch.tensor([[0.3, -0.2]])
    with torch.no_grad():
        expected = {model_id: held[model_id](inputs) for model_id in held}
    for classifier in held.values():
        _perturb(classifier, 0.1)
    shared_feature_classifier, classifiers = snapshot.select_prediction_classifiers(
        held_model_training_state_registry=registry
    )
    with torch.no_grad():
        shared_features = shared_feature_classifier.extract_shared_features(inputs)
        for model_id, classifier in classifiers.items():
            assert torch.equal(
                classifier.forward_from_shared_features(shared_features), expected[model_id]
            )


# ---- runでの確認（診断との一致、検出力、予測の後の処理、構成名） ----

# 同期の後に作ったモデル（写しを持たないモデル）を含む予測が起きる、小さい条件（2026-10-15に下調べで選んだ）。
_CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=1,
    client_count=3,
    per_client_sample_count=400,
    server_aggregation_interval_per_client_samples=50,
)


def _with_parameter_source(run_settings, parameter_source):
    participant = run_settings.run_participant_settings
    client = participant.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                prediction_combination_settings=replace(
                    client.prediction_combination_settings,
                    prediction_model_parameter_source=parameter_source,
                ),
            ),
        ),
    )


def _records_and_accuracy(experiment_run):
    records = tuple(
        client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        for client in experiment_run.measured_run.participants.client_operations
    )
    return records, experiment_run.run_metrics.prediction_accuracy


def _run_with_diagnostic_substitution(monkeypatch, run_settings):
    """手元の診断と同じ差し替え: 配布の後に保有モデル全部のstate_dictを控え、予測の間だけ読み込み、戻す。

    診断のscript（gitの対象外の結果のディレクトリ）には依らず、同じ処理をここで書き直す。新しい経路（写しの
    owner）は通らない（局所更新した現在の値で予測する設定で走らせる）。
    """
    state_dicts_by_registry = {}
    original_apply = FedsdaRunClient.apply_global_model_distribution
    original_predict = (
        observed_sample_processing.predict_observed_sample_and_update_prediction_weights
    )

    def apply_and_keep_state_dicts(self, **arguments):
        application = original_apply(self, **arguments)
        registry = self.owners.held_model_training_state_registry
        state_dicts_by_registry[id(registry)] = {
            state.model_id: (
                state.classifier,
                {k: v.detach().clone() for k, v in state.classifier.state_dict().items()},
            )
            for state in registry.snapshot_ordered_held_model_training_states()
        }
        return application

    def predict_with_kept_state_dicts(**arguments):
        registry = arguments["held_model_training_state_registry"]
        kept = state_dicts_by_registry.get(id(registry), {})
        targets = [
            (state.classifier, kept[state.model_id][1])
            for state in registry.snapshot_ordered_held_model_training_states()
            if state.model_id in kept and kept[state.model_id][0] is state.classifier
        ]
        saved = [
            (classifier, {k: v.detach().clone() for k, v in classifier.state_dict().items()})
            for classifier, _ in targets
        ]
        for classifier, state_dict in targets:
            classifier.load_state_dict(state_dict)
        try:
            return original_predict(**arguments)
        finally:
            for classifier, state_dict in saved:
                classifier.load_state_dict(state_dict)

    with monkeypatch.context() as patch:
        patch.setattr(
            FedsdaRunClient, "apply_global_model_distribution", apply_and_keep_state_dicts
        )
        patch.setattr(
            observed_sample_processing,
            "predict_observed_sample_and_update_prediction_weights",
            predict_with_kept_state_dicts,
        )
        return execute_fedsda_experiment_run(
            run_settings=_with_parameter_source(run_settings, "current_local_parameters")
        )


def _run_counting_snapshot_selections(monkeypatch, run_settings):
    counts = {"with_snapshot": 0, "with_added_model": 0}
    original_select = SynchronizedPredictionModelSnapshot.select_prediction_classifiers

    def counting_select(self, **arguments):
        selected = original_select(self, **arguments)
        registry = arguments["held_model_training_state_registry"]
        held = {
            state.model_id: state.classifier
            for state in registry.snapshot_ordered_held_model_training_states()
        }
        if any(selected[1][model_id] is not held[model_id] for model_id in held):
            counts["with_snapshot"] += 1
            if any(selected[1][model_id] is held[model_id] for model_id in held):
                counts["with_added_model"] += 1
        return selected

    with monkeypatch.context() as patch:
        patch.setattr(
            SynchronizedPredictionModelSnapshot, "select_prediction_classifiers", counting_select
        )
        experiment_run = execute_fedsda_experiment_run(
            run_settings=_with_parameter_source(run_settings, "parameters_at_last_synchronization")
        )
    return experiment_run, counts


@pytest.mark.parametrize(
    "configuration_name",
    [
        PROPOSED_FEDSDA_CONFIGURATION_NAME,
        INDEPENDENT_MODELS_CONFIGURATION_NAME,
        WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
        # 結合せず、現在のモデルだけで予測する経路。
        SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    ],
)
def test_synchronized_model_prediction_matches_diagnostic_substitution(
    monkeypatch, configuration_name
):
    run_settings = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=_CONDITIONS
    )
    experiment_run, counts = _run_counting_snapshot_selections(monkeypatch, run_settings)
    oracle_run = _run_with_diagnostic_substitution(monkeypatch, run_settings)
    assert counts["with_snapshot"] > 0
    if configuration_name == PROPOSED_FEDSDA_CONFIGURATION_NAME:
        # 同期の後に作ったモデル（写しを持たない）を含む予測の経路を通った。
        assert counts["with_added_model"] > 0
    assert _records_and_accuracy(experiment_run) == _records_and_accuracy(oracle_run)
    # 検出力: 局所更新した現在の値で予測する既定とは、予測の記録が違う。
    default_run = execute_fedsda_experiment_run(
        run_settings=_with_parameter_source(run_settings, "current_local_parameters")
    )
    assert _records_and_accuracy(default_run)[0] != _records_and_accuracy(experiment_run)[0]


def test_loss_monitoring_and_validation_use_current_classifiers(monkeypatch):
    """要求1.4: 予測の後の損失の評価（監視・候補の検証）には、写しでなく保有モデルの現在の分類器を渡す。"""
    copied_classifiers = []
    evaluated_classifiers = []
    original_capture = SynchronizedPredictionModelSnapshot.capture_held_models
    original_evaluate = observed_sample_processing.evaluate_classifier_per_sample_bounded_losses

    def capture_and_remember(self, **arguments):
        original_capture(self, **arguments)
        copied_classifiers.extend(
            copied for _, copied in self._snapshot_classifiers_by_model_id.values()
        )

    def evaluate_and_remember(**arguments):
        evaluated_classifiers.append(arguments["classifier"])
        return original_evaluate(**arguments)

    monkeypatch.setattr(
        SynchronizedPredictionModelSnapshot, "capture_held_models", capture_and_remember
    )
    monkeypatch.setattr(
        observed_sample_processing,
        "evaluate_classifier_per_sample_bounded_losses",
        evaluate_and_remember,
    )
    execute_fedsda_experiment_run(
        run_settings=build_experiment_run_settings(
            configuration_name=SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
            experiment_run_conditions=_CONDITIONS,
        )
    )
    assert copied_classifiers and evaluated_classifiers
    copied_ids = {id(classifier) for classifier in copied_classifiers}
    assert not any(id(classifier) in copied_ids for classifier in evaluated_classifiers)


def test_default_configuration_does_not_capture_snapshots(monkeypatch):
    captured = []
    monkeypatch.setattr(
        SynchronizedPredictionModelSnapshot,
        "capture_held_models",
        lambda self, **arguments: captured.append(arguments),
    )
    experiment_run = execute_fedsda_experiment_run(
        run_settings=build_experiment_run_settings(
            configuration_name=PROPOSED_FEDSDA_CONFIGURATION_NAME,
            experiment_run_conditions=_CONDITIONS,
        )
    )
    assert captured == []
    assert all(
        client.synchronized_prediction_model_snapshot._snapshot_classifiers_by_model_id == {}
        for client in experiment_run.measured_run.participants.client_operations
    )


@pytest.mark.parametrize(
    "function",
    [
        observed_sample_processing.predict_observed_sample_and_update_prediction_weights,
        observed_sample_processing.process_observed_sample,
    ],
)
def test_wrong_snapshot_owner_type_is_rejected_before_processing(function):
    with pytest.raises(TypeError, match="synchronized_prediction_model_snapshot"):
        function(synchronized_prediction_model_snapshot=object(), **_none_arguments(function))


def _none_arguments(function):
    import inspect

    return {
        name: None
        for name, parameter in inspect.signature(function).parameters.items()
        if name != "synchronized_prediction_model_snapshot"
        and parameter.default is inspect.Parameter.empty
    }


def test_configuration_differs_from_proposed_only_in_prediction_model_parameter_source():
    proposed = convert_settings_to_plain_mapping(
        build_experiment_run_settings(
            configuration_name=PROPOSED_FEDSDA_CONFIGURATION_NAME,
            experiment_run_conditions=_CONDITIONS,
        )
    )
    synchronized = convert_settings_to_plain_mapping(
        build_experiment_run_settings(
            configuration_name=SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
            experiment_run_conditions=_CONDITIONS,
        )
    )
    holders = []
    for mapping in (proposed, synchronized):
        holder = mapping["run_participant_settings"]["run_client_settings"]
        holders.append(holder["prediction_combination_settings"])
    assert holders[0]["prediction_model_parameter_source"] == "current_local_parameters"
    assert holders[1]["prediction_model_parameter_source"] == "parameters_at_last_synchronization"
    holders[1]["prediction_model_parameter_source"] = "current_local_parameters"
    assert synchronized == proposed
