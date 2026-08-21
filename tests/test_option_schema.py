from pathlib import Path

import pytest

from federated_drift_experiment import config
from federated_drift_experiment.mode_names import (
    BASELINE_MODES,
    FEDSDA_MODES,
)
from federated_drift_experiment.experiment_spec.options import (
    CAPABILITIES_BY_ID,
    CHOICE_CONSTRAINTS,
    METHODS,
    OPTIONS,
    explicit_option_ids,
    implementation_status,
    inactive_reasons,
    option,
    render_option_document,
    validate_explicit_options,
    validate_selection,
    validate_sweep_dependencies,
)


def test_option_schema_references_known_capabilities_and_options():
    option_ids = {item.id for item in OPTIONS}
    assert len(option_ids) == len(OPTIONS)
    for method in METHODS:
        assert set(method.capabilities) <= set(CAPABILITIES_BY_ID)
    for item in OPTIONS:
        assert set(item.requires_capabilities) <= set(CAPABILITIES_BY_ID)
        assert all(rule.option_id in option_ids for rule in item.active_when)
    assert all(item.option_id in option_ids for item in CHOICE_CONSTRAINTS)


def test_schema_choices_follow_runtime_configuration():
    assert option("new_model_creation_policy").choices == config.NEW_MODEL_CREATION_POLICIES
    assert option("clustering_policy").choices == config.FEDSDA_CLUSTERING_POLICIES
    assert option("clustering_decision").choices == config.FEDSDA_CLUSTERING_DECISIONS
    assert option("clustering_consolidation").choices == (
        config.FEDSDA_CLUSTERING_CONSOLIDATIONS
    )
    assert option("cluster_linkage").choices == config.FEDSDA_CLUSTER_LINKAGES
    assert option("shared_backbone_training").choices == (
        config.SHARED_BACKBONE_TRAINING_CHOICES
    )
    assert option("soft_routing_context").choices == (
        config.SOFT_ROUTING_CONTEXT_CHOICES
    )
    assert option("soft_routing_top_combination").choices == (
        config.SOFT_ROUTING_TOP_COMBINATION_CHOICES
    )
    assert option("soft_routing_meta_loss").choices == (
        config.SOFT_ROUTING_META_LOSS_CHOICES
    )
    known_modes = set(FEDSDA_MODES) | set(BASELINE_MODES) | {"FedDrift"}
    assert all(
        set(item.implemented_modes) <= known_modes
        for item in CHOICE_CONSTRAINTS
    )


def test_theoretical_and_implemented_status_are_distinct():
    assert implementation_status("new_model_forward_validation_samples", "FedSDA") == "implemented"
    assert implementation_status("new_model_forward_validation_samples", "FedDrift") == "theoretical"
    assert implementation_status("new_model_forward_validation_samples", "Oblivious") == "unavailable"


def test_forward_parameter_activation_is_machine_readable():
    assert inactive_reasons(
        "new_model_forward_validation_samples",
        {"new_model_creation_policy": "immediate"},
    ) == ("forward系のとき",)
    assert inactive_reasons(
        "new_model_forward_validation_samples",
        {"new_model_creation_policy": "forward_persistent"},
    ) == ()


def test_selection_validation_distinguishes_theory_from_implementation():
    assert validate_selection("FedDrift", {
        "new_model_creation_policy": "forward_persistent",
    }) == ("new_model_creation_policy is theoretical for FedDrift",)
    assert validate_selection("FedSDA", {
        "new_model_creation_policy": "forward_persistent",
        "new_model_forward_validation_samples": "30",
    }) == ()


def test_only_explicit_options_are_checked_against_selected_modes():
    explicit = explicit_option_ids([
        "--new-model-forward-validation-samples", "30",
    ])
    assert explicit == ("new_model_forward_validation_samples",)
    assert validate_explicit_options(
        ("FedDrift",),
        {"new_model_creation_policy": "forward_persistent"},
        explicit,
    ) == (
        "--new-model-forward-validation-samples: FedDrift: "
        "new_model_forward_validation_samples is theoretical for FedDrift",
    )
    assert validate_explicit_options(
        ("FedSDA_NoCached_ClassESR",),
        {"new_model_creation_policy": "forward_persistent"},
        explicit,
    ) == ()


def test_empty_sweep_disables_its_fixed_parameter():
    argv = ["--adwin-deltas", "--fixed-aggregation-interval", "50"]
    assert validate_sweep_dependencies(argv, {"adwin-deltas": []}) == (
        "--fixed-aggregation-interval requires a non-empty --adwin-deltas",
    )
    assert validate_sweep_dependencies(argv, {"adwin-deltas": [0.05]}) == ()


def test_soft_routing_constraints_are_explicit():
    assert validate_selection("FedSDA", {
        "routing": "restarting_soft",
        "server_flow": "NoCached",
        "detector": "ADWIN",
    }) == ("routing=restarting_soft: ClassESRが必要",)
    assert validate_selection("FedSDA", {
        "routing": "restarting_soft",
        "server_flow": "NoCached",
        "detector": "ClassESR",
    }) == ()


def test_parameter_sharing_is_limited_to_no_cached_modes():
    assert validate_selection("FedSDA", {
        "clustering_policy": "on_new_model",
        "clustering_consolidation": "parameter_share",
        "server_flow": "NoCached",
    }) == ()
    assert validate_selection("FedSDA", {
        "clustering_policy": "on_new_model",
        "clustering_consolidation": "parameter_share",
        "server_flow": "Cached",
    }) == (
        "clustering_consolidation=parameter_share: NoCachedが必要",
    )


def test_meta_loss_dependency_is_validated_from_explicit_options():
    explicit = explicit_option_ids([
        "--soft-routing-meta-loss", "zero_one",
    ])
    mode = "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    assert validate_explicit_options(
        (mode,),
        {"soft_routing_context": "global",
         "soft_routing_meta_loss": "zero_one"},
        explicit,
    ) == (
        "--soft-routing-meta-loss: " + mode + ": "
        "soft_routing_meta_loss is active only when "
        "文脈別Meta-routerを計算するとき",
    )
    assert validate_explicit_options(
        (mode,),
        {"soft_routing_context": "meta_predicted_class",
         "soft_routing_meta_loss": "zero_one"},
        explicit,
    ) == ()


def test_top_combination_requires_meta_switching():
    explicit = explicit_option_ids([
        "--soft-routing-top-combination", "mixture",
    ])
    mode = "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    assert validate_explicit_options(
        (mode,),
        {
            "soft_routing_context": "global",
            "soft_routing_top_combination": "mixture",
        },
        explicit,
    ) == (
        "--soft-routing-top-combination: " + mode + ": "
        "soft_routing_top_combination is active only when "
        "Meta-switchingを実予測へ使うとき",
    )
    assert validate_explicit_options(
        (mode,),
        {
            "soft_routing_context": "meta_switching",
            "soft_routing_top_combination": "mixture",
        },
        explicit,
    ) == ()


def test_shared_backbone_training_requires_shared_architecture():
    assert validate_selection("FedSDA", {
        "model_architecture": "independent",
        "shared_backbone_training": "joint",
    }) == (
        "shared_backbone_training is active only when 共有表現構造のとき",
    )
    assert validate_selection("FedSDA", {
        "model_architecture": "shared_backbone",
        "shared_backbone_training": "joint",
        "server_flow": "NoCached",
        "routing": "restarting_soft",
        "detector": "ClassESR",
    }) == ()


def test_routing_recalibration_requires_soft_routing():
    assert validate_selection("FedSDA", {
        "model_architecture": "residual_adapter",
        "routing": "hard",
        "server_flow": "NoCached",
        "shared_backbone_routing_recalibration": "fifo_replay",
    }) == (
        "shared_backbone_routing_recalibration is active only when "
        "Restarting SoftRoutingのとき",
    )
    assert validate_selection("FedSDA", {
        "model_architecture": "residual_adapter",
        "shared_backbone_training": "joint",
        "server_flow": "NoCached",
        "routing": "restarting_soft",
        "detector": "ClassESR",
    }) == ()


def test_pcgrad_requires_joint_shared_training():
    assert validate_selection("FedSDA", {
        "model_architecture": "residual_adapter",
        "shared_backbone_training": "sequential",
        "shared_backbone_gradient_strategy": "pcgrad",
        "server_flow": "NoCached",
        "routing": "restarting_soft",
        "detector": "ClassESR",
    }) == (
        "shared_backbone_gradient_strategy is active only when "
        "共有表現をjoint学習するとき",
    )
    assert validate_selection("FedSDA", {
        "model_architecture": "residual_adapter",
        "shared_backbone_training": "joint",
        "shared_backbone_gradient_strategy": "pcgrad",
        "server_flow": "NoCached",
        "routing": "restarting_soft",
        "detector": "ClassESR",
    }) == ()


def test_unknown_option_is_rejected():
    with pytest.raises(KeyError, match="Unknown option id"):
        option("unknown")


def test_sequential_tournament_activates_alpha_but_not_fixed_forward_count():
    selection = {
        "new_model_creation_policy": "sequential_tournament",
        "sequential_tournament_alpha": 0.05,
        "new_model_forward_validation_samples": 10,
    }

    assert inactive_reasons("sequential_tournament_alpha", selection) == ()
    assert inactive_reasons(
        "new_model_forward_validation_samples", selection
    ) == ("forward系のとき",)


def test_generated_option_document_is_current():
    path = Path(__file__).resolve().parents[1] / "docs" / "options.md"
    assert path.read_text(encoding="utf-8") == render_option_document()
