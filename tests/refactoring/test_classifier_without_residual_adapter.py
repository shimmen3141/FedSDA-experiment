"""adapterを持たない構造の分類器: 旧の、共有部と出力層だけのモデル（SharedBackboneMLP）と照合する。"""

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.models import ResidualAdapterMLP, SharedBackboneMLP
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    measure_model_computation,
)
from federated_learning_experiments.learning.models.nonlinear_residual_adapter import (
    IdentityFeatureAdapter,
    NonlinearResidualAdapter,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)

HIDDEN_LAYER_WIDTHS = (6, 4)
INPUT_FEATURE_COUNT = 3


def make_classifier(*, model_architecture_name, class_count):
    return ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name=model_architecture_name, residual_adapter_requested_rank=8
        ),
        input_feature_count=INPUT_FEATURE_COUNT,
        hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
        class_count=class_count,
    )


@pytest.mark.parametrize("class_count", (2, 4))
def test_classifier_without_adapter_matches_legacy_shared_backbone_model(class_count, monkeypatch):
    """同じ乱数の状態から作ると、パラメータ・出力・構築の後の乱数の状態が、旧のモデルと一致する。"""
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset: type(
            "DatasetSpecification",
            (),
            dict(
                input_dim=INPUT_FEATURE_COUNT,
                num_classes=class_count,
                hidden_dims=HIDDEN_LAYER_WIDTHS,
                learning_rate=None,
            ),
        )(),
    )
    monkeypatch.setattr(
        "federated_drift_experiment.models.normalize_dataset_name", lambda name: name
    )
    torch.manual_seed(31)
    legacy_model = SharedBackboneMLP(dataset="test-dataset")
    legacy_random_state_after_construction = torch.random.get_rng_state()
    torch.manual_seed(31)
    classifier = make_classifier(
        model_architecture_name="shared_backbone_without_residual_adapter", class_count=class_count
    )
    assert torch.equal(torch.random.get_rng_state(), legacy_random_state_after_construction)
    assert type(classifier.residual_adapter) is IdentityFeatureAdapter
    assert list(classifier.residual_adapter.parameters()) == []
    # パラメータの名前は、共有部と分類層だけ。値は、旧のモデルの、対応するパラメータと同じ。
    classifier_parameters = classifier.state_dict()
    assert not any(name.startswith("residual_adapter.") for name in classifier_parameters)
    legacy_parameters = legacy_model.state_dict()
    assert not any(name.startswith("adapter.") for name in legacy_parameters)
    assert len(classifier_parameters) == len(legacy_parameters)
    for classifier_parameter, legacy_parameter in zip(
        classifier_parameters.values(), legacy_parameters.values(), strict=True
    ):
        assert torch.equal(classifier_parameter, legacy_parameter)
    input_features = torch.tensor([[0.2, -0.4, 0.9], [1.5, 0.3, -0.7]], dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(classifier(input_features), legacy_model(input_features))
        shared_features = classifier.extract_shared_features(input_features)
        assert classifier.residual_adapter(shared_features) is shared_features
    # adapterを持つ構造は、同じ乱数の状態から、旧の、adapterつきのモデルと同じ数の乱数を消費する（提案構成）。
    monkeypatch.setattr(config, "SHARED_ADAPTER_RANK", 8)
    torch.manual_seed(31)
    ResidualAdapterMLP(dataset="test-dataset")
    legacy_adapter_random_state = torch.random.get_rng_state()
    torch.manual_seed(31)
    adapter_classifier = make_classifier(
        model_architecture_name="shared_backbone_residual_adapter", class_count=class_count
    )
    assert torch.equal(torch.random.get_rng_state(), legacy_adapter_random_state)
    assert type(adapter_classifier.residual_adapter) is NonlinearResidualAdapter
    assert not torch.equal(legacy_adapter_random_state, legacy_random_state_after_construction)


def test_identity_feature_adapter_validates_inputs_and_is_counted_as_concept_specific_part():
    for invalid_feature_count in (0, True, 2.0):
        with pytest.raises(ValueError):
            IdentityFeatureAdapter(feature_count=invalid_feature_count)
    identity_adapter = IdentityFeatureAdapter(feature_count=4)
    for invalid_features in (
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float64),
        [0.0, 0.0, 0.0, 0.0],
    ):
        with pytest.raises(ValueError):
            identity_adapter(invalid_features)
    # 計算量の計測は、adapterを持たない構造でも、概念固有部へ入力された標本数を数える。
    classifier = make_classifier(
        model_architecture_name="shared_backbone_without_residual_adapter", class_count=2
    )
    input_features = torch.zeros((5, INPUT_FEATURE_COUNT), dtype=torch.float32)
    with measure_model_computation() as model_computation_meter, torch.no_grad():
        classifier(input_features)
    model_computation_counts = model_computation_meter.get_model_computation_counts()
    assert model_computation_counts.shared_part_inference_example_count == 5
    assert model_computation_counts.concept_specific_part_inference_example_count == 5
    # 概念固有部の積和演算は、分類層だけ（共有部の出力4 × 出力1 × 標本5）。
    assert (
        model_computation_counts.concept_specific_part_inference_forward_multiply_accumulate_count
        == 5 * HIDDEN_LAYER_WIDTHS[-1] * 1
    )


@pytest.mark.parametrize("class_count", (2, 4))
def test_independent_classifier_matches_legacy_model_without_shared_part(class_count, monkeypatch):
    """共有部を持たない構造の分類器は、同じ乱数の状態から作ると、パラメータ・出力・乱数の状態が、旧の`SimpleMLP`と一致する。"""
    from federated_drift_experiment.models import SimpleMLP
    from federated_learning_experiments.learning.models.nonlinear_residual_adapter import (
        ModelOwnedHiddenLayers,
    )

    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset: type(
            "DatasetSpecification",
            (),
            dict(
                input_dim=INPUT_FEATURE_COUNT,
                num_classes=class_count,
                hidden_dims=HIDDEN_LAYER_WIDTHS,
                learning_rate=None,
            ),
        )(),
    )
    monkeypatch.setattr(
        "federated_drift_experiment.models.normalize_dataset_name", lambda name: name
    )
    torch.manual_seed(37)
    legacy_model = SimpleMLP(dataset="test-dataset")
    legacy_random_state_after_construction = torch.random.get_rng_state()
    torch.manual_seed(37)
    classifier = make_classifier(
        model_architecture_name="independent_models_without_shared_backbone",
        class_count=class_count,
    )
    assert torch.equal(torch.random.get_rng_state(), legacy_random_state_after_construction)
    # 特徴抽出部は、パラメータを持たず、入力をそのまま返す。隠れ層は、モデルごとの部品が持つ。
    assert list(classifier.feature_extractor.parameters()) == []
    assert classifier.feature_extractor.hidden_layer_widths == ()
    assert type(classifier.residual_adapter) is ModelOwnedHiddenLayers
    assert classifier.hidden_layer_widths == HIDDEN_LAYER_WIDTHS
    assert classifier.input_feature_count == INPUT_FEATURE_COUNT
    classifier_parameters = classifier.state_dict()
    assert all(
        name.startswith(("residual_adapter.hidden_layers.", "classification_layer."))
        for name in classifier_parameters
    )
    legacy_parameters = legacy_model.state_dict()
    assert len(classifier_parameters) == len(legacy_parameters)
    for classifier_parameter, legacy_parameter in zip(
        classifier_parameters.values(), legacy_parameters.values(), strict=True
    ):
        assert torch.equal(classifier_parameter, legacy_parameter)
    input_features = torch.tensor([[0.2, -0.4, 0.9], [1.5, 0.3, -0.7]], dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(classifier(input_features), legacy_model(input_features))
        assert classifier.extract_shared_features(input_features) is input_features
    # 同じ構造の分類器を、分類器が持つ寸法から、もう1つ作れる（特徴抽出部の寸法は、隠れ層を持たない）。
    second_classifier = ResidualAdapterClassifier(
        model_architecture_settings=classifier.model_architecture_settings,
        input_feature_count=classifier.input_feature_count,
        hidden_layer_widths=classifier.hidden_layer_widths,
        class_count=classifier.class_count,
    )
    second_classifier.load_state_dict(classifier.state_dict(), strict=True)
    # 2つの分類器は、パラメータを共有しない。パラメータを持たない特徴抽出部は、つなぎ直せる。
    assert not {id(parameter) for parameter in classifier.parameters()} & {
        id(parameter) for parameter in second_classifier.parameters()
    }
    second_classifier.attach_shared_feature_extractor(
        shared_feature_extractor=classifier.feature_extractor
    )
    assert second_classifier.feature_extractor is classifier.feature_extractor
    # 計算量の計測: 共有部の計数は0。概念固有部は、全部の層。
    with measure_model_computation() as model_computation_meter, torch.no_grad():
        classifier(input_features)
    model_computation_counts = model_computation_meter.get_model_computation_counts()
    assert model_computation_counts.shared_part_inference_example_count == 0
    assert model_computation_counts.shared_part_inference_forward_multiply_accumulate_count == 0
    assert model_computation_counts.concept_specific_part_inference_example_count == 2
    assert (
        model_computation_counts.concept_specific_part_inference_forward_multiply_accumulate_count
        == 2 * (3 * 6 + 6 * 4 + 4 * (1 if class_count == 2 else class_count))
    )
