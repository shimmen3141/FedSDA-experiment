"""共有特徴抽出部・概念別残差・分類層を構築する。"""

import torch
from torch.nn import Module, Linear, Sigmoid, Identity

from .model_architecture_settings import ModelArchitectureSettings
from .shared_feature_extractor import SharedFeatureExtractor
from .nonlinear_residual_adapter import NonlinearResidualAdapter


def _validate_shared_feature_extractor(
    *,
    shared_feature_extractor: SharedFeatureExtractor | None,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
) -> None:
    if shared_feature_extractor is None:
        return
    if type(shared_feature_extractor) is not SharedFeatureExtractor:
        raise ValueError(
            "shared_feature_extractorはexact SharedFeatureExtractorまたはNoneが必要です。"
        )
    shared_feature_extractor.validate_structure(
        input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
    )


def _validate_classifier_inputs(
    *,
    model_architecture_settings: ModelArchitectureSettings,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    shared_feature_extractor: SharedFeatureExtractor | None,
) -> ModelArchitectureSettings:
    if type(model_architecture_settings) is not ModelArchitectureSettings:
        raise ValueError("model_architecture_settingsはexact ModelArchitectureSettingsが必要です。")
    try:
        validated_model_architecture_settings = ModelArchitectureSettings(
            model_architecture_name=model_architecture_settings.model_architecture_name,
            residual_adapter_requested_rank=model_architecture_settings.residual_adapter_requested_rank,
        )
    except (TypeError, ValueError) as validation_error:
        raise ValueError(f"model_architecture_settings: {validation_error}") from validation_error
    SharedFeatureExtractor.validate_feature_dimensions(
        input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
    )
    if type(class_count) is not int or class_count < 2:
        raise ValueError("class_countはbool以外の2以上の整数が必要です。")
    _validate_shared_feature_extractor(
        shared_feature_extractor=shared_feature_extractor,
        input_feature_count=input_feature_count,
        hidden_layer_widths=hidden_layer_widths,
    )
    return validated_model_architecture_settings


class ResidualAdapterClassifier(Module):
    """共有部だけを借用し、adapterと分類層を独立に所有する。"""

    def __init__(
        self,
        *,
        model_architecture_settings: ModelArchitectureSettings,
        input_feature_count: int,
        hidden_layer_widths: tuple[int, ...],
        class_count: int,
        shared_feature_extractor: SharedFeatureExtractor | None = None,
    ):
        validated_model_architecture_settings = _validate_classifier_inputs(
            model_architecture_settings=model_architecture_settings,
            input_feature_count=input_feature_count,
            hidden_layer_widths=hidden_layer_widths,
            class_count=class_count,
            shared_feature_extractor=shared_feature_extractor,
        )
        super().__init__()
        self.model_architecture_settings = validated_model_architecture_settings
        self.class_count = class_count
        self.feature_extractor = (
            shared_feature_extractor
            if shared_feature_extractor is not None
            else SharedFeatureExtractor(
                input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
            )
        )
        self.residual_adapter = NonlinearResidualAdapter(
            feature_count=self.feature_extractor.output_feature_count,
            requested_rank=validated_model_architecture_settings.residual_adapter_requested_rank,
        )
        classification_output_count = 1 if class_count == 2 else class_count
        self.classification_layer = Linear(
            self.feature_extractor.output_feature_count,
            classification_output_count,
            device="cpu",
            dtype=torch.float32,
        )
        self.output_activation = Sigmoid() if class_count == 2 else Identity()

    def extract_shared_features(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(input_features)

    def forward_from_shared_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        adapted_features = self.residual_adapter(shared_features)
        classification_scores = self.classification_layer(adapted_features)
        return self.output_activation(classification_scores)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.forward_from_shared_features(self.extract_shared_features(input_features))
