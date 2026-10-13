"""特徴抽出部・モデルごとの部品・分類層からなる分類器を、モデル構造の設定に従って構築する。"""

import torch
from torch.nn import Identity, Linear, Module, Sigmoid

from .model_architecture_settings import ModelArchitectureSettings
from .nonlinear_residual_adapter import (
    IdentityFeatureAdapter,
    ModelOwnedHiddenLayers,
    NonlinearResidualAdapter,
)
from .shared_feature_extractor import SharedFeatureExtractor

# 共有部を持たない構造の名前。この構造の分類器は、パラメータを持たない特徴抽出部（入力をそのまま返す）と、
# モデルごとの隠れ層を持つ。
INDEPENDENT_MODELS_ARCHITECTURE_NAME = "independent_models_without_shared_backbone"


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
        # 共有部を持たない構造では、特徴抽出部は、隠れ層を持たない。
        hidden_layer_widths=(
            ()
            if validated_model_architecture_settings.model_architecture_name
            == INDEPENDENT_MODELS_ARCHITECTURE_NAME
            else hidden_layer_widths
        ),
    )
    return validated_model_architecture_settings


class ModelArchitectureClassifier(Module):
    """特徴抽出部（共有できる）を借用し、モデルごとの部品（`model_owned_feature_transform`）と分類層を独立に所有する。

    モデルごとの部品は、提案構成の構造では、残差adapter。

    adapterを持たない構造（設定の構造の名前）では、`model_owned_feature_transform`は、共有特徴をそのまま返す部品で、
    パラメータを持たない（概念固有部は、分類層だけ）。

    共有部を持たない構造では、特徴抽出部（`feature_extractor`）は、隠れ層を持たず（入力をそのまま返す。
    パラメータなし）、隠れ層は、`model_owned_feature_transform`の場所に、モデルごとに独立に持つ（`ModelOwnedHiddenLayers`）。
    概念固有部（`model_owned_feature_transform`と`classification_layer`のパラメータ）が、モデルの全パラメータになる。

    `input_feature_count`と`hidden_layer_widths`は、どの構造でも、構築のときに指定された値。同じ構造の
    分類器を作るときは、特徴抽出部の寸法でなく、これらを使う。
    """

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
        self.input_feature_count = input_feature_count
        self.hidden_layer_widths = hidden_layer_widths
        models_are_independent = (
            validated_model_architecture_settings.model_architecture_name
            == INDEPENDENT_MODELS_ARCHITECTURE_NAME
        )
        self.feature_extractor = (
            shared_feature_extractor
            if shared_feature_extractor is not None
            else SharedFeatureExtractor(
                input_feature_count=input_feature_count,
                hidden_layer_widths=() if models_are_independent else hidden_layer_widths,
            )
        )
        self.model_owned_feature_transform: (
            IdentityFeatureAdapter | ModelOwnedHiddenLayers | NonlinearResidualAdapter
        )
        if models_are_independent:
            # 隠れ層→分類層の順に作る（旧の、共有部を持たないモデルと、同じ順に乱数を消費する）。
            self.model_owned_feature_transform = ModelOwnedHiddenLayers(
                input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
            )
        elif (
            validated_model_architecture_settings.model_architecture_name
            == "shared_backbone_without_residual_adapter"
        ):
            # パラメータを作らないので、乱数も消費しない（旧の、adapterを持たないモデルと同じ）。
            self.model_owned_feature_transform = IdentityFeatureAdapter(
                feature_count=self.feature_extractor.output_feature_count
            )
        else:
            self.model_owned_feature_transform = NonlinearResidualAdapter(
                feature_count=self.feature_extractor.output_feature_count,
                requested_rank=validated_model_architecture_settings.residual_adapter_requested_rank,
            )
        classification_output_count = 1 if class_count == 2 else class_count
        self.classification_layer = Linear(
            (
                self.model_owned_feature_transform.output_feature_count
                if type(self.model_owned_feature_transform) is ModelOwnedHiddenLayers
                else self.feature_extractor.output_feature_count
            ),
            classification_output_count,
            device="cpu",
            dtype=torch.float32,
        )
        self.output_activation = Sigmoid() if class_count == 2 else Identity()

    def attach_shared_feature_extractor(
        self, *, shared_feature_extractor: SharedFeatureExtractor
    ) -> None:
        """適合する共有部への参照だけを交換し、概念固有部を保持する。"""
        if type(shared_feature_extractor) is not SharedFeatureExtractor:
            raise ValueError("shared_feature_extractorはexact SharedFeatureExtractorが必要です。")
        shared_feature_extractor.validate_structure(
            input_feature_count=self.feature_extractor.input_feature_count,
            hidden_layer_widths=self.feature_extractor.hidden_layer_widths,
        )
        self.feature_extractor = shared_feature_extractor

    def extract_shared_features(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(input_features)

    def forward_from_shared_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        transformed_features = self.model_owned_feature_transform(shared_features)
        classification_scores = self.classification_layer(transformed_features)
        return self.output_activation(classification_scores)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.forward_from_shared_features(self.extract_shared_features(input_features))
