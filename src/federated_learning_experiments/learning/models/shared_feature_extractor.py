"""明示寸法のLinear/ReLU列から共有特徴を抽出する。"""

import torch
from torch.nn import Linear, Module, ReLU, Sequential


def _validate_feature_tensor(*, input_features: torch.Tensor, expected_feature_count: int) -> None:
    if not isinstance(input_features, torch.Tensor):
        raise ValueError("input_featuresはTensorが必要です。")
    if (
        input_features.device.type != "cpu"
        or input_features.dtype != torch.float32
        or input_features.layout != torch.strided
        or input_features.is_nested
    ):
        raise ValueError("input_featuresはCPU float32 strided Tensorが必要です。")
    if input_features.dim() not in (1, 2) or input_features.shape[-1] != expected_feature_count:
        raise ValueError("input_featuresは末尾の特徴数が一致するvector/batchが必要です。")


class SharedFeatureExtractor(Module):
    """共有される特徴抽出層と、生成しない構造検査を所有する。"""

    @staticmethod
    def validate_feature_dimensions(
        *, input_feature_count: int, hidden_layer_widths: tuple[int, ...]
    ) -> None:
        if type(input_feature_count) is not int or input_feature_count < 1:
            raise ValueError("input_feature_countはbool以外の正整数が必要です。")
        if type(hidden_layer_widths) is not tuple:
            raise ValueError("hidden_layer_widthsはexact tupleが必要です。")
        for hidden_layer_width in hidden_layer_widths:
            if type(hidden_layer_width) is not int or hidden_layer_width < 1:
                raise ValueError("hidden_layer_widthsの各幅はbool以外の正整数が必要です。")

    def __init__(self, *, input_feature_count: int, hidden_layer_widths: tuple[int, ...]):
        self.validate_feature_dimensions(
            input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
        )
        super().__init__()
        self.input_feature_count = input_feature_count
        self.hidden_layer_widths = hidden_layer_widths
        hidden_layers = []
        previous_feature_count = input_feature_count
        for hidden_layer_width in hidden_layer_widths:
            hidden_layers.extend(
                (
                    Linear(
                        previous_feature_count,
                        hidden_layer_width,
                        device="cpu",
                        dtype=torch.float32,
                    ),
                    ReLU(),
                )
            )
            previous_feature_count = hidden_layer_width
        self.output_feature_count = previous_feature_count
        self.hidden_layers = Sequential(*hidden_layers)

    def validate_structure(
        self, *, input_feature_count: int, hidden_layer_widths: tuple[int, ...]
    ) -> None:
        if type(self) is not SharedFeatureExtractor:
            raise ValueError("shared_feature_extractorはexact SharedFeatureExtractorが必要です。")
        self.validate_feature_dimensions(
            input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
        )
        self.validate_feature_dimensions(
            input_feature_count=self.input_feature_count,
            hidden_layer_widths=self.hidden_layer_widths,
        )
        expected_feature_count = (
            hidden_layer_widths[-1] if hidden_layer_widths else input_feature_count
        )
        if (
            self.input_feature_count != input_feature_count
            or self.hidden_layer_widths != hidden_layer_widths
            or type(self.output_feature_count) is not int
            or self.output_feature_count != expected_feature_count
        ):
            raise ValueError("shared_feature_extractorの宣言寸法が一致しません。")
        expected_layer_count = 2 * len(hidden_layer_widths)
        if (
            type(self.hidden_layers) is not Sequential
            or len(self.hidden_layers) != expected_layer_count
        ):
            raise ValueError("shared_feature_extractor.hidden_layersの型/層数が一致しません。")
        previous_feature_count = input_feature_count
        for layer_index, hidden_layer_width in enumerate(hidden_layer_widths):
            linear_layer = self.hidden_layers[2 * layer_index]
            relu_layer = self.hidden_layers[2 * layer_index + 1]
            if type(linear_layer) is not Linear or type(relu_layer) is not ReLU:
                raise ValueError(
                    "shared_feature_extractor.hidden_layersはLinear/ReLU列が必要です。"
                )
            if (
                type(linear_layer.in_features) is not int
                or type(linear_layer.out_features) is not int
                or linear_layer.in_features != previous_feature_count
                or linear_layer.out_features != hidden_layer_width
            ):
                raise ValueError("shared_feature_extractorのLinear寸法が一致しません。")
            if not isinstance(linear_layer.weight, torch.Tensor) or not isinstance(
                linear_layer.bias, torch.Tensor
            ):
                raise ValueError("shared_feature_extractorのLinear weight/biasが必要です。")
            if (
                dict(linear_layer.named_parameters(recurse=False)).get("weight")
                is not linear_layer.weight
                or dict(linear_layer.named_parameters(recurse=False)).get("bias")
                is not linear_layer.bias
            ):
                raise ValueError(
                    "shared_feature_extractorのLinear weight/biasは登録済みParameterが必要です。"
                )
            if (
                linear_layer.weight.is_nested
                or linear_layer.bias.is_nested
                or linear_layer.weight.shape != (hidden_layer_width, previous_feature_count)
                or linear_layer.bias.shape != (hidden_layer_width,)
            ):
                raise ValueError("shared_feature_extractorのweight/bias形状が一致しません。")
            for feature_extractor_parameter in (linear_layer.weight, linear_layer.bias):
                if (
                    feature_extractor_parameter.device.type != "cpu"
                    or feature_extractor_parameter.dtype != torch.float32
                    or feature_extractor_parameter.layout != torch.strided
                ):
                    raise ValueError(
                        "shared_feature_extractorのweight/biasはCPU float32 stridedが必要です。"
                    )
            previous_feature_count = hidden_layer_width

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(
            input_features=input_features, expected_feature_count=self.input_feature_count
        )
        return self.hidden_layers(input_features)
