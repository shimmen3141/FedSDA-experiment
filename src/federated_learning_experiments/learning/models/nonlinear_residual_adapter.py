"""共有特徴と分類層の間に置く部品: 初期状態が恒等写像となる非線形残差adapterと、adapterを持たない構造の恒等の部品。"""

import torch
from torch.nn import Linear, Module, ReLU, Sequential
from torch.nn.init import zeros_


def _validate_feature_tensor(*, input_features: torch.Tensor, expected_feature_count: int) -> None:
    if not isinstance(input_features, torch.Tensor):
        raise ValueError("shared_featuresはTensorが必要です。")
    if (
        input_features.device.type != "cpu"
        or input_features.dtype != torch.float32
        or input_features.layout != torch.strided
        or input_features.is_nested
    ):
        raise ValueError("shared_featuresはCPU float32 strided Tensorが必要です。")
    if input_features.dim() not in (1, 2) or input_features.shape[-1] != expected_feature_count:
        raise ValueError("shared_featuresは末尾の特徴数が一致するvector/batchが必要です。")


class IdentityFeatureAdapter(Module):
    """adapterを持たない構造の、概念固有部の入口。共有特徴を、そのまま返す。パラメータは持たない。

    分類器が、どの構造でも、同じ場所（`residual_adapter`）に部品を持てるようにする。入力の検査は、
    残差adapterと同じ。
    """

    def __init__(self, *, feature_count: int):
        if type(feature_count) is not int or feature_count < 1:
            raise ValueError("feature_countはbool以外の正整数が必要です。")
        super().__init__()
        self.feature_count = feature_count

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(
            input_features=shared_features, expected_feature_count=self.feature_count
        )
        return shared_features


class ModelOwnedHiddenLayers(Module):
    """共有部を持たない構造の、モデルごとの隠れ層（Linear/ReLUの列）。入力の特徴から、最後の隠れ層の特徴を作る。

    層の作り方と順は、共有の特徴抽出部（`SharedFeatureExtractor`）と同じ。こちらは、モデルごとに独立に持つ、
    概念固有部の部品で、分類器の`residual_adapter`の場所に置く。
    """

    def __init__(self, *, input_feature_count: int, hidden_layer_widths: tuple[int, ...]):
        if type(input_feature_count) is not int or input_feature_count < 1:
            raise ValueError("input_feature_countはbool以外の正整数が必要です。")
        if type(hidden_layer_widths) is not tuple:
            raise ValueError("hidden_layer_widthsはexact tupleが必要です。")
        for hidden_layer_width in hidden_layer_widths:
            if type(hidden_layer_width) is not int or hidden_layer_width < 1:
                raise ValueError("hidden_layer_widthsの各幅はbool以外の正整数が必要です。")
        super().__init__()
        self.feature_count = input_feature_count
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

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(
            input_features=shared_features, expected_feature_count=self.feature_count
        )
        return self.hidden_layers(shared_features)


class NonlinearResidualAdapter(Module):
    """圧縮/ReLU/展開を通る残差を共有特徴へ加算する。"""

    def __init__(self, *, feature_count: int, requested_rank: int):
        if type(feature_count) is not int or feature_count < 1:
            raise ValueError("feature_countはbool以外の正整数が必要です。")
        if type(requested_rank) is not int or requested_rank < 1:
            raise ValueError("requested_rankはbool以外の正整数が必要です。")
        super().__init__()
        self.feature_count = feature_count
        self.requested_rank = requested_rank
        self.effective_rank = min(requested_rank, feature_count)
        self.feature_compression = Linear(
            feature_count, self.effective_rank, device="cpu", dtype=torch.float32
        )
        self.activation = ReLU()
        self.feature_expansion = Linear(
            self.effective_rank, feature_count, device="cpu", dtype=torch.float32
        )
        # 通常初期化の乱数消費後にゼロ化して旧構築順を保つ。
        zeros_(self.feature_expansion.weight)
        zeros_(self.feature_expansion.bias)

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(
            input_features=shared_features, expected_feature_count=self.feature_count
        )
        residual_features = self.feature_expansion(
            self.activation(self.feature_compression(shared_features))
        )
        return shared_features + residual_features
