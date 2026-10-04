"""初期状態が恒等写像となる非線形残差adapter。"""

import torch
from torch.nn import Module, Linear, ReLU
from torch.nn.init import zeros_


def _validate_feature_tensor(*, input_features: torch.Tensor, expected_feature_count: int) -> None:
    if not isinstance(input_features, torch.Tensor):
        raise ValueError("shared_featuresはTensorが必要です。")
    if (input_features.device.type != "cpu" or input_features.dtype != torch.float32
            or input_features.layout != torch.strided or input_features.is_nested):
        raise ValueError("shared_featuresはCPU float32 strided Tensorが必要です。")
    if input_features.dim() not in (1, 2) or input_features.shape[-1] != expected_feature_count:
        raise ValueError("shared_featuresは末尾の特徴数が一致するvector/batchが必要です。")


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
        self.feature_compression = Linear(feature_count, self.effective_rank, device="cpu", dtype=torch.float32)
        self.activation = ReLU()
        self.feature_expansion = Linear(self.effective_rank, feature_count, device="cpu", dtype=torch.float32)
        # 通常初期化の乱数消費後にゼロ化して旧構築順を保つ。
        zeros_(self.feature_expansion.weight)
        zeros_(self.feature_expansion.bias)

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(input_features=shared_features, expected_feature_count=self.feature_count)
        residual_features = self.feature_expansion(self.activation(self.feature_compression(shared_features)))
        return shared_features + residual_features
