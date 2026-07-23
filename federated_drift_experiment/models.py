"""実験で用いるモデル定義。"""
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from . import config
from .compatibility import normalize_dataset_name


class SimpleMLP(nn.Module):
    """データセット仕様に応じた分類MLP。既存の二値モデル構造は維持する。"""

    def __init__(self, input_dim=None, dataset=None):
        super(SimpleMLP, self).__init__()
        self.dataset = normalize_dataset_name(
            dataset if dataset is not None else config.DATASET
        )
        spec = config.dataset_spec(self.dataset)
        if input_dim is None:
            input_dim = spec.input_dim
        self.num_classes = spec.num_classes

        layers = []
        previous_dim = input_dim
        for hidden_dim in spec.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        output_dim = 1 if self.num_classes == 2 else self.num_classes
        layers.append(nn.Linear(previous_dim, output_dim))
        if self.num_classes == 2:
            layers.append(nn.Sigmoid())
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.net = nn.Sequential(*layers)

        default_lr = spec.learning_rate if spec.learning_rate is not None else config.BASE_LR
        self.optimizer = self._build_optimizer(default_lr)

    def _build_optimizer(self, lr):
        """config.OPTIMIZER に従って最適化器を構築する。"""
        if config.OPTIMIZER == 'adam':
            return optim.Adam(self.parameters(), lr=lr,
                              weight_decay=config.WEIGHT_DECAY, amsgrad=config.AMSGRAD)
        elif config.OPTIMIZER == 'sgd':
            return optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER!r}")

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            scores = self.forward(x)
            if self.num_classes == 2:
                out = (scores > 0.5).float()
            else:
                out = torch.argmax(scores, dim=1, keepdim=True).float()
        return out

    def per_sample_error(self, x, y):
        """各標本の予測誤差を[0,1]で返す。検出器とモデル比較で共通利用する。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.num_classes == 2:
            if y.dim() == 1:
                y = y.unsqueeze(1)
            return torch.abs(self.forward(x) - y).view(-1)

        labels = y.view(-1).long()
        probabilities = torch.softmax(self.forward(x), dim=1)
        correct_probabilities = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        return 1.0 - correct_probabilities

    def get_absolute_error(self, x, y):
        """|pred - y| の平均。[0,1] に収まるためADWINへの入力損失として使う。"""
        with torch.no_grad():
            error = self.per_sample_error(x, y)
            if error.numel() == 1:
                return error.item()
            else:
                return float(torch.mean(error).item())

    def update(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(x)
        target = y if self.num_classes == 2 else y.view(-1).long()
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reset_optimizer(self, lr=None):
        """最適化器を作り直す(新規モデルの初期学習前に内部状態をリセット)。"""
        if lr is None:
            spec_lr = config.dataset_spec(self.dataset).learning_rate
            lr = spec_lr if spec_lr is not None else config.NEW_MODEL_LR
        self.optimizer = self._build_optimizer(lr)

    def get_params(self):
        return copy.deepcopy(self.state_dict())

    def set_params(self, params):
        self.load_state_dict(params)
