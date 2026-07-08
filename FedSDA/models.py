"""実験で用いるモデル定義。"""
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from . import config


class SimpleMLP(nn.Module):
    """二値分類MLP(損失: BCE、出力: Sigmoid)。入力次元は config.DATASET に追従。"""

    def __init__(self, input_dim=None):
        super(SimpleMLP, self).__init__()
        if input_dim is None:
            input_dim = config.input_dim()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=config.BASE_LR)

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = (self.forward(x) > 0.5).float()
        return out

    def get_absolute_error(self, x, y):
        """|pred - y| の平均。[0,1] に収まるためADWINへの入力損失として使う。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        with torch.no_grad():
            pred = self.forward(x)
            error = torch.abs(pred - y)
            if error.numel() == 1:
                return error.item()
            else:
                return float(torch.mean(error).item())

    def update(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def set_optimizer_sgd(self, lr=None):
        lr = lr if lr is not None else config.NEW_MODEL_LR
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def get_params(self):
        return copy.deepcopy(self.state_dict())

    def set_params(self, params):
        self.load_state_dict(params)
