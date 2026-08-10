"""実験で用いるモデル定義。"""
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from . import config
from .data.names import normalize_dataset_name


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

    def per_sample_error_and_prediction(self, x, y):
        """1回のforwardから有界損失と予測クラスを同時に返す。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        scores = self.forward(x)
        if self.num_classes == 2:
            if y.dim() == 1:
                y = y.unsqueeze(1)
            return torch.abs(scores - y).view(-1), (scores > 0.5).float()
        labels = y.view(-1).long()
        probabilities = torch.softmax(scores, dim=1)
        correct_probabilities = probabilities.gather(
            1, labels.unsqueeze(1)
        ).squeeze(1)
        predictions = torch.argmax(scores, dim=1, keepdim=True).float()
        return 1.0 - correct_probabilities, predictions

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


class SharedFeatureBackbone(nn.Module):
    """概念別ヘッド間で共有するMLPの特徴抽出部。"""

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        self.output_dim = previous_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConceptAdapter(nn.Module):
    """共有特徴を概念ごとに補正する小さなadapter。

    元のMLPに後段の隠れ層がある場合は、その層を概念固有部分として使う。
    隠れ層が一つしかない場合は、特徴ごとのスケールとバイアスだけを学習し、
    大きな全結合層を概念数だけ複製しない。
    """

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        if hidden_dims:
            layers = []
            previous_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
                previous_dim = hidden_dim
            self.net = nn.Sequential(*layers)
            self.scale = None
            self.bias = None
        else:
            self.net = None
            self.scale = nn.Parameter(torch.ones(input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, features):
        if self.net is not None:
            return self.net(features)
        return features * self.scale + self.bias


class ResidualConceptAdapter(nn.Module):
    """完全共有表現へ概念別の低ランク非線形残差を加えるadapter。"""

    def __init__(self, feature_dim, rank):
        super().__init__()
        self.rank = min(int(rank), int(feature_dim))
        if self.rank < 1:
            raise ValueError("残差adapterのrankは1以上である必要があります")
        self.down = nn.Linear(feature_dim, self.rank)
        self.activation = nn.ReLU()
        self.up = nn.Linear(self.rank, feature_dim)
        # 初期状態を恒等写像にし、完全共有モデルから安全に学習を開始する。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        residual = self.up(self.activation(self.down(features)))
        return features + residual


class SharedBackboneMLP(SimpleMLP):
    """共有特徴抽出層と概念別出力ヘッドからなる分類モデル。

    複数インスタンスへ同じ`SharedFeatureBackbone`を接続すると、表現学習を共有しつつ
    出力ヘッドだけを概念ごとに独立して保持できる。`get_params()`はサーバ処理との
    互換性のため、共有部とヘッドを含む完全なstate dictを返す。
    """

    is_shared_backbone_model = True

    def __init__(self, input_dim=None, dataset=None, backbone=None):
        nn.Module.__init__(self)
        self.dataset = normalize_dataset_name(
            dataset if dataset is not None else config.DATASET
        )
        spec = config.dataset_spec(self.dataset)
        if input_dim is None:
            input_dim = spec.input_dim
        self.num_classes = spec.num_classes
        self.backbone = (
            backbone
            if backbone is not None
            else SharedFeatureBackbone(input_dim, spec.hidden_dims)
        )
        output_dim = 1 if self.num_classes == 2 else self.num_classes
        self.head = nn.Linear(self.backbone.output_dim, output_dim)
        self.output_activation = nn.Sigmoid() if self.num_classes == 2 else nn.Identity()
        self.loss_fn = nn.BCELoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        default_lr = spec.learning_rate if spec.learning_rate is not None else config.BASE_LR
        if not hasattr(self.backbone, "optimizer"):
            self.backbone.optimizer = self._build_component_optimizer(
                self.backbone.parameters(), default_lr
            )
        self.head_optimizer = self._build_component_optimizer(
            self.personalized_parameters(), default_lr
        )

    def personalized_parameters(self):
        """概念IDごとに独立して保持・更新するパラメータを返す。"""
        return self.head.parameters()

    @staticmethod
    def _build_component_optimizer(parameters, lr):
        """共有部とヘッドを重複なく更新するoptimizerを構築する。"""
        if config.OPTIMIZER == 'adam':
            return optim.Adam(
                parameters,
                lr=lr,
                weight_decay=config.WEIGHT_DECAY,
                amsgrad=config.AMSGRAD,
            )
        if config.OPTIMIZER == 'sgd':
            return optim.SGD(parameters, lr=lr)
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER!r}")

    def extract_features(self, x):
        return self.backbone(x)

    def forward_from_features(self, features):
        return self.output_activation(self.head(features))

    def forward(self, x):
        return self.forward_from_features(self.extract_features(x))

    def loss_from_features(self, features, y):
        """抽出済み特徴から、この概念ヘッドの学習損失を計算する。"""
        prediction = self.forward_from_features(features)
        target = y if self.num_classes == 2 else y.view(-1).long()
        return self.loss_fn(prediction, target)

    def update(self, x, y):
        """共有部を共通optimizerで、概念別ヘッドを専用optimizerで更新する。"""
        self.backbone.optimizer.zero_grad()
        self.head_optimizer.zero_grad()
        loss = self.loss_from_features(self.extract_features(x), y)
        loss.backward()
        self.backbone.optimizer.step()
        self.head_optimizer.step()
        return loss.item()

    def reset_optimizer(self, lr=None):
        """共有部に一つ、各ヘッドに一つのoptimizerを作り直す。"""
        if lr is None:
            spec_lr = config.dataset_spec(self.dataset).learning_rate
            lr = spec_lr if spec_lr is not None else config.NEW_MODEL_LR
        self.backbone.optimizer = self._build_component_optimizer(
            self.backbone.parameters(), lr
        )
        self.head_optimizer = self._build_component_optimizer(
            self.personalized_parameters(), lr
        )

    def attach_backbone(self, backbone):
        """既存ヘッドを維持したまま、共有する特徴抽出部を付け替える。"""
        self.backbone = backbone
        spec_lr = config.dataset_spec(self.dataset).learning_rate
        lr = spec_lr if spec_lr is not None else config.NEW_MODEL_LR
        if not hasattr(self.backbone, "optimizer"):
            self.backbone.optimizer = self._build_component_optimizer(
                self.backbone.parameters(), lr
            )
        self.head_optimizer = self._build_component_optimizer(
            self.personalized_parameters(), lr
        )

    @staticmethod
    def split_params(params):
        """完全state dictを共有部とヘッドへ分ける。キーは元の接頭辞を保つ。"""
        backbone = {
            name: value for name, value in params.items()
            if name.startswith("backbone.")
        }
        personalized = {
            name: value for name, value in params.items()
            if not name.startswith("backbone.")
        }
        if not backbone or not personalized:
            raise ValueError("SharedBackboneMLPのstate dictではありません")
        return backbone, personalized

    @staticmethod
    def combine_params(backbone, head):
        return {**copy.deepcopy(backbone), **copy.deepcopy(head)}


class PartialSharedAdapterMLP(SharedBackboneMLP):
    """低層表現だけを共有し、概念別adapterと分類headを持つMLP。

    二層以上のMLPでは先頭隠れ層のみを共有し、残りの隠れ層をadapterとする。
    一層MLPでは共有隠れ層の直後に特徴別アフィンadapterを置く。これにより、
    元モデルの表現容量と比較可能性を保ちつつ、概念固有の補正を許す。
    """

    def __init__(self, input_dim=None, dataset=None, backbone=None):
        nn.Module.__init__(self)
        self.dataset = normalize_dataset_name(
            dataset if dataset is not None else config.DATASET
        )
        spec = config.dataset_spec(self.dataset)
        if input_dim is None:
            input_dim = spec.input_dim
        if not spec.hidden_dims:
            raise ValueError("部分共有adapterには少なくとも一つの隠れ層が必要です")

        self.num_classes = spec.num_classes
        shared_dims = spec.hidden_dims[:1]
        adapter_dims = spec.hidden_dims[1:]
        self.backbone = (
            backbone
            if backbone is not None
            else SharedFeatureBackbone(input_dim, shared_dims)
        )
        self.adapter = ConceptAdapter(self.backbone.output_dim, adapter_dims)
        output_dim = 1 if self.num_classes == 2 else self.num_classes
        self.head = nn.Linear(self.adapter.output_dim, output_dim)
        self.output_activation = nn.Sigmoid() if self.num_classes == 2 else nn.Identity()
        self.loss_fn = nn.BCELoss() if self.num_classes == 2 else nn.CrossEntropyLoss()

        default_lr = spec.learning_rate if spec.learning_rate is not None else config.BASE_LR
        if not hasattr(self.backbone, "optimizer"):
            self.backbone.optimizer = self._build_component_optimizer(
                self.backbone.parameters(), default_lr
            )
        self.head_optimizer = self._build_component_optimizer(
            self.personalized_parameters(), default_lr
        )

    def personalized_parameters(self):
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def forward_from_features(self, features):
        adapted = self.adapter(features)
        return self.output_activation(self.head(adapted))


class ResidualAdapterMLP(SharedBackboneMLP):
    """完全共有表現にゼロ初期化の概念別低ランク残差を加えるMLP。"""

    def __init__(self, input_dim=None, dataset=None, backbone=None):
        nn.Module.__init__(self)
        self.dataset = normalize_dataset_name(
            dataset if dataset is not None else config.DATASET
        )
        spec = config.dataset_spec(self.dataset)
        if input_dim is None:
            input_dim = spec.input_dim
        self.num_classes = spec.num_classes
        self.backbone = (
            backbone
            if backbone is not None
            else SharedFeatureBackbone(input_dim, spec.hidden_dims)
        )
        self.adapter = ResidualConceptAdapter(
            self.backbone.output_dim, config.SHARED_ADAPTER_RANK
        )
        output_dim = 1 if self.num_classes == 2 else self.num_classes
        self.head = nn.Linear(self.backbone.output_dim, output_dim)
        self.output_activation = nn.Sigmoid() if self.num_classes == 2 else nn.Identity()
        self.loss_fn = nn.BCELoss() if self.num_classes == 2 else nn.CrossEntropyLoss()

        default_lr = spec.learning_rate if spec.learning_rate is not None else config.BASE_LR
        if not hasattr(self.backbone, "optimizer"):
            self.backbone.optimizer = self._build_component_optimizer(
                self.backbone.parameters(), default_lr
            )
        self.head_optimizer = self._build_component_optimizer(
            self.personalized_parameters(), default_lr
        )

    def personalized_parameters(self):
        return list(self.adapter.parameters()) + list(self.head.parameters())

    def forward_from_features(self, features):
        adapted = self.adapter(features)
        return self.output_activation(self.head(adapted))


def parameter_payload_size(params):
    """state dictに含まれるパラメータ値数と実バイト数を返す。"""
    tensors = [value for value in params.values() if torch.is_tensor(value)]
    return (
        sum(value.numel() for value in tensors),
        sum(value.numel() * value.element_size() for value in tensors),
    )


def model_collection_parameter_footprint(models):
    """モデル集合が実質的に保持する値数とバイト数を返す。"""
    if not models:
        return 0, 0
    model_values = list(models.values())
    if all(
        getattr(model, "is_shared_backbone_model", False)
        for model in model_values
    ):
        backbone, _ = SharedBackboneMLP.split_params(
            model_values[0].get_params()
        )
        values, byte_count = parameter_payload_size(backbone)
        for model in model_values:
            _, head = SharedBackboneMLP.split_params(model.get_params())
            head_values, head_bytes = parameter_payload_size(head)
            values += head_values
            byte_count += head_bytes
        return values, byte_count

    values = 0
    byte_count = 0
    for model in model_values:
        model_values_count, model_bytes = parameter_payload_size(model.get_params())
        values += model_values_count
        byte_count += model_bytes
    return values, byte_count
