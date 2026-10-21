"""新しいモデルの予測的（prequential）符号長: 標本を順に、予測してから1回学ぶ学習者の対数損失の和。

警報の後に「新しいモデルを作る」仮説の符号長に使う。学習者は、学び始めは予測を外すので、その分の符号長が、
新しいモデルを作ることの罰として働く。学習者は測るためだけに作り、残さない。
"""

from copy import deepcopy

from torch import Tensor, no_grad
from torch.nn import BCELoss, CrossEntropyLoss
from torch.nn.init import zeros_

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)
from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
    evaluate_classifier_per_sample_log_losses,
)
from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
    create_parameter_optimizer,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)


def compute_new_model_prequential_code_length(
    *,
    template_classifier: ModelArchitectureClassifier,
    input_features: Tensor,
    observed_class_labels: Tensor,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    initialization_seed: int,
) -> float:
    """新しいモデルの予測的符号長（nats）。

    学習者は、`template_classifier`の写しの、概念ごとの部分（共有部の上のadapterと出力の層）を初期化し直した
    もの。初期化は、runの乱数を変えないように、分けたCPUの乱数の状態で`initialization_seed`から行う
    （`reset_parameters`を持つ部品を、adapter、出力の層の順に、各部品の中の順に初期化し直し、残差のadapterの
    広げる側を0にする）。標本を順に、対数損失を足してから、共有部を勾配なしで計算し、候補の学習と同じ損失
    （2クラスはBCE、ほかはcross entropy）とoptimizerの設定で、共有部以外を1回更新する。写しを使うので、
    `template_classifier`は変わらない。
    """
    if type(template_classifier) is not ModelArchitectureClassifier:
        raise TypeError("template_classifier must be exact ModelArchitectureClassifier")
    if type(initialization_seed) is not int:
        raise TypeError("initialization_seed must be builtin int")
    if initialization_seed < 0:
        raise ValueError("initialization_seed must be nonnegative")
    if input_features.shape[0] < 1 or input_features.shape[0] != observed_class_labels.shape[0]:
        raise ValueError(
            "input_features and observed_class_labels must have the same nonzero sample count"
        )
    learner = deepcopy(template_classifier)
    with isolated_cpu_torch_random_state(random_seed=initialization_seed):
        for module in (learner.model_owned_feature_transform, learner.classification_layer):
            for submodule in module.modules():
                reset_parameters = getattr(submodule, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
        # 残差のadapterは、広げる側を0で始める（構造の初期化と同じ）。
        feature_expansion = getattr(
            learner.model_owned_feature_transform, "feature_expansion", None
        )
        if feature_expansion is not None:
            zeros_(feature_expansion.weight)
            zeros_(feature_expansion.bias)
    parameters = tuple(
        parameter
        for name, parameter in learner.named_parameters()
        if not name.startswith("feature_extractor.")
    )
    optimizer = create_parameter_optimizer(
        parameters=parameters, optimizer_settings=parameter_optimizer_settings
    )
    code_length = 0.0
    for sample_position in range(input_features.shape[0]):
        sample_features = input_features[sample_position : sample_position + 1]
        sample_labels = observed_class_labels[sample_position : sample_position + 1]
        code_length += float(
            evaluate_classifier_per_sample_log_losses(
                classifier=learner,
                input_features=sample_features,
                observed_class_labels=sample_labels,
            ).sum()
        )
        with no_grad():
            shared_features = learner.extract_shared_features(sample_features)
        optimizer.zero_grad()
        predictions = learner.forward_from_shared_features(shared_features)
        if learner.class_count == 2:
            loss = BCELoss()(predictions, sample_labels)
        else:
            loss = CrossEntropyLoss()(predictions, sample_labels.reshape(-1).long())
        loss.backward()
        optimizer.step()
    return code_length
