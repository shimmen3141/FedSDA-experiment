"""最後の同期の時点の保有モデルの写しを持ち、標本の予測に使う分類器を選ぶ。

予測に使うパラメータを`parameters_at_last_synchronization`にした設定で使う。写しは、配布の反映の直後に取り、
次の同期まで変えない。学習・損失の監視ほかは、写しを使わず、保有モデルの現在の分類器を使う。
"""

from copy import deepcopy

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)


class SynchronizedPredictionModelSnapshot:
    """モデルIDごとに、写しを取った時点の分類器への参照と、写した分類器を持つ。最初は空。"""

    def __init__(self) -> None:
        self._snapshot_classifiers_by_model_id: dict[
            int, tuple[ModelArchitectureClassifier, ModelArchitectureClassifier]
        ] = {}

    def capture_held_models(
        self, *, held_model_training_state_registry: HeldModelTrainingStateRegistry
    ) -> None:
        """保有モデル全部の分類器を写す（前の写しは捨てる）。

        分類器どうしが共有する共有部が、写しの中でも1つのオブジェクトになるよう、全部を1回でまとめて写す。
        """
        if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
            raise TypeError(
                "held_model_training_state_registry must be exact HeldModelTrainingStateRegistry"
            )
        held_model_training_states = (
            held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        held_classifiers = tuple(state.classifier for state in held_model_training_states)
        copied_classifiers = deepcopy(held_classifiers)
        self._snapshot_classifiers_by_model_id = {
            state.model_id: (held_classifier, copied_classifier)
            for state, held_classifier, copied_classifier in zip(
                held_model_training_states, held_classifiers, copied_classifiers, strict=True
            )
        }

    def select_prediction_classifiers(
        self, *, held_model_training_state_registry: HeldModelTrainingStateRegistry
    ) -> tuple[ModelArchitectureClassifier, dict[int, ModelArchitectureClassifier]]:
        """共有特徴を計算する分類器と、保有モデルごとの予測の分類器を返す。

        写しを取った時点の分類器のまま（同じオブジェクト）のモデルは写した分類器、それ以外（同期の後に作った
        モデル）は現在の分類器。共有特徴は、写した分類器があれば、そのうちID昇順で最初のもので計算する
        （同期の後に作ったモデルも、同じ共有部につながっているので、共有部は同期の時点の値で評価される）。
        写した分類器が1つも無ければ、ID昇順で最初の保有モデルの現在の分類器で計算する（今の予測と同じ）。
        """
        if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
            raise TypeError(
                "held_model_training_state_registry must be exact HeldModelTrainingStateRegistry"
            )
        prediction_classifiers_by_model_id: dict[int, ModelArchitectureClassifier] = {}
        snapshot_model_ids = []
        for (
            state
        ) in held_model_training_state_registry.snapshot_ordered_held_model_training_states():
            snapshot_entry = self._snapshot_classifiers_by_model_id.get(state.model_id)
            if snapshot_entry is not None and snapshot_entry[0] is state.classifier:
                prediction_classifiers_by_model_id[state.model_id] = snapshot_entry[1]
                snapshot_model_ids.append(state.model_id)
            else:
                prediction_classifiers_by_model_id[state.model_id] = state.classifier
        if not prediction_classifiers_by_model_id:
            raise ValueError("held_model_training_state_registry must hold at least one model")
        shared_feature_model_id = min(snapshot_model_ids or prediction_classifiers_by_model_id)
        return (
            prediction_classifiers_by_model_id[shared_feature_model_id],
            prediction_classifiers_by_model_id,
        )
