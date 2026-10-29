"""警報の後の変化区間の長さと帰属先を、保有モデルと新しいモデルの符号長から、最小記述長の原理で選ぶ。

純粋な比較は`minimum_description_length_assignment_selection`、標本ごとの対数損失と新しいモデルの予測的符号長の
計算はlearningの層が行う。ここでは、帰属の保留の標本の列と、保有モデルのowner・損失の統計から、それらを
まとめる。どれも、保有モデルの状態とrunの乱数を変えない。
"""

from hashlib import sha256
from math import isfinite

from torch import cat

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
    evaluate_classifier_per_sample_log_losses,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.prequential_code_length import (
    compute_reinitialized_learner_prequential_code_length,
    compute_warm_started_learner_prequential_code_length,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    REINITIALIZED_LEARNER_NAME,
    WARM_STARTED_LEARNER_NAME,
    MinimumDescriptionLengthAssignmentSelection,
    NewModelLearnerSelection,
    compute_switching_code_length,
    select_assignment_by_minimum_description_length,
    select_new_model_learner,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME,
    validate_new_model_hypothesis_learners,
)


def _validate_nonnegative_int(*, value: int, value_name: str, minimum_value: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{value_name} must be builtin int")
    if value < minimum_value:
        raise ValueError(f"{value_name} must be at least {minimum_value}")


def derive_reinitialized_learner_initialization_seed(
    *, client_id: int, alarm_sample_index: int, change_interval_sample_count: int
) -> int:
    """予測的符号長の学習者の初期化のseed。

    3つの値を10進の文字列にして`:`で区切った文字列（例: `3:1250:15`）のUTF-8のSHA-256の、先頭8 byteを
    big endianの符号なし整数にしたもの。
    """
    _validate_nonnegative_int(value=client_id, value_name="client_id")
    _validate_nonnegative_int(value=alarm_sample_index, value_name="alarm_sample_index")
    _validate_nonnegative_int(
        value=change_interval_sample_count,
        value_name="change_interval_sample_count",
        minimum_value=1,
    )
    text = f"{client_id}:{alarm_sample_index}:{change_interval_sample_count}"
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


def derive_reinitialized_learner_single_pass_initialization_seed(
    *, client_id: int, alarm_sample_index: int
) -> int:
    """1本の流れ用のseed: `client_id:alarm_sample_index`のUTF-8のSHA-256の先頭8 byteをbig endianで読む。"""
    _validate_nonnegative_int(value=client_id, value_name="client_id")
    _validate_nonnegative_int(value=alarm_sample_index, value_name="alarm_sample_index")
    text = f"{client_id}:{alarm_sample_index}"
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


def select_alarm_interval_assignment(
    *,
    client_id: int,
    alarm_sample_index: int,
    pending_sample_observations: tuple[IndexedObservedTrainingSample, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    switching_probability: float,
    change_interval_grid_sample_count: int,
    new_model_hypothesis_learners: str,
) -> MinimumDescriptionLengthAssignmentSelection:
    """帰属の保留の標本（警報の標本を最後に含む、古い順）の列で、最小記述長の原理で選ぶ。

    行き先は、損失の統計を持つ、現在のモデル以外の保有モデル（保有の順）と、新しいモデル。新しいモデルの
    予測的符号長は、刻みの倍数の変化区間の長さごとに、列の末尾の標本で計算する。学習者の組が
    `reinitialized_and_warm_started_learners`なら、初期化し直した学習者の後に、warm startの学習者でも計算し、
    短い方＋ln 2を使う（`select_new_model_learner`）。
    """
    _validate_nonnegative_int(value=client_id, value_name="client_id")
    validate_new_model_hypothesis_learners(
        new_model_hypothesis_learners=new_model_hypothesis_learners
    )
    if type(pending_sample_observations) is not tuple or not pending_sample_observations:
        raise ValueError("pending_sample_observations must be a nonempty builtin tuple")
    for observation in pending_sample_observations:
        if type(observation) is not IndexedObservedTrainingSample:
            raise TypeError(
                "pending_sample_observations must hold exact IndexedObservedTrainingSample"
            )
    if pending_sample_observations[-1].sample_index != alarm_sample_index:
        raise ValueError("the alarm sample must be the last pending sample")
    if type(switching_probability) is not float or not isfinite(switching_probability):
        raise ValueError("switching_probability must be a finite builtin float")
    _validate_nonnegative_int(
        value=change_interval_grid_sample_count,
        value_name="change_interval_grid_sample_count",
        minimum_value=1,
    )
    input_features = cat([o.training_sample.input_features for o in pending_sample_observations])
    observed_class_labels = cat(
        [o.training_sample.observed_class_labels for o in pending_sample_observations]
    )
    sequence_length = len(pending_sample_observations)
    current_model_id = current_training_model_assignment.current_training_model_id
    held_model_ids = tuple(
        state.model_id
        for state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    log_losses_by_model_id = {
        model_id: tuple(
            evaluate_classifier_per_sample_log_losses(
                classifier=held_model_training_state_registry.get_held_model_training_state(
                    model_id=model_id
                ).classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            ).tolist()
        )
        for model_id in held_model_ids
    }
    destination_model_ids = tuple(
        model_id
        for model_id in held_model_ids
        if model_id != current_model_id
        and loss_statistics_store.get_model_loss_statistics(model_id=model_id) is not None
    )
    template_classifier = held_model_training_state_registry.get_held_model_training_state(
        model_id=current_model_id
    ).classifier
    new_model_learner_selections: dict[int, NewModelLearnerSelection] = {}
    for change_interval_sample_count in range(
        change_interval_grid_sample_count,
        sequence_length + 1,
        change_interval_grid_sample_count,
    ):
        reinitialized_learner_code_length = compute_reinitialized_learner_prequential_code_length(
            template_classifier=template_classifier,
            input_features=input_features[sequence_length - change_interval_sample_count :],
            observed_class_labels=observed_class_labels[
                sequence_length - change_interval_sample_count :
            ],
            parameter_optimizer_settings=parameter_optimizer_settings,
            initialization_seed=derive_reinitialized_learner_initialization_seed(
                client_id=client_id,
                alarm_sample_index=alarm_sample_index,
                change_interval_sample_count=change_interval_sample_count,
            ),
        )
        learner_code_lengths: tuple[tuple[str, float], ...] = (
            (REINITIALIZED_LEARNER_NAME, reinitialized_learner_code_length),
        )
        if new_model_hypothesis_learners == REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME:
            learner_code_lengths += (
                (
                    WARM_STARTED_LEARNER_NAME,
                    compute_warm_started_learner_prequential_code_length(
                        template_classifier=template_classifier,
                        input_features=input_features[
                            sequence_length - change_interval_sample_count :
                        ],
                        observed_class_labels=observed_class_labels[
                            sequence_length - change_interval_sample_count :
                        ],
                        parameter_optimizer_settings=parameter_optimizer_settings,
                    ),
                ),
            )
        new_model_learner_selections[change_interval_sample_count] = select_new_model_learner(
            prequential_code_lengths_by_learner_name=learner_code_lengths
        )
    return select_assignment_by_minimum_description_length(
        current_model_log_losses=log_losses_by_model_id[current_model_id],
        held_model_log_losses_by_model_id=tuple(
            (model_id, log_losses_by_model_id[model_id]) for model_id in destination_model_ids
        ),
        new_model_code_lengths_by_change_interval_sample_count={
            change_interval_sample_count: learner_selection.description_length
            for change_interval_sample_count, learner_selection in new_model_learner_selections.items()
        },
        new_model_learner_names_by_change_interval_sample_count={
            change_interval_sample_count: learner_selection.learner_name
            for change_interval_sample_count, learner_selection in new_model_learner_selections.items()
        },
        switching_code_length=compute_switching_code_length(
            switching_probability=switching_probability,
            destination_count=len(destination_model_ids) + 1,
        ),
        change_interval_grid_sample_count=change_interval_grid_sample_count,
        change_interval_step_sample_count=change_interval_grid_sample_count,
    )
