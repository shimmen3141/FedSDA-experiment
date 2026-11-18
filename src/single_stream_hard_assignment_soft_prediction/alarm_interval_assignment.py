"""警報の時点で、変化区間の長さと帰属先を、木の符号長から最小記述長の原理で選ぶ（製品の同名の関数の木の版）。

符号長の比較は製品の`select_assignment_by_minimum_description_length`をそのまま使う。新しいモデルの予測的符号長は、
警報の時点の空の概念モデル（reset）と現在のモデルの写し（keep）に、列の標本を新しいものから順に1件ずつ
「MDL用の損失を足す→学習」して作る前置きの和（製品の既定の、1件刻みの逆順の1本の流れ）。保有モデルは変えない。
"""

from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    REINITIALIZED_LEARNER_NAME,
    WARM_STARTED_LEARNER_NAME,
    MinimumDescriptionLengthAssignmentSelection,
    compute_switching_code_length,
    select_assignment_by_minimum_description_length,
    select_new_model_learner,
)
from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
    HoeffdingTreeConceptModel,
)

PendingSample = tuple[dict[int, float], int]


def _validate_inputs(
    *,
    pending_samples: tuple[PendingSample, ...],
    held_models: tuple[tuple[int, HoeffdingTreeConceptModel], ...],
    current_model_id: int,
    minimum_change_interval_sample_count: int,
) -> dict[int, HoeffdingTreeConceptModel]:
    if type(pending_samples) is not tuple or not pending_samples:
        raise ValueError("pending_samplesは空でないtupleが必要です。")
    for sample in pending_samples:
        if type(sample) is not tuple or len(sample) != 2 or type(sample[0]) is not dict:
            raise TypeError("pending_samplesの各要素は(特徴の辞書, クラス)のtupleが必要です。")
    if type(held_models) is not tuple or not held_models:
        raise ValueError("held_modelsは空でないtupleが必要です。")
    models_by_id: dict[int, HoeffdingTreeConceptModel] = {}
    for entry in held_models:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("held_modelsの各要素は(モデルID, 概念モデル)のtupleが必要です。")
        model_id, model = entry
        if type(model_id) is not int or type(model) is not HoeffdingTreeConceptModel:
            raise TypeError("held_modelsの各要素は(int, HoeffdingTreeConceptModel)が必要です。")
        if model_id in models_by_id:
            raise ValueError("held_modelsのモデルIDは重複できません。")
        models_by_id[model_id] = model
    if current_model_id not in models_by_id:
        raise ValueError("current_model_idは保有モデルのIDが必要です。")
    if (
        type(minimum_change_interval_sample_count) is not int
        or minimum_change_interval_sample_count < 1
    ):
        raise ValueError("minimum_change_interval_sample_countは1以上のintが必要です。")
    return models_by_id


def _reverse_prefix_code_lengths(
    *, learner: HoeffdingTreeConceptModel, pending_samples: tuple[PendingSample, ...]
) -> list[float]:
    """新しいものから順に1件ずつ、MDL用の損失を足してから学習し、各件を足した直後の和を返す。learnerを更新する。"""
    prefixes: list[float] = []
    total = 0.0
    for features, observed_class_id in reversed(pending_samples):
        total += learner.description_length_loss(
            features=features, observed_class_id=observed_class_id
        )
        prefixes.append(total)
        learner.learn_one(features=features, observed_class_id=observed_class_id)
    return prefixes


def select_alarm_interval_assignment(
    *,
    pending_samples: tuple[PendingSample, ...],
    held_models: tuple[tuple[int, HoeffdingTreeConceptModel], ...],
    current_model_id: int,
    model_ids_with_loss_statistics: frozenset[int],
    switching_probability: float,
    minimum_change_interval_sample_count: int,
) -> MinimumDescriptionLengthAssignmentSelection:
    """帰属の保留の標本（警報の標本を最後に含む、古い順）の列で、最小記述長の原理で選ぶ。

    帰属先の候補は、現在のモデル以外で損失の統計を持つ保有モデル（保有の順）と、新しいモデル。
    """
    models_by_id = _validate_inputs(
        pending_samples=pending_samples,
        held_models=held_models,
        current_model_id=current_model_id,
        minimum_change_interval_sample_count=minimum_change_interval_sample_count,
    )
    if type(model_ids_with_loss_statistics) is not frozenset:
        raise TypeError("model_ids_with_loss_statisticsはfrozensetが必要です。")
    log_losses_by_model_id = {
        model_id: tuple(
            model.description_length_loss(features=features, observed_class_id=observed_class_id)
            for features, observed_class_id in pending_samples
        )
        for model_id, model in models_by_id.items()
    }
    destination_model_ids = tuple(
        model_id
        for model_id, _ in held_models
        if model_id != current_model_id and model_id in model_ids_with_loss_statistics
    )
    sequence_length = len(pending_samples)
    learner_code_lengths: dict[int, float] = {}
    learner_names: dict[int, str] = {}
    if sequence_length >= minimum_change_interval_sample_count:
        current_model = models_by_id[current_model_id]
        reset_prefixes = _reverse_prefix_code_lengths(
            learner=current_model.create_empty_like(), pending_samples=pending_samples
        )
        keep_prefixes = _reverse_prefix_code_lengths(
            learner=current_model.copy(), pending_samples=pending_samples
        )
        for change_interval_sample_count in range(
            minimum_change_interval_sample_count, sequence_length + 1
        ):
            selection = select_new_model_learner(
                prequential_code_lengths_by_learner_name=(
                    (REINITIALIZED_LEARNER_NAME, reset_prefixes[change_interval_sample_count - 1]),
                    (WARM_STARTED_LEARNER_NAME, keep_prefixes[change_interval_sample_count - 1]),
                )
            )
            learner_code_lengths[change_interval_sample_count] = selection.description_length
            learner_names[change_interval_sample_count] = selection.learner_name
    return select_assignment_by_minimum_description_length(
        current_model_log_losses=log_losses_by_model_id[current_model_id],
        held_model_log_losses_by_model_id=tuple(
            (model_id, log_losses_by_model_id[model_id]) for model_id in destination_model_ids
        ),
        new_model_code_lengths_by_change_interval_sample_count=learner_code_lengths,
        new_model_learner_names_by_change_interval_sample_count=learner_names,
        switching_code_length=compute_switching_code_length(
            switching_probability=switching_probability,
            destination_count=len(destination_model_ids) + 1,
        ),
        change_interval_grid_sample_count=minimum_change_interval_sample_count,
        change_interval_step_sample_count=1,
    )


def build_selected_new_concept_model(
    *,
    learner_name: str,
    current_model: HoeffdingTreeConceptModel,
    pending_samples: tuple[PendingSample, ...],
    change_interval_sample_count: int,
) -> tuple[HoeffdingTreeConceptModel, list[float]]:
    """選ばれた学習者と同じ作り方で新しい概念モデルを作る（測った学習者のk件目の状態を再現する）。

    resetは空の木、keepは現在のモデルの写しから始め、列の新しいものからk件を、1件ずつ「監視用の損失を記録→学習」
    する。新しいモデルと、記録した監視用の損失（学習した順）を返す。現在のモデルは変えない。
    """
    if type(current_model) is not HoeffdingTreeConceptModel:
        raise TypeError("current_modelはHoeffdingTreeConceptModelが必要です。")
    if type(pending_samples) is not tuple or not pending_samples:
        raise ValueError("pending_samplesは空でないtupleが必要です。")
    if type(
        change_interval_sample_count
    ) is not int or not 1 <= change_interval_sample_count <= len(pending_samples):
        raise ValueError("change_interval_sample_countは1以上、列の長さ以下が必要です。")
    if learner_name == REINITIALIZED_LEARNER_NAME:
        model = current_model.create_empty_like()
    elif learner_name == WARM_STARTED_LEARNER_NAME:
        model = current_model.copy()
    else:
        raise ValueError("learner_nameはresetかkeepの学習者の名前が必要です。")
    recorded_losses: list[float] = []
    for features, observed_class_id in list(reversed(pending_samples))[
        :change_interval_sample_count
    ]:
        recorded_losses.append(
            model.monitoring_loss(features=features, observed_class_id=observed_class_id)
        )
        model.learn_one(features=features, observed_class_id=observed_class_id)
    return model, recorded_losses
