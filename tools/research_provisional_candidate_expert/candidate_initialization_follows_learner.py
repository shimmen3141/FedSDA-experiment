"""IMPROVE-033。選択したMDL学習者と候補の初期値だけをrun中に結び付ける。"""

from contextlib import contextmanager
from hashlib import sha256
from unittest.mock import patch

from torch import zeros

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training import (
    prequential_code_length as training,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment import (
    minimum_description_length_assignment_selection as selection,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME,
    MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME,
)
from federated_learning_experiments.runtime import (
    alarm_buffer_response as response,
)
from federated_learning_experiments.runtime import (
    alarm_change_interval_resolution as resolution,
)
from federated_learning_experiments.runtime import (
    alarm_interval_minimum_description_length_selection as mdl,
)
from federated_learning_experiments.runtime import (
    observed_sample_processing as processing,
)

CANDIDATE_INITIALIZATION_VARIANTS = (
    "product-default-candidate-init-follows-new-model-learner",
    "product-default-candidate-init-current-model",
)


def _concept_specific(snapshot):
    return {k: v for k, v in snapshot.items() if not k.startswith("feature_extractor.")}


def _equal(left, right):
    return left.keys() == right.keys() and all(
        left[k].shape == right[k].shape
        and left[k].dtype == right[k].dtype
        and left[k].equal(right[k])
        for k in left
    )


def _digest(snapshot):
    digest = sha256()
    for name in sorted(snapshot):
        value = snapshot[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str((tuple(value.shape), value.dtype)).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def reinitialized_candidate_parameter_snapshot(
    *, classifier, initialization_seed, optimizer_settings
):
    """製品と同じ初期化を行い、学習直前で値を取り出す。runの乱数は消費しない。"""
    captured = []

    def snapshot(**arguments):
        captured.append(snapshot_classifier_parameters(classifier=arguments["learner"]))
        return 0.0

    # 製品の初期化の順・残差adapterの0初期化・乱数の隔離をそのまま使う。
    # 標本は入力検査だけに使い、予測・optimizer生成・学習には進まない。
    with patch.object(training, "_accumulate_prequential_code_length", snapshot):
        training.compute_reinitialized_learner_prequential_code_length(
            template_classifier=classifier,
            input_features=zeros(1, classifier.feature_extractor.input_feature_count),
            observed_class_labels=zeros(1, 1),
            parameter_optimizer_settings=optimizer_settings,
            initialization_seed=initialization_seed,
        )
    return captured[0]


def _seed(*, client_id, alarm_sample_index, count, search):
    if search == EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME:
        return mdl.derive_reinitialized_learner_single_pass_initialization_seed(
            client_id=client_id, alarm_sample_index=alarm_sample_index
        )
    if search == MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME:
        return mdl.derive_reinitialized_learner_initialization_seed(
            client_id=client_id,
            alarm_sample_index=alarm_sample_index,
            change_interval_sample_count=count,
        )
    raise ValueError("未対応の変化区間探索です")


@contextmanager
def candidate_initialization_policy(*, variant, diagnostics):
    """選択の実体ごとに記憶し、解決呼出し中だけ候補の初期値を差し替える。"""
    if variant not in CANDIDATE_INITIALIZATION_VARIANTS:
        raise ValueError("未対応の候補初期化の変種です")
    original_select = processing.select_alarm_interval_assignment
    original_resolve = response.resolve_alarm_change_interval
    original_initialize = resolution.select_candidate_initial_parameter_snapshot
    original_train = resolution.train_candidate_on_alarm_change_interval
    original_reinitialize = training._compute_reinitialized_learner_prequential_code_length
    pending = {}
    active = []
    diagnostics.update(candidates=[])

    def select(**arguments):
        initial_by_seed = {}

        def observe_reinitialized(**learner_arguments):
            accumulate = training._accumulate_prequential_code_length

            def observe_initial(**samples):
                seed = learner_arguments["initialization_seed"]
                if seed in initial_by_seed:
                    raise ValueError("同じ選択内で学習者のseedが重複しています")
                initial_by_seed[seed] = _concept_specific(
                    snapshot_classifier_parameters(classifier=samples["learner"])
                )
                return accumulate(**samples)

            with patch.object(training, "_accumulate_prequential_code_length", observe_initial):
                return original_reinitialize(**learner_arguments)

        with patch.object(
            training,
            "_compute_reinitialized_learner_prequential_code_length",
            observe_reinitialized,
        ):
            chosen = original_select(**arguments)
        if chosen.selected_hypothesis != "new_model":
            return chosen
        key = id(chosen)
        if key in pending:
            raise ValueError("未解決の選択を上書きできません")
        count = chosen.change_interval_sample_count
        search = arguments["change_interval_search"]
        seed = _seed(
            client_id=arguments["client_id"],
            alarm_sample_index=arguments["alarm_sample_index"],
            count=count,
            search=search,
        )
        if seed not in initial_by_seed:
            raise ValueError("選択した区間の学習者の初期値が捕捉されていません")
        pending[key] = dict(
            selection=chosen,
            client_id=arguments["client_id"],
            alarm_sample_index=arguments["alarm_sample_index"],
            change_interval_sample_count=count,
            change_interval_search=search,
            learner_initial=initial_by_seed[seed],
            learner_seed=seed,
        )
        return chosen

    def resolve(**arguments):
        chosen = arguments.get("alarm_interval_assignment_selection")
        # 現在のモデル・保持モデルへの帰属では候補を作らず、表にも入れない。
        if chosen is not None and chosen.selected_hypothesis != "new_model":
            return original_resolve(**arguments)
        entry = pending.get(id(chosen))
        if entry is None or entry["selection"] is not chosen:
            raise ValueError("解決に渡された選択が記憶した選択と一致しません")
        if arguments["proposal_sample_index"] != entry["alarm_sample_index"]:
            raise ValueError("解決に渡された警報の標本番号が一致しません")
        if (
            len(arguments["change_interval_training_samples"])
            != entry["change_interval_sample_count"]
        ):
            raise ValueError("解決に渡された変化区間の長さが一致しません")
        if active:
            raise ValueError("候補初期化の解決を入れ子にできません")
        active.append((entry, arguments))
        try:
            return original_resolve(**arguments)
        finally:
            active.pop()
            del pending[id(chosen)]

    def initialize(**arguments):
        if not active:
            raise ValueError("候補の初期化は選択と対応した解決の中だけで実行できます")
        entry, resolve_arguments = active[-1]
        if "diagnostic" in entry:
            raise ValueError("同じ警報で候補を複数回初期化できません")
        # 製品の入力検査を維持し、元の方式の値とモデルIDを診断する。
        original = original_initialize(**arguments)
        current_id = arguments["current_training_model_id"]
        losses = arguments["evaluated_mean_losses_by_model_id"]
        original_id = min(losses, key=lambda pair: pair[1])[0] if losses else current_id
        current = arguments["available_parameter_snapshots_by_model_id"][current_id]
        chosen = entry["selection"]
        learner_name = chosen.new_model_learner_name
        if learner_name not in (
            selection.REINITIALIZED_LEARNER_NAME,
            selection.WARM_STARTED_LEARNER_NAME,
        ):
            raise ValueError("新しいモデルの学習者が不明です")
        reinitialize = (
            variant == CANDIDATE_INITIALIZATION_VARIANTS[0]
            and learner_name == selection.REINITIALIZED_LEARNER_NAME
        )
        seed = entry["learner_seed"] if reinitialize else None
        if reinitialize:
            classifier = (
                resolve_arguments["held_model_training_state_registry"]
                .get_held_model_training_state(model_id=current_id)
                .classifier
            )
            initial = reinitialized_candidate_parameter_snapshot(
                classifier=classifier,
                initialization_seed=seed,
                optimizer_settings=resolve_arguments["parameter_optimizer_settings"],
            )
        else:
            initial = {name: value.detach().clone() for name, value in current.items()}
        if (
            original is None
            or initial.keys() != original.keys()
            or any(
                initial[k].shape != original[k].shape or initial[k].dtype != original[k].dtype
                for k in initial
            )
        ):
            raise ValueError("候補の初期値の形式が製品と一致しません")
        concept = _concept_specific(initial)
        matches_reinitialized = _equal(concept, entry["learner_initial"])
        if reinitialize and not matches_reinitialized:
            raise ValueError("候補とMDL学習者の最初の概念固有部が一致しません")
        diagnostic = dict(
            client_id=entry["client_id"],
            alarm_sample_index=entry["alarm_sample_index"],
            change_interval_sample_count=entry["change_interval_sample_count"],
            change_interval_search=entry["change_interval_search"],
            new_model_learner_name=learner_name,
            initialization_kind="reinitialized_concept_specific_part"
            if reinitialize
            else "current_model_copy",
            initialization_seed=seed,
            reinitialized_learner_initialization_seed=entry["learner_seed"],
            concept_specific_parameter_digest=_digest(concept),
            reinitialized_learner_initial_parameter_digest=_digest(entry["learner_initial"]),
            matches_reinitialized_learner_initial_parameters=matches_reinitialized,
            matches_selected_learner_initial_parameters=(
                matches_reinitialized
                if learner_name == selection.REINITIALIZED_LEARNER_NAME
                else _equal(concept, _concept_specific(current))
            ),
            lowest_evaluated_mean_loss_model_id=original_id,
            completed_epoch_count=None,
        )
        entry["diagnostic"] = diagnostic
        diagnostics["candidates"].append(diagnostic)
        return initial

    def train(**arguments):
        if not active or "diagnostic" not in active[-1][0]:
            raise ValueError("候補の学習と初期化の記録が対応していません")
        result = original_train(**arguments)
        active[-1][0]["diagnostic"]["completed_epoch_count"] = (
            result.candidate_epoch_training_result.completed_epoch_count
        )
        return result

    with (
        patch.object(processing, "select_alarm_interval_assignment", select),
        patch.object(response, "resolve_alarm_change_interval", resolve),
        patch.object(resolution, "select_candidate_initial_parameter_snapshot", initialize),
        patch.object(resolution, "train_candidate_on_alarm_change_interval", train),
    ):
        yield
        if pending:
            raise ValueError("contextの終了時に未解決の新しいモデルの選択が残っています")
