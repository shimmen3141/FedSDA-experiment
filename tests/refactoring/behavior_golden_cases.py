"""新実装の重要動作を基準版・数値環境ごとに固定するtest専用helper。"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from federated_learning_experiments.configuration.experiment_run_conditions import (  # noqa: E402
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (  # noqa: E402
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.data.mnist.mnist_training_data import (  # noqa: E402
    resolve_mnist_data_directory,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (  # noqa: E402
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (  # noqa: E402
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (  # noqa: E402
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
    REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
    REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
    PendingModelRegistrationSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (  # noqa: E402
    KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
)
from federated_learning_experiments.runtime import (  # noqa: E402
    alarm_occurrence_handling,
    fedsda_run_client,
    observed_sample_processing,
    server_model_registration_and_aggregation,
    server_round_synchronization,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import (  # noqa: E402
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_record_tables import (  # noqa: E402
    build_fedavg_run_record_tables,
)
from federated_learning_experiments.runtime.fedavg_run_settings import (  # noqa: E402
    FedavgRunSettings,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (  # noqa: E402
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_record_tables import (  # noqa: E402
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_settings import (  # noqa: E402
    FeddriftRunSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (  # noqa: E402
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (  # noqa: E402
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep.experiment_configurations import (  # noqa: E402
    build_experiment_run_settings,
)

# 照合する基準版の全部と、現在の版（新しい基準は、現在の版で生成する。最新のdirectoryを自動で選ばない）。
# 版1: 2026-10-13の移植完了時点の動作。FedSDAは、送信保留を1つだけ保持し、確認で現行モデルを付け替える
#      （固定旧と同じ登録の扱い）。保存した設定は、登録の扱いの項目を持たない（版1の後に足された）。
# 版2: FedSDAが、送信保留のモデルを、それ自身のIDで登録する（LEGACY-016の修正。当時の最終構成の既定）。
#      版1との差は、版1の扱いで、登録されないままの採用が起きるケースに限る。
# 版3: FedSDAが、短い警報（変化区間が最小件数より少ない警報）のあと、現行モデルへ取り込んだ前区間を、
#      保留から外す（LEGACY-002の修正。当時の最終構成の既定）。版2との差は、版2の扱いで、前区間のある
#      短い警報が起きるケースに限る。
# 版4: FedSDAが、送信保留のモデルを、現行モデルとして待った後に登録する（LEGACY-016の扱いの見直し。警報は即時。
#      この扱いを明示する）。版3との差は、版3の扱いで、待ちの間に現行でない送信保留が起きるケースに限る。
# 版5: FedSDAが、さらに、推定変化区間が最小件数に達するまで警報を待つ（IMPROVE-006。案Y。最終構成の既定。
#      ADWINの2構成は、警報を即時に明示）。版4との差は、版4の扱いで、警報を待つ場面が起きるケースに限る。
# 版6: 予測を同期時点・局所更新の混合へ、Fixed-Shareの時間尺度を120へ替える（容量は30）。
SUPPORTED_BASELINE_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
BASELINE_VERSION = 14
SCHEMA_VERSION = 1
# 版の後に足された、FedSDAのclientの設定の項目（`run_client_settings`からの経路）と、その版の動作に当たる値
# （保存の形）。保存してある設定と比べるときは、組み立てた設定の値がこの値であることを確かめてから除く（その版の
# 設定の組立ては、`build_case_settings`が、既定と違う項目だけを個別に置き換える）。
_PENDING_MODEL_REGISTRATION_SETTINGS_PATH = ("pending_model_registration_settings",)
_TOO_SHORT_CHANGE_INTERVAL_POLICY_PATH = (
    "training_data_assignment_settings",
    "pending_sample_retention_after_too_short_change_interval_policy",
)
_SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH = (
    "loss_change_detection_settings",
    "short_change_interval_alarm_policy",
)
# 予測に使うパラメータ（2026-10-15に足した項目）。版1〜5は、どれも局所更新した現在の値。
_PREDICTION_MODEL_PARAMETER_SOURCE_PATH = (
    "prediction_combination_settings",
    "prediction_model_parameter_source",
)
_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS = {
    _PREDICTION_MODEL_PARAMETER_SOURCE_PATH: "current_local_parameters",
}
_LOSS_MONITORING_BASELINE_UPDATE_POLICY_PATH = (
    "loss_change_detection_settings",
    "loss_monitoring_baseline_update_policy",
)
# Fixed-Shareの時間尺度T。版1〜5は30（帰属の保留の容量と同じ値）。
_FIXED_SHARE_TIME_SCALE_PATH = (
    "prediction_combination_settings",
    "fixed_share_weight_redistribution_time_scale_samples",
)
# 候補の学習で、共有部を更新するかどうか（2026-10-20に足した項目）。版1〜6は、どれも候補と一緒に更新する。
_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_PATH = (
    "candidate_epoch_training_settings",
    "candidate_shared_feature_update_policy",
)
_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS = {
    _CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_PATH: "update_shared_features_with_candidate",
}
# 候補の採否の方針。版1〜6は、候補検証あり（版7で、既定が、検証せずに採用する方針へ変わる）。
_CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH = ("candidate_model_acceptance_policy",)
_CANDIDATE_VALIDATION_POLICY_NAME = (
    "current_model_first_reuse_then_two_segment_candidate_validation"
)
# 版の後に足された、FedSDAの参加者の設定の項目（`run_participant_settings`からの経路）と、その版の値。
# クロス評価のモデルの送り方（2026-10-20に足した項目）。版1〜7は、どれも、候補のモデルを送る旧の扱い。
_CROSS_EVALUATION_MODEL_TRANSFER_POLICY_PATH = ("cross_evaluation_model_transfer_policy",)
# 統合での共有部の扱い（2026-10-20に足した項目）。版1〜8は、どれも、共有部も平均する旧の扱い。
_CONSOLIDATED_SHARED_FEATURE_POLICY_PATH = ("consolidated_shared_feature_policy",)
_PARTICIPANT_SETTINGS_ADDED_AFTER_BASELINE_VERSION = {
    baseline_version: {
        **(
            {
                _CROSS_EVALUATION_MODEL_TRANSFER_POLICY_PATH: (
                    "send_candidate_models_for_cross_evaluation"
                )
            }
            if baseline_version <= 7
            else {}
        ),
        _CONSOLIDATED_SHARED_FEATURE_POLICY_PATH: "average_shared_features_with_concept_specific_parts",
    }
    for baseline_version in (1, 2, 3, 4, 5, 6, 7, 8)
}
# 初期モデルの損失の統計の推定の仕方（2026-10-22に足した項目）。版1〜10は、どれも、学習に使った事前学習の標本で
# 測る再代入の推定。
_INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_PATH = ("initial_model_loss_statistics_estimation",)
for _baseline_version in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
    _PARTICIPANT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.setdefault(_baseline_version, {})[
        _INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_PATH
    ] = "resubstitution"
# 版1〜11の設定では、標本を持たないclientも候補に入れる。
for _baseline_version in range(1, 12):
    _PARTICIPANT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.setdefault(_baseline_version, {})[
        ("cross_evaluation_client_selection_policy",)
    ] = "clients_holding_target_model"
# 警報の後の変化区間の長さと帰属先の選び方（2026-10-21に足した項目）。版1〜9は、どれも、検出器の推定と
# 許容量で選ぶ今の扱い。
_ALARM_INTERVAL_ASSIGNMENT_POLICY_PATH = (
    "training_data_assignment_settings",
    "alarm_interval_assignment_policy",
)
_ALARM_INTERVAL_ASSIGNMENT_POLICY_DEFAULTS = {
    _ALARM_INTERVAL_ASSIGNMENT_POLICY_PATH: "detector_change_span_and_mean_loss_increase_tolerance",
}
# 「新しいモデル」の仮説の学習者の組（2026-10-22に足した項目）。版1〜10は、どれも、初期化し直した学習者だけ。
_NEW_MODEL_HYPOTHESIS_LEARNERS_PATH = (
    "training_data_assignment_settings",
    "new_model_hypothesis_learners",
)
# 版10で足した、最小記述長の原理による選択の記録の表。版9までの比べる形には含めない（今の扱いでは空の表）。
ALARM_INTERVAL_ASSIGNMENT_SELECTION_TABLE_PREFIX = "alarm_interval_assignment_selection."
# 版11で、選択の記録の表に足した列（「新しいモデル」で最短だった学習者）。版10までの比べる形には含めない。
NEW_MODEL_LEARNER_NAME_TABLE_NAME = "alarm_interval_assignment_selection.new_model_learner_name"
_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION = {
    1: {
        _PENDING_MODEL_REGISTRATION_SETTINGS_PATH: {
            "settings_type": "PendingModelRegistrationSettings",
            "pending_model_registration_policy": (
                KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
            ),
        },
        _TOO_SHORT_CHANGE_INTERVAL_POLICY_PATH: KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
        _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH: RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        **_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS,
        **_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS,
    },
    2: {
        _TOO_SHORT_CHANGE_INTERVAL_POLICY_PATH: KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
        _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH: RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        **_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS,
        **_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS,
    },
    3: {
        _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH: RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        **_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS,
        **_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS,
    },
    4: {
        **_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS,
        **_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS,
    },
    5: {
        **_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS,
        **_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS,
    },
    6: dict(_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS),
}
for _baseline_version in (1, 2, 3, 4, 5, 6, 7, 8, 9):
    _CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.setdefault(_baseline_version, {}).update(
        _ALARM_INTERVAL_ASSIGNMENT_POLICY_DEFAULTS
    )
for _baseline_version in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
    _CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.setdefault(_baseline_version, {})[
        _NEW_MODEL_HYPOTHESIS_LEARNERS_PATH
    ] = "reinitialized_learner"
# 版の後に、最終構成の既定の値が変わった、FedSDAのclientの設定の項目と、その版の値。保存してある設定は、
# この項目を持つので、比べる前に除かない（上の「後に足された項目」の表へ入れると、除かれて比べられなくなる）。
# 版2・3: 登録の扱いは、送信保留の全部を、それ自身のIDで登録する（現在の既定——版5——とは違う扱い）。
# 版4: 登録は、現行モデルとして待った後に登録する（版5の既定と同じ）。警報は即時（版5の既定とは違う）。
# 版1〜5: Fixed-Shareの時間尺度Tは30（版6で、既定が120へ変わる）。
# 版1〜6: 候補の採否の方針は、候補検証あり（版7で、既定が変わる）。候補検証を外した構成のケース
# （`fedsda-without-candidate-validation`）は、構成が方針を明示するので、当てない。
for _baseline_version in range(1, 13):
    _CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.setdefault(_baseline_version, {})[
        _LOSS_MONITORING_BASELINE_UPDATE_POLICY_PATH
    ] = "keep_baseline_until_alarm"

# 既存の項目の値が版13で変わる。保存してある設定からは除かず、上限3を明示して照合する。
_PARTICIPANT_SETTING_VALUES_CHANGED_AFTER_BASELINE_VERSION = {
    version: {("maximum_evaluating_client_count_per_model",): 3} for version in range(1, 13)
}

_CLIENT_SETTING_VALUES_CHANGED_AFTER_BASELINE_VERSION = {
    1: {
        _FIXED_SHARE_TIME_SCALE_PATH: 30,
        _CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH: _CANDIDATE_VALIDATION_POLICY_NAME,
    },
    2: {
        _PENDING_MODEL_REGISTRATION_SETTINGS_PATH: REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
        _FIXED_SHARE_TIME_SCALE_PATH: 30,
        _CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH: _CANDIDATE_VALIDATION_POLICY_NAME,
    },
    3: {
        _PENDING_MODEL_REGISTRATION_SETTINGS_PATH: REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
        _FIXED_SHARE_TIME_SCALE_PATH: 30,
        _CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH: _CANDIDATE_VALIDATION_POLICY_NAME,
    },
    4: {
        _PENDING_MODEL_REGISTRATION_SETTINGS_PATH: REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
        # 版4の警報の扱いは、即時（版5で、既定が、最小件数まで待つ扱いへ変わる）。
        _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH: RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        _FIXED_SHARE_TIME_SCALE_PATH: 30,
        _CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH: _CANDIDATE_VALIDATION_POLICY_NAME,
    },
    5: {
        _FIXED_SHARE_TIME_SCALE_PATH: 30,
        _CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH: _CANDIDATE_VALIDATION_POLICY_NAME,
    },
    6: {_CANDIDATE_MODEL_ACCEPTANCE_POLICY_PATH: _CANDIDATE_VALIDATION_POLICY_NAME},
}


def _validate_baseline_version(baseline_version):
    if type(baseline_version) is not int or baseline_version not in SUPPORTED_BASELINE_VERSIONS:
        raise ValueError(f"unsupported baseline version: {baseline_version!r}")


def golden_directory(baseline_version=BASELINE_VERSION):
    _validate_baseline_version(baseline_version)
    return Path(__file__).with_name("behavior_regression_goldens") / f"v{baseline_version}"


GOLDEN_DIRECTORY = golden_directory()
# 基準版1〜5のgoldenの事例にした構成（構成の一覧の、2026-10-15までの13個。並びも同じ）。後から足した構成は、
# goldenの事例にせず、下の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`の専用のtestで守る（保存済みのgoldenは、
# 事例の集合を一覧と厳密に比べ、既存の版を更新する経路も無いため）。
BASELINE_CONFIGURATION_CASE_NAMES = (
    "fedsda",
    "fedsda-without-prediction-weight-recalibration",
    "fedsda-without-candidate-validation",
    "fedsda-loss-distance-clustering",
    "fedsda-overall-loss-esr-detection",
    "fedsda-adahedge-prediction-weights",
    "fedsda-single-model-prediction",
    "fedsda-classwise-adwin-detection",
    "fedsda-overall-loss-adwin-detection",
    "fedsda-without-concept-specific-adapter",
    "fedsda-independent-models",
    "feddrift",
    "fedavg",
)
# goldenの事例にしない構成の名前と、その構成を守る専用のtestのファイル（リポジトリの根からの位置）。
CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN: dict[str, str] = {
    "fedavg-local-updates-per-aggregation": "tests/refactoring/test_fedavg_local_updates_per_aggregation.py",
    "fedsda-synchronized-model-prediction": "tests/refactoring/test_fedsda_synchronized_model_prediction.py",
    "fedsda-synchronized-local-mixture-prediction": "tests/refactoring/test_fedsda_synchronized_local_mixture_prediction.py",
    "fedsda-local-parameter-prediction": "tests/refactoring/test_fedsda_local_parameter_prediction.py",
    # 版7では、候補検証の経路を守る大きい条件のケースを、この構成で実行する（構成の名前のケースは無い）。
}
_VERSION_1_CASE_NAMES = (
    *BASELINE_CONFIGURATION_CASE_NAMES,
    "fedsda-sine2-events",
    "fedsda-mnist2-decisions",
    "feddrift-drift-tail",
    "fedavg-no-complete-interval",
)
# 版2では、`fedsda-sine2-events`（版1と同じ設定）が、登録の扱いの修正で、参照モデルの再利用の判定を通らなくなる
# （登録されずにclientへ残っていたモデルが、登録・統合されるため）。その経路は、seedの違うケースで守る。
CASE_NAMES_BY_BASELINE_VERSION = {
    1: _VERSION_1_CASE_NAMES,
    2: (*_VERSION_1_CASE_NAMES, "fedsda-sine2-reference-reuse"),
    3: (*_VERSION_1_CASE_NAMES, "fedsda-sine2-reference-reuse"),
    # 版4では、`fedsda-sine2-reference-reuse`（seed 4）が、登録の扱いの見直しで、別の保有モデルの再利用の判定を
    # 通らなくなる。その経路は、同じ設定でseedが5のケースで守る（2026-10-14、seed 0〜15を調べて選んだ）。
    4: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
    ),
    5: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
    ),
    6: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
    ),
    # 版7では、候補検証の経路を守る大きい条件の4ケースを、版6までの既定と同じ設定で記録した。新しい既定（候補検証なし＋候補の共有部の固定）の経路は、同じ条件の2ケース
    # （seedも同じ0。2026-10-20の下調べで、採用・切替え・再較正が起き、sine2では統合も起きた）で守る。
    7: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
        "fedsda-sine2-adoption-without-validation",
        "fedsda-mnist2-adoption-without-validation",
    ),
    # 版8は、クロス評価で配布するモデルを使う扱い（通信量の記録だけが版7と違う）。ケースは版7と同じ。
    8: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
        "fedsda-sine2-adoption-without-validation",
        "fedsda-mnist2-adoption-without-validation",
    ),
    # 版9は、統合で同じ共有部を平均せずに保つ扱い。ケースは版8と同じ。
    9: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
        "fedsda-sine2-adoption-without-validation",
        "fedsda-mnist2-adoption-without-validation",
    ),
    # 版10は、警報の後の変化区間の長さと帰属先を最小記述長の原理で選ぶ扱い。ケースは版9と同じ。
    10: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
        "fedsda-sine2-adoption-without-validation",
        "fedsda-mnist2-adoption-without-validation",
    ),
    # 版11は、「新しいモデル」の仮説のwarm startの学習者と、初期モデルの統計の交差検証。版10のケースに、
    # 一部のクラスだけが変わる概念で、warm startの学習者が短い選択が起きるMNIST4のケースを足す（既存の2つの
    # 大きい条件のケースでは、この選択が起きない。2026-10-22の実装の前の確認）。
    11: (
        *_VERSION_1_CASE_NAMES,
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
        "fedsda-sine2-adoption-without-validation",
        "fedsda-mnist2-adoption-without-validation",
        "fedsda-mnist4-warm-started-learner",
    ),
}
# 版12は、対象の標本を返せるclientだけを評価させる。ケースは版11と同じ。
CASE_NAMES_BY_BASELINE_VERSION[12] = CASE_NAMES_BY_BASELINE_VERSION[11]
# 版13は、損失の低下による基準の引下げと、評価clientの上限の撤廃。ケースは版12と同じ。
CASE_NAMES_BY_BASELINE_VERSION[13] = CASE_NAMES_BY_BASELINE_VERSION[12]
CASE_NAMES_BY_BASELINE_VERSION[14] = CASE_NAMES_BY_BASELINE_VERSION[13]
# warm startの学習者の経路を守るケース（版11から）。
WARM_STARTED_LEARNER_PATH_CASE_NAME = "fedsda-mnist4-warm-started-learner"

# 候補検証の経路を守る、大きい条件のFedSDAのケース。
CANDIDATE_VALIDATION_PATH_CASE_NAMES = (
    "fedsda-sine2-events",
    "fedsda-mnist2-decisions",
    "fedsda-sine2-reference-reuse",
    "fedsda-sine2-alternative-reference-reuse",
)
# 新しい既定（候補検証なし＋候補の共有部の固定）の経路を守る、大きい条件のFedSDAのケース（版7から）。
ADOPTION_WITHOUT_VALIDATION_PATH_CASE_NAMES = (
    "fedsda-sine2-adoption-without-validation",
    "fedsda-mnist2-adoption-without-validation",
)
CASE_NAMES = CASE_NAMES_BY_BASELINE_VERSION[BASELINE_VERSION]
# 全部の版のケースの名前（初めて現れた版の順）。並列の実行での配り方が使う。
# 候補検証の処理を消す基準版15のspecにより、検証ありの方針のcaseを数値照合から外す。
# 判定が0件でも方針が検証ありなら外す。保存済みgoldenと組立ての両方からtestで確かめる。
_EARLY_CANDIDATE_VALIDATION_CONFIGURATION_CASE_NAMES = (
    "fedsda",
    "fedsda-without-prediction-weight-recalibration",
    "fedsda-loss-distance-clustering",
    "fedsda-overall-loss-esr-detection",
    "fedsda-adahedge-prediction-weights",
    "fedsda-single-model-prediction",
    "fedsda-classwise-adwin-detection",
    "fedsda-overall-loss-adwin-detection",
    "fedsda-without-concept-specific-adapter",
    "fedsda-independent-models",
)
EXCLUDED_CANDIDATE_VALIDATION_CASE_NAMES_BY_BASELINE_VERSION = {
    1: (
        *_EARLY_CANDIDATE_VALIDATION_CONFIGURATION_CASE_NAMES,
        "fedsda-sine2-events",
        "fedsda-mnist2-decisions",
    ),
    2: (
        *_EARLY_CANDIDATE_VALIDATION_CONFIGURATION_CASE_NAMES,
        "fedsda-sine2-events",
        "fedsda-mnist2-decisions",
        "fedsda-sine2-reference-reuse",
    ),
    3: (
        *_EARLY_CANDIDATE_VALIDATION_CONFIGURATION_CASE_NAMES,
        "fedsda-sine2-events",
        "fedsda-mnist2-decisions",
        "fedsda-sine2-reference-reuse",
    ),
    **{
        version: (
            *_EARLY_CANDIDATE_VALIDATION_CONFIGURATION_CASE_NAMES,
            *CANDIDATE_VALIDATION_PATH_CASE_NAMES,
        )
        for version in (4, 5, 6)
    },
    **{version: CANDIDATE_VALIDATION_PATH_CASE_NAMES for version in range(7, 15)},
}
# 記録・指標・設定を消した項目の、goldenの中の完全な鍵の道（task 3・4で足した。一覧にない差は失敗のまま）。
REMOVED_CANDIDATE_VALIDATION_ITEM_PATHS: tuple[tuple[str, ...], ...] = (
    (
        "run_settings",
        "run_participant_settings",
        "run_client_settings",
        "candidate_model_training_and_acceptance_settings",
    ),
    (
        "run_settings",
        "run_participant_settings",
        "run_client_settings",
        "scalar_settings",
        "minimum_candidate_mean_loss_improvement",
    ),
    # FedAvgの実行eventの表は、廃止した終端の段だけ行が減る。保存済みgoldenは列全体のdigestを持つ。
    ("record_tables", "execution_event.stage_name"),
    ("record_tables", "execution_event.client_id"),
    ("record_tables", "execution_event.sample_index"),
    ("record_tables", "execution_event.round_index"),
    # 候補検証の判定の記録の表（task 4。保存済みのgoldenでは、検証なしのcaseでも空の16列を持つ）。
    ("record_tables", "candidate_validation_decision.candidate_accepted"),
    ("record_tables", "candidate_validation_decision.candidate_full_interval_mean_loss"),
    ("record_tables", "candidate_validation_decision.candidate_second_segment_mean_loss"),
    ("record_tables", "candidate_validation_decision.candidate_training_interval_sample_count"),
    ("record_tables", "candidate_validation_decision.client_id"),
    ("record_tables", "candidate_validation_decision.comparison_reference_model_id"),
    ("record_tables", "candidate_validation_decision.decision_reason"),
    ("record_tables", "candidate_validation_decision.detector_name"),
    ("record_tables", "candidate_validation_decision.is_incomplete_at_run_end"),
    ("record_tables", "candidate_validation_decision.matched_to_concept_change"),
    ("record_tables", "candidate_validation_decision.proposal_sample_index"),
    ("record_tables", "candidate_validation_decision.reference_full_interval_mean_loss"),
    ("record_tables", "candidate_validation_decision.reference_historical_mean_loss"),
    ("record_tables", "candidate_validation_decision.reference_second_segment_mean_loss"),
    ("record_tables", "candidate_validation_decision.resolution_sample_index"),
    ("record_tables", "candidate_validation_decision.validation_sample_count"),
    # 候補検証の指標（task 4）。
    ("run_metrics", "candidate_validation_decision_count"),
    ("run_metrics", "change_response_metrics", "candidate_validation_decision_metrics"),
    (
        "run_metrics",
        "change_response_metrics",
        "adaptation_action_metrics",
        "start_candidate_validation_count",
    ),
    (
        "run_metrics",
        "change_response_metrics",
        "adaptation_action_metrics",
        "reject_candidate_model_count",
    ),
    (
        "run_metrics",
        "change_response_metrics",
        "adaptation_action_metrics",
        "defer_during_candidate_validation_count",
    ),
    # 判定の記録の表から求めていた網羅の項目（task 4）。
    ("coverage", "candidate_accepted_count"),
    ("coverage", "candidate_rejected_count"),
    ("coverage", "decision_reasons"),
)


def without_removed_candidate_validation_items(snapshot, path=()):
    """名前を挙げた項目だけを除く。入力と、一覧にない差はそのまま残す。"""
    from copy import deepcopy

    result = deepcopy(snapshot)
    for removed_path in REMOVED_CANDIDATE_VALIDATION_ITEM_PATHS:
        if removed_path[: len(path)] != path:
            continue
        relative_path = removed_path[len(path) :]
        if not relative_path:
            continue
        holder = result
        for key in relative_path[:-1]:
            if not isinstance(holder, dict) or key not in holder:
                break
            holder = holder[key]
        else:
            if isinstance(holder, dict):
                holder.pop(relative_path[-1], None)
    return result


ALL_CASE_NAMES = tuple(
    dict.fromkeys(name for names in CASE_NAMES_BY_BASELINE_VERSION.values() for name in names)
)
ENVIRONMENT_KEYS = {"system", "machine", "python", "numpy", "torch", "device", "dtype", "threads"}
CASE_KEYS = {"run_settings", "run_metrics", "record_tables", "global_parameters", "coverage"}
DIGEST_KEYS = {"shape", "dtype", "sha256"}
TOP_LEVEL_KEYS = {
    "baseline_version",
    "schema_version",
    "source_commit",
    "environment",
    "mnist_sha256",
    "cases",
}


def _plain(value):
    return json.loads(
        json.dumps(convert_settings_to_plain_mapping(value, non_finite_float_as_none=True))
    )


def build_case_settings(case_name, baseline_version=BASELINE_VERSION):
    """builderの方式を維持して、明示した代表条件にだけ置換する。

    FedSDAのケースは、その版の後に足された設定の項目を、その版の動作に当たる値にする
    （版1: 登録の扱いは、現行モデルを付け替える。版1・2: 短い警報のあと、保留を全部残す）。
    """
    _validate_baseline_version(baseline_version)
    if case_name not in CASE_NAMES_BY_BASELINE_VERSION[baseline_version]:
        raise ValueError(f"unknown golden case: {case_name}")
    sine2_rich = case_name in (
        "fedsda-sine2-events",
        "fedsda-sine2-reference-reuse",
        "fedsda-sine2-alternative-reference-reuse",
        "fedsda-sine2-adoption-without-validation",
    )
    mnist = case_name in ("fedsda-mnist2-decisions", "fedsda-mnist2-adoption-without-validation")
    # MNIST4のケースは、MNIST2のケースと違い、学習率・事前学習を構成の値に近いまま使う（下の置き換え）。
    mnist4 = case_name == WARM_STARTED_LEARNER_PATH_CASE_NAME
    rich = sine2_rich or mnist or mnist4
    configuration = (
        "fedsda"
        if rich
        else "feddrift"
        if case_name == "feddrift-drift-tail"
        else "fedavg"
        if case_name == "fedavg-no-complete-interval"
        else case_name
    )
    conditions = ExperimentRunConditions(
        dataset_name="mnist2" if mnist else "mnist4" if mnist4 else "sine2",
        random_seed=4
        if case_name in ("fedsda-sine2-reference-reuse", WARM_STARTED_LEARNER_PATH_CASE_NAME)
        else 5
        if case_name == "fedsda-sine2-alternative-reference-reuse"
        else 0
        if rich
        else 71
        if case_name == "feddrift-drift-tail"
        else 37,
        client_count=3 if sine2_rich else 2,
        per_client_sample_count=100
        if mnist
        else 1000
        if mnist4
        else 1500
        if rich
        else 17
        if case_name == "feddrift-drift-tail"
        else 9
        if case_name == "fedavg-no-complete-interval"
        else 35,
        server_aggregation_interval_per_client_samples=50
        if rich
        else 4
        if case_name == "feddrift-drift-tail"
        else 10,
    )
    settings = build_experiment_run_settings(
        configuration_name=configuration, experiment_run_conditions=conditions
    )
    schedule = replace(
        settings.execution_settings.concept_schedule_settings,
        minimum_sample_index_gap_before_change_trial=30 if mnist or mnist4 else 100 if rich else 1,
        per_eligible_sample_concept_change_probability=0.015
        if rich
        else 1.0
        if case_name == "feddrift-drift-tail"
        else 0.5,
    )
    participant = settings.run_participant_settings
    pretraining = InitialModelPretrainingSettings(
        pretraining_sample_count=500 if mnist4 else 100 if rich else 24,
        pretraining_epoch_count=2 if mnist or not rich else 10,
        pretraining_batch_sample_count=32 if rich else 8,
    )
    changes: dict = dict(
        initial_model_pretraining_settings=pretraining,
        hidden_layer_widths=participant.hidden_layer_widths if rich else (5, 4),
    )
    if type(settings) is FedavgRunSettings:
        changes.update(
            local_training_batch_sample_count=4,
            updates_per_sample=2,
            local_update_interval_per_client_samples=3,
        )
    elif type(settings) is FeddriftRunSettings:
        threshold = (
            (0.03 if case_name == "feddrift-drift-tail" else 0.1)
            if baseline_version <= 13
            else participant.run_client_settings.loss_increase_threshold
        )
        changes.update(
            run_client_settings=replace(
                participant.run_client_settings,
                **(
                    _FEDDRIFT_SETTINGS_ADDED_AFTER_V13["run_client_settings"]
                    if baseline_version <= 13
                    else {}
                ),
                loss_increase_threshold=threshold,
                local_training_batch_sample_count=4,
                added_evaluation_batch_sample_count=3,
            ),
            server_settings=replace(
                participant.server_settings,
                **(
                    _FEDDRIFT_SETTINGS_ADDED_AFTER_V13["server_settings"]
                    if baseline_version <= 13
                    else {}
                ),
                maximum_cross_evaluation_client_count=3
                if baseline_version <= 13
                else conditions.client_count,
                maximum_same_cluster_loss_increase=threshold,
                initial_model_aggregation_weight=24,
                local_training_round_count=3 if case_name == "feddrift-drift-tail" else 2,
            ),
        )
    else:
        client = participant.run_client_settings
        if not rich:
            client = replace(
                client,
                scalar_settings=replace(
                    client.scalar_settings, local_training_batch_sample_count=4
                ),
                candidate_epoch_training_settings=replace(
                    client.candidate_epoch_training_settings, maximum_batch_sample_count=4
                ),
            )
        if mnist:
            client = replace(
                client,
                parameter_optimizer_settings=replace(
                    client.parameter_optimizer_settings, learning_rate=0.01
                ),
                rebuilt_model_parameter_optimizer_settings=replace(
                    client.rebuilt_model_parameter_optimizer_settings, learning_rate=0.01
                ),
            )
        added_settings = _CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.get(baseline_version, {})
        if _LOSS_MONITORING_BASELINE_UPDATE_POLICY_PATH in added_settings:
            client = replace(
                client,
                loss_change_detection_settings=replace(
                    client.loss_change_detection_settings,
                    loss_monitoring_baseline_update_policy="keep_baseline_until_alarm",
                ),
            )
        changed_settings = _CLIENT_SETTING_VALUES_CHANGED_AFTER_BASELINE_VERSION.get(
            baseline_version, {}
        )
        if _PENDING_MODEL_REGISTRATION_SETTINGS_PATH in changed_settings:
            client = replace(
                client,
                pending_model_registration_settings=PendingModelRegistrationSettings(
                    pending_model_registration_policy=changed_settings[
                        _PENDING_MODEL_REGISTRATION_SETTINGS_PATH
                    ]
                ),
            )
        if _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH in changed_settings:
            client = replace(
                client,
                loss_change_detection_settings=replace(
                    client.loss_change_detection_settings,
                    short_change_interval_alarm_policy=changed_settings[
                        _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH
                    ],
                ),
            )
        if _PENDING_MODEL_REGISTRATION_SETTINGS_PATH in added_settings:
            client = replace(
                client,
                pending_model_registration_settings=PendingModelRegistrationSettings(
                    pending_model_registration_policy=(
                        KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
                    )
                ),
            )
        if _SHORT_CHANGE_INTERVAL_ALARM_POLICY_PATH in added_settings:
            client = replace(
                client,
                loss_change_detection_settings=replace(
                    client.loss_change_detection_settings,
                    short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
                ),
            )
        if _FIXED_SHARE_TIME_SCALE_PATH in changed_settings:
            client = replace(
                client,
                prediction_combination_settings=replace(
                    client.prediction_combination_settings,
                    fixed_share_weight_redistribution_time_scale_samples=changed_settings[
                        _FIXED_SHARE_TIME_SCALE_PATH
                    ],
                ),
            )
        if _PREDICTION_MODEL_PARAMETER_SOURCE_PATH in added_settings:
            client = replace(
                client,
                prediction_combination_settings=replace(
                    client.prediction_combination_settings,
                    prediction_model_parameter_source=added_settings[
                        _PREDICTION_MODEL_PARAMETER_SOURCE_PATH
                    ],
                ),
            )
        # 候補の採否の方針と共有部の更新は、組合せの制約（候補検証と共有部の固定は組み合わせられない）が
        # あるので、1回の置き換えで、まとめて替える。
        candidate_changes = {}
        if _CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_PATH in added_settings:
            candidate_changes["candidate_epoch_training_settings"] = replace(
                client.candidate_epoch_training_settings,
                candidate_shared_feature_update_policy=added_settings[
                    _CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_PATH
                ],
            )
        # 警報の後の選び方と学習者の組も、組合せの制約（候補検証と最小記述長の選択、warm startの学習者と
        # 最小記述長の選択）があるので、同じ置き換えで替える。
        assignment_changes = {
            added_field_path[-1]: added_settings[added_field_path]
            for added_field_path in (
                _ALARM_INTERVAL_ASSIGNMENT_POLICY_PATH,
                _NEW_MODEL_HYPOTHESIS_LEARNERS_PATH,
            )
            if added_field_path in added_settings
        }
        if assignment_changes:
            candidate_changes["training_data_assignment_settings"] = replace(
                client.training_data_assignment_settings, **assignment_changes
            )
        if candidate_changes:
            client = replace(client, **candidate_changes)
        if _TOO_SHORT_CHANGE_INTERVAL_POLICY_PATH in added_settings:
            client = replace(
                client,
                training_data_assignment_settings=replace(
                    client.training_data_assignment_settings,
                    pending_sample_retention_after_too_short_change_interval_policy=KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
                ),
            )
        changes["run_client_settings"] = client
        # 参加者の設定の、版の後に足された項目を、その版の値に明示する。
        for (
            field_path,
            version_value,
        ) in _PARTICIPANT_SETTING_VALUES_CHANGED_AFTER_BASELINE_VERSION.get(
            baseline_version, {}
        ).items():
            changes[field_path[-1]] = version_value
        for (
            added_field_path,
            version_value,
        ) in _PARTICIPANT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.get(baseline_version, {}).items():
            changes[added_field_path[-1]] = version_value
    if case_name in CANDIDATE_VALIDATION_PATH_CASE_NAMES and baseline_version >= 7:
        from federated_learning_experiments.runtime.fedsda_run_client_settings import (
            FedsdaRunClientSettings,
        )

        client = changes["run_client_settings"]
        assert isinstance(client, FedsdaRunClientSettings)
        changes["run_client_settings"] = replace(
            client,
            candidate_epoch_training_settings=replace(
                client.candidate_epoch_training_settings,
                candidate_shared_feature_update_policy="update_shared_features_with_candidate",
            ),
            training_data_assignment_settings=replace(
                client.training_data_assignment_settings,
                alarm_interval_assignment_policy="detector_change_span_and_mean_loss_increase_tolerance",
                new_model_hypothesis_learners="reinitialized_learner",
            ),
            loss_change_detection_settings=replace(
                client.loss_change_detection_settings,
                loss_monitoring_baseline_update_policy="keep_baseline_until_alarm",
            ),
        )
        changes.update(
            initial_model_loss_statistics_estimation="resubstitution",
            cross_evaluation_client_selection_policy="clients_holding_target_model",
            maximum_evaluating_client_count_per_model=3,
        )
    return replace(
        settings,
        execution_settings=replace(settings.execution_settings, concept_schedule_settings=schedule),
        run_participant_settings=replace(participant, **changes),
        run_metric_settings=replace(
            settings.run_metric_settings, post_change_recovery_window_sample_count=50
        )
        if rich
        else settings.run_metric_settings,
    )


_FEDDRIFT_SETTINGS_ADDED_AFTER_V13 = {
    "run_client_settings": {
        "model_comparison_criterion": "bounded_loss",
        "new_model_parameter_source": "current_training_model",
        "detection_reference_after_drift": "new_model_batch_value",
    },
    "server_settings": {
        "model_comparison_criterion": "bounded_loss",
        "training_model_eligibility": "held_models_with_training_samples",
        "consolidation_weight_timing": "weights_before_current_timestep",
    },
}


def comparable_run_settings(case_name, baseline_version=BASELINE_VERSION):
    """保存してある`run_settings`と比べる形の設定（読込みの検査と、実行した結果の照合の、両方が使う）。

    FedSDAのケースは、その版の後に足された項目を、値がその版の動作に当たる値であることを確かめてから、
    除く（保存してあるその版の設定は、その項目を持たない。確かめずに除くと、別の動作の設定でも通ってしまう）。
    """
    settings = build_case_settings(case_name, baseline_version)
    plain_settings = _plain(settings)
    if type(settings) is FeddriftRunSettings and baseline_version <= 13:
        for section, fields in _FEDDRIFT_SETTINGS_ADDED_AFTER_V13.items():
            holder = plain_settings["run_participant_settings"][section]
            for name, value in fields.items():
                assert holder[name] == value
                del holder[name]
    if type(settings) not in (FedavgRunSettings, FeddriftRunSettings):
        client_settings = plain_settings["run_participant_settings"]["run_client_settings"]
        for added_field_path, version_value in _CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.get(
            baseline_version, {}
        ).items():
            holder = client_settings
            for field_name in added_field_path[:-1]:
                holder = holder[field_name]
            assert holder[added_field_path[-1]] == version_value, (
                f"{case_name}: baseline v{baseline_version} requires "
                f"{'.'.join(added_field_path)}={version_value!r}"
            )
            del holder[added_field_path[-1]]
        participant_settings = plain_settings["run_participant_settings"]
        for (
            added_field_path,
            version_value,
        ) in _PARTICIPANT_SETTINGS_ADDED_AFTER_BASELINE_VERSION.get(baseline_version, {}).items():
            holder = participant_settings
            for field_name in added_field_path[:-1]:
                holder = holder[field_name]
            assert holder[added_field_path[-1]] == version_value, (
                f"{case_name}: baseline v{baseline_version} requires "
                f"run_participant_settings.{'.'.join(added_field_path)}={version_value!r}"
            )
            del holder[added_field_path[-1]]
    return plain_settings


def numerical_environment():
    """比較対象のCPU数値計算環境。CPU型番差はCI診断の対象とする。"""
    return dict(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        device="cpu",
        dtype=str(torch.get_default_dtype()),
        threads=torch.get_num_threads(),
    )


def golden_file_name(environment):
    return (
        f"{environment['system']}-{environment['machine']}-python{environment['python']}-numpy{environment['numpy']}-torch{environment['torch']}.json"
    ).lower()


def mnist_sha256():
    directory = resolve_mnist_data_directory()
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    }


def _canonical_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def array_snapshot(array):
    """値・shape・dtypeを、ZIPやnative byte orderと無関係に固定する。"""
    if type(array) is not np.ndarray or array.dtype.kind not in "biufU":
        raise TypeError("numeric/bool/unicode ndarray is required")
    payload = dict(
        shape=list(array.shape), dtype=array.dtype.name, values=_canonical_value(array.tolist())
    )
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return dict(shape=payload["shape"], dtype=payload["dtype"], sha256=digest)


def assert_case_matches(case_name, actual, expected, path=""):
    """相違した設定・指標・列・parameterのfield pathを示す。"""
    label = f"{case_name}: {path}"
    if isinstance(expected, dict):
        assert isinstance(actual, dict), label
        assert set(actual) == set(expected), (
            f"{label} keys missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}"
        )
        for key in sorted(expected):
            assert_case_matches(case_name, actual[key], expected[key], f"{path}.{key}")
    else:
        assert type(actual) is type(expected) and actual == expected, (
            f"{label}: actual={actual!r}, expected={expected!r}"
        )


def _held_and_pending_model_ids_by_client(run):
    """FedSDAのrunの終わりの、clientごとの（保有モデルのID、送信保留のモデルのID）。"""
    held_and_pending_model_ids = []
    for run_client in run.measured_run.participants.client_operations:
        owners = run_client.owners
        held_and_pending_model_ids.append(
            (
                tuple(
                    held_state.model_id
                    for held_state in owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
                ),
                tuple(
                    pending_model_upload.model_id
                    for pending_model_upload in owners.pending_model_upload_state.get_all_pending_model_uploads()
                ),
            )
        )
    return held_and_pending_model_ids


def count_unregistered_adopted_models(run):
    """runの終わりに、clientが一時ID（負）で保有していて、送信保留でもないモデルの数（全clientの合計）。

    採用したのに、サーバへ登録されないままになったモデル（LEGACY-016）。送信保留のまま残るモデル（最後の区間の
    採用など。以後の同期が無い）は、数えない。FedDrift・FedAvgのrunは、定義により0。
    """
    if type(run.run_settings) in (FedavgRunSettings, FeddriftRunSettings):
        return 0
    return sum(
        model_id < 0 and model_id not in pending_model_ids
        for held_model_ids, pending_model_ids in _held_and_pending_model_ids_by_client(run)
        for model_id in held_model_ids
    )


def count_pending_models_at_run_end(run):
    """runの終わりに、送信保留のまま残ったモデルの数（全clientの合計）。FedDrift・FedAvgのrunは、数えない（0）。"""
    if type(run.run_settings) in (FedavgRunSettings, FeddriftRunSettings):
        return 0
    return sum(
        len(pending_model_ids)
        for _, pending_model_ids in _held_and_pending_model_ids_by_client(run)
    )


def count_repeated_training_sample_assignments(run):
    """FedSDAのrunの終わりに、clientの学習標本（全モデル）にある、同じ標本の2つ目以降の数（全clientの合計）。

    標本のobjectは、観測のたびに1つ作られ、取込みは、そのobjectを学習標本へ足す（統合と正式IDの付替えは、
    列を移すか連結するだけ）ので、2回取り込まれた標本（LEGACY-002）は、同じobjectが2つになる。
    FedDrift・FedAvgのrunは、保留の部品を持たないので、受け付けない。
    """
    if type(run.run_settings) in (FedavgRunSettings, FeddriftRunSettings):
        raise TypeError("FedSDAのrunが必要です。")
    repeated_assignment_count = 0
    for run_client in run.measured_run.participants.client_operations:
        training_sample_object_ids = [
            id(training_sample)
            for collection in run_client.owners.training_sample_store.snapshot_ordered_model_training_samples()
            for training_sample in collection.training_samples
        ]
        repeated_assignment_count += len(training_sample_object_ids) - len(
            set(training_sample_object_ids)
        )
    return repeated_assignment_count


def _coverage(case_name, run, tables, baseline_version=BASELINE_VERSION):
    result = run.measured_run.run_result
    coverage = dict(
        generated_sample_count_per_client=result.generated_sample_count_per_client,
        processed_sample_count_per_client=result.processed_sample_count_per_client,
        unprocessed_tail_sample_count_per_client=result.unprocessed_tail_sample_count_per_client,
        synchronization_round_count=result.synchronization_interval_count,
    )

    def column(name):
        return tables.get(name, np.asarray([], dtype=np.int64))

    if baseline_version >= 13 and type(run.run_settings) not in (
        FedavgRunSettings,
        FeddriftRunSettings,
    ):
        coverage["loss_monitoring_baseline_lowering_count"] = int(
            column("loss_monitoring_baseline_lowering.sample_index").size
        )

    if baseline_version >= 12 and type(run.run_settings) not in (
        FedavgRunSettings,
        FeddriftRunSettings,
    ):
        sample_counts = column("client_cross_evaluation.evaluated_sample_count")
        coverage.update(
            cross_evaluation_record_count=int(sample_counts.size),
            zero_sample_cross_evaluation_count=int(np.count_nonzero(sample_counts == 0)),
        )
    coverage.update(
        training_model_switch_count=int(column("training_model_switch.client_id").size),
        merged_model_count=int(
            np.count_nonzero(column("model_clustering.merged_with_other_models"))
        ),
        recalibration_replayed_sample_count=int(
            column(
                "client_weight_update_counts.fixed_share_weight_aggregation_recalibration_replayed_sample_count"
            ).sum()
        ),
        registered_model_count=int(column("model_registration.model_id").size),
        detection_count=run.run_metrics.training_model_switch_detection_metrics.detection_count,
    )
    if baseline_version >= 7 and type(run.run_settings) not in (
        FedavgRunSettings,
        FeddriftRunSettings,
    ):
        # 採用した候補の数（検証ありの採用も含む。新しい既定では、全部が検証なしの採用）。
        coverage.update(
            adopted_candidate_count=run.run_metrics.change_response_metrics.adaptation_action_metrics.adopt_candidate_model_count
        )
    if baseline_version >= 10 and type(run.run_settings) not in (
        FedavgRunSettings,
        FeddriftRunSettings,
    ):
        # 最小記述長の原理で選んだ仮説ごとの回数（選択の記録の表から。今の扱いの構成では、どれも0）。
        selected_hypotheses = column(
            "alarm_interval_assignment_selection.selected_hypothesis"
        ).tolist()
        coverage.update(
            selected_current_model_count=selected_hypotheses.count("current_model"),
            selected_held_model_count=selected_hypotheses.count("held_model"),
            selected_new_model_count=selected_hypotheses.count("new_model"),
        )
        if baseline_version >= 11:
            # 「新しいモデル」を選んだときに、短かった学習者ごとの回数（版11で足した列から）。
            new_model_learner_names = column(NEW_MODEL_LEARNER_NAME_TABLE_NAME).tolist()
            coverage.update(
                selected_reinitialized_learner_count=new_model_learner_names.count(
                    "reinitialized_learner"
                ),
                selected_warm_started_learner_count=new_model_learner_names.count(
                    "warm_started_learner"
                ),
            )
    if baseline_version >= 2:
        coverage.update(unregistered_adopted_model_count=count_unregistered_adopted_models(run))
    if baseline_version >= 3 and type(run.run_settings) not in (
        FedavgRunSettings,
        FeddriftRunSettings,
    ):
        # 保留の部品を持つ、FedSDAのケースだけの項目。
        coverage.update(
            repeated_training_sample_assignment_count=count_repeated_training_sample_assignments(
                run
            )
        )
    return coverage


def assert_required_paths(case_name, coverage, baseline_version=BASELINE_VERSION):
    """イベントが空のケースを重要経路の基準に採用しない。"""
    if baseline_version >= 13 and case_name.startswith("fedsda"):
        if case_name in CANDIDATE_VALIDATION_PATH_CASE_NAMES:
            # 候補検証を消したので、goldenの照合から外したこのcaseを補助のtestで流すときは、確かめない（基準版15のspec）。
            if (
                case_name
                not in EXCLUDED_CANDIDATE_VALIDATION_CASE_NAMES_BY_BASELINE_VERSION[
                    baseline_version
                ]
            ):
                assert coverage["loss_monitoring_baseline_lowering_count"] == 0, (
                    case_name,
                    coverage,
                )
        elif case_name in (
            "fedsda-sine2-adoption-without-validation",
            WARM_STARTED_LEARNER_PATH_CASE_NAME,
        ):
            assert coverage["loss_monitoring_baseline_lowering_count"] > 0, (case_name, coverage)
    if (
        baseline_version >= 12
        and case_name.startswith("fedsda")
        and case_name not in CANDIDATE_VALIDATION_PATH_CASE_NAMES
    ):
        assert coverage["zero_sample_cross_evaluation_count"] == 0, (case_name, coverage)
        if case_name in (
            "fedsda-sine2-adoption-without-validation",
            WARM_STARTED_LEARNER_PATH_CASE_NAME,
        ):
            assert coverage["cross_evaluation_record_count"] > 0, (case_name, coverage)
    if baseline_version >= 2:
        # 版2の扱いでは、採用したモデルが、登録されないまま残らない。
        assert coverage["unregistered_adopted_model_count"] == 0, (case_name, coverage)
    if baseline_version >= 3 and "repeated_training_sample_assignment_count" in coverage:
        # 版3の扱いでは、同じ標本が、2回取り込まれない。
        assert coverage["repeated_training_sample_assignment_count"] == 0, (case_name, coverage)
    # 版11の新しい既定では、MNIST2のケースの条件で警報が出ない（交差検証の統計）。このケースの採用の経路と
    # 選択は版10までに限り、版11では、MNIST4のケースが守る。
    adoption_path_case_names = (
        tuple(
            name
            for name in ADOPTION_WITHOUT_VALIDATION_PATH_CASE_NAMES
            if name != "fedsda-mnist2-adoption-without-validation"
        )
        + (WARM_STARTED_LEARNER_PATH_CASE_NAME,)
        if baseline_version >= 11
        else ADOPTION_WITHOUT_VALIDATION_PATH_CASE_NAMES
    )
    if case_name in adoption_path_case_names:
        # 検証の判定は無く、候補を、その場で採用する。学習帰属の切替えと、予測の重みの再較正も起きる。
        for key in (
            "adopted_candidate_count",
            "training_model_switch_count",
            "recalibration_replayed_sample_count",
        ):
            assert coverage[key] > 0, f"{case_name}: required {key}"
        if case_name == "fedsda-sine2-adoption-without-validation":
            assert coverage["merged_model_count"] > 0, case_name
            # 切替えが採用より多い（保有モデルの再利用の経路も通る）。
            assert coverage["training_model_switch_count"] > coverage["adopted_candidate_count"]
        if baseline_version >= 10:
            # 最小記述長の原理の選択が起きる。SINE2のケースでは、3種類（現在のモデルのまま、保有モデルへ、新しい
            # モデルへ）が、それぞれ1回以上起きる。MNIST2のケースでは、保有モデルへの切り替わりが起きない
            # （2026-10-21の実装の前の確認で0回）ので、3種類は求めない。
            selection_keys = (
                "selected_current_model_count",
                "selected_held_model_count",
                "selected_new_model_count",
            )
            assert sum(coverage[key] for key in selection_keys) > 0, case_name
            if case_name in (
                "fedsda-sine2-adoption-without-validation",
                WARM_STARTED_LEARNER_PATH_CASE_NAME,
            ):
                for key in selection_keys:
                    assert coverage[key] > 0, f"{case_name}: required {key}"
        if case_name == WARM_STARTED_LEARNER_PATH_CASE_NAME:
            # 「新しいモデル」の選択で、初期化し直した学習者とwarm startの学習者が、それぞれ1回以上短い。
            for key in (
                "selected_reinitialized_learner_count",
                "selected_warm_started_learner_count",
            ):
                assert coverage[key] > 0, f"{case_name}: required {key}"
    if case_name == "feddrift-drift-tail":
        assert coverage["detection_count"] > 0 and coverage["registered_model_count"] > 1, coverage
        assert coverage["processed_sample_count_per_client"] == 16, coverage
        assert coverage["unprocessed_tail_sample_count_per_client"] == 1, coverage
        assert coverage["synchronization_round_count"] == 4, coverage
    if case_name == "fedavg":
        assert coverage["synchronization_round_count"] == 3, coverage
        assert (
            coverage["processed_sample_count_per_client"] == 30
            and coverage["unprocessed_tail_sample_count_per_client"] == 5
        ), coverage
    if case_name == "fedavg-no-complete-interval":
        assert (
            coverage["processed_sample_count_per_client"]
            == coverage["synchronization_round_count"]
            == 0
        ), coverage
        assert coverage["unprocessed_tail_sample_count_per_client"] == 9, coverage


def capture_case(case_name, baseline_version=BASELINE_VERSION):
    """ケースを実行して、goldenと比べる形の結果を返す。"""
    return capture_case_and_count_divergence_events(case_name, baseline_version)[0]


def _pending_model_ids(pending_model_upload_state):
    return tuple(
        pending_model_upload.model_id
        for pending_model_upload in pending_model_upload_state.get_all_pending_model_uploads()
    )


class PendingModelNotCurrentWhileWaitingCounter:
    """待ちの間に現行でない送信保留の事象を、FedSDAの全体runの実行の間だけ数える（版3→4の差の説明）。

    (i) 後に同期が続く、clientの待ちの進行の直前の、現行でない送信保留の数（進行のたびに仮の数へ足し、次の
    同期の登録が呼ばれたときに、本の数へ移す。最後の同期の後の進行の分は、本の数へ入らない）。(ii) 確認で外れた
    送信保留のIDが、確認の直前の現行のIDと違った回数。読むのは、送信保留の一覧と現行のIDだけ（どの方針でも
    許される）。確認と同じ選択の関数は、観測に使わない。

    包みは、runの参加者を組み立てる前に入れる（clientの待ちの進行はclassの属性、登録と確認は、それを名前で
    importして呼ぶmoduleの名前を包む）。`confirmed_pending_model_currentness`には、確認ごとに、（外れた送信保留
    のID、確認の直前の現行のID）を残す（新しい既定の全体runの不変条件のtestが読む）。
    """

    def __init__(self):
        self.counted_event_count = 0
        self._provisional_event_count = 0
        self.confirmed_pending_model_currentness = []

    @contextmanager
    def counting(self):
        client_type = fedsda_run_client.FedsdaRunClient
        real_advance = client_type.advance_new_model_upload_wait_after_synchronization
        real_registration = server_round_synchronization.register_ready_client_models
        real_confirmation = (
            server_model_registration_and_aggregation.confirm_held_model_registration
        )
        counter = self

        def advance_counting_pending_models_that_are_not_current(run_client, **arguments):
            owners = run_client.owners
            current_training_model_id = (
                owners.current_training_model_assignment.current_training_model_id
            )
            counter._provisional_event_count += sum(
                pending_model_id != current_training_model_id
                for pending_model_id in _pending_model_ids(owners.pending_model_upload_state)
            )
            return real_advance(run_client, **arguments)

        def registration_moving_provisional_events(**arguments):
            counter.counted_event_count += counter._provisional_event_count
            counter._provisional_event_count = 0
            return real_registration(**arguments)

        def confirmation_observing_removed_pending_model(**arguments):
            pending_model_upload_state = arguments["pending_model_upload_state"]
            current_training_model_id = arguments[
                "current_training_model_assignment"
            ].current_training_model_id
            pending_model_ids_before = _pending_model_ids(pending_model_upload_state)
            assignment_change = real_confirmation(**arguments)
            pending_model_ids_after = _pending_model_ids(pending_model_upload_state)
            for removed_pending_model_id in pending_model_ids_before:
                if removed_pending_model_id in pending_model_ids_after:
                    continue
                counter.confirmed_pending_model_currentness.append(
                    (removed_pending_model_id, current_training_model_id)
                )
                counter.counted_event_count += removed_pending_model_id != current_training_model_id
            return assignment_change

        client_type.advance_new_model_upload_wait_after_synchronization = (
            advance_counting_pending_models_that_are_not_current
        )
        server_round_synchronization.register_ready_client_models = (
            registration_moving_provisional_events
        )
        server_model_registration_and_aggregation.confirm_held_model_registration = (
            confirmation_observing_removed_pending_model
        )
        try:
            yield self
        finally:
            client_type.advance_new_model_upload_wait_after_synchronization = real_advance
            server_round_synchronization.register_ready_client_models = real_registration
            server_model_registration_and_aggregation.confirm_held_model_registration = (
                real_confirmation
            )


def capture_case_and_count_divergence_events(case_name, baseline_version=BASELINE_VERSION):
    """ケースを実行して、（goldenと比べる形の結果、次の版と状態が分かれる事象の数）を返す（毎回実行する）。

    goldenの生成は、こちらを使う（2つの独立processで、それぞれ全ケースを実行して比べるため）。
    """
    settings = build_case_settings(case_name, baseline_version)
    run_outputs = _execute_case_run(case_name, settings)
    return _assemble_case_capture(case_name, baseline_version, *run_outputs)


# 同じprocessの中で、完全なrun設定（保存の形のJSON）→実行の結果（run、記録の表、パラメータ、事象の数）。
_REUSED_CASE_RUNS = {}


def capture_case_reusing_runs(case_name, baseline_version=BASELINE_VERSION):
    """`capture_case_and_count_divergence_events`と同じ結果を返す。同じprocessの中で、同じ完全なrun設定の
    runは1回だけ実行し、使い回す（経路の記録、必須の経路の確認、比べる形の設定は、版ごとに組み立てる）。

    完全なrun設定が同じなら、runの結果は同じ（乱数のseedも設定が持つ）。testの側だけが使う。
    """
    settings = build_case_settings(case_name, baseline_version)
    key = json.dumps(_plain(settings), sort_keys=True)
    if key not in _REUSED_CASE_RUNS:
        _REUSED_CASE_RUNS[key] = _execute_case_run(case_name, settings)
    run, tables, parameters, divergence_event_counts = _REUSED_CASE_RUNS[key]
    return _assemble_case_capture(
        case_name, baseline_version, run, tables, parameters, dict(divergence_event_counts)
    )


def _execute_case_run(case_name, settings):
    """run設定からケースを実行して、（run、記録の表、パラメータ、次の版と状態が分かれる事象の数）を返す。

    事象の数は、名前→数の辞書: 登録されないまま残った採用（版1→2で直した。runの終わりの状態から数える）、
    前区間のある短い警報（版2→3で、扱いが分かれる。実行の間だけ、警報の処理が呼ぶ完了処理を包んで数える）、
    同じ標本の2回目の取込み（runの終わりの状態から数える）、待ちの間に現行でない送信保留（版3→4で、扱いが
    分かれる。後に同期が続く待ちの進行の直前に、現行でない送信保留の数と、確認で外れた送信保留が、確認の直前の
    現行でなかった回数の和。実行の間だけ、clientの待ちの進行・同期が呼ぶ登録・登録が呼ぶ確認を包んで数える。
    包みが読むのは、3つの方針の全部で許される操作——送信保留の一覧と現行のID——だけ）。FedDrift・FedAvgの
    ケースは、どれも0。
    これらは、古い版の結果には入らない（古い版のファイルは、変更しない）ので、版の差の説明のtestが、ここから読む。
    """
    divergence_event_counts = dict(
        unregistered_adopted_model_count=0,
        too_short_alarm_with_earlier_interval_count=0,
        repeated_training_sample_assignment_count=0,
        pending_model_not_current_while_waiting_count=0,
        alarm_wait_situation_count=0,
    )
    if type(settings) is FedavgRunSettings:
        run = execute_fedavg_experiment_run(run_settings=settings)
        tables = build_fedavg_run_record_tables(experiment_run=run)
        parameters = {0: run.measured_run.participants.server_operations.global_parameters}
    elif type(settings) is FeddriftRunSettings:
        run = execute_feddrift_experiment_run(run_settings=settings)
        tables = build_feddrift_run_record_tables(experiment_run=run)
        assert any(
            event.stage_name == "observed_stream_finalization"
            for event in run.measured_run.run_result.execution_events
        ), case_name
        assert settings.run_participant_settings.server_settings.local_training_round_count == (
            3 if case_name == "feddrift-drift-tail" else 2
        ), case_name
        repository = run.measured_run.participants.run_server.global_model_repository
        parameters = {
            mid: repository.get_global_model_parameters(model_id=mid)
            for mid in repository.global_model_ids
        }
    else:
        # 警報の処理は、完了処理を、名前でimportしている。呼ぶ側のmoduleの名前を、この実行の間だけ包む。
        real_completion = alarm_occurrence_handling.complete_alarm_buffer_response

        def count_too_short_alarm_with_earlier_interval(**completion_arguments):
            alarm_buffer_response = completion_arguments["alarm_buffer_response"]
            divergence_event_counts["too_short_alarm_with_earlier_interval_count"] += (
                alarm_buffer_response.response_outcome == "alarm_change_interval_too_short"
                and bool(
                    alarm_buffer_response.prepared_alarm_training_intervals.earlier_observations
                )
            )
            return real_completion(**completion_arguments)

        alarm_occurrence_handling.complete_alarm_buffer_response = (
            count_too_short_alarm_with_earlier_interval
        )
        pending_model_event_counter = PendingModelNotCurrentWhileWaitingCounter()
        # 待つ扱いなら警報を抑える場面（警報が上がり、保留の件数と推定変化区間の小さい方が、最小件数より短い）。
        # 即時の扱いでは、その警報は警報の処理へ渡るので、その引数から同じ式で数える。標本1件の処理のmoduleが
        # 名前でimportしている警報の処理を、この実行の間だけ包む（e-SRの検出器のケースだけ。ADWINのケースは、
        # 版4・5とも即時の警報で、動作が変わらないので数えない）。
        real_alarm_occurrence_handling = observed_sample_processing.handle_alarm_occurrence
        counts_alarm_wait_situations = (
            settings.run_participant_settings.run_client_settings.loss_change_detection_settings.drift_detector_name
            != "adwin"
        )

        def count_alarm_wait_situation(**alarm_arguments):
            divergence_event_counts["alarm_wait_situation_count"] += (
                counts_alarm_wait_situations
                and (
                    min(
                        len(alarm_arguments["pending_sample_observations"]),
                        alarm_arguments["estimated_change_span_sample_count"],
                    )
                    < alarm_arguments["minimum_change_interval_sample_count"]
                )
            )
            return real_alarm_occurrence_handling(**alarm_arguments)

        observed_sample_processing.handle_alarm_occurrence = count_alarm_wait_situation
        try:
            with pending_model_event_counter.counting():
                run = execute_fedsda_experiment_run(run_settings=settings)
        finally:
            alarm_occurrence_handling.complete_alarm_buffer_response = real_completion
            observed_sample_processing.handle_alarm_occurrence = real_alarm_occurrence_handling
        divergence_event_counts["pending_model_not_current_while_waiting_count"] = (
            pending_model_event_counter.counted_event_count
        )
        divergence_event_counts.update(
            unregistered_adopted_model_count=count_unregistered_adopted_models(run),
            repeated_training_sample_assignment_count=count_repeated_training_sample_assignments(
                run
            ),
        )
        tables = build_fedsda_run_record_tables(experiment_run=run)
        repository = run.measured_run.participants.server_operations.owners.global_model_repository
        parameters = {
            mid: repository.get_global_model_parameters(model_id=mid)
            for mid in repository.global_model_ids
        }
    # 版11→12のtestは、goldenの要約からでなく、その場の実行の記録から0件の行を数える。
    sample_counts = tables.get(
        "client_cross_evaluation.evaluated_sample_count", np.asarray([], dtype=np.int64)
    )
    divergence_event_counts["zero_sample_cross_evaluation_count"] = int(
        np.count_nonzero(sample_counts == 0)
    )
    return run, tables, parameters, divergence_event_counts


def _assemble_case_capture(
    case_name, baseline_version, run, tables, parameters, divergence_event_counts
):
    """実行の結果から、版ごとに、goldenと比べる形の結果を組み立てる（経路の記録と必須の経路の確認を含む）。"""
    coverage = _coverage(case_name, run, tables, baseline_version)
    assert_required_paths(case_name, coverage, baseline_version)
    run_metrics = _plain(run.run_metrics)
    if baseline_version <= 12 and "loss_monitoring_computation_counts" in run_metrics:
        assert run_metrics.pop("loss_monitoring_baseline_lowering_count") == 0
        counts = run_metrics["loss_monitoring_computation_counts"]
        assert counts.pop("decrease_detector_component_update_count") == 0
        assert counts.pop("decrease_detector_evaluated_candidate_bet_count") == 0
    return (
        dict(
            run_settings=comparable_run_settings(case_name, baseline_version),
            run_metrics=run_metrics,
            record_tables={
                name: array_snapshot(value)
                for name, value in tables.items()
                if baseline_version >= 13
                or not name.startswith("loss_monitoring_baseline_lowering.")
                # 版9までの比べる形には、版10で足した選択の記録の表を含めない（今の扱いでは空の表）。版10までの
                # 比べる形には、版11で足した列を含めない。
                if (
                    baseline_version >= 10
                    or not name.startswith(ALARM_INTERVAL_ASSIGNMENT_SELECTION_TABLE_PREFIX)
                )
                and (baseline_version >= 11 or name != NEW_MODEL_LEARNER_NAME_TABLE_NAME)
            },
            global_parameters={
                f"{mid}.{name}": array_snapshot(value.detach().cpu().numpy())
                for mid, values in parameters.items()
                for name, value in values.items()
            },
            coverage=coverage,
        ),
        divergence_event_counts,
    )


def load_golden(path, baseline_version=BASELINE_VERSION):
    """未知版や破損を環境選択の前に拒否する。"""
    _validate_baseline_version(baseline_version)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(document, allow_nan=False)
        assert set(document) == TOP_LEVEL_KEYS
        for key, version in (
            ("baseline_version", baseline_version),
            ("schema_version", SCHEMA_VERSION),
        ):
            assert type(document[key]) is int and document[key] == version, key
        assert re.fullmatch(r"[0-9a-f]{40}", document["source_commit"]), "source_commit"
        assert set(document["environment"]) == ENVIRONMENT_KEYS, "environment"
        assert all(
            isinstance(v, str) for k, v in document["environment"].items() if k != "threads"
        ), "environment types"
        assert (
            type(document["environment"]["threads"]) is int
            and document["environment"]["threads"] == 1
        )
        assert (
            document["environment"]["device"] == "cpu"
            and document["environment"]["dtype"] == "torch.float32"
        )
        assert set(document["mnist_sha256"]) == {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
        }
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value) for value in document["mnist_sha256"].values()
        )
        assert set(document["cases"]) == set(CASE_NAMES_BY_BASELINE_VERSION[baseline_version]), (
            "cases"
        )
        for name, case in document["cases"].items():
            assert set(case) == CASE_KEYS, name
            if (
                name
                not in EXCLUDED_CANDIDATE_VALIDATION_CASE_NAMES_BY_BASELINE_VERSION[
                    baseline_version
                ]
            ):
                assert_case_matches(
                    name,
                    without_removed_candidate_validation_items(
                        case["run_settings"], ("run_settings",)
                    ),
                    without_removed_candidate_validation_items(
                        comparable_run_settings(name, baseline_version), ("run_settings",)
                    ),
                    "run_settings",
                )
            assert isinstance(case["run_metrics"], dict) and case["run_metrics"], name
            for section in ("record_tables", "global_parameters"):
                assert isinstance(case[section], dict) and case[section], (name, section)
                for item, digest in case[section].items():
                    assert set(digest) == DIGEST_KEYS, (name, section, item)
                    assert isinstance(digest["shape"], list) and all(
                        type(n) is int and n >= 0 for n in digest["shape"]
                    )
                    assert isinstance(digest["dtype"], str) and (
                        (
                            re.fullmatch(r"str[0-9]+", digest["dtype"])
                            and int(digest["dtype"][3:]) % 32 == 0
                        )
                        or np.dtype(digest["dtype"]).kind in "biufU"
                    )
                    assert re.fullmatch(r"[0-9a-f]{64}", digest["sha256"])
            assert_required_paths(name, case["coverage"], baseline_version)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid behavior golden: {path}: {error}") from error
    return document


def find_golden_path(environment, directory=None, baseline_version=BASELINE_VERSION):
    if directory is None:
        directory = golden_directory(baseline_version)
    matches = [
        path
        for path in sorted(directory.glob("*.json"))
        if load_golden(path, baseline_version)["environment"] == environment
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate golden environment: {matches}")
    return matches[0] if matches else None


def _write_exclusive(path, document):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def generate_golden(output, source_commit, baseline_version=BASELINE_VERSION):
    """2つの新processの全結果一致後に、新規出力へ保存する。"""
    _validate_baseline_version(baseline_version)
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full commit hash")
    git_command = ["git"]
    mounted_windows_root = re.fullmatch(r"/mnt/([a-z])/(.+)", ROOT.as_posix())
    if sys.platform == "linux" and mounted_windows_root and (ROOT / ".git").is_file():
        git_reference = (ROOT / ".git").read_text(encoding="utf-8").strip()
        if re.match(r"gitdir: [A-Za-z]:[/\\]", git_reference):
            drive, remainder = mounted_windows_root.groups()
            git_command = ["git.exe", "-C", f"{drive.upper()}:/{remainder}"]
    head = subprocess.check_output([*git_command, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source_commit != head:
        raise ValueError("source_commit must match HEAD")
    source_changes = subprocess.check_output(
        [*git_command, "status", "--porcelain", "--untracked-files=all", "--", "src"],
        cwd=ROOT,
        text=True,
    )
    if source_changes.strip():
        raise ValueError("src must match the source commit, including untracked files")
    with tempfile.TemporaryDirectory(prefix="behavior-golden-") as temporary:
        documents = []
        for iteration in (1, 2):
            candidate = Path(temporary) / f"run-{iteration}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-output",
                    str(candidate),
                    "--source-commit",
                    source_commit,
                    "--baseline-version",
                    str(baseline_version),
                ],
                cwd=ROOT,
                check=True,
            )
            documents.append(load_golden(candidate, baseline_version))
        assert_case_matches("independent-processes", documents[0], documents[1])
        _write_exclusive(output, documents[0])
    return output


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--baseline-version",
        type=int,
        default=BASELINE_VERSION,
        choices=SUPPORTED_BASELINE_VERSIONS,
    )
    args = parser.parse_args()
    if bool(args.output) == bool(args.worker_output):
        parser.error("exactly one output is required")
    environment = numerical_environment()
    if (
        environment["threads"] != 1
        or environment["dtype"] != "torch.float32"
        or any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    ):
        parser.error(
            "float32 CPU, OMP_NUM_THREADS=MKL_NUM_THREADS=1 and torch threads=1 are required"
        )
    if args.worker_output:
        cases = {}
        for name in CASE_NAMES_BY_BASELINE_VERSION[args.baseline_version]:
            cases[name] = capture_case(name, args.baseline_version)
            print(f"captured {name}", flush=True)
        _write_exclusive(
            args.worker_output,
            dict(
                baseline_version=args.baseline_version,
                schema_version=SCHEMA_VERSION,
                source_commit=args.source_commit,
                environment=environment,
                mnist_sha256=mnist_sha256(),
                cases=cases,
            ),
        )
    else:
        print(generate_golden(args.output, args.source_commit, args.baseline_version))


if __name__ == "__main__":
    _main()
