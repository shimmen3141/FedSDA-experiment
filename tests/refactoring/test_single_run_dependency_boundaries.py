"""単一runの各層で許可する依存とpackage境界を確認する。"""

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import dataclasses", False),
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from dataclasses import field", False),
        ("from typing import Any", False),
        ("from copy import deepcopy", False),
        ("from torch import nn", False),
        ("from torch import clone", False),
        ("from torch import mean", False),
        ("from torch import no_grad", False),
        ("from torch.optim import Adam", False),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("import federated_drift_experiment", False),
        ("import pathlib", False),
        ("from torch import *", False),
        ("from dataclasses import *", False),
        ("from dataclasses import dataclass", True),
        ("from torch import Tensor", True),
        ("from torch import float32, isfinite", True),
        ("from torch import strided", True),
        ("from torch import Tensor as ParameterValues", True),
    ],
)
def test_pending_model_upload_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="methods/fedsda/model_registration/pending_model_upload.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import torch.nn", False),
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from torch import nn", False),
        ("from torch import rand", False),
        ("from torch import mean", False),
        ("from torch import clamp", False),
        ("from torch.optim import Adam", False),
        ("from .residual_adapter_classifier import _validate_classifier_inputs", False),
        ("from . import residual_adapter_classifier", False),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "from ..training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from ..loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("from federated_learning_experiments.configuration import config", False),
        ("import federated_drift_experiment", False),
        ("import pathlib", False),
        ("from torch import *", False),
        ("from torch import Tensor", True),
        ("from torch import float32, isfinite", True),
        ("from torch import no_grad, strided", True),
        ("from .residual_adapter_classifier import ResidualAdapterClassifier", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as Classifier",
            True,
        ),
    ],
)
def test_classifier_parameter_snapshot_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/models/classifier_parameter_snapshot.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import torch.nn", False),
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from torch import nn", False),
        ("from torch import rand", False),
        ("from torch import clamp", False),
        ("from torch import mean", False),
        ("from torch.nn import BCELoss", False),
        ("from torch.optim import Adam", False),
        (
            "from .class_probability_calculations import compute_model_mean_bounded_losses_after_label_observation",
            False,
        ),
        ("from ..models.residual_adapter_classifier import _validate_classifier_inputs", False),
        ("from ..models import residual_adapter_classifier", False),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "from ..loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch",
            False,
        ),
        (
            "from ..training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("from federated_learning_experiments.configuration import config", False),
        ("import federated_drift_experiment", False),
        ("import pathlib", False),
        ("from torch import *", False),
        ("from torch import Tensor", True),
        ("from torch import abs, float32, isfinite", True),
        ("from torch import no_grad, softmax, strided, trunc", True),
        ("from ..models.residual_adapter_classifier import ResidualAdapterClassifier", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as Classifier",
            True,
        ),
    ],
)
def test_classifier_bounded_loss_evaluation_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/prediction/classifier_bounded_loss_evaluation.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from dataclasses import asdict", False),
        ("from torch import rand", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import AdamW", False),
        ("from .parameter_optimizer_state import _private", False),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", False),
        ("from .joint_model_parameter_update import perform_joint_model_parameter_update", False),
        ("from .model_training_sample_store import ModelTrainingSampleStore", False),
        ("from .held_model_training_binding import _private", False),
        ("from federated_learning_experiments.runtime import run", False),
        ("from federated_learning_experiments.configuration import config", False),
        ("import federated_drift_experiment", False),
        ("from . import parameter_optimizer_state", False),
        ("from dataclasses import dataclass", True),
        ("from torch import float32, strided", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Adam, SGD", True),
        ("from .parameter_optimizer_state import ParameterOptimizerState", True),
        ("from .held_model_training_binding import HeldModelTrainingBinding", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
    ],
)
def test_held_model_training_state_registry_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/held_model_training_state_registry.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text",
    [
        "import torch",
        "import torch.nn",
        "import torch.optim as optim",
        "import dataclasses",
        "import federated_learning_experiments.learning.training.parameter_optimizer_state as state",
        "from torch import nn",
        "from torch import optim",
    ],
)
def test_held_model_training_state_registry_rejects_module_imports(source_text):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/held_model_training_state_registry.py",
        source_text=source_text,
    )
    assert dependency_boundary_violations


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import numpy", False),
        ("from dataclasses import asdict", False),
        ("from torch import Tensor", False),
        ("from torch import rand", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import AdamW", False),
        ("from .joint_model_parameter_update import perform_joint_model_parameter_update", False),
        ("from .held_model_training_binding import HeldModelTrainingBinding", False),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", False),
        ("from .parameter_optimizer_state import _private", False),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _validate_classifier_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import _private",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("import federated_drift_experiment", False),
        ("from . import parameter_optimizer_state", False),
        ("from dataclasses import dataclass", False),
        ("from torch import float32, strided", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Adam, SGD", True),
        ("from .parameter_optimizer_state import ParameterOptimizerState", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
    ],
)
def test_adopted_candidate_shared_feature_integration_dependency_contract(
    source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/adopted_candidate_shared_feature_integration.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import numpy", False),
        ("from dataclasses import asdict", False),
        ("from torch import Tensor", False),
        ("from torch import rand", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import AdamW", False),
        ("from .joint_model_parameter_update import perform_joint_model_parameter_update", False),
        ("from .held_model_training_binding import HeldModelTrainingBinding", False),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", False),
        ("from .parameter_optimizer_state import _private", False),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _validate_classifier_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import _private",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("import federated_drift_experiment", False),
        ("from . import parameter_optimizer_state", False),
        ("from dataclasses import dataclass", True),
        ("from torch import float32, strided", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Adam, SGD", True),
        ("from .parameter_optimizer_state import ParameterOptimizerState", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
    ],
)
def test_held_model_shared_feature_reconnection_dependency_contract(
    source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/held_model_shared_feature_reconnection.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


def is_configuration_foundation_module(*, source_module_path):
    """設定基盤の実装と機能別の設定宣言を選ぶ。"""
    return source_module_path.startswith(("core/", "configuration/")) or (
        source_module_path.startswith(("learning/", "methods/"))
        and source_module_path.endswith("_settings.py")
    )


def resolve_imported_module_names(*, import_statement, importing_package_name):
    """fromの別名も展開し、package経由の禁止モジュールを見逃さない。"""
    if isinstance(import_statement, ast.Import):
        return tuple(imported_module_alias.name for imported_module_alias in import_statement.names)
    if not isinstance(import_statement, ast.ImportFrom):
        return ()
    relative_import_prefix = "." * import_statement.level
    imported_base_module_name = resolve_name(
        relative_import_prefix + (import_statement.module or ""),
        importing_package_name,
    )
    # package名自体は依存先の層を特定しないため、選択された属性まで確認する。
    if imported_base_module_name in (
        "federated_learning_experiments",
        "federated_learning_experiments.configuration",
        "federated_learning_experiments.core",
        "federated_learning_experiments.data",
        "federated_learning_experiments.data.concept_schedules",
        "federated_learning_experiments.data.sine",
        "federated_learning_experiments.execution",
        "federated_learning_experiments.learning",
        "federated_learning_experiments.learning.models",
        "federated_learning_experiments.runtime",
    ):
        return tuple(
            imported_base_module_name + "." + imported_module_alias.name
            for imported_module_alias in import_statement.names
        )
    return (imported_base_module_name,) + tuple(
        imported_base_module_name + "." + imported_module_alias.name
        for imported_module_alias in import_statement.names
    )


def dependency_is_allowed(*, source_module_path, imported_module_name):
    """各層の依存方向と数値ライブラリを参照できる場所を判定する。"""
    if source_module_path == "learning/training/temporary_model_id_allocation.py":
        return imported_module_name == "__future__.annotations"
    if source_module_path == "runtime/assigned_training_sample_absorption.py":
        return imported_module_name in (
            "__future__.annotations",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
        )
    if source_module_path == "runtime/adopted_candidate_local_adoption.py":
        return imported_module_name in (
            "__future__.annotations",
            "torch.Tensor",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.runtime.adopted_candidate_initial_local_registration.register_adopted_candidate_as_temporary_held_model",
        )
    if source_module_path == "runtime/adopted_candidate_initial_local_registration.py":
        return imported_module_name in (
            "__future__.annotations",
            "torch.Tensor",
            "federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization.initialize_model_and_class_loss_statistics_from_batch",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration.integrate_adopted_candidate_shared_features",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingState",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
        )
    if source_module_path == "runtime/held_model_registration_confirmation.py":
        return imported_module_name in (
            "__future__.annotations",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
        )
    if source_module_path in (
        "learning/training/model_training_and_assignment_counts.py",
        "learning/training/current_training_model_assignment.py",
    ):
        return imported_module_name in ("__future__.annotations", "dataclasses.dataclass")
    if source_module_path == "evaluation/model_evaluation_sample_records.py":
        return imported_module_name in (
            "__future__.annotations",
            "dataclasses.dataclass",
            "torch.Tensor",
        )
    if source_module_path == "evaluation/model_evaluation_sample_store.py":
        return imported_module_name in (
            "__future__.annotations",
            "random.Random",
            "federated_learning_experiments.evaluation.model_evaluation_sample_records.ObservedEvaluationSample",
            "federated_learning_experiments.evaluation.model_evaluation_sample_records.ModelEvaluationSampleCollection",
        )
    if source_module_path == "methods/fedsda/model_registration/pending_model_upload.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "torch.float32",
            "torch.isfinite",
            "torch.strided",
        )
    if source_module_path == "learning/models/classifier_parameter_snapshot.py":
        return imported_module_name in (
            "torch.Tensor",
            "torch.float32",
            "torch.isfinite",
            "torch.no_grad",
            "torch.strided",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/prediction/classifier_bounded_loss_evaluation.py":
        return imported_module_name in (
            "torch.Tensor",
            "torch.abs",
            "torch.float32",
            "torch.isfinite",
            "torch.no_grad",
            "torch.softmax",
            "torch.strided",
            "torch.trunc",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/held_model_training_state_registry.py":
        # 一覧構造と公開状態参照だけを許可し、上位処理の呼出しを防ぐ。
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.float32",
            "torch.strided",
            "torch.nn.Parameter",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.held_model_training_binding.HeldModelTrainingBinding",
        )
    if source_module_path == "learning/training/adopted_candidate_shared_feature_integration.py":
        # 外側接続はモデルとownerの公開型/属性だけを使い、学習計算へ依存しない。
        return imported_module_name in (
            "torch",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.parameter_optimizer_state",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
        )
    if source_module_path == "learning/training/held_model_shared_feature_reconnection.py":
        # 外側接続はモデルとownerの公開型/属性だけを使い、学習計算へ依存しない。
        return imported_module_name in (
            "dataclasses",
            "dataclasses.dataclass",
            "torch",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.parameter_optimizer_state",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
        )
    if source_module_path == "learning/training/parameter_optimizer_state.py":
        # 状態所有者は公開型・固定設定・既存生成関数だけへ依存する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.training.parameter_optimizer_construction",
            "federated_learning_experiments.learning.training.parameter_optimizer_construction.create_parameter_optimizer",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
        )
    if source_module_path == "learning/training/model_training_sample_store.py":
        # 構造保持は同階層の公開recordのみを参照し、演算や抽出を呼ばない。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "federated_learning_experiments.learning.training.model_training_sample_records",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_records.ModelTrainingSampleCollection",
        )
    if source_module_path == "learning/training/local_training_schedule_settings.py":
        # 機能別の設定宣言と公開値検査だけを許可する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.settings_field_validation",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/training/local_training_request_schedule.py":
        # 要求counterは自分の設定だけを参照し、学習実体や上位をimportしない。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "federated_learning_experiments.learning.training.local_training_schedule_settings",
            "federated_learning_experiments.learning.training.local_training_schedule_settings.LocalTrainingScheduleSettings",
        )
    if source_module_path == "learning/training/held_model_training_binding.py":
        # 借用参照と公開部品の接続だけに限定し、演算・上位層・private依存を拒否する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/held_model_joint_training_iterations.py":
        # 借用参照と公開部品の接続だけに限定し、演算・上位層・private依存を拒否する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "random",
            "random.Random",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.held_model_training_binding",
            "federated_learning_experiments.learning.training.held_model_training_binding.HeldModelTrainingBinding",
            "federated_learning_experiments.learning.training.model_training_sample_records",
            "federated_learning_experiments.learning.training.model_training_sample_records.ModelTrainingSampleCollection",
            "federated_learning_experiments.learning.training.participating_model_training_batch",
            "federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch",
            "federated_learning_experiments.learning.training.local_training_settings",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.held_model_training_batch_sampling",
            "federated_learning_experiments.learning.training.held_model_training_batch_sampling.sample_training_batches_for_held_models",
            "federated_learning_experiments.learning.training.joint_model_parameter_update",
            "federated_learning_experiments.learning.training.joint_model_parameter_update.perform_joint_model_parameter_update",
        )
    if source_module_path == "learning/training/model_training_sample_records.py":
        # 標本記録はdataclass宣言と借用Tensor型だけを参照する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "torch",
            "torch.Tensor",
        )
    if source_module_path == "learning/training/held_model_training_batch_sampling.py":
        # 抽出は借用Randomと明示Tensor演算・同feature公開記録だけに限定する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "random",
            "random.Random",
            "torch",
            "torch.Tensor",
            "torch.cat",
            "torch.isfinite",
            "torch.float32",
            "torch.strided",
            "federated_learning_experiments.learning.training.model_training_sample_records",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_records.ModelTrainingSampleCollection",
            "federated_learning_experiments.learning.training.model_training_sample_records.SampledModelTrainingBatch",
        )
    if source_module_path == "learning/training/participating_model_training_batch.py":
        # 借用記録は宣言に必要な公開型だけを参照する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "torch",
            "torch.Tensor",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/joint_model_parameter_update.py":
        # 共同更新は明示演算と兄弟記録/設定・モデル公開型へ限定する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch",
            "torch.Tensor",
            "torch.cat",
            "torch.isfinite",
            "torch.no_grad",
            "torch.is_grad_enabled",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.nn.BCELoss",
            "torch.nn.CrossEntropyLoss",
            "torch.optim",
            "torch.optim.Optimizer",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.training.local_training_settings",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.participating_model_training_batch",
            "federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/parameter_optimizer_settings.py":
        # optimizer設定は一般stdlib/core許可より前に公開field検査へ限定する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.settings_field_validation",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/training/parameter_optimizer_construction.py":
        # 生成部は指定Parameter/optimizerと専用公開設定型だけを参照する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Optimizer",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
        )
    if source_module_path in (
        "learning/models/shared_feature_extractor.py",
        "learning/models/nonlinear_residual_adapter.py",
        "learning/models/residual_adapter_classifier.py",
    ):
        # モデル構造は一般stdlib許可より先にexact module/public symbolで制限する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch",
            "torch.Tensor",
            "torch.float32",
            "torch.device",
            "torch.strided",
            "torch.nn",
            "torch.nn.Module",
            "torch.nn.Sequential",
            "torch.nn.Linear",
            "torch.nn.ReLU",
            "torch.nn.Sigmoid",
            "torch.nn.Identity",
            "torch.nn.init",
            "torch.nn.init.zeros_",
        ) or (
            source_module_path == "learning/models/residual_adapter_classifier.py"
            and imported_module_name
            in (
                "federated_learning_experiments.learning.models.model_architecture_settings",
                "federated_learning_experiments.learning.models.model_architecture_settings.ModelArchitectureSettings",
                "federated_learning_experiments.learning.models.shared_feature_extractor",
                "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
                "federated_learning_experiments.learning.models.nonlinear_residual_adapter",
                "federated_learning_experiments.learning.models.nonlinear_residual_adapter.NonlinearResidualAdapter",
            )
        )
    if imported_module_name.split(".")[0] in sys.stdlib_module_names:
        return True
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py"
    ):
        # 初期snapshot作成だけにtorchと同機能の公開設定型を許可する。
        return imported_module_name in (
            "torch",
            "torch.Tensor",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
        )
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py"
    ):
        # 設定宣言は既存の公開field検査関数だけを参照する。
        return imported_module_name in (
            "federated_learning_experiments.core.settings_field_validation",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/loss_statistics/server_loss_mean_aggregation.py":
        # サーバ用途の平均集約は公開集計値型だけに依存する。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
        )
    if (
        source_module_path
        == "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py"
    ):
        # ID対応後の選択は公開統計型だけを参照する。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
        )
    if source_module_path == "learning/loss_statistics/batch_loss_statistics_initialization.py":
        # batch初期化だけにtorchと既存の集計値型を許可する。
        return imported_module_name in ("torch", "torch.Tensor") or imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
        )
    if source_module_path == "learning/loss_statistics/model_and_class_loss_statistics.py":
        # 統計所有は公開集計型と一件追加だけに依存する。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.accumulate_bounded_loss_observation",
        )
    if source_module_path == "methods/fedsda/loss_statistics/loss_baseline_selection.py":
        # 基準値方針は公開集計型だけを参照し、他の数値処理へ依存しない。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
        )
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py"
    ):
        # 収集状態は同機能設定だけを参照し、数値評価へ依存しない。
        return imported_module_name in (
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
        )
    if (
        source_module_path
        == "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py"
    ):
        # 保留位置FIFOは同機能の容量条件だけを参照する。
        return (
            imported_module_name
            == "federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings"
            or imported_module_name.startswith(
                "federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings."
            )
        )
    if source_module_path == "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py":
        # 単系列の数値検出器だけにNumPyを許可する。
        return imported_module_name == "numpy" or imported_module_name.startswith("numpy.")
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py"
    ):
        # 候補loss評価だけにtorchと同機能の固定条件を許可する。
        return (
            imported_module_name == "torch"
            or imported_module_name.startswith("torch.")
            or imported_module_name
            == "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings"
            or imported_module_name.startswith(
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings."
            )
        )
    if source_module_path in (
        "learning/models/torch_random_state_scope.py",
        "learning/prediction/class_probability_calculations.py",
    ):
        # torchを必要とする責務のexact moduleだけを例外にする。
        return imported_module_name == "torch" or imported_module_name.startswith("torch.")
    if imported_module_name == "numpy" or imported_module_name.startswith("numpy."):
        return (
            source_module_path.startswith("data/")
            and not source_module_path.endswith("_settings.py")
        ) or source_module_path == "execution/run_random_sources.py"
    if not imported_module_name.startswith("federated_learning_experiments."):
        return False
    source_module_layer = source_module_path.split("/")[0]
    configuration_foundation_module = is_configuration_foundation_module(
        source_module_path=source_module_path,
    )
    if configuration_foundation_module:
        if source_module_layer == "configuration" and source_module_path != (
            "configuration/experiment_run_conditions.py"
        ):
            allowed_internal_module_prefixes = (
                "federated_learning_experiments.core.",
                "federated_learning_experiments.learning.",
                "federated_learning_experiments.methods.",
                "federated_learning_experiments.configuration.experiment_run_conditions.",
            )
            if source_module_path == "configuration/run_settings.py":
                allowed_internal_module_prefixes += (
                    "federated_learning_experiments.configuration.run_settings_validation.",
                )
        else:
            allowed_internal_module_prefixes = ("federated_learning_experiments.core.",)
    elif source_module_path.endswith("_settings.py"):
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.core.",
            "federated_learning_experiments.configuration.experiment_run_conditions.",
        )
        if source_module_layer == "execution":
            allowed_internal_module_prefixes += (
                "federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings.",
            )
    elif (
        source_module_path
        == "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py"
    ):
        # 数値状態部品は同機能の固定条件だけを参照する。
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings.",
        )
    elif (
        source_module_path
        == "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py"
    ):
        # 混合監視は同機能の検出器と固定条件だけを参照する。
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection.",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings.",
        )
    elif source_module_layer == "data":
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.data.",
            "federated_learning_experiments.configuration.experiment_run_conditions.",
        )
    elif source_module_layer == "execution":
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.execution.",
            "federated_learning_experiments.data.",
        )
        if source_module_path == "execution/run_participant_contracts.py":
            allowed_internal_module_prefixes += (
                "federated_learning_experiments.configuration.experiment_run_conditions.",
            )
    elif source_module_layer == "runtime":
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.core.",
            "federated_learning_experiments.configuration.",
            "federated_learning_experiments.data.",
            "federated_learning_experiments.execution.",
            "federated_learning_experiments.learning.models.torch_random_state_scope.",
        )
    else:
        allowed_internal_module_prefixes = ()
    return any(
        imported_module_name == imported_base_module_name.rstrip(".")
        or imported_module_name.startswith(imported_base_module_name)
        for imported_base_module_name in allowed_internal_module_prefixes
    )


def collect_dependency_boundary_violations(*, source_module_path, source_text):
    """実際のソースと注入用ソースに同じAST検査を適用する。"""
    importing_package_name = "federated_learning_experiments"
    if "/" in source_module_path:
        importing_package_name += "." + source_module_path.rsplit("/", 1)[0].replace("/", ".")
    parsed_source_module = ast.parse(source_text)
    dependency_boundary_violations = ()
    for import_statement in ast.walk(parsed_source_module):
        imported_module_names = resolve_imported_module_names(
            import_statement=import_statement,
            importing_package_name=importing_package_name,
        )
        if source_module_path in (
            "learning/training/held_model_training_state_registry.py",
            "learning/prediction/classifier_bounded_loss_evaluation.py",
            "learning/models/classifier_parameter_snapshot.py",
            "methods/fedsda/model_registration/pending_model_upload.py",
            "evaluation/model_evaluation_sample_records.py",
            "evaluation/model_evaluation_sample_store.py",
            "learning/training/model_training_and_assignment_counts.py",
            "learning/training/current_training_model_assignment.py",
            "runtime/held_model_registration_confirmation.py",
            "runtime/adopted_candidate_initial_local_registration.py",
            "learning/training/temporary_model_id_allocation.py",
            "runtime/adopted_candidate_local_adoption.py",
            "runtime/assigned_training_sample_absorption.py",
        ) and isinstance(import_statement, ast.ImportFrom):
            # 通常resolverのpackage別返却差に依存せず、束縛symbolを直接解決する。
            imported_module_names = tuple(
                resolve_name(
                    "." * import_statement.level + (import_statement.module or ""),
                    importing_package_name,
                )
                + "."
                + imported_module_alias.name
                for imported_module_alias in import_statement.names
            )
        for imported_module_name in imported_module_names:
            if (
                source_module_path
                in (
                    "learning/training/held_model_training_state_registry.py",
                    "learning/prediction/classifier_bounded_loss_evaluation.py",
                    "learning/models/classifier_parameter_snapshot.py",
                    "methods/fedsda/model_registration/pending_model_upload.py",
                    "evaluation/model_evaluation_sample_records.py",
                    "evaluation/model_evaluation_sample_store.py",
                    "learning/training/model_training_and_assignment_counts.py",
                    "learning/training/current_training_model_assignment.py",
                    "runtime/held_model_registration_confirmation.py",
                    "runtime/adopted_candidate_initial_local_registration.py",
                    "learning/training/temporary_model_id_allocation.py",
                    "runtime/adopted_candidate_local_adoption.py",
                    "runtime/assigned_training_sample_absorption.py",
                )
                and isinstance(import_statement, ast.Import)
            ) or not dependency_is_allowed(
                source_module_path=source_module_path,
                imported_module_name=imported_module_name,
            ):
                dependency_boundary_violations += (
                    (
                        imported_module_name,
                        "この層では許可されない依存先です",
                    ),
                )
    return dependency_boundary_violations


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("from torch import Tensor", False),
        ("import numpy", False),
        ("import math", False),
        ("import random", False),
        ("from random import Random", False),
        ("from dataclasses import dataclass", False),
        ("import federated_drift_experiment", False),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _validate_model_id",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment, TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from ..methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState as UploadState",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_held_model_registration_confirmation_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/held_model_registration_confirmation.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import float32",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "from torch.nn import Parameter",
            False,
        ),
        (
            "from torch.optim import Adam",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_post_alarm_candidate_losses",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import _validate_adopted_candidate_integration_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _validate_classifier_bounded_loss_inputs",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "from torch import Tensor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import integrate_adopted_candidate_shared_features",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from ..methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState as UploadState",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_adopted_candidate_initial_local_registration_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/adopted_candidate_initial_local_registration.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "import random as random_module",
            False,
        ),
        (
            "import dataclasses",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "from dataclasses import dataclass as Record",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "import torch",
            False,
        ),
        (
            "from torch import Tensor",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from __future__ import division",
            False,
        ),
        (
            "from __future__ import annotations, division",
            False,
        ),
        (
            "from __future__ import *",
            False,
        ),
        (
            "from . import current_training_model_assignment",
            False,
        ),
        (
            "from .current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from .held_model_training_state_registry import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        ("from __future__ import annotations", True),
        ("from __future__ import annotations as postponed_annotations", True),
    ],
)
def test_temporary_model_id_allocation_rejects_every_import_except_annotations(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="learning/training/temporary_model_id_allocation.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import float32",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import integrate_adopted_candidate_shared_features",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ModelTrainingSampleCollection",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import _validate_registration_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import _select_active_shared_feature_extractor",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.adopted_candidate_initial_local_registration",
            False,
        ),
        (
            "from federated_learning_experiments.runtime import adopted_candidate_initial_local_registration",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_post_alarm_candidate_losses",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import TemporaryModelIdAllocator",
            False,
        ),
        (
            "from torch import Tensor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            True,
        ),
        (
            "from .adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model as register",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_adopted_candidate_local_adoption_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/adopted_candidate_local_adoption.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import Tensor",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ModelTrainingSampleCollection",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import accumulate_bounded_loss_observation",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _validate_classifier_bounded_loss_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_local_adoption import adopt_candidate_as_current_training_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import ModelTrainingSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from ..learning.training.model_training_sample_store import ModelTrainingSampleStore as SampleStore",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_assigned_training_sample_absorption_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/assigned_training_sample_absorption.py",
            source_text=source_text,
        )
    ) == expected_acceptance


def test_single_run_package_boundaries_have_no_exports():
    """新しい境界は存在し、説明だけを持ち再exportしない。"""
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    package_boundary_paths = (
        "data",
        "data/concept_schedules",
        "data/sine",
        "execution",
        "runtime",
        "learning/prediction",
        "evaluation",
    )
    for package_boundary_path in package_boundary_paths:
        source_file_path = package_source_directory / package_boundary_path / "__init__.py"
        assert source_file_path.is_file(), package_boundary_path
        parsed_source_module = ast.parse(source_file_path.read_text(encoding="utf-8"))
        assert ast.get_docstring(parsed_source_module), package_boundary_path
        assert len(parsed_source_module.body) == 1, package_boundary_path


def test_single_run_layers_import_only_allowed_dependencies():
    """後続で追加されるモジュールも毎回全走査して境界を検査する。"""
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    for source_file_path in sorted(package_source_directory.rglob("*.py")):
        source_module_path = source_file_path.relative_to(package_source_directory).as_posix()
        dependency_boundary_violations = collect_dependency_boundary_violations(
            source_module_path=source_module_path,
            source_text=source_file_path.read_text(encoding="utf-8"),
        )
        assert not dependency_boundary_violations, (
            source_module_path,
            dependency_boundary_violations,
        )


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_imported_module_name",
    [
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/training/model_training_sample_records.py",
                "learning/training/held_model_training_batch_sampling.py",
            )
            for source_text, expected_imported_module_name in (
                ("import math", "math"),
                ("import numpy", "numpy"),
                ("import torch._C", "torch._C"),
                ("import torch.optim", "torch.optim"),
                ("from torch.nn import Module", "torch.nn"),
                (
                    "import federated_drift_experiment.clients.base",
                    "federated_drift_experiment.clients.base",
                ),
                (
                    "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
                    "federated_learning_experiments.learning.models.residual_adapter_classifier",
                ),
                (
                    "import federated_learning_experiments.runtime.single_run_execution",
                    "federated_learning_experiments.runtime.single_run_execution",
                ),
                (
                    "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
                    "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
                ),
                (
                    "from .joint_model_parameter_update import perform_joint_model_parameter_update",
                    "federated_learning_experiments.learning.training.joint_model_parameter_update",
                ),
            )
        ],
        (
            "learning/training/model_training_sample_records.py",
            "from dataclasses import field",
            "dataclasses.field",
        ),
        (
            "learning/training/model_training_sample_records.py",
            "from torch import cat",
            "torch.cat",
        ),
        (
            "learning/training/model_training_sample_records.py",
            "from random import Random",
            "random",
        ),
        (
            "learning/training/model_training_sample_records.py",
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            "federated_learning_experiments.learning.training.held_model_training_batch_sampling",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import sample",
            "random.sample",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import seed",
            "random.seed",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import getstate",
            "random.getstate",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import setstate",
            "random.setstate",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import SystemRandom",
            "random.SystemRandom",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import _inst",
            "random._inst",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "import random.child",
            "random.child",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from .model_training_sample_records import Tensor",
            "federated_learning_experiments.learning.training.model_training_sample_records.Tensor",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from .model_training_sample_records import _private",
            "federated_learning_experiments.learning.training.model_training_sample_records._private",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records.child",
            "federated_learning_experiments.learning.training.model_training_sample_records.child",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/training/participating_model_training_batch.py",
                "learning/training/joint_model_parameter_update.py",
            )
            for source_text, expected_imported_module_name in (
                ("import math", "math"),
                ("import random", "random"),
                ("import numpy", "numpy"),
                ("import torch._C", "torch._C"),
                ("import torch.optim.lr_scheduler", "torch.optim.lr_scheduler"),
                ("from torch.optim import RMSprop", "torch.optim.RMSprop"),
                ("import federated_drift_experiment.models", "federated_drift_experiment.models"),
                (
                    "import federated_learning_experiments.runtime.single_run_execution",
                    "federated_learning_experiments.runtime.single_run_execution",
                ),
                (
                    "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                    "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                ),
                (
                    "from .parameter_optimizer_construction import create_parameter_optimizer",
                    "federated_learning_experiments.learning.training.parameter_optimizer_construction",
                ),
                (
                    "import federated_learning_experiments.learning.training",
                    "federated_learning_experiments.learning.training",
                ),
            )
        ],
        (
            "learning/training/participating_model_training_batch.py",
            "from dataclasses import field",
            "dataclasses.field",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch import cat",
            "torch.cat",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch.optim import Adam",
            "torch.optim.Adam",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch.nn import Parameter",
            "torch.nn.Parameter",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from .local_training_settings import LocalTrainingSettings",
            "federated_learning_experiments.learning.training.local_training_settings",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from ..models.shared_feature_extractor import SharedFeatureExtractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from ..models.residual_adapter_classifier import _validate_classifier_inputs",
            "federated_learning_experiments.learning.models.residual_adapter_classifier._validate_classifier_inputs",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .participating_model_training_batch import Tensor",
            "federated_learning_experiments.learning.training.participating_model_training_batch.Tensor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .local_training_settings import validate_settings_field_values",
            "federated_learning_experiments.learning.training.local_training_settings.validate_settings_field_values",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from ..models.shared_feature_extractor import _validate_feature_tensor",
            "federated_learning_experiments.learning.models.shared_feature_extractor._validate_feature_tensor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "import federated_learning_experiments.learning.training.participating_model_training_batch.child",
            "federated_learning_experiments.learning.training.participating_model_training_batch.child",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "import federated_learning_experiments.learning.models.residual_adapter_classifier.child",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.child",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/training/parameter_optimizer_settings.py",
                "learning/training/parameter_optimizer_construction.py",
            )
            for source_text, expected_imported_module_name in (
                ("import math", "math"),
                ("import random", "random"),
                ("import numpy", "numpy"),
                ("import torch._C", "torch._C"),
                ("import federated_drift_experiment.models", "federated_drift_experiment.models"),
                (
                    "from ..models import residual_adapter_classifier",
                    "federated_learning_experiments.learning.models.residual_adapter_classifier",
                ),
                (
                    "import federated_learning_experiments.data.sine.sine_sample_generation",
                    "federated_learning_experiments.data.sine.sine_sample_generation",
                ),
                (
                    "import federated_learning_experiments.runtime.single_run_execution",
                    "federated_learning_experiments.runtime.single_run_execution",
                ),
                (
                    "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                    "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                ),
            )
        ],
        ("learning/training/parameter_optimizer_settings.py", "import torch", "torch"),
        ("learning/training/parameter_optimizer_settings.py", "import torch.optim", "torch.optim"),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from dataclasses import asdict",
            "dataclasses.asdict",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.settings_field_validation import _private",
            "federated_learning_experiments.core.settings_field_validation._private",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.settings_field_validation import fields",
            "federated_learning_experiments.core.settings_field_validation.fields",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "import federated_learning_experiments.core.settings_field_validation.child",
            "federated_learning_experiments.core.settings_field_validation.child",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.configuration_errors import RunSettingsValidationError",
            "federated_learning_experiments.core.configuration_errors",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch.optim import RMSprop",
            "torch.optim.RMSprop",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "import torch.optim.lr_scheduler",
            "torch.optim.lr_scheduler",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "import torch.optim.optimizer",
            "torch.optim.optimizer",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch.nn import Module",
            "torch.nn.Module",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from .parameter_optimizer_settings import _private",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings._private",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from .parameter_optimizer_settings import validate_settings_field_values",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.validate_settings_field_values",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.child",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.child",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/models/shared_feature_extractor.py",
                "learning/models/nonlinear_residual_adapter.py",
                "learning/models/residual_adapter_classifier.py",
            )
            for source_text, expected_imported_module_name in (
                ("import numpy", "numpy"),
                ("import random", "random"),
                ("import torch.optim", "torch.optim"),
                ("import torch._C", "torch._C"),
                ("from torch.nn import Dropout", "torch.nn.Dropout"),
                ("from torch.nn.init import kaiming_uniform_", "torch.nn.init.kaiming_uniform_"),
                ("import federated_drift_experiment.models", "federated_drift_experiment.models"),
            )
        ],
        (
            "learning/models/shared_feature_extractor.py",
            "from .residual_adapter_classifier import ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
        ),
        (
            "learning/models/nonlinear_residual_adapter.py",
            "from .shared_feature_extractor import SharedFeatureExtractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .shared_feature_extractor import _validate_feature_tensor",
            "federated_learning_experiments.learning.models.shared_feature_extractor._validate_feature_tensor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .model_architecture_settings import validate_settings_field_values",
            "federated_learning_experiments.learning.models.model_architecture_settings.validate_settings_field_values",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "import federated_learning_experiments.learning.models.shared_feature_extractor.child",
            "federated_learning_experiments.learning.models.shared_feature_extractor.child",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "import federated_learning_experiments.runtime.single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from ...methods.fedsda.candidate_model_selection import candidate_parameter_initialization",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import torch._C",
            "torch._C",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from torch import nn",
            "torch.nn",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .candidate_parameter_initialization_settings import _private",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings._private",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .candidate_parameter_initialization_settings import validate_settings_field_values",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.validate_settings_field_values",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.child",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.child",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import federated_drift_experiment.config",
            "federated_drift_experiment.config",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import federated_learning_experiments.runtime.single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import fields",
            "federated_learning_experiments.core.settings_field_validation.fields",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import _private",
            "federated_learning_experiments.core.settings_field_validation._private",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core import configuration_errors",
            "federated_learning_experiments.core.configuration_errors",
        ),
        ("learning/loss_statistics/server_loss_mean_aggregation.py", "import torch", "torch"),
        ("learning/loss_statistics/server_loss_mean_aggregation.py", "import numpy", "numpy"),
        ("learning/loss_statistics/server_loss_mean_aggregation.py", "import config", "config"),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import accumulate_bounded_loss_observation",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.accumulate_bounded_loss_observation",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import torch",
            "torch",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import numpy",
            "numpy",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import config",
            "config",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import _copy_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics._copy_loss_moments",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .bounded_loss_moments import BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import arbitrary_public_name",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.arbitrary_public_name",
        ),
        ("learning/prediction/class_probability_calculations.py", "import numpy", "numpy"),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.internal",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.internal",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from .candidate_model_training_and_acceptance_settings import UnsupportedSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.UnsupportedSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import config",
            "config",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from federated_drift_experiment import provisional_model",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from . import post_alarm_candidate_loss_evaluation",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from ..training_data_assignment import pending_training_assignment_buffer",
            "federated_learning_experiments.methods.fedsda.training_data_assignment",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from . import another_loss_collection",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "import config",
            "config",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from federated_drift_experiment import clients",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from ..loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from . import another_assignment_buffer",
            "federated_learning_experiments.methods.fedsda.training_data_assignment",
        ),
        ("learning/prediction/class_probability_calculations.py", "import config", "config"),
        (
            "learning/prediction/class_probability_calculations.py",
            "from federated_drift_experiment import expert_routing",
            "federated_drift_experiment",
        ),
        (
            "learning/prediction/class_probability_calculations.py",
            "from ... import runtime",
            "federated_learning_experiments.runtime",
        ),
        (
            "learning/prediction/class_probability_calculations.py",
            "from ...methods.fedsda.prediction_combination.fixed_share_prediction_weights import FixedSharePredictionWeightController",
            "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights",
        ),
        ("learning/prediction/another_prediction.py", "import torch", "torch"),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "import config",
            "config",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from federated_drift_experiment import provisional_model",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from ..loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from ....learning.prediction import class_probability_calculations",
            "federated_learning_experiments.learning.prediction",
        ),
        ("methods/fedsda/candidate_model_selection/another_candidate.py", "import torch", "torch"),
        (
            "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py",
            "from .loss_change_detection_settings import LossChangeDetectionSettings",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from ..prediction_combination import fixed_share_prediction_weights",
            "federated_learning_experiments.methods.fedsda.prediction_combination",
        ),
        ("methods/fedsda/loss_change_detection/another_detector.py", "import numpy", "numpy"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import numpy", "numpy"),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import numpy",
            "numpy",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import torch._C",
            "torch._C",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import config",
            "config",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import accumulate_bounded_loss_observation",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.accumulate_bounded_loss_observation",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .model_and_class_loss_statistics import _copy_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics._copy_loss_moments",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        ("learning/loss_statistics/model_and_class_loss_statistics.py", "import numpy", "numpy"),
        ("learning/loss_statistics/model_and_class_loss_statistics.py", "import torch", "torch"),
        ("learning/loss_statistics/model_and_class_loss_statistics.py", "import config", "config"),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from ...methods.fedsda.loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from .bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from .bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        ("learning/loss_statistics/bounded_loss_moments.py", "import torch", "torch"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import config", "config"),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ...methods.fedsda.loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ...methods.fedsda.candidate_model_selection import post_alarm_candidate_loss_evaluation",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ..prediction import class_probability_calculations",
            "federated_learning_experiments.learning.prediction",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        ("methods/fedsda/loss_statistics/loss_baseline_selection.py", "import numpy", "numpy"),
        ("methods/fedsda/loss_statistics/loss_baseline_selection.py", "import torch", "torch"),
        ("methods/fedsda/loss_statistics/loss_baseline_selection.py", "import config", "config"),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ..loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ..candidate_model_selection import post_alarm_candidate_loss_evaluation",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.another_module",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.another_module",
        ),
        ("data/sine/sine_sample_generation.py", "import torch", "torch"),
        ("data/observed_streams.py", "import config", "config"),
        ("data/observed_streams.py", "from clients import fedsda", "clients"),
        (
            "data/observed_streams.py",
            "from federated_learning_experiments import runtime",
            "federated_learning_experiments.runtime",
        ),
        (
            "data/sine/sine_sample_generation.py",
            "from ... import runtime",
            "federated_learning_experiments.runtime",
        ),
        (
            "data/sine/sine_sample_generation.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "data/observed_streams.py",
            "from federated_learning_experiments.learning.models import torch_random_state_scope",
            "federated_learning_experiments.learning.models.torch_random_state_scope",
        ),
        (
            "data/observed_streams.py",
            "from federated_learning_experiments.configuration import run_settings",
            "federated_learning_experiments.configuration.run_settings",
        ),
        ("data/concept_schedules/random_concept_schedule_settings.py", "import numpy", "numpy"),
        (
            "data/concept_schedules/random_concept_schedule_settings.py",
            "from .. import observed_streams",
            "federated_learning_experiments.data.observed_streams",
        ),
        ("execution/stream_protocol_execution_settings.py", "import numpy", "numpy"),
        (
            "execution/stream_protocol_execution_settings.py",
            "from federated_learning_experiments.data.sine import sine_sample_generation",
            "federated_learning_experiments.data.sine.sine_sample_generation",
        ),
        ("execution/run_random_sources.py", "import torch", "torch"),
        ("execution/run_participant_contracts.py", "import numpy", "numpy"),
        (
            "execution/stream_protocol_execution_loop.py",
            "from ..runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "execution/stream_protocol_execution_loop.py",
            "from federated_learning_experiments import methods",
            "federated_learning_experiments.methods",
        ),
        (
            "execution/run_execution_records.py",
            "from .. import learning",
            "federated_learning_experiments.learning",
        ),
        (
            "execution/stream_protocol_execution_loop.py",
            "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
            "federated_learning_experiments.configuration.experiment_run_conditions",
        ),
        ("runtime/single_run_execution.py", "import torch", "torch"),
        ("runtime/single_run_execution.py", "import numpy", "numpy"),
        (
            "runtime/single_run_execution.py",
            "from ..learning.models import model_architecture_settings",
            "federated_learning_experiments.learning.models.model_architecture_settings",
        ),
        ("learning/models/torch_random_state_scope.py", "import numpy", "numpy"),
        (
            "learning/models/torch_random_state_scope.py",
            "from ... import runtime",
            "federated_learning_experiments.runtime",
        ),
        ("learning/models/another_model.py", "import torch", "torch"),
        ("core/configuration_errors.py", "import numpy", "numpy"),
        ("configuration/experiment_run_conditions.py", "import numpy", "numpy"),
        ("learning/models/model_architecture_settings.py", "import torch", "torch"),
        ("methods/fedsda/consolidation/model_consolidation_settings.py", "import numpy", "numpy"),
        ("data/observed_streams.py", "if False:\n    import torch", "torch"),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from federated_drift_experiment import expert_routing",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from federated_learning_experiments.runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from federated_learning_experiments.methods.fedsda.consolidation import model_consolidation_settings",
            "federated_learning_experiments.methods.fedsda.consolidation",
        ),
    ],
)
def test_single_run_dependency_checker_rejects_forbidden_imports(
    source_module_path,
    source_text,
    expected_imported_module_name,
):
    """productionを変更せず、実際の検査に禁止依存を注入する。"""
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path,
        source_text=source_text,
    )
    assert any(
        imported_module_name == expected_imported_module_name
        for imported_module_name, dependency_boundary_violation_reason in dependency_boundary_violations
    ), dependency_boundary_violations


@pytest.mark.parametrize(
    "source_module_path,source_text",
    [
        ("learning/training/model_training_sample_records.py", "from dataclasses import dataclass"),
        ("learning/training/model_training_sample_records.py", "from torch import Tensor"),
        (
            "learning/training/model_training_sample_records.py",
            "from __future__ import annotations",
        ),
        ("learning/training/held_model_training_batch_sampling.py", "from random import Random"),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from torch import Tensor, cat, isfinite, float32, strided",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from .model_training_sample_records import ObservedTrainingSample, ModelTrainingSampleCollection, SampledModelTrainingBatch",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample, ModelTrainingSampleCollection, SampledModelTrainingBatch",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from __future__ import annotations",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from dataclasses import dataclass",
        ),
        ("learning/training/participating_model_training_batch.py", "from torch import Tensor"),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch.optim import Optimizer",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from __future__ import annotations",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from torch import Tensor, cat, isfinite, no_grad, is_grad_enabled, float32, strided",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from torch.nn import Parameter, BCELoss, CrossEntropyLoss",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from torch.optim import Optimizer, Adam, SGD",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .local_training_settings import LocalTrainingSettings",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .participating_model_training_batch import ParticipatingModelTrainingBatch",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from ..models.shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.training.local_training_settings import LocalTrainingSettings",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.training.participating_model_training_batch import ParticipatingModelTrainingBatch",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        ("learning/training/joint_model_parameter_update.py", "from __future__ import annotations"),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from dataclasses import dataclass, field",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.settings_field_validation import validate_settings_field_values",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values",
        ),
        ("learning/training/parameter_optimizer_settings.py", "from __future__ import annotations"),
        ("learning/training/parameter_optimizer_construction.py", "import torch"),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch import float32, strided",
        ),
        ("learning/training/parameter_optimizer_construction.py", "from torch.nn import Parameter"),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch.optim import Optimizer, Adam, SGD",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from .parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from __future__ import annotations",
        ),
        *[
            (source_module_path, source_text)
            for source_module_path in (
                "learning/models/shared_feature_extractor.py",
                "learning/models/nonlinear_residual_adapter.py",
                "learning/models/residual_adapter_classifier.py",
            )
            for source_text in (
                "import torch",
                "from torch import Tensor, float32, device, strided",
                "from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid, Identity",
                "from torch.nn.init import zeros_",
                "from __future__ import annotations",
            )
        ],
        (
            "learning/models/residual_adapter_classifier.py",
            "from .model_architecture_settings import ModelArchitectureSettings",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .nonlinear_residual_adapter import NonlinearResidualAdapter",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from federated_learning_experiments.learning.models.model_architecture_settings import ModelArchitectureSettings",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from federated_learning_experiments.learning.models.nonlinear_residual_adapter import NonlinearResidualAdapter",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import math",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from torch import Tensor",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from dataclasses import dataclass, field",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from dataclasses import dataclass",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from collections import defaultdict",
        ),
        ("learning/loss_statistics/bounded_loss_moments.py", "from dataclasses import dataclass"),
        ("learning/loss_statistics/batch_loss_statistics_initialization.py", "import torch"),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from torch import Tensor",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from dataclasses import dataclass",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from dataclasses import dataclass",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments, accumulate_bounded_loss_observation",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from .bounded_loss_moments import BoundedLossMoments, accumulate_bounded_loss_observation",
        ),
        ("learning/loss_statistics/bounded_loss_moments.py", "import math"),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ....learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        ("learning/prediction/class_probability_calculations.py", "import torch"),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from .candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from dataclasses import dataclass",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from .training_data_assignment_settings import TrainingDataAssignmentSettings",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from collections import deque",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "import torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from .candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
        ),
        ("learning/prediction/class_probability_calculations.py", "from torch import Tensor"),
        (
            "learning/prediction/class_probability_calculations.py",
            "from collections.abc import Mapping",
        ),
        ("methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py", "import numpy"),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from .bounded_loss_e_sr_detection import BoundedLossESRDetector",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from .loss_change_detection_settings import LossChangeDetectionSettings",
        ),
        ("data/observed_streams.py", "from dataclasses import dataclass"),
        ("data/sine/sine_sample_generation.py", "import numpy as np"),
        ("data/sine/sine_sample_generation.py", "from ..observed_streams import ObservedSample"),
        ("data/sine/sine_sample_generation.py", "from .. import observed_streams"),
        (
            "data/concept_schedules/random_concept_schedule_generation.py",
            "from ...configuration.experiment_run_conditions import ExperimentRunConditions",
        ),
        (
            "data/concept_schedules/random_concept_schedule_settings.py",
            "from ...core.settings_field_validation import validate_settings_field_values",
        ),
        (
            "execution/stream_protocol_execution_settings.py",
            "from ..data.concept_schedules.random_concept_schedule_settings import RandomConceptScheduleSettings",
        ),
        (
            "execution/stream_protocol_execution_settings.py",
            "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
        ),
        ("execution/run_random_sources.py", "from numpy.random import RandomState"),
        (
            "execution/run_participant_contracts.py",
            "from .run_random_sources import RunRandomSources",
        ),
        (
            "execution/run_participant_contracts.py",
            "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
        ),
        ("execution/stream_protocol_execution_loop.py", "from . import run_participant_contracts"),
        (
            "execution/stream_protocol_execution_loop.py",
            "from ..data.observed_streams import ClientObservedStream",
        ),
        ("learning/models/torch_random_state_scope.py", "import torch"),
        ("learning/models/torch_random_state_scope.py", "from torch import random"),
        (
            "runtime/single_run_execution.py",
            "from ..learning.models.torch_random_state_scope import isolated_cpu_torch_random_state",
        ),
        (
            "runtime/single_run_execution.py",
            "from ..learning.models import torch_random_state_scope",
        ),
        ("runtime/single_run_execution.py", "from ..execution import run_random_sources"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "import math"),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from .prediction_combination_settings import PredictionCombinationSettings",
        ),
    ],
)
def test_single_run_dependency_checker_accepts_allowed_imports(source_module_path, source_text):
    """許可された型依存・NumPy例外・専用torch境界を拒否しない。"""
    assert (
        collect_dependency_boundary_violations(
            source_module_path=source_module_path,
            source_text=source_text,
        )
        == ()
    )


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        ("learning/training/held_model_training_binding.py", "import math", False),
        ("learning/training/held_model_training_binding.py", "from random import random", False),
        (
            "learning/training/held_model_training_binding.py",
            "from random import SystemRandom",
            False,
        ),
        ("learning/training/held_model_training_binding.py", "from torch import cat", False),
        ("learning/training/held_model_training_binding.py", "from torch.optim import Adam", False),
        (
            "learning/training/held_model_training_binding.py",
            "from .joint_model_parameter_update import _validate_training_tensor",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from .held_model_training_batch_sampling import Tensor",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ..models.shared_feature_extractor import Module",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ...runtime import single_run_execution",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "import federated_drift_experiment.clients.base",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ...methods.fedsda.training_data_assignment import pending_training_assignment_buffer",
            False,
        ),
        ("learning/training/held_model_joint_training_iterations.py", "import math", False),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from random import random",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from random import SystemRandom",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from torch import cat",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from torch.optim import Adam",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .joint_model_parameter_update import _validate_training_tensor",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .held_model_training_batch_sampling import Tensor",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ..models.shared_feature_extractor import Module",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ...runtime import single_run_execution",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "import federated_drift_experiment.clients.base",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ...methods.fedsda.training_data_assignment import pending_training_assignment_buffer",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from torch.optim import Optimizer",
            True,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from __future__ import annotations",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from random import Random",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from torch.optim import Optimizer",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ..models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .held_model_training_binding import HeldModelTrainingBinding",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .model_training_sample_records import ModelTrainingSampleCollection",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .participating_model_training_batch import ParticipatingModelTrainingBatch",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .local_training_settings import LocalTrainingSettings",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .joint_model_parameter_update import perform_joint_model_parameter_update",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from __future__ import annotations",
            True,
        ),
    ],
)
def test_joint_training_iteration_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import copy", False),
        ("import dataclasses", False),
        ("import torch", False),
        ("import numpy", False),
        ("import federated_drift_experiment", False),
        ("from torch.optim import Adam", False),
        ("from torch.nn import Linear", False),
        ("from .parameter_optimizer_construction import _validate_optimizer_settings", False),
        ("from .parameter_optimizer_settings import field", False),
        ("from .parameter_optimizer_construction.child import create_parameter_optimizer", False),
        (
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models import ResidualAdapterClassifier",
            False,
        ),
        ("from __future__ import annotations", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Optimizer", True),
        ("from .parameter_optimizer_settings import AdamParameterOptimizerSettings", True),
        ("from .parameter_optimizer_settings import SgdParameterOptimizerSettings", True),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", True),
    ],
)
def test_parameter_optimizer_state_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/parameter_optimizer_state.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import dataclasses", False),
        ("import torch", False),
        ("import numpy", False),
        ("import federated_drift_experiment", False),
        (
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            False,
        ),
        ("from .local_training_schedule_settings import LocalTrainingScheduleSettings", False),
        ("from .model_training_sample_records import Tensor", False),
        ("from .model_training_sample_records import SampledModelTrainingBatch", False),
        ("from .model_training_sample_records import _private", False),
        ("from .model_training_sample_records.child import ObservedTrainingSample", False),
        ("from federated_learning_experiments.runtime import run_experiment", False),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            False,
        ),
        ("from __future__ import annotations", True),
        ("from .model_training_sample_records import ObservedTrainingSample", True),
        ("from .model_training_sample_records import ModelTrainingSampleCollection", True),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
    ],
)
def test_model_training_sample_store_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/model_training_sample_store.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        *[
            (source_module_path, source_text, False)
            for source_module_path in (
                "evaluation/model_evaluation_sample_records.py",
                "evaluation/model_evaluation_sample_store.py",
            )
            for source_text in (
                "import math",
                "import random",
                "from random import sample",
                "from random import SystemRandom",
                "import torch",
                "from torch import clone",
                "import numpy",
                "import federated_drift_experiment",
                "from dataclasses import asdict",
                "from torch import *",
                "from .model_evaluation_sample_records import _private",
                "from .model_evaluation_sample_records import Tensor",
                "from .model_evaluation_sample_records.child import ObservedEvaluationSample",
                "import federated_learning_experiments.evaluation.model_evaluation_sample_records",
                "from federated_learning_experiments.evaluation import model_evaluation_sample_records",
                "from federated_learning_experiments.runtime import run_experiment",
                "from ..learning.training.model_training_sample_store import ModelTrainingSampleStore",
                "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            )
        ],
        (
            "evaluation/model_evaluation_sample_records.py",
            "from dataclasses import dataclass",
            True,
        ),
        ("evaluation/model_evaluation_sample_records.py", "from torch import Tensor", True),
        ("evaluation/model_evaluation_sample_records.py", "from random import Random", False),
        ("evaluation/model_evaluation_sample_store.py", "from dataclasses import dataclass", False),
        ("evaluation/model_evaluation_sample_store.py", "from random import Random", True),
        ("evaluation/model_evaluation_sample_store.py", "from __future__ import annotations", True),
        (
            "evaluation/model_evaluation_sample_store.py",
            "from .model_evaluation_sample_records import ObservedEvaluationSample",
            True,
        ),
        (
            "evaluation/model_evaluation_sample_store.py",
            "from .model_evaluation_sample_records import ModelEvaluationSampleCollection",
            True,
        ),
    ],
)
def test_model_evaluation_sample_dependencies(source_module_path, source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("from random import Random", False),
        ("from collections import Counter", False),
        ("import dataclasses", False),
        ("from dataclasses import asdict", False),
        ("from dataclasses import *", False),
        ("import numpy", False),
        ("import torch", False),
        ("from torch.optim import Optimizer", False),
        ("import federated_drift_experiment", False),
        ("from .local_training_request_schedule import LocalTrainingRequestSchedule", False),
        ("from .local_training_settings import LocalTrainingSettings", False),
        ("from .model_training_sample_store import ModelTrainingSampleStore", False),
        ("from federated_learning_experiments.runtime import run", False),
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as Record", True),
        ("from __future__ import annotations", True),
    ],
)
@pytest.mark.parametrize(
    "source_module_path",
    [
        "learning/training/model_training_and_assignment_counts.py",
        "learning/training/current_training_model_assignment.py",
    ],
)
def test_training_state_owner_dataclass_only_dependency_contract(
    source_text, expected_acceptance, source_module_path
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path,
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        ("learning/training/local_training_schedule_settings.py", "import math", False),
        ("learning/training/local_training_schedule_settings.py", "import random", False),
        ("learning/training/local_training_schedule_settings.py", "import torch", False),
        (
            "learning/training/local_training_schedule_settings.py",
            "from dataclasses import asdict",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ...core.settings_field_validation import fields",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ...configuration.run_settings import ValidatedExperimentRunSettingsSubset",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from .held_model_joint_training_iterations import perform_held_model_joint_training_iterations",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from .local_training_schedule_settings import field",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "import federated_drift_experiment.config",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from .local_training_request_schedule import _private",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
            False,
        ),
        ("learning/training/local_training_schedule_settings.py", "import numpy", False),
        ("learning/training/local_training_request_schedule.py", "import math", False),
        ("learning/training/local_training_request_schedule.py", "import random", False),
        ("learning/training/local_training_request_schedule.py", "import torch", False),
        (
            "learning/training/local_training_request_schedule.py",
            "from dataclasses import asdict",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from ...core.settings_field_validation import fields",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from ...configuration.run_settings import ValidatedExperimentRunSettingsSubset",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .held_model_joint_training_iterations import perform_held_model_joint_training_iterations",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .local_training_schedule_settings import field",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "import federated_drift_experiment.config",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .local_training_request_schedule import _private",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
            False,
        ),
        ("learning/training/local_training_request_schedule.py", "import numpy", False),
        (
            "learning/training/local_training_schedule_settings.py",
            "from dataclasses import dataclass, field",
            True,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ...core.settings_field_validation import validate_settings_field_values",
            True,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from __future__ import annotations",
            True,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .local_training_schedule_settings import LocalTrainingScheduleSettings",
            True,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from federated_learning_experiments.learning.training.local_training_schedule_settings import LocalTrainingScheduleSettings",
            True,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from __future__ import annotations",
            True,
        ),
    ],
)
def test_local_training_schedule_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance
