"""単一runの各層で許可する依存とpackage境界を確認する。"""

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest


def is_configuration_foundation_module(*, source_module_path):
    """設定基盤の実装と機能別の設定宣言を選ぶ。"""
    return source_module_path.startswith(("core/", "configuration/")) or (
        source_module_path.startswith(("learning/", "methods/"))
        and source_module_path.endswith("_settings.py")
    )


def resolve_imported_module_names(*, import_statement, importing_package_name):
    """fromの別名も展開し、package経由の禁止モジュールを見逃さない。"""
    if isinstance(import_statement, ast.Import):
        return tuple(
            imported_module_alias.name for imported_module_alias in import_statement.names
        )
    if not isinstance(import_statement, ast.ImportFrom):
        return ()
    relative_import_prefix = "." * import_statement.level
    imported_base_module_name = resolve_name(
        relative_import_prefix + (import_statement.module or ""), importing_package_name,
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
    if imported_module_name.split(".")[0] in sys.stdlib_module_names:
        return True
    if source_module_path == "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py":
        # 収集状態は同機能設定だけを参照し、数値評価へ依存しない。
        return imported_module_name in (
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
        )
    if source_module_path == "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py":
        # 保留位置FIFOは同機能の容量条件だけを参照する。
        return (
            imported_module_name == "federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings"
            or imported_module_name.startswith("federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings.")
        )
    if source_module_path == "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py":
        # 単系列の数値検出器だけにNumPyを許可する。
        return imported_module_name == "numpy" or imported_module_name.startswith("numpy.")
    if source_module_path == "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py":
        # 候補loss評価だけにtorchと同機能の固定条件を許可する。
        return (
            imported_module_name == "torch"
            or imported_module_name.startswith("torch.")
            or imported_module_name == "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings"
            or imported_module_name.startswith("federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.")
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
    elif source_module_path == "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py":
        # 数値状態部品は同機能の固定条件だけを参照する。
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings.",
        )
    elif source_module_path == "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py":
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
            import_statement=import_statement, importing_package_name=importing_package_name,
        )
        for imported_module_name in imported_module_names:
            if not dependency_is_allowed(
                source_module_path=source_module_path, imported_module_name=imported_module_name,
            ):
                dependency_boundary_violations += ((
                    imported_module_name, "この層では許可されない依存先です",
                ),)
    return dependency_boundary_violations


def test_single_run_package_boundaries_have_no_exports():
    """新しい境界は存在し、説明だけを持ち再exportしない。"""
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    package_boundary_paths = (
        "data", "data/concept_schedules", "data/sine", "execution", "runtime",
        "learning/prediction",
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
            source_module_path, dependency_boundary_violations,
        )


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_imported_module_name",
    [
        ("learning/prediction/class_probability_calculations.py", "import numpy", "numpy"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "import numpy", "numpy"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.internal", "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.internal"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from .candidate_model_training_and_acceptance_settings import UnsupportedSettings", "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.UnsupportedSettings"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "import torch", "torch"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "import config", "config"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from federated_drift_experiment import provisional_model", "federated_drift_experiment"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from ....runtime import single_run_execution", "federated_learning_experiments.runtime.single_run_execution"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from . import post_alarm_candidate_loss_evaluation", "federated_learning_experiments.methods.fedsda.candidate_model_selection"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from ..training_data_assignment import pending_training_assignment_buffer", "federated_learning_experiments.methods.fedsda.training_data_assignment"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from . import another_loss_collection", "federated_learning_experiments.methods.fedsda.candidate_model_selection"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "import numpy", "numpy"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "import torch", "torch"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "import config", "config"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "from federated_drift_experiment import clients", "federated_drift_experiment"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "from ....runtime import single_run_execution", "federated_learning_experiments.runtime.single_run_execution"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "from ..loss_change_detection import overall_and_true_class_loss_monitoring", "federated_learning_experiments.methods.fedsda.loss_change_detection"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "from . import another_assignment_buffer", "federated_learning_experiments.methods.fedsda.training_data_assignment"),
        ("learning/prediction/class_probability_calculations.py", "import config", "config"),
        ("learning/prediction/class_probability_calculations.py", "from federated_drift_experiment import expert_routing", "federated_drift_experiment"),
        ("learning/prediction/class_probability_calculations.py", "from ... import runtime", "federated_learning_experiments.runtime"),
        ("learning/prediction/class_probability_calculations.py", "from ...methods.fedsda.prediction_combination.fixed_share_prediction_weights import FixedSharePredictionWeightController", "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights"),
        ("learning/prediction/another_prediction.py", "import torch", "torch"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "import numpy", "numpy"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "import config", "config"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "from federated_drift_experiment import provisional_model", "federated_drift_experiment"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "from ....runtime import single_run_execution", "federated_learning_experiments.runtime.single_run_execution"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "from ..loss_change_detection import overall_and_true_class_loss_monitoring", "federated_learning_experiments.methods.fedsda.loss_change_detection"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "from ....learning.prediction import class_probability_calculations", "federated_learning_experiments.learning.prediction"),
        ("methods/fedsda/candidate_model_selection/another_candidate.py", "import torch", "torch"),
        ("methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py", "import torch", "torch"),
        ("methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py", "from federated_drift_experiment import config", "federated_drift_experiment"),
        ("methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py", "from .loss_change_detection_settings import LossChangeDetectionSettings", "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings"),
        ("methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py", "import numpy", "numpy"),
        ("methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py", "import torch", "torch"),
        ("methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py", "from ....runtime import single_run_execution", "federated_learning_experiments.runtime.single_run_execution"),
        ("methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py", "from ..prediction_combination import fixed_share_prediction_weights", "federated_learning_experiments.methods.fedsda.prediction_combination"),
        ("methods/fedsda/loss_change_detection/another_detector.py", "import numpy", "numpy"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import numpy", "numpy"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import torch", "torch"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import config", "config"),
        ("learning/loss_statistics/bounded_loss_moments.py", "from federated_drift_experiment import config", "federated_drift_experiment"),
        ("learning/loss_statistics/bounded_loss_moments.py", "from ...runtime import single_run_execution", "federated_learning_experiments.runtime.single_run_execution"),
        ("learning/loss_statistics/bounded_loss_moments.py", "from ...methods.fedsda.loss_change_detection import overall_and_true_class_loss_monitoring", "federated_learning_experiments.methods.fedsda.loss_change_detection"),
        ("learning/loss_statistics/bounded_loss_moments.py", "from ...methods.fedsda.candidate_model_selection import post_alarm_candidate_loss_evaluation", "federated_learning_experiments.methods.fedsda.candidate_model_selection"),
        ("learning/loss_statistics/bounded_loss_moments.py", "from ..prediction import class_probability_calculations", "federated_learning_experiments.learning.prediction"),
        ("learning/loss_statistics/bounded_loss_moments.py", "from . import another_loss_statistics", "federated_learning_experiments.learning.loss_statistics"),
        ("data/sine/sine_sample_generation.py", "import torch", "torch"),
        ("data/observed_streams.py", "import config", "config"),
        ("data/observed_streams.py", "from clients import fedsda", "clients"),
        ("data/observed_streams.py", "from federated_learning_experiments import runtime",
         "federated_learning_experiments.runtime"),
        ("data/sine/sine_sample_generation.py", "from ... import runtime",
         "federated_learning_experiments.runtime"),
        ("data/sine/sine_sample_generation.py", "from ...runtime import single_run_execution",
         "federated_learning_experiments.runtime.single_run_execution"),
        ("data/observed_streams.py", "from federated_learning_experiments.learning.models import torch_random_state_scope",
         "federated_learning_experiments.learning.models.torch_random_state_scope"),
        ("data/observed_streams.py", "from federated_learning_experiments.configuration import run_settings",
         "federated_learning_experiments.configuration.run_settings"),
        ("data/concept_schedules/random_concept_schedule_settings.py", "import numpy", "numpy"),
        ("data/concept_schedules/random_concept_schedule_settings.py", "from .. import observed_streams",
         "federated_learning_experiments.data.observed_streams"),
        ("execution/stream_protocol_execution_settings.py", "import numpy", "numpy"),
        ("execution/stream_protocol_execution_settings.py", "from federated_learning_experiments.data.sine import sine_sample_generation",
         "federated_learning_experiments.data.sine.sine_sample_generation"),
        ("execution/run_random_sources.py", "import torch", "torch"),
        ("execution/run_participant_contracts.py", "import numpy", "numpy"),
        ("execution/stream_protocol_execution_loop.py", "from ..runtime import single_run_execution",
         "federated_learning_experiments.runtime.single_run_execution"),
        ("execution/stream_protocol_execution_loop.py", "from federated_learning_experiments import methods",
         "federated_learning_experiments.methods"),
        ("execution/run_execution_records.py", "from .. import learning",
         "federated_learning_experiments.learning"),
        ("execution/stream_protocol_execution_loop.py", "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
         "federated_learning_experiments.configuration.experiment_run_conditions"),
        ("runtime/single_run_execution.py", "import torch", "torch"),
        ("runtime/single_run_execution.py", "import numpy", "numpy"),
        ("runtime/single_run_execution.py", "from ..learning.models import model_architecture_settings",
         "federated_learning_experiments.learning.models.model_architecture_settings"),
        ("learning/models/torch_random_state_scope.py", "import numpy", "numpy"),
        ("learning/models/torch_random_state_scope.py", "from ... import runtime",
         "federated_learning_experiments.runtime"),
        ("learning/models/another_model.py", "import torch", "torch"),
        ("core/configuration_errors.py", "import numpy", "numpy"),
        ("configuration/experiment_run_conditions.py", "import numpy", "numpy"),
        ("learning/models/model_architecture_settings.py", "import torch", "torch"),
        ("methods/fedsda/consolidation/model_consolidation_settings.py", "import numpy", "numpy"),
        ("data/observed_streams.py", "if False:\n    import torch", "torch"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "import torch", "torch"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "import numpy", "numpy"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "from federated_drift_experiment import expert_routing",
         "federated_drift_experiment"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "from federated_learning_experiments.runtime import single_run_execution",
         "federated_learning_experiments.runtime.single_run_execution"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "from federated_learning_experiments.methods.fedsda.consolidation import model_consolidation_settings",
         "federated_learning_experiments.methods.fedsda.consolidation"),
    ],
)
def test_single_run_dependency_checker_rejects_forbidden_imports(
    source_module_path, source_text, expected_imported_module_name,
):
    """productionを変更せず、実際の検査に禁止依存を注入する。"""
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text,
    )
    assert any(
        imported_module_name == expected_imported_module_name
        for imported_module_name, dependency_boundary_violation_reason
        in dependency_boundary_violations
    ), dependency_boundary_violations


@pytest.mark.parametrize(
    "source_module_path,source_text",
    [
        ("learning/loss_statistics/bounded_loss_moments.py", "from dataclasses import dataclass"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import math"),
        ("learning/prediction/class_probability_calculations.py", "import torch"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from .candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py", "from dataclasses import dataclass"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "from .training_data_assignment_settings import TrainingDataAssignmentSettings"),
        ("methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py", "from collections import deque"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "import torch"),
        ("methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py", "from .candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings"),
        ("learning/prediction/class_probability_calculations.py", "from torch import Tensor"),
        ("learning/prediction/class_probability_calculations.py", "from collections.abc import Mapping"),
        ("methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py", "import numpy"),
        ("methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py", "from .bounded_loss_e_sr_detection import BoundedLossESRDetector"),
        ("methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py", "from .loss_change_detection_settings import LossChangeDetectionSettings"),
        ("data/observed_streams.py", "from dataclasses import dataclass"),
        ("data/sine/sine_sample_generation.py", "import numpy as np"),
        ("data/sine/sine_sample_generation.py", "from ..observed_streams import ObservedSample"),
        ("data/sine/sine_sample_generation.py", "from .. import observed_streams"),
        ("data/concept_schedules/random_concept_schedule_generation.py", "from ...configuration.experiment_run_conditions import ExperimentRunConditions"),
        ("data/concept_schedules/random_concept_schedule_settings.py", "from ...core.settings_field_validation import validate_settings_field_values"),
        ("execution/stream_protocol_execution_settings.py", "from ..data.concept_schedules.random_concept_schedule_settings import RandomConceptScheduleSettings"),
        ("execution/stream_protocol_execution_settings.py", "from ..configuration.experiment_run_conditions import ExperimentRunConditions"),
        ("execution/run_random_sources.py", "from numpy.random import RandomState"),
        ("execution/run_participant_contracts.py", "from .run_random_sources import RunRandomSources"),
        ("execution/run_participant_contracts.py", "from ..configuration.experiment_run_conditions import ExperimentRunConditions"),
        ("execution/stream_protocol_execution_loop.py", "from . import run_participant_contracts"),
        ("execution/stream_protocol_execution_loop.py", "from ..data.observed_streams import ClientObservedStream"),
        ("learning/models/torch_random_state_scope.py", "import torch"),
        ("learning/models/torch_random_state_scope.py", "from torch import random"),
        ("runtime/single_run_execution.py", "from ..learning.models.torch_random_state_scope import isolated_cpu_torch_random_state"),
        ("runtime/single_run_execution.py", "from ..learning.models import torch_random_state_scope"),
        ("runtime/single_run_execution.py", "from ..execution import run_random_sources"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "import math"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "from .prediction_combination_settings import PredictionCombinationSettings"),
    ],
)
def test_single_run_dependency_checker_accepts_allowed_imports(source_module_path, source_text):
    """許可された型依存・NumPy例外・専用torch境界を拒否しない。"""
    assert collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text,
    ) == ()
