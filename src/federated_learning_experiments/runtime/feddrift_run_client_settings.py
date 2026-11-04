"""FedDrift clientの組立と進行に必要な条件。"""

from dataclasses import dataclass, field
from math import isfinite

from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.feddrift.adaptive_fedavg_learning_rate import (
    AdaptiveFedavgLearningRateSettings,
)
from federated_learning_experiments.methods.feddrift.driftsurf_state import DriftSurfSettings


@dataclass(frozen=True, kw_only=True)
class FeddriftRunClientSettings:
    """一clientの検出・学習・評価保持条件。"""

    loss_increase_threshold: float
    detection_batch_sample_count: int
    local_training_batch_sample_count: int
    updates_per_sample: int
    maximum_stored_evaluation_sample_count_per_model: int
    added_evaluation_batch_sample_count: int
    new_model_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings
    model_comparison_criterion: str
    new_model_parameter_source: str
    detection_reference_after_drift: str
    model_assignment_policy: str = field(
        default="detection_batch_assignment", metadata={"omit_default_from_saved_settings": True}
    )
    training_sample_window: str = field(
        default="all_timesteps", metadata={"omit_default_from_saved_settings": True}
    )
    cluster_model_initialization: str = field(
        default="all_pretrained", metadata={"omit_default_from_saved_settings": True}
    )

    adaptive_learning_rate_settings: AdaptiveFedavgLearningRateSettings | None = field(
        default=None, metadata={"omit_default_from_saved_settings": True}
    )

    driftsurf_settings: DriftSurfSettings | None = field(
        default=None, metadata={"omit_default_from_saved_settings": True}
    )

    cfl_split_gamma: float = field(default=0.1, metadata={"omit_default_from_saved_settings": True})

    fedccfa_dbscan_eps: float = field(
        default=0.1, metadata={"omit_default_from_saved_settings": True}
    )
    fedccfa_alignment_gamma: float = field(
        default=20.0, metadata={"omit_default_from_saved_settings": True}
    )
    fedccfa_alignment_temperature: float = field(
        default=0.1, metadata={"omit_default_from_saved_settings": True}
    )
    fedccfa_alignment_start: str = field(
        default="fixed_round_20", metadata={"omit_default_from_saved_settings": True}
    )

    def __post_init__(self) -> None:
        for name, default in (
            ("fedccfa_dbscan_eps", 0.1),
            ("fedccfa_alignment_gamma", 20.0),
            ("fedccfa_alignment_temperature", 0.1),
        ):
            value = getattr(self, name)
            if type(value) is not float or not isfinite(value) or value <= 0:
                raise ValueError(f"{name}は正の有限builtin floatが必要です。")
            if self.model_assignment_policy != "fedccfa_classifier_clustering" and value != default:
                raise ValueError(f"{name}はFedCCFAだけで有効です。")
        if self.fedccfa_alignment_start not in ("fixed_round_20", "scaled_to_run_rounds"):
            raise ValueError("FedCCFAの整合の開始が未対応です。")
        if (
            self.model_assignment_policy != "fedccfa_classifier_clustering"
            and self.fedccfa_alignment_start != "fixed_round_20"
        ):
            raise ValueError("整合の開始はFedCCFAだけで有効です。")

        if (
            type(self.cfl_split_gamma) is not float
            or not isfinite(self.cfl_split_gamma)
            or not 0 <= self.cfl_split_gamma <= 1
        ):
            raise ValueError("CFLのγは0以上1以下の有限builtin floatが必要です。")
        if (
            self.model_assignment_policy != "cfl_bipartition_assignment"
            and self.cfl_split_gamma != 0.1
        ):
            raise ValueError("CFLのγはCFLの帰属だけで有効です。")
        if self.driftsurf_settings is not None:
            if type(self.driftsurf_settings) is not DriftSurfSettings:
                raise TypeError("exact DriftSurfSettingsが必要です。")
            self.driftsurf_settings.__post_init__()
        if (self.model_assignment_policy == "driftsurf_global_role_assignment") != (
            self.driftsurf_settings is not None
        ):
            raise ValueError("DriftSurfの設定は役割の帰属と組み合わせてください。")
        if self.adaptive_learning_rate_settings is not None:
            if type(self.adaptive_learning_rate_settings) is not AdaptiveFedavgLearningRateSettings:
                raise TypeError("exact AdaptiveFedavgLearningRateSettingsが必要です。")
            self.adaptive_learning_rate_settings.__post_init__()
            if self.model_assignment_policy != "single_global_model_assignment":
                raise ValueError("学習率の調整は単一モデルの構成専用です。")
        for name, choices in (
            (
                "model_assignment_policy",
                (
                    "detection_batch_assignment",
                    "true_concept_assignment",
                    "minimum_loss_cluster_assignment",
                    "single_global_model_assignment",
                    "driftsurf_global_role_assignment",
                    "cfl_bipartition_assignment",
                    "fedem_responsibility_assignment",
                    "fedccfa_classifier_clustering",
                ),
            ),
            (
                "training_sample_window",
                (
                    "all_timesteps",
                    "current_timestep",
                    "role_timestep_window",
                    "all_observed_history",
                    "recent_observed_samples",
                ),
            ),
            (
                "cluster_model_initialization",
                ("all_pretrained", "all_random", "pretrained_cluster0"),
            ),
            ("model_comparison_criterion", ("bounded_loss", "classification_error_rate")),
            (
                "new_model_parameter_source",
                ("current_training_model", "minimum_criterion_held_model"),
            ),
            (
                "detection_reference_after_drift",
                ("new_model_batch_value", "minimum_held_model_batch_value"),
            ),
        ):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name}はbuiltin strが必要です。")
            if value not in choices:
                raise ValueError(f"{name}が未対応です。")
        if self.model_assignment_policy == "fedccfa_classifier_clustering":
            if (
                self.training_sample_window not in ("current_timestep", "recent_observed_samples")
                or self.cluster_model_initialization != "all_pretrained"
            ):
                raise ValueError("FedCCFAは観測済みの窓と共通の事前学習の初期値が必要です。")
        elif self.model_assignment_policy == "fedem_responsibility_assignment":
            if (
                self.training_sample_window not in ("current_timestep", "all_observed_history")
                or self.cluster_model_initialization == "all_pretrained"
            ):
                raise ValueError("FedEMは時刻または全履歴の標本と成分ごとの初期化が必要です。")
        elif self.model_assignment_policy == "minimum_loss_cluster_assignment":
            if (
                self.training_sample_window != "current_timestep"
                or self.cluster_model_initialization == "all_pretrained"
            ):
                raise ValueError("IFCA+Windowには1時刻の窓とクラスタごとの初期化が必要です。")
        elif self.model_assignment_policy == "driftsurf_global_role_assignment":
            if (
                self.training_sample_window != "role_timestep_window"
                or self.cluster_model_initialization != "all_pretrained"
            ):
                raise ValueError("DriftSurfは役割の窓と共通の初期値が必要です。")
        elif self.model_assignment_policy in (
            "single_global_model_assignment",
            "cfl_bipartition_assignment",
        ):
            if (
                self.training_sample_window != "current_timestep"
                or self.cluster_model_initialization != "all_pretrained"
            ):
                raise ValueError("単一モデルには1時刻の窓と事前学習の初期化が必要です。")
        elif (
            self.training_sample_window != "all_timesteps"
            or self.cluster_model_initialization != "all_pretrained"
        ):
            raise ValueError("窓とクラスタごとの初期化はIFCA+Window専用です。")
        if type(self.loss_increase_threshold) is not float or (
            not isfinite(self.loss_increase_threshold) or self.loss_increase_threshold < 0
        ):
            raise ValueError("損失増分閾値は非負有限のbuiltin floatが必要です。")
        for name, minimum in (
            ("detection_batch_sample_count", 1),
            ("local_training_batch_sample_count", 1),
            ("updates_per_sample", 0),
            ("maximum_stored_evaluation_sample_count_per_model", 1),
            ("added_evaluation_batch_sample_count", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}は{minimum}以上のbuiltin intが必要です。")
        if type(self.new_model_optimizer_settings) not in (
            AdamParameterOptimizerSettings,
            SgdParameterOptimizerSettings,
        ):
            raise TypeError("新規optimizerにはexact Adam/SGD設定が必要です。")
        self.new_model_optimizer_settings.__post_init__()
