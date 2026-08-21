"""単一runへ解決済みの実験設定。"""

from contextlib import contextmanager
from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class AlgorithmOptions:
    """掃引値とは独立したアルゴリズムの選択肢。"""

    clustering_policy: str
    clustering_decision: str
    detection_episodes: bool
    new_model_creation_policy: str
    fifo_size: int
    new_model_validation_fraction: float
    new_model_forward_validation_samples: int
    shared_backbone_training: str
    shared_backbone_routing_recalibration: str
    sequential_tournament_alpha: float = 0.05
    shared_adapter_rank: int = 8
    shared_backbone_gradient_strategy: str = "mean"
    soft_routing_context: str = "global"
    soft_routing_top_combination: str = "leader"
    soft_routing_meta_loss: str = "zero_one"
    cluster_linkage: str | None = None
    clustering_consolidation: str = "merge"

    @classmethod
    def from_current_config(cls):
        return cls(
            clustering_policy=config.FEDSDA_CLUSTERING_POLICY,
            clustering_decision=config.FEDSDA_CLUSTERING_DECISION,
            detection_episodes=config.FEDSDA_DETECTION_EPISODES_ENABLED,
            new_model_creation_policy=config.NEW_MODEL_CREATION_POLICY,
            fifo_size=config.FIFO_BUFFER_SIZE,
            new_model_validation_fraction=config.NEW_MODEL_VALIDATION_FRACTION,
            new_model_forward_validation_samples=(
                config.NEW_MODEL_FORWARD_VALIDATION_SAMPLES
            ),
            sequential_tournament_alpha=config.SEQUENTIAL_TOURNAMENT_ALPHA,
            shared_backbone_training=config.SHARED_BACKBONE_TRAINING,
            shared_backbone_routing_recalibration=(
                config.SHARED_BACKBONE_ROUTING_RECALIBRATION
            ),
            shared_adapter_rank=config.SHARED_ADAPTER_RANK,
            shared_backbone_gradient_strategy=(
                config.SHARED_BACKBONE_GRADIENT_STRATEGY
            ),
            soft_routing_context=config.SOFT_ROUTING_CONTEXT,
            soft_routing_top_combination=(
                config.SOFT_ROUTING_TOP_COMBINATION
            ),
            soft_routing_meta_loss=config.SOFT_ROUTING_META_LOSS,
            cluster_linkage=None,
            clustering_consolidation=(
                config.FEDSDA_CLUSTERING_CONSOLIDATION
            ),
        )

    def config_overrides(self):
        """既存実行層へ渡す設定名と値を返す。"""
        overrides = {
            "FEDSDA_CLUSTERING_POLICY": self.clustering_policy,
            "FEDSDA_CLUSTERING_DECISION": self.clustering_decision,
            "FEDSDA_CLUSTERING_CONSOLIDATION": (
                self.clustering_consolidation
            ),
            "FEDSDA_DETECTION_EPISODES_ENABLED": self.detection_episodes,
            "NEW_MODEL_CREATION_POLICY": self.new_model_creation_policy,
            "FIFO_BUFFER_SIZE": self.fifo_size,
            "NEW_MODEL_VALIDATION_FRACTION": self.new_model_validation_fraction,
            "NEW_MODEL_FORWARD_VALIDATION_SAMPLES": (
                self.new_model_forward_validation_samples
            ),
            "SEQUENTIAL_TOURNAMENT_ALPHA": self.sequential_tournament_alpha,
            "SHARED_BACKBONE_TRAINING": self.shared_backbone_training,
            "SHARED_BACKBONE_ROUTING_RECALIBRATION": (
                self.shared_backbone_routing_recalibration
            ),
            "SHARED_ADAPTER_RANK": self.shared_adapter_rank,
            "SHARED_BACKBONE_GRADIENT_STRATEGY": (
                self.shared_backbone_gradient_strategy
            ),
            "SOFT_ROUTING_CONTEXT": self.soft_routing_context,
            "SOFT_ROUTING_TOP_COMBINATION": (
                self.soft_routing_top_combination
            ),
            "SOFT_ROUTING_META_LOSS": self.soft_routing_meta_loss,
        }
        if self.cluster_linkage is not None:
            overrides["FEDSDA_CLUSTER_LINKAGE"] = self.cluster_linkage
            overrides["FEDDRIFT_CLUSTER_LINKAGE"] = self.cluster_linkage
        return overrides


@dataclass(frozen=True)
class ParameterAssignment:
    """一つの正規パラメータIDに対する解決済み値。"""

    parameter_id: str
    value: int | float


@dataclass(frozen=True)
class ExperimentConfiguration:
    """一つのmode・dataset・seedを実行するための完全な設定。"""

    mode: str
    dataset: str
    seed: int
    concept_schedule: str
    series: str
    sweep_parameter: str | None
    sweep_value: int | float | None
    parameters: tuple[ParameterAssignment, ...]
    algorithm: AlgorithmOptions

    def parameter_value(self, parameter_id, default=None):
        for assignment in self.parameters:
            if assignment.parameter_id == parameter_id:
                return assignment.value
        return default

    def config_overrides(self):
        """このrunだけで有効にする既存config互換の設定を返す。"""
        overrides = {
            "DATASET": self.dataset,
            "CONCEPT_SCHEDULE": self.concept_schedule,
            **self.algorithm.config_overrides(),
        }
        for assignment in self.parameters:
            overrides[_CONFIG_ATTRIBUTES[assignment.parameter_id]] = assignment.value
        return overrides

    @contextmanager
    def activated(self):
        """run設定を一時的に有効化し、終了時には例外時も元へ戻す。"""
        with temporary_config(self.config_overrides()):
            yield self


_CONFIG_ATTRIBUTES = {
    "adwin_delta": "ADWIN_DELTA",
    "aggregation_interval": "AGGREGATION_INTERVAL",
    "fedsda_distance_threshold": "FEDSDA_DISTANCE_THRESHOLD",
    "feddrift_detection_batch_size": "FEDDRIFT_DETECTION_BATCH_SIZE",
    "feddrift_distance_threshold": "FEDDRIFT_DISTANCE_THRESHOLD",
}


@contextmanager
def temporary_config(overrides):
    """移行中の実行層に対して、漏れない設定スコープを提供する。"""
    previous = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)
