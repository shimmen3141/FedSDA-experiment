"""予測混合方式・予測状態管理方針・share時間尺度の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class PredictionCombinationSettings:
    """予測処理の状態を持たず、構築時に正式名と時間尺度を検証する。"""

    prediction_combination_strategy: str = field(
        metadata={
            "allowed_parameter_values": (
                # Fixed-Shareの重みで、全保有モデルの確率を結合して予測する（提案構成）。
                "fixed_share_weighted_prediction",
                # AdaHedgeの重みで結合して予測する。Fixed-Shareの重みも更新し続け、比較用に記録する。
                "adahedge_weighted_prediction",
                # 結合しない。現在の学習帰属のモデルだけで予測する（重みは、使わない・更新しない）。
                "current_training_model_prediction",
            )
        },
    )
    prediction_mixture_activation_policy: str = field(
        metadata={"allowed_parameter_values": ("always",)},
    )
    prediction_weight_recalibration_after_aggregation_policy: str = field(
        metadata={
            "allowed_parameter_values": (
                # 保留中の標本の損失を、集約後のモデルで計算し直して、重みを作り直す（提案構成）。
                "recompute_buffer_losses_and_replay_weight_updates",
                # 集約の後も、重みを、作り直さずに、そのまま使う（再較正を外した構成）。
                "keep_weights_without_recalibration",
            )
        },
    )
    prediction_state_reset_on_training_assignment_change_policy: str = field(
        metadata={
            "allowed_parameter_values": ("restart_adahedge_preserve_fixed_share_prediction_state",)
        },
    )
    prediction_model_parameter_source: str = field(
        metadata={
            "allowed_parameter_values": (
                # 予測（と予測重みの更新）に、局所更新した現在のパラメータを使う（提案構成）。
                "current_local_parameters",
                # 予測（と予測重みの更新）に、最後の同期の時点のパラメータを使う。同期の後に作ったモデルは、
                # 共有部を同期の時点の値、モデル固有部を現在の値で評価する。学習・検出ほかは、現在の値のまま。
                "parameters_at_last_synchronization",
            )
        },
    )
    fixed_share_weight_redistribution_time_scale_samples: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 2,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
