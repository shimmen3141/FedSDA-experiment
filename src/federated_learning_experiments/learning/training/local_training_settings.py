"""ローカル学習のパラメータ更新と共有部の勾配統合、保有モデルの学習batchの抜き出し方に関する固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class LocalTrainingSettings:
    """共同学習と概念別勾配の統合方式を保持し、構築時に正式値を検証する。"""

    local_model_parameter_update_strategy: str = field(
        metadata={
            "allowed_parameter_values": (
                # 更新ごとに、全保有モデルのbatchの損失を平均し、共有部を1回、各概念固有部を1回ずつ更新する（提案構成）。
                "joint_backbone_adapter_and_head_training",
                # モデルごとに順に、そのモデルのbatchだけで更新する（共有部を持たない構造）。
                "sequential_per_model_training",
            ),
        },
    )
    shared_backbone_gradient_combination_strategy: str = field(
        metadata={
            "allowed_parameter_values": ("sample_weighted_mean_per_concept_gradients",),
        },
    )

    # 保有モデルの学習batchの抜き出し方。全履歴から一様（既定）、直近K件から一様、新しい標本ほど選ばれやすい
    # 指数減衰（各回、まだ選んでいない位置の中で、列の末尾からa件前の位置がαのa乗に比例して選ばれる）。
    training_sample_selection_policy: str = field(
        metadata={
            "allowed_parameter_values": (
                "uniform_over_all_stored_samples",
                "uniform_over_most_recent_stored_samples",
                "recency_decayed_stored_samples",
            ),
        },
    )
    # 直近K件のK（直近K件のときだけ）。
    most_recent_stored_sample_count: int | None = field(
        metadata={
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
            "parameter_unit": "sample",
        },
    )
    # 指数減衰のα（指数減衰のときだけ。0<α<1）。
    stored_sample_recency_decay_factor: float | None = field(
        metadata={
            "minimum_allowed_value": 0.0,
            "minimum_value_is_inclusive": False,
            "maximum_allowed_value": 1.0,
            "maximum_value_is_inclusive": False,
            "parameter_unit": "dimensionless",
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
        # 共有の検査は、int宣言で派生型を、float宣言でintも受けるので、型を厳密に確かめる。
        for field_name, expected_type in (
            ("most_recent_stored_sample_count", int),
            ("stored_sample_recency_decay_factor", float),
        ):
            field_value = getattr(self, field_name)
            if field_value is not None and type(field_value) is not expected_type:
                raise RunSettingsValidationError(
                    configuration_parameter_name=field_name,
                    specified_parameter_value=field_value,
                    validation_failure_reason=f"builtin {expected_type.__name__}かNoneを指定してください。",
                )
        expected_value_field_name = {
            "uniform_over_all_stored_samples": None,
            "uniform_over_most_recent_stored_samples": "most_recent_stored_sample_count",
            "recency_decayed_stored_samples": "stored_sample_recency_decay_factor",
        }[self.training_sample_selection_policy]
        for field_name in (
            "most_recent_stored_sample_count",
            "stored_sample_recency_decay_factor",
        ):
            field_value = getattr(self, field_name)
            if (field_name == expected_value_field_name) != (field_value is not None):
                raise RunSettingsValidationError(
                    configuration_parameter_name=field_name,
                    specified_parameter_value=field_value,
                    validation_failure_reason=(
                        "抜き出し方に合う値だけを指定してください（直近K件ならKだけ、指数減衰ならαだけ、"
                        "全履歴ならどちらもNone）。"
                        f"training_sample_selection_policy={self.training_sample_selection_policy!r}。"
                    ),
                )
