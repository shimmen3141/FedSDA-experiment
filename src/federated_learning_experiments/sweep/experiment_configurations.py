"""実験の構成の名前（手法と、その選択肢の組）から、完全なrun設定を作る。

構成は、実行条件（dataset、seed、規模、集約間隔）とは別のもの。構成を足すときは、ここへ足す。
提案構成以外の構成は、提案構成のrun設定の、1つの要素だけを置き換える（要素を外す構成は`fedsda-without-…`）。
"""

from dataclasses import replace

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime.feddrift_default_run_settings import (
    build_default_feddrift_run_settings,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings

# 提案構成（docs/overview/proposed-method.mdの「最終提案構成」）。
PROPOSED_FEDSDA_CONFIGURATION_NAME = "fedsda"
# 提案構成から、集約後の、予測重みの再較正を外した構成。
PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME = (
    "fedsda-without-prediction-weight-recalibration"
)
# 提案構成から、候補検証を外した構成（合う保有モデルが無いとき、学習した候補を、その場で採用する）。
CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME = "fedsda-without-candidate-validation"
# 提案構成の、クラスタリングのモデル対の比べ方を、損失の増加量（互いの標本での平均損失の増加）へ替えた構成。
LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME = "fedsda-loss-distance-clustering"
# 提案構成の、損失の監視の対象を、全体の損失の系列だけにした構成（正解クラス別の系列を混合しない）。
OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME = "fedsda-overall-loss-esr-detection"
# 提案構成の、予測に使う重みを、Fixed-Shareの重みから、AdaHedgeの重みへ替えた構成。
ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME = "fedsda-adahedge-prediction-weights"
# 提案構成から、モデルの予測の結合を外した構成（現在の学習帰属のモデルだけで予測する）。
SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME = "fedsda-single-model-prediction"
# 提案構成の、損失の監視の検出器を、e-SRから、ADWINへ替えた構成（全体と、正解クラス別の系列を監視する）。
CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME = "fedsda-classwise-adwin-detection"
# 同じく、ADWINで、全体の損失の系列だけを監視する構成。
OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME = "fedsda-overall-loss-adwin-detection"
# 提案構成のモデルから、概念固有の残差adapterを外した構成（概念固有部は、出力層だけ）。
WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME = "fedsda-without-concept-specific-adapter"
# 提案構成のモデルを、共有部を持たないモデル（隠れ層と分類層の全部を、モデルごとに独立に持つ）へ替えた構成。
# ローカル学習は、モデルごとに順に行い、集約後の、予測の重みの再較正は、行わない。
INDEPENDENT_MODELS_CONFIGURATION_NAME = "fedsda-independent-models"
# コマンドで指定できる構成の名前。
EXPERIMENT_CONFIGURATION_NAMES: tuple[str, ...] = (
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
    CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME,
    LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME,
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
    ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME,
    SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
    WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
    INDEPENDENT_MODELS_CONFIGURATION_NAME,
    "feddrift",
)


# 損失の監視を替える構成: 検出の設定の、替えるfieldと、検出器の表示名。
_LOSS_CHANGE_DETECTION_CHANGES_BY_CONFIGURATION_NAME: dict[str, tuple[dict[str, str], str]] = {
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME: (
        {"loss_monitoring_scope": "overall_loss"},
        "e-SR",
    ),
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME: ({"drift_detector_name": "adwin"}, "ADWIN"),
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME: (
        {"drift_detector_name": "adwin", "loss_monitoring_scope": "overall_loss"},
        "ADWIN",
    ),
}


def _replace_prediction_combination_settings(
    *, run_settings: FedsdaRunSettings, **changed_fields: object
) -> FedsdaRunSettings:
    """完全なrun設定の、予測の結合の設定の一部を置き換える（設定の検証は、もう一度行われる）。"""
    run_participant_settings = run_settings.run_participant_settings
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            run_participant_settings,
            run_client_settings=replace(
                run_client_settings,
                prediction_combination_settings=replace(
                    run_client_settings.prediction_combination_settings, **changed_fields
                ),
            ),
        ),
    )


def _replace_candidate_model_training_and_acceptance_settings(
    *, run_settings: FedsdaRunSettings, **changed_fields: object
) -> FedsdaRunSettings:
    """完全なrun設定の、候補モデルの学習と採否の設定の一部を置き換える。"""
    run_participant_settings = run_settings.run_participant_settings
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            run_participant_settings,
            run_client_settings=replace(
                run_client_settings,
                candidate_model_training_and_acceptance_settings=replace(
                    run_client_settings.candidate_model_training_and_acceptance_settings,
                    **changed_fields,
                ),
            ),
        ),
    )


def build_experiment_run_settings(
    *, configuration_name: str, experiment_run_conditions: ExperimentRunConditions
) -> FedsdaRunSettings | FeddriftRunSettings:
    """構成の名前と実行条件から、1 runの完全なrun設定を作る。"""
    if type(configuration_name) is not str:
        raise TypeError("configuration_name must be builtin str")
    if type(experiment_run_conditions) is not ExperimentRunConditions:
        raise TypeError("experiment_run_conditions must be exact ExperimentRunConditions")
    if configuration_name not in EXPERIMENT_CONFIGURATION_NAMES:
        raise ValueError(
            f"unknown experiment configuration: {configuration_name!r} "
            f"(available: {', '.join(EXPERIMENT_CONFIGURATION_NAMES)})"
        )
    if configuration_name == "feddrift":
        return build_default_feddrift_run_settings(
            experiment_run_conditions=experiment_run_conditions
        )
    proposed_run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=experiment_run_conditions
    )
    if configuration_name == PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME:
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_weight_recalibration_after_aggregation_policy=(
                "keep_weights_without_recalibration"
            ),
        )
    if configuration_name == CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME:
        return _replace_candidate_model_training_and_acceptance_settings(
            run_settings=proposed_run_settings,
            candidate_model_acceptance_policy=(
                "current_model_first_reuse_then_candidate_adoption_without_validation"
            ),
        )
    if configuration_name == LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME:
        # サーバが読む判定の基準と、統合の方式の宣言の、両方を替える。
        run_participant_settings = proposed_run_settings.run_participant_settings
        return replace(
            proposed_run_settings,
            run_participant_settings=replace(
                run_participant_settings,
                model_clustering_criteria=replace(
                    run_participant_settings.model_clustering_criteria,
                    model_pair_comparison_strategy="mean_loss_increase",
                ),
            ),
            model_consolidation_settings=replace(
                proposed_run_settings.model_consolidation_settings,
                model_pair_comparison_strategy="mean_loss_increase",
            ),
        )
    if configuration_name in _LOSS_CHANGE_DETECTION_CHANGES_BY_CONFIGURATION_NAME:
        # 監視の設定（対象、検出器）と、それに対応する検出器の表示名を、替える。
        changed_detection_settings_fields, detector_name = (
            _LOSS_CHANGE_DETECTION_CHANGES_BY_CONFIGURATION_NAME[configuration_name]
        )
        run_participant_settings = proposed_run_settings.run_participant_settings
        run_client_settings = run_participant_settings.run_client_settings
        return replace(
            proposed_run_settings,
            run_participant_settings=replace(
                run_participant_settings,
                run_client_settings=replace(
                    run_client_settings,
                    loss_change_detection_settings=replace(
                        run_client_settings.loss_change_detection_settings,
                        **changed_detection_settings_fields,
                    ),
                    scalar_settings=replace(
                        run_client_settings.scalar_settings, detector_name=detector_name
                    ),
                ),
            ),
        )
    if configuration_name == ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME:
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_combination_strategy="adahedge_weighted_prediction",
        )
    if configuration_name == SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME:
        # 重みを持たないので、集約後の、重みの再較正も行わない。
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_combination_strategy="current_training_model_prediction",
            prediction_weight_recalibration_after_aggregation_policy=(
                "keep_weights_without_recalibration"
            ),
        )
    if configuration_name == WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME:
        run_participant_settings = proposed_run_settings.run_participant_settings
        return replace(
            proposed_run_settings,
            run_participant_settings=replace(
                run_participant_settings,
                model_architecture_settings=replace(
                    run_participant_settings.model_architecture_settings,
                    model_architecture_name="shared_backbone_without_residual_adapter",
                ),
            ),
        )
    if configuration_name == INDEPENDENT_MODELS_CONFIGURATION_NAME:
        without_recalibration_run_settings = _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_weight_recalibration_after_aggregation_policy=(
                "keep_weights_without_recalibration"
            ),
        )
        run_participant_settings = without_recalibration_run_settings.run_participant_settings
        run_client_settings = run_participant_settings.run_client_settings
        return replace(
            without_recalibration_run_settings,
            run_participant_settings=replace(
                run_participant_settings,
                model_architecture_settings=replace(
                    run_participant_settings.model_architecture_settings,
                    model_architecture_name="independent_models_without_shared_backbone",
                ),
                run_client_settings=replace(
                    run_client_settings,
                    local_training_settings=replace(
                        run_client_settings.local_training_settings,
                        local_model_parameter_update_strategy="sequential_per_model_training",
                    ),
                ),
            ),
        )
    return proposed_run_settings
