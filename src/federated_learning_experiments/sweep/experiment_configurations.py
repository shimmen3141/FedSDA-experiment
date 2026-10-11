"""実験の構成の名前（手法と、その選択肢の組）から、完全なrun設定を作る。

構成は、実行条件（dataset、seed、規模、集約間隔）とは別のもの。構成を足すときは、ここへ足す。
提案構成以外の構成は、提案構成のrun設定の、1つの要素だけを置き換える（要素を外す構成は`fedsda-without-…`）。
"""

from dataclasses import replace

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
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
# コマンドで指定できる構成の名前。
EXPERIMENT_CONFIGURATION_NAMES: tuple[str, ...] = (
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
    CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME,
    LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME,
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
)


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
) -> FedsdaRunSettings:
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
    if configuration_name == OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME:
        # 監視の対象と、それに対応する検出器の表示名を、替える。
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
                        loss_monitoring_scope="overall_loss",
                    ),
                    scalar_settings=replace(
                        run_client_settings.scalar_settings, detector_name="e-SR"
                    ),
                ),
            ),
        )
    return proposed_run_settings
