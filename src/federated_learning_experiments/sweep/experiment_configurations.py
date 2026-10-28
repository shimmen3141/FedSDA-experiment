"""実験の構成の名前（手法と、その選択肢の組）から、完全なrun設定を作る。

構成は、実行条件（dataset、seed、規模、集約間隔）とは別のもの。構成を足すときは、ここへ足す。
提案構成以外のFedSDAの構成は、提案構成のrun設定の、1つの要素だけを置き換える（要素を外す構成は
`fedsda-without-…`）。FedDrift・FedAvgの構成は、それぞれの既定のrun設定から作る。
"""

from dataclasses import replace

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
)
from federated_learning_experiments.runtime.fedavg_run_settings import (
    FedavgRunSettings,
    build_default_fedavg_run_settings,
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
# 候補検証を外し、候補の学習で共有部も更新する構成（合う保有モデルが無いとき、学習した候補を、その場で採用する）。
# 2026-10-20に、提案構成が候補検証なし（共有部は固定）になったので、提案構成からは、候補の共有部の固定を外した
# 構成に当たる。設定は、これまでと同じ（名前は、過去の結果との連続のために残す）。
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
# 提案構成の、予測（と予測重みの更新）を、最後の同期の時点のモデルで行う構成。学習・検出・候補の検証は、
# 局所更新した現在のモデルのまま。
SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME = "fedsda-synchronized-model-prediction"
# 提案構成の予測を、同期の時点のモデルの結合予測と、局所更新したモデルの結合予測を、clientごとの重みで
# 混ぜて行う構成。学習・検出ほかは、局所更新した現在のモデルのまま。2026-10-16に、この予測が提案構成の既定に
# なったので、提案構成と同じ設定（名前だけが違う。これまでのコマンドと文書のために残す。この名前の過去の結果は
# Fixed-Shareの時間尺度30で、これからは既定の120なので、値は連続しない）。
SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME = (
    "fedsda-synchronized-local-mixture-prediction"
)
# 提案構成から、予測に使うパラメータだけを、局所更新した現在の値にした構成（2026-10-15までの既定の予測。
# 混ぜる予測の寄与を比べる用）。全モデルの結合予測は、提案構成と同じ（`fedsda-single-model-prediction`は、
# 結合をせず、現在の学習帰属のモデルだけで予測する）。
LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME = "fedsda-local-parameter-prediction"
# `fedavg`の、局所更新の間隔を、集約の間隔と同じにした構成。roundの中は、そのroundの始めに配布されたモデルで
# 予測し、局所更新は、roundの最後の標本の処理でまとめて行う（`fedavg`は、1標本ごとに局所更新する）。
FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME = "fedavg-local-updates-per-aggregation"
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
    "fedavg",
    FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
    SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
    SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
    LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
)


# 損失の監視を替える構成: 検出の設定の、替えるfieldと、検出器の表示名。
_LOSS_CHANGE_DETECTION_CHANGES_BY_CONFIGURATION_NAME: dict[str, tuple[dict[str, str], str]] = {
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME: (
        {"loss_monitoring_scope": "overall_loss"},
        "e-SR",
    ),
    # ADWINは、最小件数まで警報を待つ扱い（最終構成の既定）と組み合わせられないので、警報を即時に明示する。
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME: (
        {
            "drift_detector_name": "adwin",
            "loss_monitoring_baseline_update_policy": "keep_baseline_until_alarm",
            "short_change_interval_alarm_policy": RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        },
        "ADWIN",
    ),
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME: (
        {
            "drift_detector_name": "adwin",
            "loss_monitoring_baseline_update_policy": "keep_baseline_until_alarm",
            "loss_monitoring_scope": "overall_loss",
            "short_change_interval_alarm_policy": RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
        },
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


def _replace_candidate_acceptance_and_shared_feature_update(
    *,
    run_settings: FedsdaRunSettings,
    candidate_shared_feature_update_policy: str,
    alarm_interval_assignment_policy: str,
    new_model_hypothesis_learners: str,
    initial_model_loss_statistics_estimation: str,
    cross_evaluation_client_selection_policy: str,
    loss_monitoring_baseline_update_policy: str,
    maximum_evaluating_client_count_per_model: int,
) -> FedsdaRunSettings:
    """完全なrun設定の、候補の採否の方針と、候補の学習での共有部の更新と、警報の後の選び方と、「新しいモデル」の
    仮説の学習者の組と、初期モデルの損失の統計の推定の仕方と、評価のclientの選び方を、一緒に置き換える。

    値には組合せの制約（候補検証は、共有部の固定とも、最小記述長の原理で選ぶ扱いとも組み合わせられない。
    warm startの学習者は、最小記述長の原理で選ぶ扱いだけが使う）があるので、片方ずつ置き換えると、途中の設定が
    拒否されることがある。
    """
    run_participant_settings = run_settings.run_participant_settings
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            run_participant_settings,
            initial_model_loss_statistics_estimation=initial_model_loss_statistics_estimation,
            cross_evaluation_client_selection_policy=cross_evaluation_client_selection_policy,
            maximum_evaluating_client_count_per_model=maximum_evaluating_client_count_per_model,
            run_client_settings=replace(
                run_client_settings,
                loss_change_detection_settings=replace(
                    run_client_settings.loss_change_detection_settings,
                    loss_monitoring_baseline_update_policy=loss_monitoring_baseline_update_policy,
                ),
                candidate_epoch_training_settings=replace(
                    run_client_settings.candidate_epoch_training_settings,
                    candidate_shared_feature_update_policy=candidate_shared_feature_update_policy,
                ),
                training_data_assignment_settings=replace(
                    run_client_settings.training_data_assignment_settings,
                    alarm_interval_assignment_policy=alarm_interval_assignment_policy,
                    new_model_hypothesis_learners=new_model_hypothesis_learners,
                ),
            ),
        ),
    )


def build_experiment_run_settings(
    *, configuration_name: str, experiment_run_conditions: ExperimentRunConditions
) -> FedsdaRunSettings | FeddriftRunSettings | FedavgRunSettings:
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
    if configuration_name == "fedavg":
        return build_default_fedavg_run_settings(
            experiment_run_conditions=experiment_run_conditions
        )
    if configuration_name == FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME:
        fedavg_run_settings = build_default_fedavg_run_settings(
            experiment_run_conditions=experiment_run_conditions
        )
        return replace(
            fedavg_run_settings,
            run_participant_settings=replace(
                fedavg_run_settings.run_participant_settings,
                local_update_interval_per_client_samples=(
                    experiment_run_conditions.server_aggregation_interval_per_client_samples
                ),
            ),
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
        return _replace_candidate_acceptance_and_shared_feature_update(
            run_settings=proposed_run_settings,
            candidate_shared_feature_update_policy="update_shared_features_with_candidate",
            # 候補検証を外した構成は、警報の後の選び方、学習者の組、初期モデルの統計の推定を、既定から引き継ぐ。
            alarm_interval_assignment_policy=(
                proposed_run_settings.run_participant_settings.run_client_settings.training_data_assignment_settings.alarm_interval_assignment_policy
            ),
            new_model_hypothesis_learners=(
                proposed_run_settings.run_participant_settings.run_client_settings.training_data_assignment_settings.new_model_hypothesis_learners
            ),
            initial_model_loss_statistics_estimation=(
                proposed_run_settings.run_participant_settings.initial_model_loss_statistics_estimation
            ),
            cross_evaluation_client_selection_policy=(
                proposed_run_settings.run_participant_settings.cross_evaluation_client_selection_policy
            ),
            loss_monitoring_baseline_update_policy=(
                proposed_run_settings.run_participant_settings.run_client_settings.loss_change_detection_settings.loss_monitoring_baseline_update_policy
            ),
            maximum_evaluating_client_count_per_model=(
                proposed_run_settings.run_participant_settings.maximum_evaluating_client_count_per_model
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
    if configuration_name == SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME:
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_model_parameter_source="parameters_at_last_synchronization",
        )
    if configuration_name == SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME:
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_model_parameter_source="mixture_of_synchronized_and_local_parameters",
        )
    if configuration_name == LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME:
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_model_parameter_source="current_local_parameters",
        )
    if configuration_name == ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME:
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_combination_strategy="adahedge_weighted_prediction",
        )
    if configuration_name == SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME:
        # 重みを持たないので、集約後の、重みの再較正も行わない。混ぜる予測（既定）とは組み合わせられないので、
        # 予測に使うパラメータは、局所更新した現在の値に明示する。
        return _replace_prediction_combination_settings(
            run_settings=proposed_run_settings,
            prediction_combination_strategy="current_training_model_prediction",
            prediction_model_parameter_source="current_local_parameters",
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
