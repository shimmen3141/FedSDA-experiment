"""最終構成のFedSDAの、runごとの参加者（全clientとサーバ）の初期準備と、その設定の束。"""

from dataclasses import dataclass
from typing import cast

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.observed_sample_generation import (
    OBSERVED_SAMPLE_GENERATOR_TYPES,
    ObservedSampleGenerator,
    is_observed_sample_generator_of_dataset,
)
from federated_learning_experiments.execution.run_participant_contracts import (
    RunClientOperations,
    RunParticipants,
)
from federated_learning_experiments.execution.run_random_sources import RunRandomSources
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.runtime.fedsda_run_client import assemble_fedsda_run_client
from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientSettings,
)
from federated_learning_experiments.runtime.fedsda_run_server import assemble_fedsda_run_server
from federated_learning_experiments.runtime.initial_model_pretraining import (
    FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME,
    INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT,
    INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES,
    estimate_cross_validated_loss_statistics,
    pretrain_initial_model,
)
from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
    CONSOLIDATED_SHARED_FEATURE_POLICY_NAMES,
)
from federated_learning_experiments.runtime.model_cross_evaluation import (
    CROSS_EVALUATION_CLIENT_SELECTION_POLICY_NAMES,
)
from federated_learning_experiments.runtime.server_round_synchronization import (
    CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES,
)

# 初期モデルの正式ID。
_INITIAL_MODEL_ID = 0

_REQUIRED_SETTINGS_TYPE_BY_FIELD_NAME = {
    "run_client_settings": FedsdaRunClientSettings,
    "initial_model_pretraining_settings": InitialModelPretrainingSettings,
    "model_architecture_settings": ModelArchitectureSettings,
    "model_clustering_criteria": ModelClusteringCriteria,
}


@dataclass(frozen=True, kw_only=True)
class FedsdaRunParticipantSettings:
    """全体runの参加者の準備に要る設定。生成時に全部を確かめる。

    事前学習のoptimizerの設定は、clientの設定の束の`rebuilt_model_parameter_optimizer_settings`を使う
    （初期モデルと、配布で作り直すモデルは、同じ学習率にする）。
    """

    run_client_settings: FedsdaRunClientSettings
    initial_model_pretraining_settings: InitialModelPretrainingSettings
    model_architecture_settings: ModelArchitectureSettings
    hidden_layer_widths: tuple[int, ...]
    model_clustering_criteria: ModelClusteringCriteria
    # サーバのクロス評価で、1つのモデルを評価するclientの上限。
    maximum_evaluating_client_count_per_model: int
    # クロス評価のモデルの送り方。`send_candidate_models_for_cross_evaluation`は、評価するclientへ候補のモデルを
    # 送る（旧と同じ）。`reuse_distributed_models_for_cross_evaluation`は、クロス評価の前に全モデルを配布し、
    # clientは受け取ったモデルを候補として評価する（クロス評価ではメッセージだけを送り、統合が起きたら、
    # 値が変わった代表のモデルを送り直す）。状態の更新は同じで、通信量の記録だけが違う。
    cross_evaluation_model_transfer_policy: str
    # クロス評価の対象を保有するclientか、対象を保有し標本を返せるclientか。
    cross_evaluation_client_selection_policy: str
    # 統合での共有部の扱い。`average_shared_features_with_concept_specific_parts`は、共有部も概念固有部と一緒に
    # 加重平均する（旧と同じ）。`keep_identical_shared_features`は、メンバーの共有部が代表とビット単位で同じなら、
    # 平均せずに代表の値を使う（同じ値の加重平均の、浮動小数点の誤差を入れない）。
    consolidated_shared_feature_policy: str
    # 初期モデルの損失の統計（検出器の基準）の推定の仕方。`resubstitution`は、学習に使った事前学習の標本で、
    # 学習の後に測る（旧と同じ）。`five_fold_cross_validation`は、事前学習の標本の5分割交差検証で、各標本を
    # 学習に使っていないモデルで測る（初期モデルそのものは、どちらも同じ）。
    initial_model_loss_statistics_estimation: str

    def __post_init__(self) -> None:
        for settings_field_name, required_type in _REQUIRED_SETTINGS_TYPE_BY_FIELD_NAME.items():
            component_settings = getattr(self, settings_field_name)
            if type(component_settings) is not required_type:
                raise RunSettingsValidationError(
                    configuration_parameter_name=settings_field_name,
                    specified_parameter_value=component_settings,
                    validation_failure_reason=(
                        f"{required_type.__name__}型の値を指定してください。派生型は受理しません。"
                    ),
                )
            # frozenを回避して組み立てた値も拒否できるよう、各設定の検査をもう一度行う。
            try:
                component_settings.__post_init__()
            except RunSettingsValidationError:
                raise
            except (TypeError, ValueError) as caught_exception:
                raise RunSettingsValidationError(
                    configuration_parameter_name=settings_field_name,
                    specified_parameter_value=component_settings,
                    validation_failure_reason=str(caught_exception),
                ) from caught_exception
        if (
            type(self.hidden_layer_widths) is not tuple
            or not self.hidden_layer_widths
            or any(
                type(hidden_layer_width) is not int or hidden_layer_width < 1
                for hidden_layer_width in self.hidden_layer_widths
            )
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="hidden_layer_widths",
                specified_parameter_value=self.hidden_layer_widths,
                validation_failure_reason="正のbuiltin intの、空でないtupleを指定してください。",
            )
        if (
            type(self.maximum_evaluating_client_count_per_model) is not int
            or self.maximum_evaluating_client_count_per_model < 1
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="maximum_evaluating_client_count_per_model",
                specified_parameter_value=self.maximum_evaluating_client_count_per_model,
                validation_failure_reason="正のbuiltin intを指定してください。",
            )
        if (
            type(self.cross_evaluation_model_transfer_policy) is not str
            or self.cross_evaluation_model_transfer_policy
            not in CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="cross_evaluation_model_transfer_policy",
                specified_parameter_value=self.cross_evaluation_model_transfer_policy,
                validation_failure_reason=(
                    "次のどれかを指定してください: "
                    + "、".join(CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES)
                    + "。"
                ),
            )
        if (
            type(self.cross_evaluation_client_selection_policy) is not str
            or self.cross_evaluation_client_selection_policy
            not in CROSS_EVALUATION_CLIENT_SELECTION_POLICY_NAMES
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="cross_evaluation_client_selection_policy",
                specified_parameter_value=self.cross_evaluation_client_selection_policy,
                validation_failure_reason="次のどれかを指定してください: "
                + "、".join(CROSS_EVALUATION_CLIENT_SELECTION_POLICY_NAMES)
                + "。",
            )
        if (
            type(self.consolidated_shared_feature_policy) is not str
            or self.consolidated_shared_feature_policy
            not in CONSOLIDATED_SHARED_FEATURE_POLICY_NAMES
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="consolidated_shared_feature_policy",
                specified_parameter_value=self.consolidated_shared_feature_policy,
                validation_failure_reason=(
                    "次のどれかを指定してください: "
                    + "、".join(CONSOLIDATED_SHARED_FEATURE_POLICY_NAMES)
                    + "。"
                ),
            )
        if (
            type(self.initial_model_loss_statistics_estimation) is not str
            or self.initial_model_loss_statistics_estimation
            not in INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="initial_model_loss_statistics_estimation",
                specified_parameter_value=self.initial_model_loss_statistics_estimation,
                validation_failure_reason=(
                    "次のどれかを指定してください: "
                    + "、".join(INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES)
                    + "。"
                ),
            )
        pretraining_sample_count = self.initial_model_pretraining_settings.pretraining_sample_count
        if (
            self.initial_model_loss_statistics_estimation
            == FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME
            and pretraining_sample_count < INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="initial_model_loss_statistics_estimation",
                specified_parameter_value=self.initial_model_loss_statistics_estimation,
                validation_failure_reason=(
                    "交差検証には、分割の数以上の事前学習の標本が要ります。"
                    f"pretraining_sample_count={pretraining_sample_count!r}、"
                    f"分割の数={INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT!r}。"
                ),
            )
        maximum_same_cluster_decision_score = (
            self.model_clustering_criteria.maximum_same_cluster_decision_score
        )
        maximum_tolerated_mean_loss_increase = (
            self.run_client_settings.scalar_settings.maximum_tolerated_mean_loss_increase
        )
        if maximum_same_cluster_decision_score != maximum_tolerated_mean_loss_increase:
            raise RunSettingsValidationError(
                configuration_parameter_name="maximum_same_cluster_decision_score",
                specified_parameter_value=maximum_same_cluster_decision_score,
                validation_failure_reason=(
                    "最終FedSDA構成ではクラスタリングの閾値を、clientの許容する平均損失の増加と同値にしてください。"
                    f"maximum_tolerated_mean_loss_increase={maximum_tolerated_mean_loss_increase!r}。"
                ),
            )
        # 共有部を持たないモデル構造は、モデルごとに順に学習する方式と、集約後の重みの再較正をしない方針との組で、
        # 旧実装と照合してある。ほかの組は、受け付けない。
        models_are_independent = (
            self.model_architecture_settings.model_architecture_name
            == "independent_models_without_shared_backbone"
        )
        local_model_parameter_update_strategy = (
            self.run_client_settings.local_training_settings.local_model_parameter_update_strategy
        )
        if models_are_independent != (
            local_model_parameter_update_strategy == "sequential_per_model_training"
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="local_model_parameter_update_strategy",
                specified_parameter_value=local_model_parameter_update_strategy,
                validation_failure_reason=(
                    "sequential_per_model_trainingは、モデル構造が"
                    "independent_models_without_shared_backboneのときだけ、指定してください"
                    "（その構造では、必ず、この方式を指定してください）。"
                ),
            )
        prediction_weight_recalibration_policy = self.run_client_settings.prediction_combination_settings.prediction_weight_recalibration_after_aggregation_policy
        if (
            models_are_independent
            and prediction_weight_recalibration_policy != "keep_weights_without_recalibration"
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="prediction_weight_recalibration_after_aggregation_policy",
                specified_parameter_value=prediction_weight_recalibration_policy,
                validation_failure_reason=(
                    "モデル構造がindependent_models_without_shared_backboneのときは、"
                    "keep_weights_without_recalibrationを指定してください。"
                ),
            )


class FedsdaRunParticipantFactory:
    """単一runの実行の枠へ、最終構成のFedSDAの参加者を準備して渡す。

    準備のたびに、新しい参加者（新しいowner）を作る。直前に準備した参加者は、結果を読むために公開する。
    """

    def __init__(self, *, run_participant_settings: FedsdaRunParticipantSettings) -> None:
        self._run_participant_settings = run_participant_settings
        self._prepared_run_participants: RunParticipants | None = None

    @property
    def prepared_run_participants(self) -> RunParticipants | None:
        """直前に準備した参加者。まだ準備していなければNone。"""
        return self._prepared_run_participants

    def validate_configuration(self) -> None:
        """設定の束を再検査する。乱数を消費せず、参加者を作らない。"""
        run_participant_settings = self._run_participant_settings
        if type(run_participant_settings) is not FedsdaRunParticipantSettings:
            raise RunSettingsValidationError(
                configuration_parameter_name="run_participant_settings",
                specified_parameter_value=run_participant_settings,
                validation_failure_reason="FedsdaRunParticipantSettingsを指定してください。",
            )
        run_participant_settings.__post_init__()

    def prepare_run(
        self,
        *,
        experiment_run_conditions: ExperimentRunConditions,
        run_random_sources: RunRandomSources,
        sample_generator: ObservedSampleGenerator,
    ) -> RunParticipants:
        """初期モデルを事前学習し、全clientを0から順に組み立て、サーバを組み立てて、参加者を返す。

        乱数は、事前学習だけが消費する（分類器の初期化＝torchの全体の乱数、標本の生成＝標本生成器、
        並べ替え＝runのPythonの乱数生成器）。サーバと全clientは、runの同じPythonの乱数生成器を借りる。
        """
        self.validate_configuration()
        for argument_name, argument, required_type in (
            ("experiment_run_conditions", experiment_run_conditions, ExperimentRunConditions),
            ("run_random_sources", run_random_sources, RunRandomSources),
        ):
            if type(argument) is not required_type:
                raise TypeError(f"{argument_name} must be exact {required_type.__name__}")
        if type(sample_generator) not in OBSERVED_SAMPLE_GENERATOR_TYPES:
            raise TypeError("sample_generator must be a generator of a supported dataset")
        # 定義のないdatasetは、ここで拒否する（入力の特徴数とクラス数は、定義から取る）。
        dataset_definition = get_dataset_definition(
            dataset_name=experiment_run_conditions.dataset_name
        )
        if not is_observed_sample_generator_of_dataset(
            sample_generator=sample_generator,
            dataset_name=experiment_run_conditions.dataset_name,
        ):
            raise ValueError("sample_generator must be the generator of the dataset")
        run_participant_settings = self._run_participant_settings
        run_client_settings = run_participant_settings.run_client_settings
        python_random_generator = run_random_sources.python_random_generator
        pretrained_initial_model = pretrain_initial_model(
            initial_model_pretraining_settings=(
                run_participant_settings.initial_model_pretraining_settings
            ),
            model_architecture_settings=run_participant_settings.model_architecture_settings,
            hidden_layer_widths=run_participant_settings.hidden_layer_widths,
            input_feature_count=dataset_definition.input_feature_count,
            class_count=dataset_definition.class_count,
            parameter_optimizer_settings=(
                run_client_settings.rebuilt_model_parameter_optimizer_settings
            ),
            sample_generator=sample_generator,
            python_random_generator=python_random_generator,
        )
        # 初期モデルの損失の統計。交差検証では、一時的な分類器をrunの乱数から切り離した乱数で学習し、統計だけを
        # 替える（初期モデルとrunの乱数は変えない）。
        initial_loss_statistics = pretrained_initial_model.loss_statistics
        if (
            run_participant_settings.initial_model_loss_statistics_estimation
            == FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME
        ):
            initial_loss_statistics = estimate_cross_validated_loss_statistics(
                pretraining_samples=pretrained_initial_model.pretraining_samples,
                fold_count=INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT,
                initial_model_pretraining_settings=(
                    run_participant_settings.initial_model_pretraining_settings
                ),
                model_architecture_settings=run_participant_settings.model_architecture_settings,
                input_feature_count=dataset_definition.input_feature_count,
                hidden_layer_widths=run_participant_settings.hidden_layer_widths,
                class_count=dataset_definition.class_count,
                parameter_optimizer_settings=(
                    run_client_settings.rebuilt_model_parameter_optimizer_settings
                ),
            )
        run_clients = tuple(
            assemble_fedsda_run_client(
                client_id=client_id,
                initial_model_id=_INITIAL_MODEL_ID,
                initial_classifier=pretrained_initial_model.classifier,
                initial_concept_specific_parameter_optimizer_state=(
                    pretrained_initial_model.concept_specific_parameter_optimizer_state
                ),
                initial_shared_parameter_optimizer_state=(
                    pretrained_initial_model.shared_parameter_optimizer_state
                ),
                initial_loss_statistics=initial_loss_statistics,
                run_client_settings=run_client_settings,
                python_random_generator=python_random_generator,
            )
            for client_id in range(experiment_run_conditions.client_count)
        )
        run_server = assemble_fedsda_run_server(
            run_clients=run_clients,
            initial_model_id=_INITIAL_MODEL_ID,
            initial_parameter_snapshot=snapshot_classifier_parameters(
                classifier=pretrained_initial_model.classifier
            ),
            initial_loss_statistics=initial_loss_statistics,
            model_clustering_criteria=run_participant_settings.model_clustering_criteria,
            maximum_evaluating_client_count_per_model=(
                run_participant_settings.maximum_evaluating_client_count_per_model
            ),
            cross_evaluation_model_transfer_policy=(
                run_participant_settings.cross_evaluation_model_transfer_policy
            ),
            cross_evaluation_client_selection_policy=run_participant_settings.cross_evaluation_client_selection_policy,
            consolidated_shared_feature_policy=(
                run_participant_settings.consolidated_shared_feature_policy
            ),
            python_random_generator=python_random_generator,
        )
        prepared_run_participants = RunParticipants(
            # clientの操作は、実行の枠が使わない戻り値（処理の結果）と、任意の引数（診断用の真の概念）を持つ。
            # 枠の契約の型へ対応付ける。値は変更しない。
            client_operations=cast(tuple[RunClientOperations, ...], run_clients),
            server_operations=run_server,
        )
        self._prepared_run_participants = prepared_run_participants
        return prepared_run_participants
