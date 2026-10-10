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
from federated_learning_experiments.runtime.initial_model_pretraining import pretrain_initial_model

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
                initial_loss_statistics=pretrained_initial_model.loss_statistics,
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
            initial_loss_statistics=pretrained_initial_model.loss_statistics,
            model_clustering_criteria=run_participant_settings.model_clustering_criteria,
            maximum_evaluating_client_count_per_model=(
                run_participant_settings.maximum_evaluating_client_count_per_model
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
