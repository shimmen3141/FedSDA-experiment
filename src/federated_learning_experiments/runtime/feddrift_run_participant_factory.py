"""FedDriftのrunごとの全参加者を準備する。"""

from dataclasses import dataclass
from random import Random

from numpy.random import RandomState

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.observed_sample_generation import (
    ObservedSampleGenerator,
    is_observed_sample_generator_of_dataset,
)
from federated_learning_experiments.execution.run_random_sources import RunRandomSources
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    INDEPENDENT_MODELS_ARCHITECTURE_NAME,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.runtime.feddrift_run_client import (
    FeddriftRunClient,
    assemble_feddrift_run_client,
)
from federated_learning_experiments.runtime.feddrift_run_client_settings import (
    FeddriftRunClientSettings,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.runtime.feddrift_server_settings import FeddriftServerSettings
from federated_learning_experiments.runtime.initial_model_pretraining import pretrain_initial_model


@dataclass(frozen=True, kw_only=True)
class FeddriftRunParticipantSettings:
    """初期準備に必要な既存設定の束。"""

    run_client_settings: FeddriftRunClientSettings
    server_settings: FeddriftServerSettings
    initial_model_pretraining_settings: InitialModelPretrainingSettings
    hidden_layer_widths: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, required_type in (
            ("run_client_settings", FeddriftRunClientSettings),
            ("server_settings", FeddriftServerSettings),
            ("initial_model_pretraining_settings", InitialModelPretrainingSettings),
        ):
            settings = getattr(self, name)
            if type(settings) is not required_type:
                raise TypeError(f"{name}にはexact {required_type.__name__}が必要です。")
            settings.__post_init__()
        if (
            type(self.hidden_layer_widths) is not tuple
            or not self.hidden_layer_widths
            or any(type(width) is not int or width < 1 for width in self.hidden_layer_widths)
        ):
            raise ValueError("隠れ層の幅には正のbuiltin intの非空tupleが必要です。")
        if (
            self.server_settings.initial_model_aggregation_weight
            != self.initial_model_pretraining_settings.pretraining_sample_count
        ):
            raise ValueError("初期集約重みを事前学習標本数と一致させてください。")
        if (
            self.run_client_settings.loss_increase_threshold
            != self.server_settings.maximum_same_cluster_loss_increase
        ):
            raise ValueError("client検出とserver距離の閾値を一致させてください。")


@dataclass(frozen=True, kw_only=True)
class FeddriftRunParticipants:
    """全clientとserverへの参照を固定する。"""

    run_clients: tuple[FeddriftRunClient, ...]
    run_server: FeddriftRunServer

    def __post_init__(self) -> None:
        if (
            type(self.run_clients) is not tuple
            or not self.run_clients
            or any(type(client) is not FeddriftRunClient for client in self.run_clients)
            or type(self.run_server) is not FeddriftRunServer
        ):
            raise TypeError("exact FedDrift clientの非空tupleとserverが必要です。")
        if any(
            type(client.client_id) is not int or client.client_id != index
            for index, client in enumerate(self.run_clients)
        ):
            raise ValueError("client IDは0から連続する昇順にしてください。")
        if self.run_server.run_clients is not self.run_clients:
            raise ValueError("serverは同じclient tupleを参照する必要があります。")


class FeddriftRunParticipantFactory:
    """初期準備を行い、直前に成功した集合だけを公開する。"""

    def __init__(self, *, run_participant_settings: FeddriftRunParticipantSettings) -> None:
        self._settings = run_participant_settings
        self._prepared: FeddriftRunParticipants | None = None

    @property
    def prepared_run_participants(self) -> FeddriftRunParticipants | None:
        return self._prepared

    def validate_configuration(self) -> None:
        """生成や乱数消費なしで設定の束を再検査する。"""
        if type(self._settings) is not FeddriftRunParticipantSettings:
            raise TypeError("exact FeddriftRunParticipantSettingsが必要です。")
        self._settings.__post_init__()

    def validate_run_conditions(
        self, *, experiment_run_conditions: ExperimentRunConditions
    ) -> None:
        """生成前にrunと検出chunkの対応を再検査する。"""
        self.validate_configuration()
        if type(experiment_run_conditions) is not ExperimentRunConditions:
            raise TypeError("exact ExperimentRunConditionsが必要です。")
        experiment_run_conditions.__post_init__()
        if (
            experiment_run_conditions.server_aggregation_interval_per_client_samples
            != self._settings.run_client_settings.detection_batch_sample_count
        ):
            raise ValueError("FedDriftの集約間隔は検出バッチ件数と一致させてください。")

    def prepare_run(
        self,
        *,
        experiment_run_conditions: ExperimentRunConditions,
        run_random_sources: RunRandomSources,
        sample_generator: ObservedSampleGenerator,
    ) -> FeddriftRunParticipants:
        """概念0の事前学習後にclient、serverを組み立てる。seedは呼出元が設定する。"""
        self.validate_run_conditions(experiment_run_conditions=experiment_run_conditions)
        if type(run_random_sources) is not RunRandomSources:
            raise TypeError("exact RunRandomSourcesが必要です。")
        if (
            type(run_random_sources.python_random_generator) is not Random
            or type(run_random_sources.numpy_random_generator) is not RandomState
        ):
            raise TypeError("runの乱数源にはexact Random/RandomStateが必要です。")
        definition = get_dataset_definition(dataset_name=experiment_run_conditions.dataset_name)
        if not is_observed_sample_generator_of_dataset(
            sample_generator=sample_generator,
            dataset_name=experiment_run_conditions.dataset_name,
        ):
            raise ValueError("標本生成器をrunのdatasetと一致させてください。")
        if (
            getattr(sample_generator, "numpy_random_generator", None)
            is not run_random_sources.numpy_random_generator
        ):
            raise ValueError("生成器はrunの同じNumPy乱数源を借用してください。")
        settings = self._settings
        initial = pretrain_initial_model(
            initial_model_pretraining_settings=settings.initial_model_pretraining_settings,
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name=INDEPENDENT_MODELS_ARCHITECTURE_NAME,
                residual_adapter_requested_rank=1,
            ),
            input_feature_count=definition.input_feature_count,
            hidden_layer_widths=settings.hidden_layer_widths,
            class_count=definition.class_count,
            parameter_optimizer_settings=settings.server_settings.rebuilt_model_optimizer_settings,
            sample_generator=sample_generator,
            python_random_generator=run_random_sources.python_random_generator,
        )
        clients = tuple(
            assemble_feddrift_run_client(
                client_id=client_id,
                initial_classifier=initial.classifier,
                initial_parameter_optimizer_state=initial.concept_specific_parameter_optimizer_state,
                initial_loss_statistics=initial.loss_statistics,
                run_client_settings=settings.run_client_settings,
                python_random_generator=run_random_sources.python_random_generator,
            )
            for client_id in range(experiment_run_conditions.client_count)
        )
        server = FeddriftRunServer(
            run_clients=clients,
            server_settings=settings.server_settings,
            initial_parameter_snapshot=snapshot_classifier_parameters(
                classifier=initial.classifier
            ),
            initial_loss_statistics=initial.loss_statistics,
            python_random_generator=run_random_sources.python_random_generator,
        )
        participants = FeddriftRunParticipants(run_clients=clients, run_server=server)
        self._prepared = participants
        return participants
