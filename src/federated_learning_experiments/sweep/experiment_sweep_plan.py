"""掃引の計画（構成・dataset・seed・規模・集約間隔の組合せ）と、runの識別子。"""

from dataclasses import dataclass
from hashlib import sha256
from json import dumps

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

# 識別子のhashの桁数（16進）。
_RUN_IDENTIFIER_HASH_LENGTH = 12


def _validate_distinct_values(*, values: object, value_type: type, field_name: str) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{field_name} must be exact tuple")
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    for value in values:
        if type(value) is not value_type:
            raise TypeError(f"{field_name} must hold builtin {value_type.__name__}")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not hold duplicates")


@dataclass(frozen=True, kw_only=True)
class ExperimentSweepPlan:
    """掃引する値の一覧。全部の組合せが、1つずつのrunになる。

    client数と、clientあたりの標本数は、全runで共通。値の範囲（知らないdataset、0以下の件数ほか）は、
    runを作るときに、実行条件の検証が確かめる。
    """

    configuration_names: tuple[str, ...]
    dataset_names: tuple[str, ...]
    random_seeds: tuple[int, ...]
    client_count: int
    per_client_sample_count: int
    server_aggregation_intervals_per_client_samples: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_distinct_values(
            values=self.configuration_names, value_type=str, field_name="configuration_names"
        )
        for configuration_name in self.configuration_names:
            if configuration_name not in EXPERIMENT_CONFIGURATION_NAMES:
                raise ValueError(
                    f"unknown experiment configuration: {configuration_name!r} "
                    f"(available: {', '.join(EXPERIMENT_CONFIGURATION_NAMES)})"
                )
        _validate_distinct_values(
            values=self.dataset_names, value_type=str, field_name="dataset_names"
        )
        _validate_distinct_values(
            values=self.random_seeds, value_type=int, field_name="random_seeds"
        )
        _validate_distinct_values(
            values=self.server_aggregation_intervals_per_client_samples,
            value_type=int,
            field_name="server_aggregation_intervals_per_client_samples",
        )
        for count_name in ("client_count", "per_client_sample_count"):
            if type(getattr(self, count_name)) is not int:
                raise TypeError(f"{count_name} must be builtin int")


@dataclass(frozen=True, kw_only=True)
class PlannedExperimentRun:
    """計画の中の1 run。識別子は、保存するファイルの名前になる。"""

    configuration_name: str
    run_identifier: str
    run_settings: FedsdaRunSettings


def make_run_identifier(*, configuration_name: str, run_settings: FedsdaRunSettings) -> str:
    """`<dataset>_interval<集約間隔>_seed<seed>_<hash>`を返す。

    hashは、構成の名前と、完全なrun設定（保存用の辞書）から計算する。読める部分が同じでも、
    ほかの条件（規模、設定の値）が違えば、hashが違う。
    """
    if type(configuration_name) is not str:
        raise TypeError("configuration_name must be builtin str")
    if not configuration_name:
        raise ValueError("configuration_name must not be empty")
    if type(run_settings) is not FedsdaRunSettings:
        raise TypeError("run_settings must be exact FedsdaRunSettings")
    canonical_text = dumps(
        {
            "configuration_name": configuration_name,
            "run_settings": convert_settings_to_plain_mapping(run_settings),
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    settings_hash = sha256(canonical_text.encode("ascii")).hexdigest()[:_RUN_IDENTIFIER_HASH_LENGTH]
    experiment_run_conditions = run_settings.execution_settings.experiment_run_conditions
    return (
        f"{experiment_run_conditions.dataset_name}"
        f"_interval{experiment_run_conditions.server_aggregation_interval_per_client_samples}"
        f"_seed{experiment_run_conditions.random_seed}"
        f"_{settings_hash}"
    )


def build_planned_experiment_runs(
    *, sweep_plan: ExperimentSweepPlan
) -> tuple[PlannedExperimentRun, ...]:
    """計画の全部の組合せを、構成・dataset・集約間隔・seedの順で、runにする。

    実行条件と、完全なrun設定の検証は、ここで、全部のrunについて済む（不正な値は、実行の前に拒否する）。
    """
    if type(sweep_plan) is not ExperimentSweepPlan:
        raise TypeError("sweep_plan must be exact ExperimentSweepPlan")
    # 手で壊した計画も、値を使う前に検査し直す。
    sweep_plan.__post_init__()
    planned_runs: list[PlannedExperimentRun] = []
    for configuration_name in sweep_plan.configuration_names:
        for dataset_name in sweep_plan.dataset_names:
            for (
                server_aggregation_interval
            ) in sweep_plan.server_aggregation_intervals_per_client_samples:
                for random_seed in sweep_plan.random_seeds:
                    run_settings = build_experiment_run_settings(
                        configuration_name=configuration_name,
                        experiment_run_conditions=ExperimentRunConditions(
                            dataset_name=dataset_name,
                            random_seed=random_seed,
                            client_count=sweep_plan.client_count,
                            per_client_sample_count=sweep_plan.per_client_sample_count,
                            server_aggregation_interval_per_client_samples=(
                                server_aggregation_interval
                            ),
                        ),
                    )
                    planned_runs.append(
                        PlannedExperimentRun(
                            configuration_name=configuration_name,
                            run_identifier=make_run_identifier(
                                configuration_name=configuration_name, run_settings=run_settings
                            ),
                            run_settings=run_settings,
                        )
                    )
    return tuple(planned_runs)
