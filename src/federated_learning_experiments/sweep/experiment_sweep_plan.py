"""掃引の計画（構成・dataset・seed・規模・集約間隔の組合せ）と、runの識別子。"""

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from math import isfinite

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedavg_run_settings import FedavgRunSettings
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

# 保有モデルの学習batchの抜き出し方の、掃引の軸の書き方と、設定の値。
_UNIFORM_SELECTION_TEXT = "uniform"
_RECENT_SELECTION_PREFIX = "recent:"
_DECAY_SELECTION_PREFIX = "decay:"
_SELECTION_POLICY_BY_KIND = {
    "uniform": "uniform_over_all_stored_samples",
    "recent": "uniform_over_most_recent_stored_samples",
    "decay": "recency_decayed_stored_samples",
}
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


def parse_fedsda_training_sample_selection(text: str) -> tuple[str, int | None, float | None]:
    """掃引の軸の書き方（`uniform`・`recent:<K>`・`decay:<α>`）を、設定の3つの値へ変換する。

    Kは先頭が0でない10進の数字だけ。αは`0.`に続く1桁以上の10進の数字だけ（0は不可）。それ以外は拒否する。
    Kとbatchの件数の関係は、作ったrun設定の検査が確かめる。
    """
    if type(text) is not str:
        raise TypeError("training sample selection must be builtin str")
    if text == _UNIFORM_SELECTION_TEXT:
        return _SELECTION_POLICY_BY_KIND["uniform"], None, None
    if text.startswith(_RECENT_SELECTION_PREFIX):
        digits = text[len(_RECENT_SELECTION_PREFIX) :]
        if digits.isascii() and digits.isdigit() and not digits.startswith("0"):
            return _SELECTION_POLICY_BY_KIND["recent"], int(digits), None
    if text.startswith(_DECAY_SELECTION_PREFIX):
        number = text[len(_DECAY_SELECTION_PREFIX) :]
        fraction_digits = number[2:]
        if (
            number.startswith("0.")
            and fraction_digits.isascii()
            and fraction_digits.isdigit()
            and fraction_digits.strip("0")
        ):
            return _SELECTION_POLICY_BY_KIND["decay"], None, float(number)
    raise ValueError(
        f"invalid FedSDA training sample selection: {text!r} "
        "(use uniform, recent:<K> with a positive integer K, or decay:<α> written as 0.<digits>)"
    )


def describe_fedsda_training_sample_selection(
    *, policy: str, most_recent_count: int | None, decay_factor: float | None
) -> str:
    """設定の3つの値を、掃引の軸の書き方（`uniform`・`recent:<K>`・`decay:<α>`。αは`repr`）にする。"""
    if policy == _SELECTION_POLICY_BY_KIND["recent"]:
        return f"{_RECENT_SELECTION_PREFIX}{most_recent_count}"
    if policy == _SELECTION_POLICY_BY_KIND["decay"]:
        return f"{_DECAY_SELECTION_PREFIX}{decay_factor!r}"
    return _UNIFORM_SELECTION_TEXT


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
    feddrift_loss_increase_thresholds: tuple[float, ...] | None = None
    # FedSDAの構成の、保有モデルの学習batchの抜き出し方（`uniform`・`recent:<K>`・`decay:<α>`）。
    fedsda_training_sample_selections: tuple[str, ...] | None = None

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
        if self.feddrift_loss_increase_thresholds is not None:
            _validate_distinct_values(
                values=self.feddrift_loss_increase_thresholds,
                value_type=float,
                field_name="feddrift_loss_increase_thresholds",
            )
            if "feddrift" not in self.configuration_names:
                raise ValueError(
                    "feddrift_loss_increase_thresholds requires feddrift configuration"
                )
            if any(
                not isfinite(value) or value < 0 for value in self.feddrift_loss_increase_thresholds
            ):
                raise ValueError("feddrift_loss_increase_thresholds must be finite and nonnegative")
        if self.fedsda_training_sample_selections is not None:
            _validate_distinct_values(
                values=self.fedsda_training_sample_selections,
                value_type=str,
                field_name="fedsda_training_sample_selections",
            )
            parsed_selections = [
                parse_fedsda_training_sample_selection(text)
                for text in self.fedsda_training_sample_selections
            ]
            # 書き方が違っても、同じ値（`decay:0.5`と`decay:0.50`ほか）は重複とする。
            if len(set(parsed_selections)) != len(parsed_selections):
                raise ValueError("fedsda_training_sample_selections must not hold duplicate values")


@dataclass(frozen=True, kw_only=True)
class PlannedExperimentRun:
    """計画の中の1 run。識別子は、保存するファイルの名前になる。"""

    configuration_name: str
    run_identifier: str
    run_settings: FedsdaRunSettings | FeddriftRunSettings | FedavgRunSettings


def make_run_identifier(
    *,
    configuration_name: str,
    run_settings: FedsdaRunSettings | FeddriftRunSettings | FedavgRunSettings,
) -> str:
    """`<dataset>_interval<集約間隔>_seed<seed>_<hash>`を返す。

    hashは、構成の名前と、完全なrun設定（保存用の辞書）から計算する。読める部分が同じでも、
    ほかの条件（規模、設定の値）が違えば、hashが違う。
    """
    if type(configuration_name) is not str:
        raise TypeError("configuration_name must be builtin str")
    if not configuration_name:
        raise ValueError("configuration_name must not be empty")
    if type(run_settings) not in (FedsdaRunSettings, FeddriftRunSettings, FedavgRunSettings):
        raise TypeError(
            "run_settings must be exact FedsdaRunSettings, FeddriftRunSettings or FedavgRunSettings"
        )
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
    # 抜き出し方の軸を当てたFedSDAのrun（構成の名前は多いので、run設定の型で見分ける）。
    built_fedsda_run_count = 0
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
                    if type(run_settings) is FeddriftRunSettings:
                        participant = run_settings.run_participant_settings
                        settings_variants = tuple(
                            replace(
                                run_settings,
                                run_participant_settings=replace(
                                    participant,
                                    run_client_settings=replace(
                                        participant.run_client_settings,
                                        loss_increase_threshold=threshold,
                                    ),
                                    server_settings=replace(
                                        participant.server_settings,
                                        maximum_same_cluster_loss_increase=threshold,
                                    ),
                                ),
                            )
                            for threshold in (
                                sweep_plan.feddrift_loss_increase_thresholds or (0.1,)
                            )
                        )
                    elif (
                        type(run_settings) is FedsdaRunSettings
                        and sweep_plan.fedsda_training_sample_selections is not None
                    ):
                        built_fedsda_run_count += 1
                        participant = run_settings.run_participant_settings
                        client = participant.run_client_settings
                        settings_variants = tuple(
                            replace(
                                run_settings,
                                run_participant_settings=replace(
                                    participant,
                                    run_client_settings=replace(
                                        client,
                                        local_training_settings=replace(
                                            client.local_training_settings,
                                            training_sample_selection_policy=policy,
                                            most_recent_stored_sample_count=most_recent_count,
                                            stored_sample_recency_decay_factor=decay_factor,
                                        ),
                                    ),
                                ),
                            )
                            for policy, most_recent_count, decay_factor in (
                                parse_fedsda_training_sample_selection(text)
                                for text in sweep_plan.fedsda_training_sample_selections
                            )
                        )
                    else:
                        settings_variants = (run_settings,)
                    for settings_variant in settings_variants:
                        planned_runs.append(
                            PlannedExperimentRun(
                                configuration_name=configuration_name,
                                run_identifier=make_run_identifier(
                                    configuration_name=configuration_name,
                                    run_settings=settings_variant,
                                ),
                                run_settings=settings_variant,
                            )
                        )
    if sweep_plan.fedsda_training_sample_selections is not None and not built_fedsda_run_count:
        raise ValueError("fedsda_training_sample_selections requires a FedSDA configuration")
    return tuple(planned_runs)
