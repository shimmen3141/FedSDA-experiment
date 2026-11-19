"""FedHASP（構成`fedsda`）から「細部」を全部外した簡素版のrun設定を、既存の1つ外した構成の差分を重ねて作る。

各「外した構成」について、同じ実行条件の`fedsda`のrun設定との差を、入れ子のdataclassの葉（dataclassでない値）の
単位で取り、全部の差を`fedsda`のrun設定に重ねる。2つの構成が同じ葉を違う値に変えようとしたら止める。
1つの構成だけを重ねた結果は、その構成のrun設定と一致する（検査で確かめる）。製品のコードは変えない。
"""

from dataclasses import fields, is_dataclass, replace

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)

BASE_CONFIGURATION_NAME = "fedsda"
# 基準版16の構成要素の比較（docs/research/fedsda-component-ablation-v16-20261030.md）で、1つずつ外して
# 10データの平均の精度の差が±0.4pp以内だった「細部」と、予測の側の再較正。
DETAIL_ABLATION_CONFIGURATION_NAMES = (
    "fedsda-overall-loss-esr-detection",
    "fedsda-minimum-interval-multiples-change-interval-search",
    "fedsda-loss-distance-clustering",
    "fedsda-without-candidate-validation",
    "fedsda-without-warm-started-learner",
    "fedsda-without-prediction-weight-recalibration",
)
ADAPTER_ABLATION_CONFIGURATION_NAME = "fedsda-without-concept-specific-adapter"
VARIANT_ABLATIONS = {
    "simplified": DETAIL_ABLATION_CONFIGURATION_NAMES,
    "simplified-without-adapter": (*DETAIL_ABLATION_CONFIGURATION_NAMES, ADAPTER_ABLATION_CONFIGURATION_NAME),
}


def leaf_differences(base, other, path=()):
    """2つの設定の、違う葉の{経路: otherの値}。dataclassの型が違えば、その位置を葉とみなす。"""
    if is_dataclass(base) and is_dataclass(other) and type(base) is type(other):
        differences = {}
        for field in fields(base):
            differences.update(
                leaf_differences(getattr(base, field.name), getattr(other, field.name), (*path, field.name))
            )
        return differences
    return {} if base == other else {path: other}


def apply_leaf_values(settings, values):
    """{経路: 値}を入れ子のdataclassへ重ねた、新しい設定を返す。"""
    by_head = {}
    for path, value in values.items():
        if not path:
            return value
        by_head.setdefault(path[0], {})[path[1:]] = value
    return replace(
        settings,
        **{
            name: apply_leaf_values(getattr(settings, name), nested)
            for name, nested in by_head.items()
        },
    )


def merged_ablation_differences(*, configuration_names, experiment_run_conditions):
    """各構成の`fedsda`との差を集める。同じ葉を違う値に変える構成があれば止める。"""
    base = build_experiment_run_settings(
        configuration_name=BASE_CONFIGURATION_NAME, experiment_run_conditions=experiment_run_conditions
    )
    merged = {}
    sources = {}
    for name in configuration_names:
        other = build_experiment_run_settings(
            configuration_name=name, experiment_run_conditions=experiment_run_conditions
        )
        if type(other) is not type(base):
            raise ValueError(f"{name}のrun設定の型が{BASE_CONFIGURATION_NAME}と違います")
        differences = leaf_differences(base, other)
        if not differences:
            raise ValueError(f"{name}は{BASE_CONFIGURATION_NAME}と同じrun設定です")
        for path, value in differences.items():
            if path in merged and merged[path] != value:
                raise ValueError(f"{sources[path]}と{name}が同じ設定{'.'.join(path)}を違う値に変えます")
            for existing in merged:
                if existing != path and (existing[: len(path)] == path or path[: len(existing)] == existing):
                    raise ValueError(f"{sources[existing]}と{name}の差が入れ子で重なります: {'.'.join(path)}")
            merged[path] = value
            sources[path] = name
    return base, merged, sources


def build_simplified_run_settings(*, variant, experiment_run_conditions):
    if variant not in VARIANT_ABLATIONS:
        raise ValueError(f"知らない変種です: {variant}")
    if type(experiment_run_conditions) is not ExperimentRunConditions:
        raise TypeError("experiment_run_conditionsはExperimentRunConditionsが必要です")
    base, merged, _ = merged_ablation_differences(
        configuration_names=VARIANT_ABLATIONS[variant], experiment_run_conditions=experiment_run_conditions
    )
    return apply_leaf_values(base, merged)
