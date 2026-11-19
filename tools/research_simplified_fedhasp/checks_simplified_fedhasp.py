"""簡素版FedHASPの研究ツールの検査（明示実行。通常のpytestの対象外）。

    python -m pytest tools/research_simplified_fedhasp/checks_simplified_fedhasp.py -q -p no:cacheprovider
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from federated_learning_experiments.configuration.experiment_run_conditions import (  # noqa: E402
    ExperimentRunConditions,
)
from federated_learning_experiments.sweep import experiment_sweep_plan  # noqa: E402
from federated_learning_experiments.sweep.experiment_configurations import (  # noqa: E402
    build_experiment_run_settings,
)
from tools.research_simplified_fedhasp import settings as simplified  # noqa: E402
from tools.research_simplified_fedhasp.run import make_patched_build  # noqa: E402

CONDITIONS = [
    ExperimentRunConditions(
        dataset_name=dataset,
        random_seed=seed,
        client_count=10,
        per_client_sample_count=5000,
        server_aggregation_interval_per_client_samples=interval,
    )
    for dataset, interval, seed in (("sine2", 50, 0), ("mnist4", 500, 3), ("fmnist2", 50, 7), ("sea4", 500, 1))
]
ALL_ABLATIONS = (
    *simplified.VARIANT_ABLATIONS["simplified-without-adapter"],
    "fedsda-local-parameter-prediction",
    "fedsda-single-model-prediction",
)


@pytest.mark.parametrize("conditions", CONDITIONS, ids=lambda c: c.dataset_name)
@pytest.mark.parametrize("configuration_name", ALL_ABLATIONS)
def test_one_ablation_reproduces_that_configuration_exactly(conditions, configuration_name):
    base, merged, _ = simplified.merged_ablation_differences(
        configuration_names=(configuration_name,), experiment_run_conditions=conditions
    )
    rebuilt = simplified.apply_leaf_values(base, merged)
    expected = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=conditions
    )
    assert rebuilt == expected


@pytest.mark.parametrize("conditions", CONDITIONS, ids=lambda c: c.dataset_name)
def test_simplified_differs_from_fedhasp_exactly_by_the_union_of_ablations(conditions):
    base = build_experiment_run_settings(configuration_name="fedsda", experiment_run_conditions=conditions)
    expected = {}
    for name in simplified.VARIANT_ABLATIONS["simplified-without-adapter"]:
        expected.update(
            simplified.leaf_differences(
                base, build_experiment_run_settings(configuration_name=name, experiment_run_conditions=conditions)
            )
        )
    for variant, names in simplified.VARIANT_ABLATIONS.items():
        if variant not in ("simplified", "simplified-without-adapter"):
            continue
        settings = simplified.build_simplified_run_settings(variant=variant, experiment_run_conditions=conditions)
        differences = simplified.leaf_differences(base, settings)
        wanted = {}
        for name in names:
            wanted.update(
                simplified.leaf_differences(
                    base, build_experiment_run_settings(configuration_name=name, experiment_run_conditions=conditions)
                )
            )
        assert differences == wanted and differences
        # adapterの差だけが2つの変種を分ける。
        assert set(expected) - set(simplified.leaf_differences(base, settings)) == (
            set()
            if variant == "simplified-without-adapter"
            else set(
                simplified.leaf_differences(
                    base,
                    build_experiment_run_settings(
                        configuration_name=simplified.ADAPTER_ABLATION_CONFIGURATION_NAME,
                        experiment_run_conditions=conditions,
                    ),
                )
            )
        )


def test_conflicting_and_nested_differences_are_rejected(monkeypatch):
    @dataclass(frozen=True)
    class Inner:
        a: int = 0
        b: int = 0

    @dataclass(frozen=True)
    class Outer:
        inner: Inner = Inner()
        c: int = 0

    table = {
        "fedsda": Outer(),
        "x": Outer(inner=Inner(a=1)),
        "y": Outer(inner=Inner(a=2)),
        "z": Outer(inner=Inner(a=1), c=3),
        "same": Outer(),
    }
    monkeypatch.setattr(
        simplified,
        "build_experiment_run_settings",
        lambda *, configuration_name, experiment_run_conditions: table[configuration_name],
    )
    with pytest.raises(ValueError, match="違う値"):
        simplified.merged_ablation_differences(configuration_names=("x", "y"), experiment_run_conditions=None)
    with pytest.raises(ValueError, match="同じrun設定"):
        simplified.merged_ablation_differences(configuration_names=("same",), experiment_run_conditions=None)
    base, merged, _ = simplified.merged_ablation_differences(
        configuration_names=("x", "z"), experiment_run_conditions=None
    )
    assert simplified.apply_leaf_values(base, merged) == Outer(inner=Inner(a=1), c=3)


def test_patched_plan_uses_simplified_settings_only_for_fedsda():
    build = make_patched_build("simplified")
    conditions = CONDITIONS[0]
    assert build(configuration_name="fedsda", experiment_run_conditions=conditions) == (
        simplified.build_simplified_run_settings(variant="simplified", experiment_run_conditions=conditions)
    )
    assert build(configuration_name="fedavg", experiment_run_conditions=conditions) == (
        build_experiment_run_settings(configuration_name="fedavg", experiment_run_conditions=conditions)
    )
    plain = experiment_sweep_plan.make_run_identifier(
        configuration_name="fedsda",
        run_settings=build_experiment_run_settings(configuration_name="fedsda", experiment_run_conditions=conditions),
    )
    patched = experiment_sweep_plan.make_run_identifier(
        configuration_name="fedsda",
        run_settings=build(configuration_name="fedsda", experiment_run_conditions=conditions),
    )
    assert plain != patched and plain.rsplit("_", 1)[0] == patched.rsplit("_", 1)[0]


@pytest.mark.parametrize("conditions", CONDITIONS, ids=lambda c: c.dataset_name)
@pytest.mark.parametrize(
    "variant, prediction_configuration",
    [
        ("simplified-local-model-mixture", "fedsda-local-parameter-prediction"),
        ("simplified-hard-prediction", "fedsda-single-model-prediction"),
    ],
)
def test_prediction_ladder_variants_are_simplified_plus_that_prediction(conditions, variant, prediction_configuration):
    """予測の段の変種は、簡素版のrun設定に、その予測の構成のfedsdaとの差を重ねたものと同じ。"""
    base = build_experiment_run_settings(configuration_name="fedsda", experiment_run_conditions=conditions)
    settings = simplified.build_simplified_run_settings(variant=variant, experiment_run_conditions=conditions)
    expected = simplified.leaf_differences(
        base, simplified.build_simplified_run_settings(variant="simplified", experiment_run_conditions=conditions)
    )
    expected.update(
        simplified.leaf_differences(
            base,
            build_experiment_run_settings(configuration_name=prediction_configuration, experiment_run_conditions=conditions),
        )
    )
    assert simplified.leaf_differences(base, settings) == expected

