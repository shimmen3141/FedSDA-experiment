"""研究の基点の組と、実際のrun直前の設定・測定の有無を検査する。"""

from contextlib import contextmanager

import pytest

from federated_learning_experiments.runtime import fedsda_experiment_run as product
from tools import research_loss_monitor_rebaseline as entry
from tools.research_provisional_candidate_expert import run as research

FORWARD = "minimum_interval_multiples_with_forward_prequential_code"
REVERSE = "every_sample_with_reverse_prequential_code"


class _Captured(BaseException):
    pass


def test_variant_groups_partition_all_declared_variants():
    names = set(research._VARIANTS)
    c = set(research.REVERSE_PREQUENTIAL_VARIANTS)
    b = set(research.LOSS_MONITOR_REBASELINE_VARIANTS) | {
        "product-default-all-evaluating-clients",
        "product-default-all-evaluating-clients-candidate-shared-feature-update",
    }
    a = names - set(research._PRODUCT_DEFAULT_BASED_VARIANTS) - c - b
    d = {"product-default", "every-sample-change-interval-search-configuration"}
    e = set(research._PRODUCT_DEFAULT_BASED_VARIANTS) - b - c - d
    groups = [a, b, c, d, e]
    assert set.union(*groups) == names
    assert sum(map(len, groups)) == len(names)
    assert set(entry._VARIANTS) == {"product-default", *research.LOSS_MONITOR_REBASELINE_VARIANTS}


@pytest.mark.parametrize(
    "variant,expected,measured",
    [
        ("default", FORWARD, True),
        ("product-default-loss-monitor-decrease-rebaseline", FORWARD, True),
        ("product-default-all-evaluating-clients", FORWARD, True),
        ("product-default-all-evaluating-clients-candidate-shared-feature-update", FORWARD, True),
        *[(name, FORWARD, True) for name in research.REVERSE_PREQUENTIAL_VARIANTS],
        ("product-default", REVERSE, False),
        ("every-sample-change-interval-search-configuration", REVERSE, False),
        ("product-default-class-conditioned-prediction", REVERSE, False),
    ],
)
def test_cli_passes_group_settings_and_only_measures_forward_or_group_c(
    monkeypatch, variant, expected, measured
):
    contexts = []

    @contextmanager
    def measurement(**kwargs):
        contexts.append("reverse_prequential_policy")
        yield

    def stop(*, run_settings):
        assignment = run_settings.run_participant_settings.run_client_settings.training_data_assignment_settings
        assert assignment.change_interval_search == expected
        assert bool(contexts) is measured
        raise _Captured

    monkeypatch.setattr(research, "reverse_prequential_policy", measurement)
    monkeypatch.setattr(product, "execute_fedsda_experiment_run", stop)
    with pytest.raises(_Captured):
        research._execute_job(
            dict(
                variant=variant,
                dataset="sine2",
                seed=0,
                interval=100,
                sample_count=600,
                check=False,
            )
        )


@pytest.mark.parametrize("variant", entry._VARIANTS)
def test_rebaseline_entry_passes_group_b_or_d_settings(monkeypatch, tmp_path, variant):
    contexts = []

    @contextmanager
    def policy(**kwargs):
        contexts.append(kwargs["variant"])
        yield

    def stop(*, planned_run, **kwargs):
        assignment = planned_run.run_settings.run_participant_settings.run_client_settings.training_data_assignment_settings
        assert assignment.change_interval_search == (
            REVERSE if variant == "product-default" else FORWARD
        )
        assert contexts == [variant]
        raise _Captured

    monkeypatch.setattr(entry, "loss_monitor_rebaseline_policy", policy)
    # _run_variantのfinallyが参照する関数も差し替えるので、製品側の参照をtestの終了時に戻す。
    monkeypatch.setattr(
        entry.experiment_sweep_execution,
        "execute_planned_experiment_run",
        entry.experiment_sweep_execution.execute_planned_experiment_run,
    )
    monkeypatch.setattr(entry, "_execute_planned_experiment_run", stop)
    monkeypatch.setattr(entry, "_provenance", lambda: {})
    arguments = entry._run_parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--datasets",
            "sine2",
            "--intervals",
            "100",
            "--seeds",
            "0",
            "--sample-count",
            "600",
            "--client-count",
            "10",
            "--workers",
            "1",
            "--variants",
            variant,
        ]
    )
    with pytest.raises(_Captured):
        entry._run_variant(arguments, variant)
