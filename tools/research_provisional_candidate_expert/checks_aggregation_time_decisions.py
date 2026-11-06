"""集約時判断の固定入力の境界検査と、計画書の小さいrunの照合。"""

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    MinimumDescriptionLengthAssignmentSelection,
)
from federated_learning_experiments.runtime import fedsda_run_client as client_module
from federated_learning_experiments.runtime import (
    observed_sample_processing as processing,
)
from tools.research_provisional_candidate_expert import (
    aggregation_time_decisions as policy,
)
from tools.research_provisional_candidate_expert import run as research


def job(variant="product-default", dataset="sine2", interval=50, sample_count=1000):
    return dict(
        variant=variant,
        dataset=dataset,
        interval=interval,
        sample_count=sample_count,
        seed=0,
        check=False,
        aggregation_time_sample_diagnostics=True,
        **({"record_aggregation_time_diagnostics": True} if variant == "product-default" else {}),
    )


def _force_alarm(monkeypatch, sample):
    original = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation

    def observe(self, **kwargs):
        return replace(original(self, **kwargs), drift_detected=kwargs["sample_index"] == sample)

    monkeypatch.setattr(
        OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", observe
    )


def _force_new_model(monkeypatch):
    def select(**kwargs):
        return MinimumDescriptionLengthAssignmentSelection(
            selected_hypothesis="new_model",
            change_interval_sample_count=len(kwargs["pending_sample_observations"]),
            destination_model_id=None,
            description_length=0.0,
            new_model_learner_name="warm_started_learner",
        )

    # 実際の選択APIの引数を保持する（研究側はsignatureで受け渡す）。
    from functools import wraps

    select = wraps(processing.select_alarm_interval_assignment)(select)
    monkeypatch.setattr(processing, "select_alarm_interval_assignment", select)


@pytest.mark.parametrize("fail", [False, True])
def test_wrappers_restored_on_exit_and_exception(fail):
    original = client_module.FedsdaRunClient.flush_pending_local_updates
    validate = processing._validate_observed_sample_processing_inputs
    with pytest.raises(LookupError) if fail else nullcontext():
        with policy.aggregation_time_policy(variant="aggregation-time-decisions", diagnostics={}):
            assert client_module.FedsdaRunClient.flush_pending_local_updates is not original
            if fail:
                raise LookupError("復元の検査")
    assert client_module.FedsdaRunClient.flush_pending_local_updates is original
    assert processing._validate_observed_sample_processing_inputs is validate


def test_removed_snapshot_model_is_excluded_and_new_model_is_not_added():
    assert policy.surviving_snapshot_weights({0: 0.2, 1: 0.8}, (0, 2)) == {0: 1.0}
    assert policy.surviving_snapshot_weights({0: 0.2, 1: 0.8}, (0, 1, 2)) == {
        0: 0.2,
        1: 0.8,
    }
    with pytest.raises(RuntimeError):
        policy.surviving_snapshot_weights({0: 1.0}, (1,))


def test_prediction_uses_snapshot_weights_with_current_probabilities():
    synchronized = {0: torch.tensor([[0.1]]), 1: torch.tensor([[0.9]]), 2: torch.tensor([[1.0]])}
    local = {0: torch.tensor([[0.3]]), 1: torch.tensor([[0.7]]), 2: torch.tensor([[1.0]])}
    values = dict(
        synchronized=synchronized,
        local=local,
        weights={0: 0.75, 1: 0.25},
        beta=(0.2, 0.8),
        class_count=2,
    )
    assert torch.allclose(policy.combine_snapshot_predictions(**values), torch.tensor([[0.38]]))
    local[0] = torch.tensor([[0.9]])
    assert torch.allclose(policy.combine_snapshot_predictions(**values), torch.tensor([[0.74]]))


@pytest.mark.parametrize("recalibrates", [True, False])
def test_snapshot_is_taken_once_after_recalibration_or_distribution(monkeypatch, recalibrates):
    expected = []
    method_name = (
        "recalibrate_prediction_state_after_aggregation"
        if recalibrates
        else "apply_global_model_distribution"
    )
    cls = client_module.FedsdaRunClient
    original = getattr(cls, method_name)

    def observe(self, **kwargs):
        result = original(self, **kwargs)
        weights = self.owners.fixed_share_weight_controller.weights_by_model_id
        expected.append(
            (
                self.client_id,
                weights,
                self.synchronized_and_local_prediction_mixture.snapshot_weights(),
            )
        )
        return result

    monkeypatch.setattr(cls, method_name, observe)
    if not recalibrates:
        from federated_learning_experiments.sweep import (
            experiment_configurations as configurations,
        )

        build = configurations.build_final_configuration_fedsda_run_settings

        def settings(**kwargs):
            value = build(**kwargs)
            participant = value.run_participant_settings
            client = participant.run_client_settings
            return replace(
                value,
                run_participant_settings=replace(
                    participant,
                    run_client_settings=replace(
                        client,
                        prediction_combination_settings=replace(
                            client.prediction_combination_settings,
                            prediction_weight_recalibration_after_aggregation_policy="keep_weights_without_recalibration",
                        ),
                    ),
                ),
            )

        monkeypatch.setattr(
            configurations, "build_final_configuration_fedsda_run_settings", settings
        )
    result = research._execute_job(job("aggregation-time-prediction", interval=10, sample_count=20))
    snapshots = [
        s for s in result["aggregation_time_diagnostics"]["snapshots"] if s["source"] != "initial"
    ]
    assert len(snapshots) == len(expected) == 20
    for snapshot, (client_id, weights, beta) in zip(snapshots, expected, strict=True):
        assert snapshot["client"] == client_id and snapshot["beta"] == beta
        assert snapshot["weights"] == weights


def test_four_variants_share_one_cli_execution(tmp_path):
    variants = ["product-default", *policy.AGGREGATION_TIME_VARIANTS]
    destination = tmp_path / "four-variants"
    command = [
        sys.executable,
        "-m",
        "tools.research_provisional_candidate_expert.run",
        "--output",
        str(destination),
        "--datasets",
        "sine2",
        "--intervals",
        "10",
        "--seeds",
        "0",
        "--sample-count",
        "10",
        "--workers",
        "1",
        "--aggregation-time-sample-diagnostics",
        "--variants",
        *variants,
    ]
    plan = subprocess.run([*command, "--print-plan"], capture_output=True, check=True)
    assert len(json.loads(plan.stdout)["jobs"]) == 4
    assert not destination.exists()
    subprocess.run(command, capture_output=True, check=True)
    results = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    assert {r["variant"] for r in results} == set(variants)
    assert not (destination / "failures.json").exists()
    assert all(len(r["aggregation_time_diagnostics"]["samples"]) == 100 for r in results)


def test_cli_plan_enables_default_diagnostics_only_for_comparison(monkeypatch, capsys):
    destination = ".tmp/aggregation-plan-does-not-create-output"
    common = [
        "run",
        "--output",
        destination,
        "--datasets",
        "sine2",
        "--intervals",
        "50",
        "--seeds",
        "0",
        "--print-plan",
        "--variants",
    ]
    monkeypatch.setattr(
        sys, "argv", [*common, "product-default", *policy.AGGREGATION_TIME_VARIANTS]
    )
    research.main()
    jobs = json.loads(capsys.readouterr().out)["jobs"]
    assert len(jobs) == 4
    assert all("aggregation_time_sample_diagnostics" not in j for j in jobs)
    assert next(j for j in jobs if j["variant"] == "product-default")[
        "record_aggregation_time_diagnostics"
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "product-default",
            *policy.AGGREGATION_TIME_VARIANTS,
            "--aggregation-time-sample-diagnostics",
        ],
    )
    research.main()
    assert all(
        j["aggregation_time_sample_diagnostics"]
        for j in json.loads(capsys.readouterr().out)["jobs"]
    )
    monkeypatch.setattr(sys, "argv", [*common, "product-default"])
    research.main()
    assert (
        "record_aggregation_time_diagnostics" not in json.loads(capsys.readouterr().out)["jobs"][0]
    )
    assert not Path(destination).exists()


@pytest.mark.parametrize("variant", ["product-default", *policy.AGGREGATION_TIME_VARIANTS])
def test_default_diagnostics_omit_sample_records_and_keep_identical_summaries(monkeypatch, variant):
    _force_alarm(monkeypatch, 0)
    detailed_job = job(variant, sample_count=50, interval=50)
    detailed = research._execute_job(detailed_job)
    summary_job = dict(detailed_job)
    del summary_job["aggregation_time_sample_diagnostics"]
    summary = research._execute_job(summary_job)
    a, b = (r["aggregation_time_diagnostics"] for r in (detailed, summary))
    assert a["samples"]
    assert "samples" not in b and "detector_waits" not in b
    assert "snapshots" not in b
    assert not b["sample_diagnostics_enabled"]
    assert b["alarm_count"] == 10
    assert sum(b["training_attempt_count_by_client"].values()) == 500
    for key in (
        "selection_count",
        "selection_outcomes",
        "new_model_adoption_count",
        "final_held_model_counts",
        "final_training_counts",
        "training_mass_purity",
        "communication_volume",
        "model_computation_counts",
    ):
        assert key in b
    assert {
        k: v
        for k, v in a.items()
        if k not in ("samples", "detector_waits", "snapshots", "sample_diagnostics_enabled")
    } == {k: v for k, v in b.items() if k != "sample_diagnostics_enabled"}
    assert {
        k: v for k, v in detailed.items() if k not in ("seconds", "aggregation_time_diagnostics")
    } == {k: v for k, v in summary.items() if k not in ("seconds", "aggregation_time_diagnostics")}


def test_single_default_keeps_existing_output_and_four_jobs_can_run_sequentially():
    plain_job = job(sample_count=10, interval=10)
    del plain_job["record_aggregation_time_diagnostics"]
    plain = research._execute_job(plain_job)
    assert "aggregation_time_diagnostics" not in plain
    results = [
        research._execute_job(job(variant, sample_count=10, interval=10))
        for variant in ("product-default", *policy.AGGREGATION_TIME_VARIANTS)
    ]
    assert all(len(r["aggregation_time_diagnostics"]["samples"]) == 100 for r in results)
    assert {
        k: v for k, v in results[0].items() if k not in ("seconds", "aggregation_time_diagnostics")
    } == {k: v for k, v in plain.items() if k != "seconds"}


def test_alarm_leaves_fifo_detector_stops_and_only_boundary_selects(monkeypatch):
    _force_alarm(monkeypatch, 0)
    result = research._execute_job(job("aggregation-time-assignment", interval=50, sample_count=50))
    d = result["aggregation_time_diagnostics"]
    assert len(d["alarms"]) == len(d["selections"]) == 10
    assert len(d["detector_waits"]) == 490
    for alarm in d["alarms"]:
        waits = [w for w in d["detector_waits"] if w["client"] == alarm["client"]]
        assert all(w["alarm"] == 0 and w["state_digest"] == alarm["state_digest"] for w in waits)
    for selection in d["selections"]:
        assert selection["sample"] == 49 and selection["before_aggregation"]
        assert selection["window"] == list(range(20, 50))
        assert selection["pending_alarm"] == 0
        samples = [s for s in d["samples"] if s["client"] == selection["client"]]
        assert [i for s in samples for i in s["released"]] == list(range(20))
        assert samples[-1]["stores"][0] == 20
        # 毎標本1回試行する。batch件数に届かないストアは製品も更新しない。
        assert all(s["training_attempts"] == 1 for s in samples)
        assert all(s["joint_updates"] == 0 for s in samples)
    json.dumps(d, allow_nan=False)


def test_no_pending_alarm_means_no_selection(monkeypatch):
    _force_alarm(monkeypatch, -1)
    result = research._execute_job(job("aggregation-time-assignment", interval=10, sample_count=20))
    d = result["aggregation_time_diagnostics"]
    assert not d["alarms"] and not d["selections"] and not d.get("responses")
    assert [s["after_sample"] for s in d["synchronizations"]] == [9, 19]


def test_short_window_uses_product_short_alarm_response(monkeypatch):
    _force_alarm(monkeypatch, 0)
    result = research._execute_job(job("aggregation-time-assignment", interval=4, sample_count=8))
    d = result["aggregation_time_diagnostics"]
    assert len(d["selections"]) == 10
    first = [r for r in d["responses"] if r["sample"] == 3]
    assert len(first) == 10 and all(r["outcome"] == "wait_for_minimum" for r in first)
    assert all(s["pending_alarm"] == 0 for s in d["samples"])
    assert all(s["sample"] == 7 for s in d["selections"])


def test_empty_window_clears_wait_without_selection_and_resets_detector(monkeypatch):
    captured = {}

    class Captured(BaseException):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured

    with monkeypatch.context() as m:
        m.setattr(client_module, "process_observed_sample", capture)
        m.setattr(research, "aggregation_time_policy", lambda **kwargs: nullcontext())
        with pytest.raises(Captured):
            research._execute_job(job(sample_count=10, interval=10))
    monitor = captured["loss_change_monitor"]
    monitor.observe_loss_after_label_observation(
        observed_loss=1.0,
        observed_class_id=0,
        sample_index=0,
        current_model_baseline_loss_mean=0.1,
    )
    assert monitor.last_observation is not None
    monkeypatch.setattr(
        processing,
        "select_alarm_interval_assignment",
        lambda **kwargs: pytest.fail("空のFIFOで選択しました"),
    )
    assert policy.resolve_pending_alarm(captured) is None
    assert monitor.last_observation is None


def test_new_model_registration_and_distribution_match_last_sample_creation(
    monkeypatch,
):
    results = []
    for variant, sample in (("product-default", 9), ("aggregation-time-assignment", 0)):
        with monkeypatch.context() as m:
            _force_alarm(m, sample)
            _force_new_model(m)
            results.append(research._execute_job(job(variant, interval=10, sample_count=30)))
    a, b = (r["aggregation_time_diagnostics"] for r in results)
    assert len(a["creations"]) == len(b["creations"]) == 10
    assert a["creations"] == b["creations"]
    assert [(s["round"], s["registered"]) for s in a["synchronizations"]] == [
        (s["round"], s["registered"]) for s in b["synchronizations"]
    ]
    assert any(s["registered"] for s in a["synchronizations"])
    assert a["distributions"] == b["distributions"]


def _assert_freezing(diagnostics):
    assert diagnostics["sample_diagnostics_enabled"] and diagnostics["samples"]
    snapshots = diagnostics["snapshots"]
    for sample in diagnostics["samples"]:
        eligible = [
            s
            for s in snapshots
            if s["client"] == sample["client"]
            and (s["after_sample"] is None or s["after_sample"] < sample["sample"])
        ]
        snapshot = eligible[-1]
        assert sample["weights"] == snapshot["weights"]
        assert sample["beta"] == snapshot["beta"]


@pytest.fixture(
    scope="module",
    params=[("sine2", 50), ("sine2", 200), ("mnist4", 50), ("mnist4", 200)],
)
def small_runs(request):
    dataset, interval = request.param
    return [
        research._execute_job(job(v, dataset=dataset, interval=interval))
        for v in (
            "product-default",
            "aggregation-time-prediction",
            "aggregation-time-assignment",
        )
    ]


@pytest.mark.slow
def test_prediction_small_runs_keep_assignment_training_and_communication(small_runs):
    default, predicted, _ = small_runs
    a, b = (r["aggregation_time_diagnostics"] for r in (default, predicted))
    _assert_freezing(b)
    assert a["snapshots"] == b["snapshots"]
    for field in (
        "alarms",
        "selections",
        "creations",
        "candidate_training",
        "synchronizations",
        "distributions",
        "final_training_counts",
        "training_mass_by_client_model_concept",
    ):
        assert a[field] == b[field], field
    for x, y in zip(a["samples"], b["samples"], strict=True):
        assert {k: v for k, v in x.items() if k not in ("weights", "beta")} == {
            k: v for k, v in y.items() if k not in ("weights", "beta")
        }
    for key in (
        "communication_volume",
        "server_parameter_digest",
        "candidate_adoption_count",
        "detection_count",
        "model_computation_counts",
    ):
        assert default[key] == predicted[key], key


@pytest.mark.slow
def test_assignment_small_runs_follow_state_transitions(small_runs):
    result = small_runs[2]
    d = result["aggregation_time_diagnostics"]
    interval = result["interval"]
    assert len(d["samples"]) == 10000
    assert [s["after_sample"] for s in d["synchronizations"]] == list(
        range(interval - 1, 1000, interval)
    )
    assert all(
        s["before_aggregation"] and (s["sample"] + 1) % interval == 0 and len(s["window"]) <= 30
        for s in d["selections"]
    )
    assert all(
        w["state_digest"]
        == next(
            a["state_digest"]
            for a in d["alarms"]
            if a["client"] == w["client"] and a["sample"] == w["alarm"]
        )
        for w in d["detector_waits"]
    )
    assert all(s["joint_updates"] in (0, 1) for s in d["samples"])
    assert all(s["training_attempts"] == 1 for s in d["samples"])


@pytest.mark.slow
def test_combined_small_run():
    result = research._execute_job(job("aggregation-time-decisions", interval=200))
    d = result["aggregation_time_diagnostics"]
    _assert_freezing(d)
    assert d["selections"]
    assert all(s["before_aggregation"] and (s["sample"] + 1) % 200 == 0 for s in d["selections"])


@pytest.mark.slow
def test_product_default_diagnostics_do_not_change_product_results(monkeypatch):
    measured = research._execute_job(job())
    with monkeypatch.context() as m:
        m.setattr(research, "aggregation_time_policy", lambda **kwargs: nullcontext())
        plain = research._execute_job(job())
    assert {
        k: v for k, v in measured.items() if k not in ("seconds", "aggregation_time_diagnostics")
    } == {k: v for k, v in plain.items() if k != "seconds"}


@pytest.fixture(scope="module")
def saved_baseline16_results():
    path = os.environ.get("FDE_AGGREGATION_TIME_REFERENCE_RESULTS")
    if path is None:
        pytest.skip(
            "同条件の保存済み基準版16 results.json を FDE_AGGREGATION_TIME_REFERENCE_RESULTS で指定してください"
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset",
    [
        "sine2",
        "circle2",
        "sea2",
        "sea4",
        "mnist2",
        "mnist4",
        "fmnist2",
        "fmnist4",
        "rotmnist2",
        "rotmnist4",
    ],
)
@pytest.mark.parametrize("interval", [50, 200])
def test_product_default_matches_saved_baseline16(saved_baseline16_results, dataset, interval):
    matching = [
        r
        for r in saved_baseline16_results
        if r["dataset"] == dataset
        and r["interval"] == interval
        and r["seed"] == 0
        and r["sample_count"] == 1000
        and r["variant"] in ("product-default", "every-sample-change-interval-search-configuration")
    ]
    assert len(matching) == 1, "保存済み結果には同条件の基準版16を1件だけ含めてください"
    expected = matching[0]
    actual = research._execute_job(job(dataset=dataset, interval=interval))
    for key in (
        "accuracy",
        "prediction_record_digest",
        "communication_volume",
        "model_computation_counts",
        "registered_model_count",
        "candidate_adoption_count",
        "detection_count",
        "server_parameter_digest",
        "alarm_interval_assignment_selection",
    ):
        assert actual[key] == expected[key], key
