"""来歴IDのscope・代表として残る統合・ゼロ重み・実runの非介入を検証する。"""

from types import SimpleNamespace
import json
from pathlib import Path
from uuid import uuid4

import pytest

from tools.research_shadow_model_lifecycle import LifecycleTracker, execute_tracking_job


def add(tracker, client, model, sample=10):
    tracker.positions[client] = sample
    tracker.adopted(
        dict(
            episode=dict(
                adopted_model_id=model,
                episode_id=len(tracker.records),
                adoption_sample_index=sample,
            )
        ),
        dict(
            client_id=client,
            held_model_training_state_registry=None,
            pending_model_upload_state=None,
        ),
    )


def prediction(weights):
    return SimpleNamespace(
        prediction_weights_by_model_id=weights,
        sample_prediction_record=SimpleNamespace(
            maximum_weight_model_id=max(weights, key=weights.get)
        ),
    )


def test_negative_ids_are_client_scoped_and_global_ids_reach_other_clients():
    t = LifecycleTracker()
    add(t, 0, -1)
    add(t, 1, -1)
    t.predicted(0, 11, prediction({-1: 1.0}), (0.25, 0.75))
    assert not t.records[1]["usage"]
    t.registered(0, -1, 4)
    t.predicted(1, 12, prediction({4: 0.25, -1: 0.75}), (0.4, 0.6))
    a = t.records[0]["usage"]["direct:global:1"]
    assert (
        a["weight_sum"] == 0.25
        and a["synchronized_weight_sum"] == 0.1
        and a["local_weight_sum"] == 0.15
    )
    assert t.records[1]["usage"]["direct:temporary:1"]["weight_sum"] == 0.75
    assert (0, -1) not in t.local and (1, -1) in t.local


def test_surviving_representative_is_also_merged_and_lineage_is_not_duplicated():
    t = LifecycleTracker()
    add(t, 0, -1)
    add(t, 1, -1)
    t.registered(0, -1, 0)
    t.registered(1, -1, 4)
    t.consolidated(((0, 4),), {0: 0, 4: 0})
    assert t.global_ids == {0: {0, 1}}
    assert all(r["first_merge"] is not None for r in t.records)
    assert t.records[0]["events"][-1]["absorbed"] is False
    assert t.records[1]["events"][-1]["absorbed"] is True
    t.predicted(2, 20, prediction({0: 1.0}), (0.5, 0.5))
    assert all(list(r["usage"]) == ["descendant:global:2"] for r in t.records)
    t.consolidated(((0, 9),), {0: 0, 9: 0})
    assert t.global_ids == {0: {0, 1}}


def test_zero_prediction_weight_is_evaluated_but_not_positive_use():
    t = LifecycleTracker()
    add(t, 0, -1)
    t.predicted(0, 11, prediction({-1: 0.0, 0: 1.0}), (0.2, 0.8))
    u = t.records[0]["usage"]["direct:temporary:0"]
    assert (
        u["evaluated_samples"] == 1
        and u["positive_weight_samples"] == 0
        and u["weight_sum"] == 0
    )
    t.remove_missing_temporary(0, {0})
    assert not t.local and t.records[0]["events"][-1]["kind"] == "temporary_removed"


def test_reference_subset_rejects_missing_and_changed_pairs(monkeypatch):
    from tools.research_shadow_learner_diagnostic import run_followup
    from tools.research_shadow_model_lifecycle import analyze

    monkeypatch.setattr(run_followup, "analyze", lambda output: None)
    # Windows sandboxでpytestのmode=0700一時dirを読めないため、通常権限で研究成果物内へ作る。
    tmp_path = (
        Path(__file__).resolve().parents[1] / "results" / "_test_tmp" / uuid4().hex
    )
    tmp_path.mkdir(parents=True)
    reference = tmp_path / "reference"
    reference.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    job = dict(dataset="sea4", stream="drift", interval=100, seed=1)
    row = dict(
        job=job,
        baseline=dict(accuracy=0.8, seconds=1),
        measured=dict(accuracy=0.9, seconds=2),
        shadow_diagnostic=dict(episodes=[]),
        model_lifecycle=[],
    )

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    write(
        output / "research-source.json",
        dict(
            sha256={}, jobs=[job], model_lifecycle=dict(reference_result=str(reference))
        ),
    )
    write(output / "results.json", [row])
    extra = {**row, "job": {**job, "seed": 2}}
    write(reference / "results.json", [row, extra])
    analyze(output)
    assert (
        json.loads((output / "summary_lifecycle.json").read_text())[
            "reference_pairs_verified"
        ]
        == 1
    )
    write(reference / "results.json", [extra])
    with pytest.raises(KeyError):
        analyze(output)
    write(reference / "results.json", [{**row, "measured": dict(accuracy=0.1)}])
    with pytest.raises(ValueError, match="全保存指標"):
        analyze(output)
    write(reference / "results.json", [row, row])
    with pytest.raises(ValueError, match="重複"):
        analyze(output)


@pytest.mark.parametrize(
    "dataset,stream,fixed_concept,registered",
    [("sea4", "drift", None, False), ("sine2", "fixed1", 1, True)],
)
def test_real_tracking_matches_all_uninstrumented_metrics_and_adoption_path(
    dataset, stream, fixed_concept, registered
):
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    job = dict(
        dataset=dataset,
        stream=stream,
        fixed_concept=fixed_concept,
        interval=100,
        seed=1,
        sample_count=600,
        horizon_samples=240,
        task="adoption",
        disabled=False,
    )
    old = execute_job(job)
    new = execute_tracking_job(job)
    for part in ("baseline", "measured"):
        assert {k: v for k, v in old[part].items() if k != "seconds"} == {
            k: v for k, v in new[part].items() if k != "seconds"
        }
    assert old["shadow_diagnostic"] == new["shadow_diagnostic"]
    records = new["model_lifecycle"]
    assert records and len(records) == sum(
        "adopted_model_id" in e for e in new["shadow_diagnostic"]["episodes"]
    )
    if registered:
        completed = [r for r in records if r["registered_model_id"] is not None]
        assert completed
        for r in completed:
            assert r["first_merge"] is not None
            assert {"registered", "uploaded_update", "consolidated", "distributed"} <= {
                e["kind"] for e in r["events"]
            }
            adopted = next(e for e in r["events"] if e["kind"] == "adopted")
            registration = next(e for e in r["events"] if e["kind"] == "registered")
            upload = next(e for e in r["events"] if e["kind"] == "uploaded_update")
            merge = next(e for e in r["events"] if e["kind"] == "consolidated")
            distribution = next(e for e in r["events"] if e["kind"] == "distributed")
            assert (
                adopted["model_id"]
                == registration["previous_model_id"]
                == r["temporary_model_id"]
            )
            assert (
                registration["model_id"]
                == upload["model_id"]
                == merge["previous_model_id"]
                == r["registered_model_id"]
            )
            assert upload["sending_client_id"] == r["client_id"]
            assert upload["training_samples"] > 0
            assert distribution["model_id"] == merge["model_id"]
            assert (
                adopted["event_index"]
                < registration["event_index"]
                < upload["event_index"]
                < merge["event_index"]
                < distribution["event_index"]
            )
    else:
        # この短いSEA4 runでは未登録の保留と予測利用を打切りとして残す。
        assert all(r["registered_model_id"] is None for r in records)
        assert all(
            r["temporary_model_id"] in r["end_origin_pending_ids"] for r in records
        )
    assert any(r["usage"] for r in records)
