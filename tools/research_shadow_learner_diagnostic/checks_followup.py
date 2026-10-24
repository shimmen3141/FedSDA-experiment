"""研究用採用の実経路と、無効化・optimizer対照を確認する。"""

from copy import deepcopy
import torch


def test_real_adoption_preserves_active_shared_and_optimizer_and_pending_accounting(
    monkeypatch,
):
    import tools.research_shadow_learner_diagnostic.adoption as adoption
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    original = adoption.adopt_shadow
    calls = []

    def checked(*, state, arguments):
        a = arguments
        current = (
            a["held_model_training_state_registry"]
            .get_held_model_training_state(
                model_id=a[
                    "current_training_model_assignment"
                ].current_training_model_id
            )
            .classifier
        )
        shared = deepcopy(current.feature_extractor.state_dict())
        optimizer = deepcopy(state["optimizer"].state_dict())
        pending = a[
            "pending_sample_observation_store"
        ].snapshot_pending_sample_observations()
        original(state=state, arguments=a)
        adopted = a["held_model_training_state_registry"].get_held_model_training_state(
            model_id=a["current_training_model_assignment"].current_training_model_id
        )
        assert adopted.classifier is state["learner"]
        assert adopted.classifier.feature_extractor is current.feature_extractor
        assert all(
            torch.equal(v, current.feature_extractor.state_dict()[k])
            for k, v in shared.items()
        )
        after = adopted.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
        assert after["param_groups"] == optimizer["param_groups"]
        for param, values in optimizer["state"].items():
            assert all(
                torch.equal(v, after["state"][param][k]) for k, v in values.items()
            )
        assert (
            a["pending_sample_observation_store"].snapshot_pending_sample_observations()
            == ()
        )
        assert (
            a["pending_training_assignment_buffer"]
            .get_state_snapshot()
            .pending_sample_indices
            == ()
        )
        assert state["episode"]["adoption_pending_sample_count"] == len(pending)
        calls.append(state["episode"]["episode_id"])

    monkeypatch.setattr(adoption, "adopt_shadow", checked)
    r = execute_job(
        dict(
            dataset="sea4",
            stream="drift",
            fixed_concept=None,
            interval=100,
            seed=1,
            sample_count=600,
            horizon_samples=240,
            task="adoption",
            disabled=False,
        )
    )
    assert calls
    assert len(calls) == sum(
        "adopted_model_id" in e for e in r["shadow_diagnostic"]["episodes"]
    )
    assert (
        r["baseline"]["prediction_record_digest"]
        != r["measured"]["prediction_record_digest"]
    )


def test_disabled_adoption_exactly_matches_product():
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    r = execute_job(
        dict(
            dataset="sea4",
            stream="drift",
            fixed_concept=None,
            interval=100,
            seed=1,
            sample_count=600,
            horizon_samples=240,
            task="adoption",
            disabled=True,
        )
    )
    assert r["shadow_diagnostic"]["trace_row_count"] > 0
    assert not any("adopted_model_id" in e for e in r["shadow_diagnostic"]["episodes"])
    assert r["baseline"]["accuracy"] == r["measured"]["accuracy"]


def test_optimizer_control_resets_same_weights_without_product_changes():
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    r = execute_job(
        dict(
            dataset="sea4",
            stream="drift",
            fixed_concept=None,
            interval=100,
            seed=1,
            sample_count=600,
            horizon_samples=240,
            task="optimizer",
            disabled=False,
        )
    )
    d = r["shadow_diagnostic"]
    assert d["trace_row_count"] > 0
    assert (
        d["extra_model_computation_counts"][
            "concept_specific_parameter_optimizer_step_count"
        ]
        == 3 * d["trace_row_count"]
    )
    assert all("reset_control_score" in e for e in d["episodes"])
