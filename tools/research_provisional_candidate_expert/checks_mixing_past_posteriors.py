"""E1a専用の検証。通常のtestsへ組み込まず、明示実行する。"""

from copy import deepcopy
from math import exp
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from tools.research_provisional_candidate_expert.mixing_past_posteriors import (
    mixing_past_posteriors_policy,
)


def controller():
    return FixedShareWeightController(
        prediction_combination_settings=PredictionCombinationSettings(
            prediction_combination_strategy="fixed_share_weighted_prediction",
            prediction_mixture_activation_policy="always",
            prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
            prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
            prediction_model_parameter_source="mixture_of_synchronized_and_local_parameters",
            fixed_share_weight_redistribution_time_scale_samples=10,
        )
    )


def observe(c, losses=None):
    losses = {0: 0.0, 1: 1.0} if losses is None else losses
    c.update_weights_after_loss_observation(
        observed_losses_by_model_id=losses,
        prediction_weights_by_model_id=c.get_weights_before_label_observation(
            model_ids=losses
        ),
    )


def test_first_update_matches_declared_ema_mixture():
    c = controller()
    diagnostics = {}
    with mixing_past_posteriors_policy(
        memory_fraction=0.5, diagnostics=diagnostics
    ) as memory:
        observe(c)
        p = 1.0 / (1.0 + exp(-1.0))
        r = 0.9 * 0.5 + 0.1 * p
        assert memory[c][0] == pytest.approx(r)
        assert c.weights_by_model_id[0] == pytest.approx(
            0.9 * p + 0.1 * (0.5 * r + 0.25)
        )
        assert diagnostics["real_observation_updates"] == 1


def test_replay_and_reset_do_not_accumulate_memory():
    c = controller()
    diagnostics = {}
    with mixing_past_posteriors_policy(
        memory_fraction=0.5, diagnostics=diagnostics
    ) as memory:
        for _ in range(8):
            observe(c)
        previous = deepcopy(memory[c])
        c.reset_weights_after_aggregation()
        c.replay_observed_losses_after_aggregation(
            observed_loss_sequence=({0: 1.0, 1: 0.0},) * 3
        )
        assert memory[c] == previous
        assert diagnostics["real_observation_updates"] == 8
        assert diagnostics["replay_updates"] == 3
        assert c.aggregation_recalibration_count == 2
        assert sum(c.weights_by_model_id.values()) == pytest.approx(1.0)


def test_disabled_policy_is_exact_and_restores_patches():
    expected, actual = controller(), controller()
    for _ in range(8):
        observe(expected)
    original = FixedShareWeightController.update_weights_after_loss_observation
    with mixing_past_posteriors_policy(memory_fraction=0.0, diagnostics={}):
        for _ in range(8):
            observe(actual)
    assert actual.__dict__ == expected.__dict__
    assert FixedShareWeightController.update_weights_after_loss_observation is original


def test_pool_change_projects_memory_without_resurrecting_removed_ids():
    c = controller()
    with mixing_past_posteriors_policy(memory_fraction=0.5, diagnostics={}) as memory:
        observe(c)
        c.get_weights_before_label_observation(model_ids=(0, 2))
        assert memory[c] == {0: 1.0, 2: 0.0}
        c.get_weights_before_label_observation(model_ids=(0, 1, 2))
        assert memory[c] == {0: 1.0, 1: 0.0, 2: 0.0}


def test_canonical_inputs_and_leader_switch_count():
    a, b = controller(), controller()
    expected_switches = 0
    with mixing_past_posteriors_policy(memory_fraction=0.5, diagnostics={}):
        for index in range(20):
            losses = {0: index % 2, 1: (index + 1) % 2}
            previous = a.select_maximum_weight_model_id(
                prediction_weights_by_model_id=a.get_weights_before_label_observation(
                    model_ids=losses
                )
            )
            observe(a, losses)
            observe(b, {k: float(v) for k, v in losses.items()})
            final = a.select_maximum_weight_model_id(
                prediction_weights_by_model_id=a.weights_by_model_id
            )
            expected_switches += previous != final
            assert a.leader_switch_count == expected_switches
            assert a.weights_by_model_id == b.weights_by_model_id


def test_registration_is_scoped_and_merge_remaps_before_replay():
    import federated_learning_experiments.runtime.observed_sample_processing as processing
    import federated_learning_experiments.runtime.server_model_registration_and_aggregation as registration
    from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

    a, b = controller(), controller()
    ra = SimpleNamespace(
        snapshot_ordered_held_model_training_states=lambda: (
            SimpleNamespace(model_id=-100),
        )
    )
    rb = SimpleNamespace(
        snapshot_ordered_held_model_training_states=lambda: (
            SimpleNamespace(model_id=-100),
        )
    )
    client = SimpleNamespace(owners=SimpleNamespace(fixed_share_weight_controller=a))

    def confirm(**kw):
        kw[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states = lambda: (
            SimpleNamespace(model_id=3),
        )

    observed = {}

    def distribute(self, **kw):
        # 配布の中で再較正される時点で、統合済みの記憶が必要。
        observed.update(memory[a])
        a.replay_observed_losses(observed_loss_sequence=({3: 0.0},))

    with (
        patch.object(registration, "confirm_held_model_registration", confirm),
        patch.object(
            processing,
            "predict_observed_sample_and_update_prediction_weights",
            lambda **kw: None,
        ),
        patch.object(FedsdaRunClient, "apply_global_model_distribution", distribute),
    ):
        with mixing_past_posteriors_policy(
            memory_fraction=0.5, diagnostics={}
        ) as memory:
            for c, registry in ((a, ra), (b, rb)):
                processing.predict_observed_sample_and_update_prediction_weights(
                    held_model_training_state_registry=registry,
                    fixed_share_weight_controller=c,
                )
                observe(c, {-100: 0.0, 2: 1.0})
            before_b = deepcopy(memory[b])
            registration.confirm_held_model_registration(
                held_model_training_state_registry=ra
            )
            assert set(memory[a]) == {3, 2}
            assert memory[b] == before_b
            FedsdaRunClient.apply_global_model_distribution(
                client, model_id_mapping={2: 3}
            )
            assert observed == {3: pytest.approx(1.0)}
            assert set(memory[a]) == {3}


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_bad_fraction_is_rejected(fraction):
    with (
        pytest.raises(ValueError),
        mixing_past_posteriors_policy(memory_fraction=fraction, diagnostics={}),
    ):
        pass


@pytest.mark.parametrize("dataset", ["sine2", "circle2", "sea4", "mnist2"])
def test_disabled_full_run_matches_product(dataset):
    from tools.research_provisional_candidate_expert.run import _execute_job

    job = dict(
        dataset=dataset,
        interval=100,
        seed=0,
        sample_count=600,
        check=False,
        record_prediction_period_metrics=True,
    )
    expected = _execute_job({**job, "variant": "product-default"})
    actual = _execute_job(
        {**job, "variant": "product-default-mixing-past-posteriors-disabled"}
    )
    for record in (actual, expected):
        for key in ("variant", "seconds"):
            record.pop(key)
    assert actual == expected
