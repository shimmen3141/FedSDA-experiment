"""Fixed-Share予測重みの明示条件・集合・状態所有を確認する。"""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment.expert_routing import SwitchingExpertRouter
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)


def capture_fixed_share_controller_state(*, controller):
    """所有する重み・分散・全計数を独立した診断値で観測する。"""
    return {
        "weights_by_model_id": controller.weights_by_model_id,
        "cumulative_observed_loss_variance": controller.cumulative_observed_loss_variance,
        "model_pool_reset_count": controller.model_pool_reset_count,
        "prediction_weight_leader_switch_count": controller.prediction_weight_leader_switch_count,
        "aggregation_recalibration_count": controller.aggregation_recalibration_count,
        "aggregation_recalibration_sample_count": controller.aggregation_recalibration_sample_count,
    }


def assert_fixed_share_state_matches_reference(*, controller, reference_router):
    """旧基準の重み・分散・全計数を丸めず照合する。"""
    assert capture_fixed_share_controller_state(controller=controller) == {
        "weights_by_model_id": reference_router.weights,
        "cumulative_observed_loss_variance": reference_router.cumulative_variance,
        "model_pool_reset_count": reference_router.pool_reset_count,
        "prediction_weight_leader_switch_count": reference_router.leader_switch_count,
        "aggregation_recalibration_count": reference_router.aggregation_recalibration_count,
        "aggregation_recalibration_sample_count": reference_router.aggregation_recalibration_sample_count,
    }


@pytest.mark.parametrize("fixed_share_weight_redistribution_time_scale_samples", [2, 30, 10000])
def test_fixed_share_prediction_weights_match_reference_before_and_after_each_observation(
    valid_run_settings_mapping,
    fixed_share_weight_redistribution_time_scale_samples,
):
    """損失制限・交代・同率・集合差・単一モデルを各標本の前後で照合する。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=replace(
            valid_run_settings_mapping["prediction_combination_settings"],
            fixed_share_weight_redistribution_time_scale_samples=fixed_share_weight_redistribution_time_scale_samples,
        )
    )
    reference_router = SwitchingExpertRouter(fixed_share_weight_redistribution_time_scale_samples)
    for observed_losses_by_model_id in (
        {2: 0.5, -3: 0.5, 0: 0.5},
        *({2: 1, -3: 0, 0: 0.3} for _observation_index in range(8)),
        *({0: 0, -3: 1, 2: 0.4} for _observation_index in range(8)),
        {2: 1e308, 0: -1e308, -3: 0.2},
        {5: 0, -3: 1},
        {-3: 1e308},
        {-3: -1e308},
        {0: 0, -3: 0},
    ):
        returned_prediction_weights = controller.get_prediction_weights_before_label_observation(
            model_ids=observed_losses_by_model_id,
        )
        reference_prediction_weights = reference_router.probabilities(observed_losses_by_model_id)
        assert returned_prediction_weights == reference_prediction_weights
        assert_fixed_share_state_matches_reference(
            controller=controller, reference_router=reference_router
        )
        controller.update_weights_after_loss_observation(
            observed_losses_by_model_id=observed_losses_by_model_id,
            prediction_weights_by_model_id=returned_prediction_weights,
        )
        reference_router.update(observed_losses_by_model_id, reference_prediction_weights)
        assert_fixed_share_state_matches_reference(
            controller=controller, reference_router=reference_router
        )


@pytest.mark.parametrize(
    "prediction_weights_by_model_id",
    [
        {0: 0.0, -1: 1.0},
        {0: 0.8, -1: 0.2},
        {0: 0.5, -1: 0.5 + 5e-13},
    ],
)
def test_fixed_share_updates_use_the_supplied_pre_observation_weight_snapshot(
    valid_run_settings_mapping,
    prediction_weights_by_model_id,
):
    """外部のゼロ確率と許容誤差内の確率も再正規化せず旧順序で使う。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    reference_router = SwitchingExpertRouter(2)
    controller.get_prediction_weights_before_label_observation(model_ids=(0, -1))
    reference_router.probabilities((0, -1))
    reference_prediction_weights = dict(prediction_weights_by_model_id)
    observed_losses_by_model_id = {0: 1.0, -1: 0.0}
    controller.update_weights_after_loss_observation(
        observed_losses_by_model_id=observed_losses_by_model_id,
        prediction_weights_by_model_id=prediction_weights_by_model_id,
    )
    reference_router.update(observed_losses_by_model_id, prediction_weights_by_model_id)
    assert_fixed_share_state_matches_reference(
        controller=controller, reference_router=reference_router
    )
    assert observed_losses_by_model_id == {0: 1.0, -1: 0.0}
    assert prediction_weights_by_model_id == reference_prediction_weights
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    assert (
        controller.get_prediction_weights_before_label_observation(model_ids=(-1, 0))
        == controller.weights_by_model_id
    )
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )


@pytest.mark.parametrize(
    "invalid_observed_losses,invalid_prediction_weights",
    [
        ({2: float("nan"), 3: 0}, {2: 0.5, 3: 0.5}),
        ({2: float("inf"), 3: 0}, {2: 0.5, 3: 0.5}),
        ({2: 10**400, 3: 0}, {2: 0.5, 3: 0.5}),
        ({2: True, 3: 0}, {2: 0.5, 3: 0.5}),
        ({2: "0", 3: 0}, {2: 0.5, 3: 0.5}),
        ({2: 0, 3: 1}, {2: float("nan"), 3: 0.5}),
        ({2: 0, 3: 1}, {2: -float("inf"), 3: 0.5}),
        ({2: 0, 3: 1}, {2: 10**400, 3: 0}),
        ({2: 0, 3: 1}, {2: True, 3: 0}),
        ({2: 0, 3: 1}, {2: -0.1, 3: 1.1}),
        ({2: 0, 3: 1}, {2: 0, 3: 0}),
        ({2: 0, 3: 1}, {2: 0.5, 3: 0.5 + 2e-12}),
        ({2: 0, 3: 1}, {2: 0.5, 4: 0.5}),
        ({True: 0, 3: 1}, {1: 0.5, 3: 0.5}),
        ({2: 0, 3: 1}, {2.0: 0.5, 3: 0.5}),
        ({}, {}),
        ([], {2: 1}),
        ({2: 0}, []),
    ],
)
def test_fixed_share_invalid_observation_inputs_leave_all_state_unchanged(
    valid_run_settings_mapping,
    invalid_observed_losses,
    invalid_prediction_weights,
):
    """新しい集合の不正入力を、累積証拠と全計数を保持して拒否する。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    controller.get_prediction_weights_before_label_observation(model_ids=(9,))
    controller.update_weights_after_loss_observation(
        observed_losses_by_model_id={0: 0, -1: 1},
        prediction_weights_by_model_id={0: 0.5, -1: 0.5},
    )
    assert controller.model_pool_reset_count == 1
    assert controller.prediction_weight_leader_switch_count == 1
    assert controller.cumulative_observed_loss_variance > 0
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    with pytest.raises((TypeError, ValueError)):
        controller.update_weights_after_loss_observation(
            observed_losses_by_model_id=invalid_observed_losses,
            prediction_weights_by_model_id=invalid_prediction_weights,
        )
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )


@pytest.mark.parametrize(
    "preferred_model_id,expected_model_id", [(None, -3), (2, 2), (0, -3), (99, -3)]
)
def test_fixed_share_leader_selection_resolves_ties_without_changing_state(
    valid_run_settings_mapping,
    preferred_model_id,
    expected_model_id,
):
    """公開の同率優先と最小ID選択は、状態を変更しない。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    controller.get_prediction_weights_before_label_observation(model_ids=(0,))
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    prediction_weights_by_model_id = {2: 0.4, 0: 0.2, -3: 0.4}
    assert (
        FixedSharePredictionWeightController.select_maximum_weight_model_id(
            prediction_weights_by_model_id=prediction_weights_by_model_id,
            preferred_model_id=preferred_model_id,
        )
        == expected_model_id
    )
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )
    for preferred_model_id in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            controller.select_maximum_weight_model_id(
                prediction_weights_by_model_id=prediction_weights_by_model_id,
                preferred_model_id=preferred_model_id,
            )
    for invalid_prediction_weights in (
        {},
        {True: 1},
        {0: float("nan")},
        {0: 10**400},
        {0: 0},
        {0: 1.1},
        {0: True},
    ):
        with pytest.raises((TypeError, ValueError)):
            controller.select_maximum_weight_model_id(
                prediction_weights_by_model_id=invalid_prediction_weights
            )
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )


@pytest.mark.parametrize(
    "configuration_parameter_name,specified_parameter_value",
    [
        ("prediction_combination_settings", None),
        ("prediction_combination_settings", {}),
        (
            "prediction_combination_settings",
            SimpleNamespace(fixed_share_weight_redistribution_time_scale_samples=30),
        ),
        ("prediction_combination_strategy", "unknown"),
        ("prediction_mixture_activation_policy", "drift_recovery"),
        ("prediction_weight_recalibration_after_aggregation_policy", "none"),
        ("prediction_state_reset_on_training_assignment_change_policy", "clear"),
        ("fixed_share_weight_redistribution_time_scale_samples", 1),
        ("fixed_share_weight_redistribution_time_scale_samples", True),
        ("fixed_share_weight_redistribution_time_scale_samples", 2.0),
        ("fixed_share_weight_redistribution_time_scale_samples", "30"),
        ("fixed_share_weight_redistribution_time_scale_samples", 10**400),
    ],
)
def test_fixed_share_controller_requires_explicit_valid_settings(
    valid_run_settings_mapping,
    configuration_parameter_name,
    specified_parameter_value,
):
    """渡された設定の全項目を再検証し、数値変換できない時間尺度も拒否する。"""
    prediction_combination_settings = replace(
        valid_run_settings_mapping["prediction_combination_settings"]
    )
    if configuration_parameter_name == "prediction_combination_settings":
        prediction_combination_settings = specified_parameter_value
        with pytest.raises(TypeError, match="prediction_combination_settings"):
            FixedSharePredictionWeightController(
                prediction_combination_settings=prediction_combination_settings
            )
    else:
        # 実行境界の再検証を確認するため、構築後の不正値をテスト側で注入する。
        object.__setattr__(
            prediction_combination_settings, configuration_parameter_name, specified_parameter_value
        )
        if specified_parameter_value == 10**400:
            with pytest.raises(
                ValueError, match="fixed_share_weight_redistribution_time_scale_samples"
            ):
                FixedSharePredictionWeightController(
                    prediction_combination_settings=prediction_combination_settings
                )
        else:
            with pytest.raises(RunSettingsValidationError) as exception_info:
                FixedSharePredictionWeightController(
                    prediction_combination_settings=prediction_combination_settings
                )
            assert exception_info.value.configuration_parameter_name == configuration_parameter_name
            assert exception_info.value.specified_parameter_value is specified_parameter_value
        assert (
            getattr(prediction_combination_settings, configuration_parameter_name)
            is specified_parameter_value
        )


def test_fixed_share_prediction_weight_calls_require_keyword_arguments(valid_run_settings_mapping):
    """設定を省略せず、構築と予測前取得はkeyword-onlyで使う。"""
    prediction_combination_settings = valid_run_settings_mapping["prediction_combination_settings"]
    with pytest.raises(TypeError):
        FixedSharePredictionWeightController()
    with pytest.raises(TypeError):
        FixedSharePredictionWeightController(prediction_combination_settings)
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=prediction_combination_settings
    )
    with pytest.raises(TypeError):
        controller.get_prediction_weights_before_label_observation((0, 1))
    with pytest.raises(TypeError):
        controller.update_weights_after_loss_observation({0: 0}, {0: 1})
    with pytest.raises(TypeError):
        controller.update_weights_after_loss_observation(observed_losses_by_model_id={0: 0})
    with pytest.raises(TypeError):
        controller.select_maximum_weight_model_id({0: 1})
    with pytest.raises(TypeError):
        controller.replay_observed_losses(({0: 0.2},))
    with pytest.raises(TypeError):
        controller.replay_observed_losses_after_aggregation(({0: 0.2},))


def test_fixed_share_model_pool_changes_preserve_order_and_reset_only_when_needed(
    valid_run_settings_mapping,
):
    """負の一時IDを含む集合を昇順で一様化し、初回と同集合をreset数に含めない。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    assert capture_fixed_share_controller_state(controller=controller) == {
        "weights_by_model_id": {},
        "cumulative_observed_loss_variance": 0.0,
        "model_pool_reset_count": 0,
        "prediction_weight_leader_switch_count": 0,
        "aggregation_recalibration_count": 0,
        "aggregation_recalibration_sample_count": 0,
    }
    model_ids = [2, -3, 0]
    returned_prediction_weights = controller.get_prediction_weights_before_label_observation(
        model_ids=model_ids
    )
    assert tuple(returned_prediction_weights) == (-3, 0, 2)
    assert returned_prediction_weights == {-3: 1.0 / 3, 0: 1.0 / 3, 2: 1.0 / 3}
    assert model_ids == [2, -3, 0]
    assert controller.model_pool_reset_count == 0
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    assert (
        controller.get_prediction_weights_before_label_observation(model_ids=(0, 2, -3))
        == returned_prediction_weights
    )
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )
    assert controller.get_prediction_weights_before_label_observation(model_ids=(5, -3)) == {
        -3: 0.5,
        5: 0.5,
    }
    assert controller.model_pool_reset_count == 1
    assert controller.cumulative_observed_loss_variance == 0.0
    assert controller.get_prediction_weights_before_label_observation(model_ids=(-3,)) == {-3: 1.0}
    assert controller.model_pool_reset_count == 2
    assert controller.get_prediction_weights_before_label_observation(model_ids=iter((-3,))) == {
        -3: 1.0
    }
    assert controller.model_pool_reset_count == 2


@pytest.mark.parametrize(
    "invalid_model_ids",
    [
        (),
        [],
        (0, 0),
        (-1, 0, -1),
        (True, 1),
        (False,),
        (0.0, 1),
        ("0",),
        (None,),
        ([],),
        None,
        2,
        map(lambda model_id: 1 // model_id, (1, 0)),
    ],
)
def test_fixed_share_invalid_model_ids_leave_all_state_unchanged(
    valid_run_settings_mapping, invalid_model_ids
):
    """初期化済み集合の不正な再指定でも、重み・分散・全計数を保持する。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    controller.get_prediction_weights_before_label_observation(model_ids=(0,))
    controller.get_prediction_weights_before_label_observation(model_ids=(-1, 0))
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    with pytest.raises((TypeError, ValueError, ZeroDivisionError)):
        controller.get_prediction_weights_before_label_observation(model_ids=invalid_model_ids)
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )


def test_fixed_share_returned_weights_and_diagnostics_do_not_expose_owned_state(
    valid_run_settings_mapping,
):
    """取得値と診断dictへの変更・読取property代入で所有状態を変更できない。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    returned_prediction_weights = controller.get_prediction_weights_before_label_observation(
        model_ids=(-1, 0)
    )
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    returned_prediction_weights[-1] = 0.0
    returned_prediction_weights[8] = 1.0
    returned_prediction_weights = controller.weights_by_model_id
    returned_prediction_weights.clear()
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )
    for record_field_name in controller_state_before_call:
        with pytest.raises(AttributeError):
            setattr(controller, record_field_name, None)
        with pytest.raises(AttributeError):
            delattr(controller, record_field_name)
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )


@pytest.mark.parametrize("replay_after_aggregation", [False, True])
def test_fixed_share_replay_matches_reference_for_ordered_and_changing_model_pools(
    valid_run_settings_mapping,
    replay_after_aggregation,
):
    """順序を保つ再生と途中の集合差を、非ゼロの既存証拠から照合する。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    reference_router = SwitchingExpertRouter(2)
    observed_loss_sequence = (
        {2: 0.9, -1: 0.1},
        {2: 0.2, -1: 0.8},
        {4: -1.0, -1: 3.0},
        {-1: 0.3},
        {4: 0.2, -1: 0.4},
    )
    for observed_losses_by_model_id in observed_loss_sequence[:2]:
        returned_prediction_weights = controller.get_prediction_weights_before_label_observation(
            model_ids=observed_losses_by_model_id,
        )
        controller.update_weights_after_loss_observation(
            observed_losses_by_model_id=observed_losses_by_model_id,
            prediction_weights_by_model_id=returned_prediction_weights,
        )
        reference_router.update(
            observed_losses_by_model_id, reference_router.probabilities(observed_losses_by_model_id)
        )
    if replay_after_aggregation:
        controller.replay_observed_losses_after_aggregation(
            observed_loss_sequence=iter(observed_loss_sequence)
        )
        reference_router.replay_after_aggregation(observed_loss_sequence)
    else:
        controller.replay_observed_losses(observed_loss_sequence=iter(observed_loss_sequence))
        reference_router.replay(observed_loss_sequence)
    assert_fixed_share_state_matches_reference(
        controller=controller, reference_router=reference_router
    )
    assert observed_loss_sequence == (
        {2: 0.9, -1: 0.1},
        {2: 0.2, -1: 0.8},
        {4: -1.0, -1: 3.0},
        {-1: 0.3},
        {4: 0.2, -1: 0.4},
    )


def test_fixed_share_aggregation_reset_and_replay_counters_match_reference(
    valid_run_settings_mapping,
):
    """集約reset、通常再生、集約再生の計数と証拠を区別する。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    reference_router = SwitchingExpertRouter(2)
    observed_loss_sequence = ({-1: 0, 1: 1}, {-1: 1, 1: 0})
    for _observation_index in range(2):
        controller.reset_weights_after_aggregation()
        reference_router.restart_after_aggregation()
        assert_fixed_share_state_matches_reference(
            controller=controller, reference_router=reference_router
        )
    controller.replay_observed_losses(observed_loss_sequence=observed_loss_sequence)
    reference_router.replay(observed_loss_sequence)
    assert_fixed_share_state_matches_reference(
        controller=controller, reference_router=reference_router
    )
    controller.replay_observed_losses_after_aggregation(
        observed_loss_sequence=observed_loss_sequence
    )
    reference_router.replay_after_aggregation(observed_loss_sequence)
    assert_fixed_share_state_matches_reference(
        controller=controller, reference_router=reference_router
    )
    assert controller.aggregation_recalibration_count == 3
    assert controller.aggregation_recalibration_sample_count == 2
    controller.reset_weights_after_aggregation()
    reference_router.restart_after_aggregation()
    assert_fixed_share_state_matches_reference(
        controller=controller, reference_router=reference_router
    )
    assert controller.prediction_weight_leader_switch_count > 0


@pytest.mark.parametrize("replay_after_aggregation", [False, True])
def test_fixed_share_empty_replay_preserves_evidence_and_all_counters(
    valid_run_settings_mapping,
    replay_after_aggregation,
):
    """初期状態と非ゼロ証拠のどちらも空列の受領で変化しない。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    for _observation_index in range(2):
        controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
        if replay_after_aggregation:
            controller.replay_observed_losses_after_aggregation(observed_loss_sequence=iter(()))
        else:
            controller.replay_observed_losses(observed_loss_sequence=iter(()))
        assert (
            capture_fixed_share_controller_state(controller=controller)
            == controller_state_before_call
        )
        controller.replay_observed_losses_after_aggregation(
            observed_loss_sequence=({0: 1, -1: 0}, {0: 0, -1: 1}, {0: 0, 2: 1}),
        )


@pytest.mark.parametrize("replay_after_aggregation", [False, True])
@pytest.mark.parametrize(
    "invalid_observed_loss_sequence",
    [
        ({0: 0.2, 1: 0.8}, {}),
        ({0: 0.2, 1: 0.8}, {0: float("nan"), 1: 0.5}),
        ({0: 0.2, 1: 0.8}, {True: 0.2}),
        ({0: 0.2, 1: 0.8}, {0: 10**400}),
        ({0: 0.2, 1: 0.8}, []),
        None,
        "raise_while_iterating",
    ],
)
def test_fixed_share_invalid_later_replay_rows_leave_all_state_unchanged(
    valid_run_settings_mapping,
    replay_after_aggregation,
    invalid_observed_loss_sequence,
):
    """後段の不正行でも先行行を適用せず、全証拠・計数を保持する。"""
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=valid_run_settings_mapping[
            "prediction_combination_settings"
        ],
    )
    controller.replay_observed_losses_after_aggregation(
        observed_loss_sequence=({-1: 0, 9: 1}, {-1: 1, 9: 0}, {9: 0}),
    )
    controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
    if invalid_observed_loss_sequence == "raise_while_iterating":
        invalid_observed_loss_sequence = map(
            lambda observed_losses_by_model_id: observed_losses_by_model_id[0],
            ({0: {0: 0.2, 1: 0.8}}, {}),
        )
    with pytest.raises((TypeError, ValueError, KeyError)):
        if replay_after_aggregation:
            controller.replay_observed_losses_after_aggregation(
                observed_loss_sequence=invalid_observed_loss_sequence
            )
        else:
            controller.replay_observed_losses(observed_loss_sequence=invalid_observed_loss_sequence)
    assert (
        capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
    )


def test_fixed_share_controller_instances_and_caller_random_state_are_independent(
    valid_run_settings_mapping,
):
    """同じ不変設定を渡した別実体と呼出元乱数へ状態を持ち出さない。"""
    global_python_random_state = random.getstate()
    prediction_combination_settings = valid_run_settings_mapping["prediction_combination_settings"]
    controller = FixedSharePredictionWeightController(
        prediction_combination_settings=prediction_combination_settings
    )
    repeated_controller = FixedSharePredictionWeightController(
        prediction_combination_settings=prediction_combination_settings
    )
    controller.get_prediction_weights_before_label_observation(model_ids=(-1, 0))
    controller.get_prediction_weights_before_label_observation(model_ids=(0, 1))
    assert repeated_controller.weights_by_model_id == {}
    assert repeated_controller.model_pool_reset_count == 0
    assert repeated_controller.get_prediction_weights_before_label_observation(model_ids=(0,)) == {
        0: 1.0
    }
    assert controller.weights_by_model_id == {0: 0.5, 1: 0.5}
    assert controller.model_pool_reset_count == 1
    assert random.getstate() == global_python_random_state
    assert prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples == 2
