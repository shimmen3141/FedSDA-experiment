"""集約後の予測の重みと診断証拠の再較正を、実旧のサーバのrun_round（クラスタリングなし）と、実旧のclient複数に対して照合する。"""

import random

import pytest
import torch
from test_fedsda_run_client import (
    assert_run_client_matches_legacy,
    assert_run_client_state_unchanged,
    run_in_both,
    snapshot_run_client_state,
)
from test_global_model_distribution import (
    DIFFERENT_LEARNING_RATES,
    assert_all_states_match_legacy_after_distribution,
    make_client_streams,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_server_model_registration_and_aggregation import (
    SERVER_ROUND_CONDITIONS,
    build_server_round_oracle,
    process_round_samples_in_both,
)

from federated_learning_experiments.runtime.global_model_distribution import (
    distribute_global_models_to_clients,
)
from federated_learning_experiments.runtime.post_aggregation_prediction_recalibration import (
    PostAggregationPredictionRecalibration,
    recalibrate_prediction_state_after_aggregation,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    aggregate_client_models_into_global_models,
    register_ready_client_models,
)

# 旧の最終構成の再較正の方式。
FIFO_REPLAY_RECALIBRATION = dict(routing_recalibration="fifo_replay")


def assert_recalibration_result_matches_legacy_state(recalibration, legacy_client):
    """再較正の結果の記録を、再較正の後の実旧clientの状態（保留標本、保有モデル、真の概念別のルータ）と照合する。"""
    assert type(recalibration) is PostAggregationPredictionRecalibration
    if legacy_client.buffer and len(legacy_client.models) > 1:
        assert recalibration.replayed_sample_count == len(legacy_client.buffer)
        assert recalibration.replayed_model_ids == tuple(sorted(legacy_client.models))
    else:
        assert recalibration.replayed_sample_count == 0
        assert recalibration.replayed_model_ids == ()
    assert recalibration.restarted_true_concept_ids == tuple(
        legacy_client.oracle_concept_expert_routers
    )


def run_round_with_recalibration_in_both(*, server_round_oracle, client_streams, round_index):
    """1ラウンド: 標本処理→保留中の学習→状態の報告→（旧はrun_round、新は登録→集約→配布→全clientの再較正）→送信待ちの進行。

    旧のサーバの処理と、新の一連の処理を、同じ乱数の状態から行い、その後に全状態を照合する。
    戻り値: 新の、clientごとの再較正の結果。
    """
    run_clients = server_round_oracle["run_clients"]
    legacy_clients = server_round_oracle["legacy_clients"]
    legacy_server = server_round_oracle["legacy_server"]
    global_model_repository = server_round_oracle["global_model_repository"]
    communication_volume_record_store = server_round_oracle["communication_volume_record_store"]
    process_round_samples_in_both(
        server_round_oracle=server_round_oracle,
        client_streams=client_streams,
        round_index=round_index,
    )
    legacy_server.record_client_state_summaries()
    communication_volume_record_store.record_messages(
        transfer_direction="upload", message_count=len(run_clients)
    )

    def synchronize_and_recalibrate():
        register_ready_client_models(
            run_clients=run_clients,
            global_model_repository=global_model_repository,
            round_index=round_index,
        )
        aggregate_client_models_into_global_models(
            run_clients=run_clients,
            global_model_repository=global_model_repository,
            communication_volume_record_store=communication_volume_record_store,
        )
        distribute_global_models_to_clients(
            run_clients=run_clients,
            global_model_repository=global_model_repository,
            communication_volume_record_store=communication_volume_record_store,
            model_id_mapping={},
        )
        return tuple(
            run_client.recalibrate_prediction_state_after_aggregation()
            for run_client in run_clients
        )

    recalibrations, _ = run_in_both(
        run_client=None,
        legacy_client=None,
        python_random_generator=server_round_oracle["python_random_generator"],
        legacy_operation=lambda: legacy_server.run_round(round_index, clustering_enabled=False),
        operation=synchronize_and_recalibrate,
    )
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    for recalibration, legacy_client in zip(recalibrations, legacy_clients, strict=True):
        assert_recalibration_result_matches_legacy_state(recalibration, legacy_client)
    for run_client, legacy_client in zip(run_clients, legacy_clients, strict=True):
        legacy_client.promote_pending_to_ready()
        run_client.advance_new_model_upload_wait_after_synchronization(round_index=round_index)
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    return recalibrations


RECALIBRATION_ROUND_COUNT = 13
OBSERVED_COVERAGE_BY_CONDITION = {}
# 登録と集約の対照の条件（クラス数, 概念の区間長, 標本列のseed）に、学習率の2つの設定が違う条件を足す。
RECALIBRATION_ROUND_CONDITIONS = [
    (*server_round_condition, {}) for server_round_condition in SERVER_ROUND_CONDITIONS
] + [(2, 16, 23, {}), (2, 22, 11, DIFFERENT_LEARNING_RATES), (4, 30, 3, DIFFERENT_LEARNING_RATES)]


@pytest.mark.parametrize(
    "class_count,concept_block_length,stream_seed,condition_overrides",
    RECALIBRATION_ROUND_CONDITIONS,
)
def test_round_with_recalibration_matches_real_legacy_server_run_round(
    class_count,
    concept_block_length,
    stream_seed,
    condition_overrides,
    monkeypatch,
    valid_run_settings_mapping,
):
    client_streams = make_client_streams(
        round_count=RECALIBRATION_ROUND_COUNT,
        concept_block_length=concept_block_length,
        stream_seed=stream_seed,
    )
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        **FIFO_REPLAY_RECALIBRATION,
        **condition_overrides,
    )
    legacy_clients = server_round_oracle["legacy_clients"]
    observed_paths = {f"class_count_{class_count}"}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(RECALIBRATION_ROUND_COUNT):
            recalibrations = run_round_with_recalibration_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
            )
            for recalibration, legacy_client in zip(recalibrations, legacy_clients, strict=True):
                if recalibration.replayed_sample_count > 0:
                    observed_paths.add("losses_replayed")
                    if any(model_id < 0 for model_id in recalibration.replayed_model_ids):
                        observed_paths.add("temporary_model_losses_replayed")
                    if len(recalibration.replayed_model_ids) >= 3:
                        observed_paths.add("three_or_more_models_replayed")
                elif len(legacy_client.models) == 1 and legacy_client.buffer:
                    observed_paths.add("single_model_without_replay")
                if recalibration.restarted_true_concept_ids:
                    observed_paths.add("true_concept_evidence_restarted")
                # 再生は、実旧の計数にも現れている。
                assert legacy_client.switching_expert_router.aggregation_recalibration_count == (
                    legacy_client.expert_router.aggregation_recalibration_count
                )
    assert any(
        legacy_client.expert_router.aggregation_recalibration_sample_count > 0
        for legacy_client in legacy_clients
    ) == ("losses_replayed" in observed_paths)
    OBSERVED_COVERAGE_BY_CONDITION[
        (class_count, concept_block_length, stream_seed, tuple(condition_overrides))
    ] = observed_paths


def test_recalibration_round_conditions_cover_required_paths():
    """上の対照が、要求の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(OBSERVED_COVERAGE_BY_CONDITION) < len(RECALIBRATION_ROUND_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    for required_path in (
        "losses_replayed",
        "temporary_model_losses_replayed",
        "three_or_more_models_replayed",
        "single_model_without_replay",
        "true_concept_evidence_restarted",
    ):
        # 2値と多クラスの両方で通っていること（一時IDのモデルを含む再生は、どちらかでよい）。
        observed_class_counts = {
            condition[0]
            for condition, observed_paths in OBSERVED_COVERAGE_BY_CONDITION.items()
            if required_path in observed_paths
        }
        if required_path == "temporary_model_losses_replayed":
            assert observed_class_counts, required_path
        else:
            assert observed_class_counts == {2, 4}, (required_path, observed_class_counts)


def build_recalibrated_server_round_oracle(*, monkeypatch, valid_run_settings_mapping):
    """再較正つきで10ラウンド進めた状態。全clientが、2つ以上のモデルと、保留中の標本を持つ。"""
    client_streams = make_client_streams(round_count=10, concept_block_length=10, stream_seed=50)
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
        **FIFO_REPLAY_RECALIBRATION,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(10):
            run_round_with_recalibration_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
            )
    for legacy_client in server_round_oracle["legacy_clients"]:
        assert len(legacy_client.models) >= 2
        assert legacy_client.buffer
    return server_round_oracle


# 再較正が変える状態（読取りの項目名）。ほかは変えない。
STATES_CHANGED_BY_RECALIBRATION = ("diagnostics", "fixed_share_prediction_weights")


def test_client_recalibration_matches_real_legacy_client_and_keeps_other_states(
    monkeypatch, valid_run_settings_mapping
):
    """clientの再較正だけを、実旧のclientの再較正と照合する。予測の重みと診断証拠のほかは、何も変えない。"""
    server_round_oracle = build_recalibrated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    global_python_random_state = random.getstate()
    for run_client, legacy_client in zip(
        server_round_oracle["run_clients"], server_round_oracle["legacy_clients"], strict=True
    ):
        owners = run_client.owners
        held_model_training_states = (
            owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        training_modes = [
            held_state.classifier.training for held_state in held_model_training_states
        ]
        gradients = [
            None if parameter.grad is None else parameter.grad.clone()
            for held_state in held_model_training_states
            for parameter in held_state.classifier.parameters()
        ]
        for _ in range(2):
            state_snapshot = snapshot_run_client_state(
                run_client=run_client, python_random_generator=python_random_generator
            )
            recalibration_count = (
                owners.fixed_share_prediction_weight_controller.aggregation_recalibration_count
            )
            recalibration, _ = run_in_both(
                run_client=run_client,
                legacy_client=legacy_client,
                python_random_generator=python_random_generator,
                legacy_operation=legacy_client.recalibrate_routing_after_aggregation,
                operation=run_client.recalibrate_prediction_state_after_aggregation,
            )
            assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
            assert_recalibration_result_matches_legacy_state(recalibration, legacy_client)
            assert recalibration.replayed_sample_count > 0
            assert (
                owners.fixed_share_prediction_weight_controller.aggregation_recalibration_count
                == recalibration_count + 1
            )
            current_snapshot = snapshot_run_client_state(
                run_client=run_client, python_random_generator=python_random_generator
            )
            # 再較正が変える2つの状態を、前の読取りへ写してから、全項目が変わっていないことを確かめる。
            assert_run_client_state_unchanged(
                state_snapshot=state_snapshot
                | {
                    state_name: current_snapshot[state_name]
                    for state_name in STATES_CHANGED_BY_RECALIBRATION
                },
                run_client=run_client,
                python_random_generator=python_random_generator,
            )
        # 損失の計算は、訓練の別と勾配を変えない。
        assert training_modes == [
            held_state.classifier.training for held_state in held_model_training_states
        ]
        for gradient, parameter in zip(
            gradients,
            (
                parameter
                for held_state in held_model_training_states
                for parameter in held_state.classifier.parameters()
            ),
            strict=True,
        ):
            assert (gradient is None and parameter.grad is None) or torch.equal(
                gradient, parameter.grad
            )
    assert random.getstate() == global_python_random_state


RECALIBRATION_OWNER_ARGUMENT_NAMES = (
    "held_model_training_state_registry",
    "pending_sample_observation_store",
    "fixed_share_prediction_weight_controller",
    "diagnostic_evidence_collection",
)


def test_recalibration_rejects_invalid_owners_before_any_update(
    monkeypatch, valid_run_settings_mapping
):
    """不正なownerは、どの状態も変える前に拒否する（1つの状態で、全条件を順に確かめる）。"""
    server_round_oracle = build_recalibrated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    run_client = server_round_oracle["run_clients"][1]
    arguments = {
        owner_argument_name: getattr(run_client.owners, owner_argument_name)
        for owner_argument_name in RECALIBRATION_OWNER_ARGUMENT_NAMES
    }
    state_snapshot = snapshot_run_client_state(
        run_client=run_client, python_random_generator=python_random_generator
    )
    for owner_argument_name in RECALIBRATION_OWNER_ARGUMENT_NAMES:
        for invalid_owner in (None, make_subclass_copy(arguments[owner_argument_name])):
            with pytest.raises(TypeError, match=owner_argument_name):
                recalibrate_prediction_state_after_aggregation(
                    **arguments | {owner_argument_name: invalid_owner}
                )
            assert_run_client_state_unchanged(
                state_snapshot=state_snapshot,
                run_client=run_client,
                python_random_generator=python_random_generator,
            )
    assert_all_states_match_legacy_after_distribution(server_round_oracle)


def test_recalibration_changes_nothing_when_loss_evaluation_fails(
    monkeypatch, valid_run_settings_mapping
):
    """損失の計算が失敗したら（非有限のパラメータ）、診断証拠も予測の重みも変わらない。"""
    server_round_oracle = build_recalibrated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    run_client = server_round_oracle["run_clients"][0]
    # IDが最大の保有モデルの分類層を非有限にする（先のモデルの損失は、計算できてしまう）。
    held_model_training_states = run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    last_held_state = max(held_model_training_states, key=lambda held_state: held_state.model_id)
    with torch.no_grad():
        next(iter(last_held_state.classifier.classification_layer.parameters())).fill_(float("nan"))
    state_snapshot = snapshot_run_client_state(
        run_client=run_client, python_random_generator=python_random_generator
    )
    with pytest.raises(ValueError):
        run_client.recalibrate_prediction_state_after_aggregation()
    current_snapshot = snapshot_run_client_state(
        run_client=run_client, python_random_generator=python_random_generator
    )
    for state_name in (
        *STATES_CHANGED_BY_RECALIBRATION,
        "python_random_state",
        "pending_sample_observations",
    ):
        assert current_snapshot[state_name] == state_snapshot[state_name], state_name
    assert torch.equal(
        current_snapshot["torch_random_state"][0], state_snapshot["torch_random_state"][0]
    )
