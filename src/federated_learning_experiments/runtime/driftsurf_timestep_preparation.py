"""DriftSurfの正解数の報告・役割の登録と窓を既存の学習へ接続する。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from torch import cat, no_grad

from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)

if TYPE_CHECKING:
    from .feddrift_run_client import FeddriftRunClient
    from .feddrift_run_server import FeddriftRunServer


def count_driftsurf_role_correct_predictions(
    *, run_client: FeddriftRunClient, model_id: int
) -> tuple[int, int]:
    """学習前の役割のモデルで、今の時刻の全標本をeval/no_gradで数える。"""
    samples = run_client.stream_processor._driftsurf_pending_samples
    classifier = run_client.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=model_id
    ).classifier
    if not samples:
        return 0, 0
    previous_training = classifier.training
    classifier.eval()
    try:
        with no_grad():
            scores = classifier(cat([s.input_features for s in samples]))
            # 手元の二値モデルはP(y=1)の1列。クラスごとの出力へ対応付ける。
            if classifier.class_count == 2:
                scores = cat((1 - scores, scores), dim=-1)
            predictions = scores.argmax(dim=-1).view(-1)
            labels = cat([s.observed_class_labels for s in samples]).view(-1)
            return int((predictions == labels).sum().item()), len(samples)
    finally:
        classifier.train(previous_training)


def prepare_driftsurf_timestep(*, run_server: FeddriftRunServer, timestep_index: int) -> None:
    """状態遷移を先に行い、学習するIDの窓だけでstoreを作り直す。"""
    server = run_server
    state = server._driftsurf_state
    assert state is not None
    roles = server._driftsurf_model_ids
    before = state.state
    accuracies: dict[str, float] = {}
    for role in state.scoring_roles if timestep_index else ():
        mid = roles[role]
        assert mid is not None
        reports = [
            count_driftsurf_role_correct_predictions(run_client=c, model_id=mid)
            for c in server.run_clients
        ]
        total = sum(n for _, n in reports)
        accuracies[role] = sum(correct for correct, _ in reports) / total if total else 0.0
    for client in server.run_clients:
        stream = client.stream_processor
        stream._driftsurf_timestep_samples.append_timestep(
            timestep_index=timestep_index, training_samples=tuple(stream._driftsurf_pending_samples)
        )
        stream._driftsurf_pending_samples.clear()
    transition = (
        state.advance(timestep_index=timestep_index, accuracies=accuracies)
        if timestep_index
        else None
    )
    reset_ids = set()
    if timestep_index == 0:
        reset_ids.update((0, 1))
    for role in transition.reset_roles if transition else ():
        mid = server.global_model_repository.allocate_global_model_id()
        roles[role] = mid
        reset_ids.add(mid)
        server.global_model_repository.set_global_model_parameters(
            model_id=mid, parameter_snapshot=server._driftsurf_initial_parameters
        )
        server.global_model_repository.set_global_model_loss_statistics(
            model_id=mid, loss_statistics=server._driftsurf_initial_statistics
        )
        server.global_model_repository.record_model_registration(
            model_id=mid, registered_round_index=timestep_index, registering_client_id=None
        )
    promoted_id = None
    if transition and transition.promoted:
        promoted_id = roles["reac"]
        roles["pred"] = promoted_id
        roles["reac"] = None
    retained = {mid for mid in roles.values() if mid is not None}
    for mid in server.global_model_repository.global_model_ids:
        if mid not in retained:
            server.global_model_repository.remove_global_model(model_id=mid)
            server._model_weights.pop(mid, None)
    # 昇格時のtrain_keysは公式どおりpred,reac。同じ昇格モデルへの学習は1回。
    training_ids = frozenset(
        roles[r] if roles[r] is not None else promoted_id for r in state.train_keys
    )
    assert None not in training_ids
    training_ids = frozenset(mid for mid in training_ids if mid is not None)
    prediction_id = roles[state.model_key]
    assert prediction_id is not None
    for client in server.run_clients:
        owners = client.owners
        registry = owners.held_model_training_state_registry
        existing = registry.snapshot_ordered_held_model_training_states()
        rebuilt = [s for s in existing if s.model_id in retained]
        for mid in sorted(reset_ids):
            if mid in {s.model_id for s in rebuilt}:
                continue
            classifier = deepcopy(existing[0].classifier)
            classifier.load_state_dict(server._driftsurf_initial_parameters, strict=True)
            optimizer = ParameterOptimizerState(
                parameters=tuple(classifier.parameters()),
                optimizer_settings=server._settings.rebuilt_model_optimizer_settings,
            )
            rebuilt.append(
                HeldModelTrainingState(
                    model_id=mid,
                    classifier=classifier,
                    concept_specific_parameter_optimizer_state=optimizer,
                )
            )
        registry.replace_held_model_training_states(held_model_training_states=tuple(rebuilt))
        owners.loss_statistics_store.replace_model_loss_statistics(
            loss_statistics_by_model_id=tuple(
                (mid, statistics)
                for mid, statistics in server.global_model_repository.snapshot_global_model_loss_statistics()
                if mid in retained
            )
        )
        owners.current_training_model_assignment.assign_model_for_training(model_id=prediction_id)
        owners.training_sample_store.clear_model_training_samples()
        for mid in sorted(training_ids):
            assert mid is not None
            role = next(
                r
                for r in state.train_keys
                if (roles[r] if roles[r] is not None else promoted_id) == mid
            )
            window = state.train_data_dict[role]
            assert window is not None
            samples = client.stream_processor._driftsurf_timestep_samples.samples_for_window(
                timestep_indices=tuple(window)
            )
            owners.training_sample_store.append_model_training_samples(
                model_id=mid, training_samples=samples
            )
    server._driftsurf_training_model_ids = frozenset(mid for mid in training_ids if mid is not None)
    server._driftsurf_reset_model_ids = frozenset(reset_ids)
    record: dict[str, object] = dict(
        timestep_index=timestep_index,
        state_before=before,
        state_after=state.state,
        accuracy_best=state.acc_best,
        training_roles=",".join(state.train_keys),
        prediction_role=state.model_key,
        reactive_counter=state.reac_ctr if state.reac_ctr is not None else -1,
        entered_reactive=transition.entered_reactive if transition else False,
        promoted=transition.promoted if transition else False,
    )
    for role in ("pred", "stab", "reac"):
        window = state.train_data_dict[role] or []
        record["accuracy_" + role] = accuracies.get(role, float("nan"))
        record[role + "_model_id"] = roles[role] if roles[role] is not None else -1
        record[role + "_window_timesteps"] = ",".join(str(t) for t in window)
    server._driftsurf_records.append(record)
