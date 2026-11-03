"""CFL判定を集約前へ接続し、分割時だけ全モデルの集約を省く。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)

if TYPE_CHECKING:
    from .feddrift_run_server import FeddriftRunServer


def split_cfl_training_round(
    *, run_server: FeddriftRunServer, timestep_index: int, active_ids: tuple[int, ...]
) -> bool:
    """学習の標本があるclientの更新を判定し、分割の記録とθ0を配布する。"""
    server = run_server
    state = server._cfl_state
    assert state is not None
    updates: dict[int, dict[int, np.ndarray]] = {mid: {} for mid in active_ids}
    uploads = []
    for mid in active_ids:
        base = server.global_model_repository.get_global_model_parameters(model_id=mid)
        for client in server.run_clients:
            if client.owners.current_training_model_assignment.current_training_model_id != mid:
                continue
            samples = next(
                (
                    c.training_samples
                    for c in client.owners.training_sample_store.snapshot_ordered_model_training_samples()
                    if c.model_id == mid
                ),
                (),
            )
            if not samples:
                continue
            classifier = (
                client.owners.held_model_training_state_registry.get_held_model_training_state(
                    model_id=mid
                ).classifier
            )
            local = snapshot_classifier_parameters(classifier=classifier)
            updates[mid][client.client_id] = np.concatenate(
                [(local[k] - base[k]).detach().cpu().numpy().flatten() for k in base]
            )
            uploads.append(local)
    records = state.check(
        updates_by_model=updates,
        gamma=server.run_clients[0]._settings.cfl_split_gamma,
        maximum_model_count=server._maximum_cfl_model_count,
    )
    server._cfl_records.extend((timestep_index, row) for row in records)
    splits = [row for row in records if row[8]]
    if not splits:
        return False
    for row in splits:
        mid, new = row[0], row[9]
        allocated = server.global_model_repository.allocate_global_model_id()
        assert allocated == new
        for reset in (mid, new):
            server.global_model_repository.set_global_model_parameters(
                model_id=reset, parameter_snapshot=server._cfl_initial_parameters
            )
            server.global_model_repository.set_global_model_loss_statistics(
                model_id=reset, loss_statistics=server._cfl_initial_statistics
            )
            server._model_weights[reset] = server._settings.initial_model_aggregation_weight
        server.global_model_repository.record_model_registration(
            model_id=new, registered_round_index=timestep_index, registering_client_id=None
        )
    # 上りの学習と通信は既に起きている。集約のMACは計上しない。
    for parameters in uploads:
        server._record_parameter_transfer("upload", parameters, 1)
    # active_idsを空にすると既存の配布部品は集約せず全モデルを再構築・配布する。
    server._aggregate_and_distribute_models(active_ids=(), update_model_weights=False)
    for row in splits:
        for client in server.run_clients:
            if client.client_id in row[11]:
                client.owners.current_training_model_assignment.assign_model_for_training(
                    model_id=row[9]
                )
    return True
