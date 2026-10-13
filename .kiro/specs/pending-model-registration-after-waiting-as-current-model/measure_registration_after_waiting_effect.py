"""登録の扱い（3つ）で、1 runの結果がどう変わるかを測る（測定用。sourceは変えない）。

使い方（worktreeのルートで、検証用の環境変数を置いて）:
    python .kiro/specs/pending-model-registration-after-waiting-as-current-model/measure_registration_after_waiting_effect.py \
        <dataset> <seed> <clientあたりの標本数> <集約間隔> <登録の扱いの値> [<client数>]

最終構成の設定の、登録の扱いだけを替えて1回実行し、結果を1行のJSONで出す。登録の扱いの値は、
`keep_latest_pending_model_and_reassign_current_model_id`（固定旧と同じ）、
`register_each_pending_model_by_its_own_id`（基準版2・3の扱い）、
`register_pending_model_after_waiting_as_current_model`（FedSDAの既定）のどれか。
"""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

for thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_count_variable_name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from federated_learning_experiments.configuration.experiment_run_conditions import (  # noqa: E402
    ExperimentRunConditions,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (  # noqa: E402
    PendingModelRegistrationSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (  # noqa: E402
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (  # noqa: E402
    build_final_configuration_fedsda_run_settings,
)


def measure(
    *,
    dataset_name,
    random_seed,
    per_client_sample_count,
    aggregation_interval,
    pending_model_registration_policy,
    client_count,
):
    default_run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset_name,
            random_seed=random_seed,
            client_count=client_count,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=aggregation_interval,
        )
    )
    run_participant_settings = default_run_settings.run_participant_settings
    experiment_run = execute_fedsda_experiment_run(
        run_settings=replace(
            default_run_settings,
            run_participant_settings=replace(
                run_participant_settings,
                run_client_settings=replace(
                    run_participant_settings.run_client_settings,
                    pending_model_registration_settings=PendingModelRegistrationSettings(
                        pending_model_registration_policy=pending_model_registration_policy
                    ),
                ),
            ),
        )
    )
    unregistered_adopted_model_count = 0
    pending_model_count_at_run_end = 0
    held_model_count_at_run_end = 0
    for run_client in experiment_run.measured_run.participants.client_operations:
        owners = run_client.owners
        pending_model_ids = {
            pending_model_upload.model_id
            for pending_model_upload in owners.pending_model_upload_state.get_all_pending_model_uploads()
        }
        held_model_ids = [
            held_state.model_id
            for held_state in owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        ]
        pending_model_count_at_run_end += len(pending_model_ids)
        held_model_count_at_run_end += len(held_model_ids)
        unregistered_adopted_model_count += sum(
            model_id < 0 and model_id not in pending_model_ids for model_id in held_model_ids
        )
    run_metrics = experiment_run.run_metrics
    communication_volume = run_metrics.communication_volume
    global_model_repository = (
        experiment_run.measured_run.participants.server_operations.owners.global_model_repository
    )
    return dict(
        prediction_accuracy=run_metrics.prediction_accuracy,
        stable_period_prediction_accuracy=run_metrics.stable_period_prediction_accuracy,
        uploaded_byte_count=communication_volume.uploaded_byte_count,
        downloaded_byte_count=communication_volume.downloaded_byte_count,
        final_global_model_count=run_metrics.final_global_model_count,
        registered_model_count=len(global_model_repository.snapshot_model_registration_records()),
        held_model_count_at_run_end=held_model_count_at_run_end,
        unregistered_adopted_model_count=unregistered_adopted_model_count,
        pending_model_count_at_run_end=pending_model_count_at_run_end,
        detection_count=run_metrics.training_model_switch_detection_metrics.detection_count,
    )


if __name__ == "__main__":
    dataset_name, random_seed, per_client_sample_count, aggregation_interval, policy_name = (
        sys.argv[1:6]
    )
    client_count = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    print(
        "MEASURED",
        json.dumps(
            dict(
                dataset_name=dataset_name,
                random_seed=int(random_seed),
                per_client_sample_count=int(per_client_sample_count),
                aggregation_interval=int(aggregation_interval),
                client_count=client_count,
                pending_model_registration_policy=policy_name,
                result=measure(
                    dataset_name=dataset_name,
                    random_seed=int(random_seed),
                    per_client_sample_count=int(per_client_sample_count),
                    aggregation_interval=int(aggregation_interval),
                    pending_model_registration_policy=policy_name,
                    client_count=client_count,
                ),
            )
        ),
        flush=True,
    )
