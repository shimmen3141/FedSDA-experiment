"""登録の扱い（LEGACY-016の修正の前後）で、1 runの結果がどう変わるかを測る（測定用。sourceは変えない）。

使い方（worktreeのルートで、検証用の環境変数を置いて）:
    python .kiro/specs/pending-model-registration-by-own-id/measure_registration_policy_effect.py \
        <dataset> <seed> <clientあたりの標本数> <集約間隔> [<client数>]

最終構成の設定の、登録の扱いだけを替えて、2回実行し、結果を1行のJSONで出す。
`keep_latest_pending_model_and_reassign_current_model_id`が、固定旧と同じ扱い（修正の前）、
`register_each_pending_model_by_its_own_id`が、既定（修正の後）。
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
    PENDING_MODEL_REGISTRATION_POLICIES,
    PendingModelRegistrationSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (  # noqa: E402
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (  # noqa: E402
    build_final_configuration_fedsda_run_settings,
)


def measure(*, dataset_name, random_seed, per_client_sample_count, aggregation_interval, client_count):
    default_run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset_name,
            random_seed=random_seed,
            client_count=client_count,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=aggregation_interval,
        )
    )
    results = {}
    for policy_name in PENDING_MODEL_REGISTRATION_POLICIES:
        run_participant_settings = default_run_settings.run_participant_settings
        experiment_run = execute_fedsda_experiment_run(
            run_settings=replace(
                default_run_settings,
                run_participant_settings=replace(
                    run_participant_settings,
                    run_client_settings=replace(
                        run_participant_settings.run_client_settings,
                        pending_model_registration_settings=PendingModelRegistrationSettings(
                            pending_model_registration_policy=policy_name
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
        results[policy_name] = dict(
            prediction_accuracy=run_metrics.prediction_accuracy,
            stable_period_prediction_accuracy=run_metrics.stable_period_prediction_accuracy,
            uploaded_byte_count=communication_volume.uploaded_byte_count,
            downloaded_byte_count=communication_volume.downloaded_byte_count,
            final_global_model_count=run_metrics.final_global_model_count,
            held_model_count_at_run_end=held_model_count_at_run_end,
            unregistered_adopted_model_count=unregistered_adopted_model_count,
            pending_model_count_at_run_end=pending_model_count_at_run_end,
            detection_count=run_metrics.training_model_switch_detection_metrics.detection_count,
        )
    return results


if __name__ == "__main__":
    dataset_name, random_seed, per_client_sample_count, aggregation_interval = sys.argv[1:5]
    client_count = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    print(
        "MEASURED",
        json.dumps(
            dict(
                dataset_name=dataset_name,
                random_seed=int(random_seed),
                per_client_sample_count=int(per_client_sample_count),
                aggregation_interval=int(aggregation_interval),
                client_count=client_count,
                results=measure(
                    dataset_name=dataset_name,
                    random_seed=int(random_seed),
                    per_client_sample_count=int(per_client_sample_count),
                    aggregation_interval=int(aggregation_interval),
                    client_count=client_count,
                ),
            )
        ),
        flush=True,
    )
