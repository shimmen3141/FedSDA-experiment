"""短い警報のあとの保留の扱い（LEGACY-002の修正の前後）で、1 runの結果がどう変わるかを測る（測定用。sourceは変えない）。

使い方（worktreeのルートで、検証用の環境変数を置いて）:
    python .kiro/specs/too-short-alarm-earlier-sample-release/measure_too_short_alarm_policy_effect.py \
        <構成の名前> <dataset> <seed> <clientあたりの標本数> <集約間隔> [<client数>]

構成の設定（登録の扱いは既定）の、短い警報のあとの保留の扱いだけを替えて、2回実行し、結果を1行のJSONで出す。
`keep_all_pending_samples`が、固定旧と同じ扱い（修正の前）、
`release_earlier_samples_assigned_to_current_model`が、既定（修正の後）。
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
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (  # noqa: E402
    PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (  # noqa: E402
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep.experiment_configurations import (  # noqa: E402
    build_experiment_run_settings,
)


def measure(
    *,
    configuration_name,
    dataset_name,
    random_seed,
    per_client_sample_count,
    aggregation_interval,
    client_count,
):
    default_run_settings = build_experiment_run_settings(
        configuration_name=configuration_name,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset_name,
            random_seed=random_seed,
            client_count=client_count,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=aggregation_interval,
        ),
    )
    results = {}
    for policy_name in PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES:
        run_participant_settings = default_run_settings.run_participant_settings
        run_client_settings = run_participant_settings.run_client_settings
        experiment_run = execute_fedsda_experiment_run(
            run_settings=replace(
                default_run_settings,
                run_participant_settings=replace(
                    run_participant_settings,
                    run_client_settings=replace(
                        run_client_settings,
                        training_data_assignment_settings=replace(
                            run_client_settings.training_data_assignment_settings,
                            pending_sample_retention_after_too_short_change_interval_policy=policy_name,
                        ),
                    ),
                ),
            )
        )
        training_sample_count = 0
        repeated_training_sample_assignment_count = 0
        for run_client in experiment_run.measured_run.participants.client_operations:
            training_sample_object_ids = [
                id(training_sample)
                for collection in run_client.owners.training_sample_store.snapshot_ordered_model_training_samples()
                for training_sample in collection.training_samples
            ]
            training_sample_count += len(training_sample_object_ids)
            repeated_training_sample_assignment_count += len(training_sample_object_ids) - len(
                set(training_sample_object_ids)
            )
        run_metrics = experiment_run.run_metrics
        communication_volume = run_metrics.communication_volume
        change_response_metrics = run_metrics.change_response_metrics
        results[policy_name] = dict(
            prediction_accuracy=run_metrics.prediction_accuracy,
            stable_period_prediction_accuracy=run_metrics.stable_period_prediction_accuracy,
            uploaded_byte_count=communication_volume.uploaded_byte_count,
            downloaded_byte_count=communication_volume.downloaded_byte_count,
            final_global_model_count=run_metrics.final_global_model_count,
            detection_count=run_metrics.training_model_switch_detection_metrics.detection_count,
            too_short_alarm_count=change_response_metrics.adaptation_action_metrics.skip_too_short_change_interval_count,
            training_sample_count=training_sample_count,
            repeated_training_sample_assignment_count=repeated_training_sample_assignment_count,
            change_point_mean_absolute_error_sample_count=change_response_metrics.change_point_estimation_metrics.mean_absolute_error_sample_count,
        )
    return results


if __name__ == "__main__":
    configuration_name, dataset_name, random_seed, per_client_sample_count, aggregation_interval = (
        sys.argv[1:6]
    )
    client_count = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    print(
        "MEASURED",
        json.dumps(
            dict(
                configuration_name=configuration_name,
                dataset_name=dataset_name,
                random_seed=int(random_seed),
                per_client_sample_count=int(per_client_sample_count),
                aggregation_interval=int(aggregation_interval),
                client_count=client_count,
                results=measure(
                    configuration_name=configuration_name,
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
