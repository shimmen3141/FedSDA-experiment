"""新実装の掃引の結果（run_summary.csv）を、旧実装の結果のCSV（run_pareto_sweep.pyのpareto_*.csv）と比べる。

同じdataset・seed・集約間隔のrunどうしで、主な指標が、完全に同じ値かを調べる（提案構成の、新旧の一致の確認用）。

使い方:
    python compare_sweep_results_with_legacy.py <新のrun_summary.csv> <旧のpareto_*.csv> [<旧のCSV> ...]

終了コード: 0＝対応する全部のrunで、全部の指標が一致。1＝違いがある、または、対応するrunがない。
"""

import csv
import sys

# 旧のCSVの列の名前から、新の一覧の表の列の名前への対応。
NEW_COLUMN_BY_LEGACY_COLUMN = {
    "accuracy": "prediction_accuracy",
    "stable_accuracy": "stable_period_prediction_accuracy",
    "precision": "training_model_switch_detection_metrics.detection_precision",
    "recall": "training_model_switch_detection_metrics.detection_recall",
    "f1": "training_model_switch_detection_metrics.detection_f1",
    "total_detect": "training_model_switch_detection_metrics.detection_count",
    "final_model_count": "final_global_model_count",
    "comm_models_up": "communication_volume.uploaded_model_count",
    "comm_models_down": "communication_volume.downloaded_model_count",
    "comm_bytes_up": "communication_volume.uploaded_byte_count",
    "comm_bytes_down": "communication_volume.downloaded_byte_count",
}


def read_rows(csv_path):
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main(arguments):
    if len(arguments) < 2:
        print(__doc__)
        return 1
    new_rows = {
        (
            row["dataset_name"],
            int(row["random_seed"]),
            int(row["server_aggregation_interval_per_client_samples"]),
        ): row
        for row in read_rows(arguments[0])
    }
    legacy_rows = {}
    for legacy_csv_path in arguments[1:]:
        for row in read_rows(legacy_csv_path):
            legacy_rows[
                (row["dataset"], int(row["seed"]), int(float(row["aggregation_interval"])))
            ] = row
    compared_keys = sorted(set(new_rows) & set(legacy_rows))
    print(f"new runs: {len(new_rows)}, legacy runs: {len(legacy_rows)}, compared: {len(compared_keys)}")
    for missing_key in sorted(set(new_rows) - set(legacy_rows)):
        print(f"only in new: {missing_key}")
    for missing_key in sorted(set(legacy_rows) - set(new_rows)):
        print(f"only in legacy: {missing_key}")
    differing_run_count = 0
    largest_difference_by_column = {}
    for run_key in compared_keys:
        differences = []
        for legacy_column, new_column in NEW_COLUMN_BY_LEGACY_COLUMN.items():
            legacy_value = float(legacy_rows[run_key][legacy_column])
            new_value = float(new_rows[run_key][new_column])
            if legacy_value != new_value:
                differences.append(f"{legacy_column}: legacy={legacy_value!r} new={new_value!r}")
                largest_difference_by_column[legacy_column] = max(
                    largest_difference_by_column.get(legacy_column, 0.0),
                    abs(legacy_value - new_value),
                )
        if differences:
            differing_run_count += 1
            print(f"DIFFERENT {run_key}: " + "; ".join(differences))
    print(f"runs with differences: {differing_run_count} of {len(compared_keys)}")
    for legacy_column, largest_difference in largest_difference_by_column.items():
        print(f"largest absolute difference in {legacy_column}: {largest_difference!r}")
    if compared_keys and not differing_run_count:
        print(f"ALL {len(compared_keys)} COMPARED RUNS MATCH in {len(NEW_COLUMN_BY_LEGACY_COLUMN)} metrics")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
