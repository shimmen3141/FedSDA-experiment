# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `run_provenance`（module） | `core/run_provenance.py` | 1 runを作ったコードの来歴（gitの版と、期待する動作の基準版） |
| `BEHAVIOR_BASELINE_VERSION` | 同上 | 今のコードが満たす、期待する動作の基準版（重要動作goldenの現在の版と同じ番号） |
| `RunProvenance` | 同上 | 1 runの来歴の値（`code_commit`・`has_uncommitted_changes`・`code_revision_unavailable_reason`・`behavior_baseline_version`） |
| `convert_run_provenance_to_mapping`・`read_run_provenance` | 同上 | 要約の`run_provenance`の辞書との変換 |
| `ExistingRunBehaviorBaselineMismatchError` | `sweep/experiment_sweep_execution.py` | 出力先の済んだrunの基準版が今と違うため、掃引を始めずに止めたことを表す例外 |
| `build_run_provenance` | `sweep/experiment_sweep_records.py` | gitの版と基準版から、runの来歴を組み立てる |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `save_{fedsda,feddrift,fedavg}_experiment_run` | 要約は、設定・指標・環境 | 必須の`run_provenance`を受け、要約へ書く（形式の版2） |
| `load_{fedsda,feddrift,fedavg}_run_artifacts` | 形式の版1だけを読む | 版1（来歴なし）と版2を読む |
| `find_existing_run_state` | `missing`・`completed`・`conflicting` | 基準版の違う済んだrunを`different_behavior_baseline`とする |
| `execute_experiment_sweep`・`execute_planned_experiment_run` | — | 必須の`run_provenance`を受ける。基準版の違う済んだrunがあれば、何も実行せずに例外 |
| `describe_code_revision` | 未commitの変更に、追跡しないファイルを含む | 追跡するファイルの変更だけ |
| `write_run_summary_table` | 識別子・構成・条件・指標 | 条件の次に、来歴の3列 |
