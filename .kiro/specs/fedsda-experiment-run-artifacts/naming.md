# 命名表: fedsda-experiment-run-artifacts

sourceの公開する名前（module、class、公開の関数）と、保存する表の名前を載せる。列の名前は、design.mdの表。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `fedsda_experiment_run` / `FedsdaExperimentRun` / `execute_fedsda_experiment_run` | `runtime/` | 完全なrun設定から、計測つきの全体runと指標の導出までを行い、設定・結果・指標をまとめて返す |
| `fedsda_run_record_tables` / `build_fedsda_run_record_tables` | `runtime/` | 1 runの結果を、表ごと・列ごとの配列にする |
| `fedsda_run_artifact_files` / `save_fedsda_experiment_run` / `load_fedsda_run_artifacts` / `FedsdaRunArtifactPaths` / `RUN_ARTIFACT_FORMAT_VERSION` | `runtime/` | 要約（JSON）と記録の表（NPZ）の保存と読込み |

## 保存する表の名前

`sample_prediction`、`concept_change`、`training_model_switch`、`adaptation`、`candidate_validation_decision`、`model_registration`、`model_clustering`、`model_pair_clustering`、`round`、`round_local_processing_computation`、`round_synchronization_computation`。

## 判断が必要な点

- 「experiment run」は、設定から指標までの1回の実行（既存の「stream protocol run」は、実行の枠の1回、「measured run」は、計測つきの実行の結果）。
- 「record table」は、記録を、列ごとの配列にしたもの。「artifact」は、保存したファイル（既存の規約の「実験成果物」）。
- 表と列の名前は、新の記録の型とfieldの名前に合わせる。旧の配列の名前（`history_accuracy`ほか）は、使わない。
- `adaptation_action`の値は、新の語（`maintain_current_model`ほか）。旧の`action`の値との対応は、(A2-変換)で持つ。
