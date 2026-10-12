# 公開名

全て`federated_learning_experiments.runtime`配下。既存FedSDAの対応する単一run境界と役割を揃え、型は手法を区別する。

| 公開名 | 役割・区別 |
|---|---|
| `feddrift_run_settings` | FedDriftの1 runの完全条件。client/serverだけの設定とは異なる |
| `FeddriftRunSettings` | 実行・参加者・指標条件を束ねて検証する型 |
| `feddrift_experiment_run` | 完全条件から実行と指標を得る入口 |
| `FeddriftExperimentRun` | 条件・計測結果・指標を持つ結果。計数だけのMeasuredRunとは異なる |
| `execute_feddrift_experiment_run` | 既存計測runと既存指標導出を接続する |
| `feddrift_run_record_tables` | FedDriftの記録の表化。判断や状態変更はしない |
| `build_feddrift_run_record_tables` | 名前付き配列を構築する |
| `feddrift_run_artifact_files` | FedDriftの新JSON/NPZを保存・読込む |
| `FeddriftRunArtifactPaths` | 保存した要約と表のファイル位置 |
| `save_feddrift_experiment_run` | 既存出力を拒否して実験runを保存する |
| `load_feddrift_run_artifacts` | 要約と対応する新記録表を読込む |
