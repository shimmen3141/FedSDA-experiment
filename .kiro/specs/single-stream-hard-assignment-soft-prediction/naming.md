# 命名と役割

パッケージ名は、手法の中身（固い割り当て＋柔らかい予測、hard assignment and soft prediction）と、単一ストリームであることを表す。論文での呼び名（FedHASP-HTなど）は論文側で決め、コードの名前には連合学習（Fed）を含めない。

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `single_stream_hard_assignment_soft_prediction` | `src/`の最上位パッケージ | FedHASPの1クライアントの純粋な形をHoeffding treeで動かす単一ストリームの手法 | 製品の`federated_learning_experiments`（連合学習、神経網）とは別。製品をimportするが、製品からはimportされない |
| `hoeffding_tree_concept_model` / `HoeffdingTreeConceptModel` | 同上 | 概念ごとのモデル1つ（Hoeffding tree）。平滑化した確率、監視用・MDL用の損失、1件の学習、写し、空の新しいモデル | 製品の`ModelArchitectureClassifier`（神経網）の代わり。共有部を持たない |
| `single_stream_settings` / `SingleStreamHardAssignmentSoftPredictionSettings` / `build_default_single_stream_settings` | 同上 | 手法の設定と既定値の組立て（監視・Fixed-Shareの設定は製品の型をそのまま持つ） | 製品の`FedsdaRunSettings`（連合学習のrun全体）ではない |
| `alarm_interval_assignment` / `select_alarm_interval_assignment` | 同上 | 警報の時点の列で、木の損失と予測的符号長を求め、製品の比較の関数で変化区間と帰属先を選ぶ | 製品の`runtime.alarm_interval_minimum_description_length_selection.select_alarm_interval_assignment`と同じ役割の木の版。名前は同じだが別のmoduleで、符号長を神経網ではなく木で計算し、torchを使わない。製品のものと混同しないよう、importはmodule名つきで書く |
| `single_stream_processing` / `SingleStreamHardAssignmentSoftPredictionProcessor` | 同上 | 1標本の処理（予測→重みの更新→監視→保留→警報の判定→割り当て・学習）と記録を所有する | 製品の`process_observed_sample`の木の版。同期・集約を持たない |
| `SingleStreamHardAssignmentSoftPredictionProcessor.process_sample` | 同上 | 標本1件を、上の順で処理し、その標本の記録を返す | 製品の`process_observed_sample`（関数）に当たるメソッド |
| `SampleProcessingRecord` | 同上 | 標本ごとの記録（予測クラス、正誤、観測クラス、真のコンセプト、現在のモデル、保有モデル数、監視の対数e値） | 製品の`SamplePredictionRecord`より項目が少ない |
| `AlarmRecord` | 同上 | 警報ごとの記録（標本の番号、選んだ仮説、k、帰属先、新しいモデルの学習者、符号長、警報としたか） | 製品の警報の記録と選択の記録を1つにしたもの |
| `single_stream_run` / `run_single_stream` / `SingleStreamRunResult` / `compute_prequential_metrics` | 同上 | 系列1本の処理、記録、指標（精度、Kappa、Kappa-M、Kappa-Temporal）、経過時間 | 製品の`execute_fedsda_experiment_run`（連合学習の全体run）ではない |
| `single_stream_artifacts` / `save_single_stream_run` / `load_single_stream_run` | 同上 | 成果物のJSON・NPZの保存と読込み | 製品の成果物の形式とは別 |
| `single_stream_command` / `main` | 同上 | 系列のファイルを並列に処理し、成果物と一覧のCSVを書く実行コマンド | 製品の`run_experiment_sweep.py`（連合学習の掃引）ではない |
