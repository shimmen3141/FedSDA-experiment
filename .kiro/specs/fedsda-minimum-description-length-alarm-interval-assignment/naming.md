# 命名と役割

命名の方針（2026-10-21ユーザー指示）: 会話の中の造語（「区分けの比較」「統合推論」「案A」など）を使わず、理解しやすい語と、研究分野で一般的な語を使う。中心の語は、最小記述長（minimum description length、MDL）、符号長（description length / code length）、予測的符号長（prequential code length）、切り替わり（switching）、変化点・変化区間（change point / change interval）。

## 新しい公開名

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `minimum_description_length_assignment_selection` | `methods/fedsda/training_data_assignment/`のmodule | 警報の後の変化区間の長さと帰属先を、最小記述長の原理で選ぶ純粋な計算（torch・ownerを使わない）と、選び方の設定の値 | `training_data_assignment_settings`は設定の型。こちらは計算 |
| `DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME`、`MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME`、`ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES` | 同じmoduleの定数 | 選び方の2つの値（今の扱い＝検出器の推定の変化の長さと、履歴の平均損失からの増加の許容量で選ぶ／最小記述長の原理で選ぶ）と、その組 | |
| `validate_alarm_interval_assignment_policy` | 同じmoduleの関数 | 選び方が、builtin strの2つの値のどちらかであることを検査する | |
| `compute_switching_code_length` | 同じmoduleの関数 | 帰属先が1回切り替わることの符号長（−ln(切り替わりの確率)＋ln(切り替わり先の数)） | 予測の重み（Fixed-Share）は、同じ確率を重みの分け合いに使う。こちらは符号長 |
| `MinimumDescriptionLengthAssignmentSelection` | 同じmoduleの型（frozen dataclass） | 選んだ仮説（現在のモデルのまま／保有モデルへ切り替わる／新しいモデルへ切り替わる）、変化区間の長さ、行き先の保有モデルのID、符号長 | 今の`AlarmIntervalModelReuseAssessment`は、履歴の基準と許容量で「合う」モデルの列を持つ評価。こちらは選択の結果 |
| `select_assignment_by_minimum_description_length` | 同じmoduleの関数 | 標本ごとの対数損失の列と、新しいモデルの予測的符号長から、符号長が最短の仮説を選ぶ | |
| `alarm_interval_minimum_description_length_selection` | `runtime/`のmodule | 帰属の保留の標本の列と、保有モデルのowner・損失の統計から、learningの層の計算をまとめて、選択を返す | 純粋な計算のmoduleと分ける（ownerを使う側） |
| `classifier_log_loss_evaluation` | `learning/prediction/`のmodule | 分類器の標本ごとの対数損失 | 既存の`classifier_bounded_loss_evaluation`は有界の損失 |
| `evaluate_classifier_per_sample_log_losses` | `classifier_log_loss_evaluation`の関数 | 標本ごとの対数損失（−log 真のクラスの確率）。既存の有界の損失（1−真のクラスの確率）から計算する | 既存の`evaluate_classifier_per_sample_bounded_losses`は有界の損失 |
| `prequential_code_length` | `learning/training/`のmodule | 新しいモデルの予測的符号長の計算 | |
| `compute_new_model_prequential_code_length` | `prequential_code_length`の関数 | 概念ごとの部分を初期化し直した学習者で、標本を予測してから学ぶ形の符号長（予測的符号長）。runの乱数を変えない | 候補の学習（`candidate_epoch_training`）は候補のパラメータを作る。こちらは符号長を測るだけで、モデルを残さない |
| `derive_new_model_initialization_seed` | `alarm_interval_minimum_description_length_selection`の関数 | 予測的符号長の学習者の初期化のseedを、clientの番号・警報の標本の番号・変化区間の長さから決める | |
| `select_alarm_interval_assignment` | `alarm_interval_minimum_description_length_selection`の関数 | 帰属の保留の標本の列で、上の計算をまとめて行い、選択を返す | |
| `AlarmIntervalAssignmentSelectionRecordStore` | `evaluation/`の新しいmoduleの型 | 最小記述長の選択ごとの記録（警報の標本の番号、選んだ仮説、変化区間の長さ、行き先、符号長、警報にしたか）を持つ | 警報の記録（`LossChangeAlarmRecordStore`）は検出器の警報。こちらは警報の後の選択 |
| 記録の表`alarm_interval_assignment_selection.*` | runの記録の表 | 上の記録の列 | |
| 基準版10 | goldenの版 | 最小記述長の原理で選ぶ扱い（新しい既定）の値 | |

設定の項目`TrainingDataAssignmentSettings.alarm_interval_assignment_policy`と、関数の引数は、設計に書く。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `process_observed_sample` | 警報の変化区間の長さは、検出器の推定 | 新しい扱いでは、最小記述長の選択の長さ。選択を警報の処理へ渡す |
| `handle_alarm_occurrence`、`respond_to_alarm_with_buffered_samples`、`resolve_alarm_change_interval` | 再利用は、履歴の基準と許容量で選ぶ | 選択が渡されれば、その行き先で再利用か新しいモデルかを決める |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 9 | 10 |
