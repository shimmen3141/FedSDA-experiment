# Design Document

## Overview
警報の後の変化点と帰属先を、最小記述長の原理で選ぶ扱いを足し、既定にする。研究ツールの変種`default-detector-gated-segment-assignment`（`tools/research_provisional_candidate_expert/run.py`の`_detector_gated_segment_assignment_policy`）の計算を、製品の部品として書き直す。検出器が警報を出す時点は変えない（誤警報の率を抑える検出器に任せる）。

| 設定 | これまで | 新しい既定 |
|---|---|---|
| 警報の後の変化点と帰属先の選び方（新しい項目`TrainingDataAssignmentSettings.alarm_interval_assignment_policy`） | （項目なし。`detector_change_span_and_mean_loss_increase_tolerance`に当たる） | `minimum_description_length` |

## Boundary Commitments
### This Spec Owns
新しい設定の項目と検査、最小記述長の選択の計算（純粋な計算と、モデルを使う計算）、標本の処理（`process_observed_sample`）と警報の処理（`handle_alarm_occurrence`→`respond_to_alarm_with_buffered_samples`→`resolve_alarm_change_interval`）への受け渡し、最終構成の既定と構成の扱い、旧と比べる設定の入口、基準版のhelper（版1〜9の明示、版10）、既定に依るtest、研究ツールの変種の乱数の扱いの揃え、文書。
### Out of Boundary
検出器、帰属の保留、区間の準備と取込み、候補の学習と採用、登録、クロス評価、サーバのクラスタリングと統合、予測、候補検証、FedDrift・FedAvg、固定旧。
### Allowed Dependencies
新しいmoduleは2つ（下）。依存境界の許可へ、新しくimportする名前を登録する。純粋な計算のmoduleは、torchもownerもimportしない。
### Revalidation Triggers
標本の処理と警報の処理を通る全test、既定のbuilderを通るtest、掃引のrunの識別子、基準版のhelper、研究ツールの変種。

## Architecture
### 設定（Requirement 1.1、2.1）
- `TrainingDataAssignmentSettings`に必須の項目`alarm_interval_assignment_policy`（値は`detector_change_span_and_mean_loss_increase_tolerance`と`minimum_description_length`）を足す。値の組の定数と検査は、純粋な計算のmodule（下）に置き、設定の型の`allowed_parameter_values`はその組を使う。
- 最終構成の既定は`minimum_description_length`。構成`fedsda-with-candidate-validation`は`detector_change_span_and_mean_loss_increase_tolerance`を明示する。候補検証との組合せを拒否するので、構成の組立ての`_replace_candidate_acceptance_and_shared_feature_update`（候補の採否の方針と共有部の更新を一緒に置き換える関数）に、この項目も一緒に置き換える引数を足す（片方ずつ置き換えると、途中の設定が拒否されるため）。ほかのFedSDAの構成（候補検証を外した構成`fedsda-without-candidate-validation`を含む）は既定を引き継ぐ。
- 新しい扱いで使う既存の値: 切り替わりの確率＝1/（`PredictionCombinationSettings.fixed_share_weight_redistribution_time_scale_samples`）、τの刻み＝`FedsdaRunClientScalarSettings.minimum_change_interval_sample_count`、学習者の更新＝`parameter_optimizer_settings`。新しい数値の設定は足さない。
- 新しい扱いと、候補検証の方針（`candidate_model_acceptance_policy`が検証ありのもの）の組合せは、設定の検査で拒否する（未検証のため。候補検証ありの構成は今の扱いを使う）。

### 純粋な計算（新しいmodule `methods/fedsda/training_data_assignment/minimum_description_length_assignment_selection.py`）
- 定数: `DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME`、`MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME`、`ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES`、検査の関数`validate_alarm_interval_assignment_policy`。
- `compute_switching_code_length(*, switching_probability, destination_count)`: −ln(switching_probability)＋ln(destination_count)。確率は(0, 1)、数は1以上。
- `MinimumDescriptionLengthAssignmentSelection`（frozen dataclass）: `change_interval_sample_count`（変化区間の長さ）、`destination_model_id`（行き先の保有モデルのID。新しいモデルならNone）、`selected_hypothesis`（`current_model`・`held_model`・`new_model`のどれか）、`description_length`（選んだ仮説の符号長）。
- `select_assignment_by_minimum_description_length(*, current_model_log_losses, held_model_log_losses_by_model_id, new_model_code_lengths_by_change_interval_sample_count, switching_code_length, change_interval_grid_sample_count)`:
  - 入力は、列（帰属の保留の標本と警報の標本、古い順）の長さnの、現在のモデルの標本ごとの対数損失、行き先の保有モデルごとの標本ごとの対数損失（行き先の順）、変化区間の長さごとの新しいモデルの予測的符号長。
  - 「現在のモデルのまま」を最初の最良とし、変化区間の長さkをgridからnまでgridごとに、各kで行き先の保有モデル（順に）、新しいモデルの順に、符号長＝Σ_{前のn−k} 現在＋Σ_{後のk} 行き先＋切り替わりの符号長、を計算し、最良より**厳密に短い**ときだけ置き換える（同じ長さは、先に計算したものを優先。研究ツールと同じ順）。
  - 「現在のモデルのまま」を選んだら、変化区間の長さはn。
  - 足す順は、研究ツールと同じ（Pythonの`sum`で、前の部分の和に切り替わりの符号長を足し、後の部分の和を足す）にし、浮動小数点の結果を揃える。

### モデルを使う計算（torchを使う計算はlearningの層、ownerをまとめる計算はruntimeの層）
依存境界の規則（runtimeの層は、moduleごとに許可した名前のほか、torchを直接importしない）に合わせ、torchを使う計算を、learningの層の2つの新しいmoduleへ置く（Task 1で、仕様レビューの後に置き場所を決め直した）。
- `learning/prediction/classifier_log_loss_evaluation.py`の`evaluate_classifier_per_sample_log_losses(*, classifier, input_features, observed_class_labels)`: 標本ごとの対数損失＝−log(max(1−有界の損失, 1e-12))（有界の損失は既存の`evaluate_classifier_per_sample_bounded_losses`。1−有界の損失は真のクラスの確率）。研究ツールと同じ式。
- `learning/training/prequential_code_length.py`の`compute_new_model_prequential_code_length(*, template_classifier, input_features, observed_class_labels, parameter_optimizer_settings, initialization_seed)`: 現在のモデルの分類器の写しを作り、既存の`isolated_cpu_torch_random_state`（`torch.random.fork_rng`の中でCPUの乱数だけを設定する。CPUでは、研究ツールの`torch.manual_seed`と同じ値）の中で`initialization_seed`を設定して、概念ごとの部分（`model_owned_feature_transform`と`classification_layer`の、`reset_parameters`を持つ全部の部品）を初期化し直し、残差のadapterの広げる側（`feature_expansion`）を0にする。共有部（`feature_extractor`）以外のパラメータのoptimizerを`create_parameter_optimizer`で作る。標本を順に、対数損失を足してから、共有部を勾配なしで計算し、候補の学習と同じ損失（2クラスはBCE、ほかはcross entropy）で1回更新する。符号長（対数損失の和）を返す。runの乱数（torch・Python・NumPy）は変えない。
- 以下の2つは`runtime/alarm_interval_minimum_description_length_selection.py`（標本の列の連結に`torch.cat`だけを、moduleごとの許可で使う）。
- `derive_new_model_initialization_seed(*, client_id, alarm_sample_index, change_interval_sample_count)`: 3つの非負の整数を、10進の文字列にして`:`で区切って連結した文字列（例: `"3:1250:15"`。区切りがあるので、(1, 23, 4)と(12, 3, 4)は別の文字列になる）のUTF-8のSHA-256を取り、先頭8 byteをbig endianの符号なし整数にしたもの。形式と値をtestで固定する。
- `select_alarm_interval_assignment(*, client_id, alarm_sample_index, pending_sample_observations, held_model_training_state_registry, loss_statistics_store, current_training_model_assignment, parameter_optimizer_settings, switching_probability, change_interval_grid_sample_count)`:
  - 列＝帰属の保留の標本（警報の標本を足した後の、保留の全部。古い順）。
  - 保有モデル（保有の順）ごとに、列の対数損失を計算する（前向きの計算だけ）。
  - 行き先＝損失の統計を持つ（`loss_statistics_store.get_model_loss_statistics`がNoneでない）、現在のモデル以外の保有モデル（保有の順）。
  - 変化区間の長さk（gridの倍数、n以下）ごとに、列の末尾k件で新しいモデルの予測的符号長を計算する（seedは`derive_new_model_initialization_seed`）。
  - 切り替わりの符号長は`compute_switching_code_length(switching_probability, 行き先の数＋1)`。
  - `select_assignment_by_minimum_description_length`の結果を返す。

### 標本の処理への接続（Requirement 1.2〜1.5、1.7）
- `process_observed_sample`に引数`client_id`と`alarm_interval_assignment_policy`を足す（clientが渡す）。
- 検出器の合図（`alarm_raised`）が真で、新しい扱いのとき、保留へ足した後（今の(3)の後）、「短い警報は待つ」の判定の前に、`select_alarm_interval_assignment`を呼ぶ。以後、検出器の推定の変化の長さ（`loss_monitoring_observation.estimated_change_span_sample_count`）の代わりに、選んだ変化区間の長さを使う（「短い警報は待つ」の判定、警報の記録の推定の変化点、`handle_alarm_occurrence`の`estimated_change_span_sample_count`）。選択の結果を`handle_alarm_occurrence`へ渡す。
  - 研究ツールは、選択を損失の監視の戻り値の中で行い、推定の変化の長さを置き換えていた（保留へ足す前に、保留の標本と警報の標本を合わせた列で計算）。列と、モデルの状態（その間にモデルは更新されない）は同じ。
  - 列の長さが刻みより短いと、kの候補が無いので「現在のモデルのまま」で長さn（＜刻み）になり、「短い警報は待つ」の判定で警報にならない（Requirement 1.5。研究ツールと同じ）。
- 今の扱いのときは、選択を呼ばず、今と同じ（Requirement 1.7）。

### 警報の処理への接続（Requirement 1.3、1.4）
- `handle_alarm_occurrence`・`respond_to_alarm_with_buffered_samples`・`resolve_alarm_change_interval`に、引数`alarm_interval_assignment_selection: MinimumDescriptionLengthAssignmentSelection | None`（既定はNone）を足し、受け渡す。
- `resolve_alarm_change_interval`は、今と同じく`evaluate_held_models_for_alarm_interval_reuse`で保有モデルを評価する（区間の平均損失は、候補の初期値の選択——区間の平均損失が最小の保有モデル——に使う。計算も今と同じ）。選択がNoneでなければ、評価の結果の「合うモデルの列」を、選択の行き先だけにする: 行き先が現在のモデル（「現在のモデルのまま」）か保有モデルで、履歴の基準を持つモデルの列に含まれれば、その1件。新しいモデル、または含まれなければ、空（新しいモデル）。研究ツールと同じ。
  - 研究ツールの240 runでは、行き先が履歴の基準を持たない場合は0回だった。「現在のモデルのまま」で現在のモデルが履歴の基準を持たない場合は、新しいモデルになる（研究ツールと同じ。頻度は未確認）。
- それ以後（取込み・切替え・候補の学習と採用）は今と同じ。

### 選択の記録（Requirement 1.8）
- clientのownerに、選択の記録の置き場（新しい型`AlarmIntervalAssignmentSelectionRecordStore`、`evaluation/`の記録の置き場と同じ形）を足し、`process_observed_sample`で選択を行ったときに、警報の標本の番号・選んだ仮説・変化区間の長さ・行き先の保有モデルのID（無ければ−1）・符号長を1行ずつ足す（選択の後、短い警報の判定の前。短い警報で警報にならなかった選択も記録し、その旨の列——警報にしたか——を持つ）。
- FedSDAのrunの記録の表（`build_fedsda_run_record_tables`）へ、表`alarm_interval_assignment_selection.*`として足す（今の扱いでは、空の表）。表の追加は、goldenの`record_tables`の項目を増やすので、版1〜9の照合では、今の扱いで空の表を持つことを、goldenと比べる前に取り除く（基準版のhelperで、版9までの比べる形に、この表を含めない）。

### 旧と比べる設定の入口（Requirement 3.1）
- `use_legacy_compatible_policies`は、`alarm_interval_assignment_policy`を今の扱いにする。実旧との照合に使う設定の組（`test_fedsda_stream_protocol_run.py`の組立て、fresh script）は、今の扱いを明示する。

### 基準版（Requirement 3.2、4）
- 版1〜9: clientの設定の「後に足された項目」の表へ、`alarm_interval_assignment_policy`＝今の扱いを、版1〜9とも足す。
- 版10を足す（ケースは版9と同じ21）。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を10へ上げるのは、goldenを置いた後。
- 版9→10の差の説明のtest: 両方の版を実行した結果（`capture_case_once`の版9と版10）を比べる。FedDrift・FedAvgのケースと、候補検証ありの構成の4ケースは、`run_metrics`・`record_tables`・`global_parameters`・`coverage`が版9と同じ。`record_tables`は、版10の結果から表`alarm_interval_assignment_selection.*`を取り除いてから比べ、取り除いた表が空であること（今の扱いでは選択を行わない）を確かめる（FedDrift・FedAvgのケースは、この表を持たない）。版9の結果は、基準版のhelperが版9までの比べる形で作るので、この表を含まない。`coverage`は、版10で足す項目（選んだ仮説ごとの回数）を取り除いてから比べる。設定は、別のtestで、新しい既定を引き継ぐFedSDAのケースの新しい項目だけが違うことを確かめる。
- 経路の記録（`coverage`）へ、版10から、選択の記録（下の「選択の記録」）の選んだ仮説ごとの回数（`current_model`・`held_model`・`new_model`）を足し、新しい既定の大きい条件のケース（`fedsda-sine2-adoption-without-validation`と`fedsda-mnist2-adoption-without-validation`）で、それぞれ1回以上を必須の経路にする（Requirement 4.3）。その後に実際に行われた処理（現在のモデルの維持・保有モデルの再利用・候補の採用）は、既存の指標`change_response_metrics.adaptation_action_metrics`で別に確かめる（選んだ仮説が「現在のモデルのまま」でも、現在のモデルが履歴の基準を持たなければ新しいモデルになるので、両者は一致しないことがある）。指定したケースで1回も起きない種類があれば、ケースの選び直しを、実装の前の確認で決める。

### 研究ツールとの照合（Requirement 5）
- 研究ツールの変種の新しいモデルの学習者のseedを、`derive_new_model_initialization_seed`と同じ式（clientの番号・警報の標本の番号・変化区間の長さ）に替える（これまでは、run全体の通し番号）。clientの番号は、`process_observed_sample`の呼出しの`indexed_observation`の持ち主から取れないので、研究ツールでは`client_module.FedsdaRunClient`の処理を包んで取る（実装で決める）。
- 照合のscript（Git管理外、Implementation Notesに場所を書く）で、手元の2500標本の条件（MNIST2・SINE2・CIRCLE2・SEA4、間隔100、seed 0）で、研究ツールの変種と製品の既定の全体runの、精度・通信量・予測の記録のdigest・最終のモデルのパラメータのdigest・計算回数が一致することを確かめる。

## 旧と違う点
既定を使う実行のFedSDAの構成で、警報の後の変化区間の長さと帰属先が変わる。その後の学習・登録・通信・予測が変わる。候補検証ありの構成と、FedDrift・FedAvgは変わらない。

## Validation Design
- 純粋な計算: 仮説の符号長と選択（3種類の選択、同じ長さの順、刻みより短い列）、切り替わりの符号長、検査。手で作った小さい対数損失の列を入力にし、期待値は研究ツールの関数を使わずに、testの中で直接書く。
- モデルを使う計算: 対数損失の式、予測的符号長（runの乱数を変えない、同じseedで同じ値、共有部を更新しない、候補の学習と同じ損失）、seedの導出。
- 接続: 今の扱いでは、標本の処理と警報の処理が今と同じ（既存の照合testが通る）。今の扱いで、最小記述長の選択の計算（`select_alarm_interval_assignment`）が1回も呼ばれないことを、呼ばれたら失敗する差し替えで確かめる。計算回数（`model_computation_counts`）は、基準版1〜9の照合（指標の全部を比べる）で、今と同じことを確かめる。新しい扱いで、選択の結果が推定の変化の長さと帰属先に使われる（現在のまま・保有モデル・新しいモデルの3つ）こと、選択の記録が残ることを確かめる。
- 設定の検査（候補検証との組合せの拒否を含む）、既定、全構成の組立て。
- 基準版: 版1〜9の照合（値を変えずに）、版10の照合、版9→10の差の説明、必須の経路。
- 研究ツールとの照合（Requirement 5）。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜10と、このspecのtest）、fresh smoke。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/training_data_assignment/`: 新規`minimum_description_length_assignment_selection.py`、`training_data_assignment_settings.py`。
- `src/federated_learning_experiments/learning/`: 新規`prediction/classifier_log_loss_evaluation.py`、`training/prequential_code_length.py`。
- `src/federated_learning_experiments/runtime/`: 新規`alarm_interval_minimum_description_length_selection.py`、`observed_sample_processing.py`、`alarm_occurrence_handling.py`、`alarm_buffer_response.py`、`alarm_change_interval_resolution.py`、`fedsda_run_client.py`、`fedsda_run_client_settings.py`（組合せの検査）、`fedsda_final_configuration_run_settings.py`、`core/run_provenance.py`。
- `src/federated_learning_experiments/sweep/experiment_configurations.py`（候補検証ありの構成）。
- `tests/refactoring/`: 新規の部品と接続のtest、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v10/`、既定に依るtest、依存境界、`fresh_process_smoke.py`。
- `tools/research_provisional_candidate_expert/run.py`（seedの揃え）。
- 文書: `docs/overview/proposed-method.md`、`docs/components/`（帰属の説明）、`docs/experiments/refactoring-baseline.md`、`docs/experiments/experiment-sweep.md`、研究記録（IMPROVE-021、設計の文書）、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.8 | 設定、純粋な計算、モデルを使う計算、標本の処理と警報の処理への接続、選択の記録、部品と接続のtest |
| 2.1 | 既定、構成の扱い、構成のtest |
| 3.1〜3.2 | 旧と比べる入口、版1〜9の明示と照合 |
| 4.1〜4.4 | 版10、版9→10の差の説明、必須の経路、版1〜9の照合と、選択を呼ばないことのtest |
| 5.1〜5.2 | 研究ツールのseedの揃え、照合のscript |
| 6.1 | 文書 |
