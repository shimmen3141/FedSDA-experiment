# Design Document

## Overview
研究ツールの変種R1（`tools/research_provisional_candidate_expert/reverse_prequential.py`の`default-reverse-prequential-grid1`）の計算を、製品の部品として書き直し、設定の選択肢と構成として足す。既定は変えない。

| 設定 | 既定（基準版15、変えない） | 新しい選択肢 |
|---|---|---|
| 変化区間の候補の探し方（新しい項目`TrainingDataAssignmentSettings.change_interval_search`） | `minimum_interval_multiples_with_forward_prequential_code` | `every_sample_with_reverse_prequential_code` |

## Boundary Commitments
### This Spec Owns
設定の項目と値の定数・検査・組合せの検査、学習者の予測的符号長の途中までの和（逆向きの1本の流れの計算の部品）、初期化し直した学習者の1本の流れ用のseed、最小記述長の選択への受け渡しと、kの候補の刻み、新しい構成、基準版1〜15の「後に足された項目」の表、runの識別子のtest、研究ツールとの照合、文書。
### Out of Boundary
既定の値、goldenの版、検出器、現在のモデルのまま・保有モデルへの仮説の計算、切り替わりの符号長、同じ長さのときの優先順、学習者の種類と選択の規則、件数不足の判定、帰属の保留、候補の学習と採用、登録、クロス評価、統合、予測、FedDrift・FedAvg、固定旧。
### Allowed Dependencies
新しいmoduleは作らない（既存のmoduleに関数・定数・引数を足す）。依存境界の許可へ、新しくimportする名前を登録する。純粋な計算のmoduleは、今と同じくtorchもownerもimportしない。
### Revalidation Triggers
最小記述長の選択を通る全test、既定のbuilderと構成の一覧を通るtest、掃引のrunの識別子、基準版のhelper、研究ツールの変種。

## Architecture
### 設定（Requirement 1.1、1.2、3）
- `TrainingDataAssignmentSettings`に項目`change_interval_search`を足す。値の定数`MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME`（`"minimum_interval_multiples_with_forward_prequential_code"`）、`EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME`（`"every_sample_with_reverse_prequential_code"`）、組`CHANGE_INTERVAL_SEARCH_NAMES`、検査`validate_change_interval_search`は、同じmodule（`training_data_assignment_settings.py`。`new_model_hypothesis_learners`の定数・検査と同じ置き場所）に置く。項目の定義の形（必須か既定値つきか、検査の呼び方）は、既存の項目`new_model_hypothesis_learners`にそろえる。
- `alarm_interval_assignment_policy`が`detector_change_span_and_mean_loss_increase_tolerance`で、`change_interval_search`が`every_sample_with_reverse_prequential_code`の組合せは、clientの設定の検査（`FedsdaRunClientSettings`の、`new_model_hypothesis_learners`の組合せの検査と同じ場所）で拒否する。
- 最終構成の既定は`minimum_interval_multiples_with_forward_prequential_code`（今の扱い）を明示する。既存の全部の構成は既定を引き継ぐ。
- 構成`fedsda-every-sample-change-interval-search`（定数`EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`）を`sweep/experiment_configurations.py`に足し、`EXPERIMENT_CONFIGURATION_NAMES`へ加える。組立ては、最終構成の既定から`training_data_assignment_settings.change_interval_search`だけを`every_sample_with_reverse_prequential_code`に替える（`fedsda-loss-distance-clustering`と同じく、`build_experiment_run_settings`の分岐で`replace`する）。
- 新しい数値の設定は足さない（kの最小は今の最小変化区間、刻みは1）。

### 予測的符号長の途中までの和（`learning/training/prequential_code_length.py`、Requirement 2.2、2.4）
- 今の非公開の関数`_accumulate_prequential_code_length`（標本を渡された順に、予測して対数損失を足してから1回更新する）を、各標本の後の途中までの和の列も作れるように直す。今の2つの関数（`compute_reinitialized_learner_prequential_code_length`、`compute_warm_started_learner_prequential_code_length`）の値・処理・浮動小数点の足す順は変えない。
- 新しい関数（入力は今の関数と同じ。標本の順は呼出し側が決める）:
  - `compute_reinitialized_learner_prequential_code_length_prefixes(*, template_classifier, input_features, observed_class_labels, parameter_optimizer_settings, initialization_seed)`
  - `compute_warm_started_learner_prequential_code_length_prefixes(*, template_classifier, input_features, observed_class_labels, parameter_optimizer_settings)`
  - 戻り値は、標本の数と同じ長さのbuiltin floatのtupleで、i番目（0始まり）は、渡された順の先頭i+1件の予測的符号長。最後の要素は、同じ入力での今の関数の値と同じ。学習者の初期化・共有部の固定・損失・optimizerの設定・batchの作り方は、今の関数と同じ。runの乱数を消費せず、`template_classifier`を変えない。
  - 途中までの和は、今の`_accumulate_prequential_code_length`の足し方（標本1件ずつ、`evaluate_classifier_per_sample_log_losses`の1件分の値の`.sum()`をbuiltin floatにして足す）のまま、各標本を足した直後の値を順に残して作る。研究ツールの`reverse_prequential_prefixes`も、製品の学習の中の同じ評価の値を、同じ順に1件ずつ足しているので、値は完全に一致する（許容誤差を置かない）。

### seed（`runtime/alarm_interval_minimum_description_length_selection.py`、Requirement 2.4）
- 新しい関数`derive_reinitialized_learner_single_pass_initialization_seed(*, client_id, alarm_sample_index)`: 2つの値を10進の文字列にして`:`で区切った文字列（例: `3:1250`）のUTF-8のSHA-256の先頭8 byteを、big endianの符号なし整数にしたもの（研究ツールの変種と同じ値）。今の`derive_reinitialized_learner_initialization_seed`（kを含む）は変えない。

### 最小記述長の選択への受け渡し（Requirement 1.3、2）
- `select_alarm_interval_assignment`に引数`change_interval_search`を足す（`process_observed_sample`が、clientの設定から渡す）。
- `minimum_interval_multiples_with_forward_prequential_code`: 今と同じ（kごとに前向きに測る。値・乱数・計算回数は基準版15と同じ）。
- `every_sample_with_reverse_prequential_code`:
  - 列Wの長さnが最小変化区間（`change_interval_grid_sample_count`）以上なら、列Wを新しい方から古い方へ並べ替えた入力で、初期化し直した学習者の途中までの和を`compute_reinitialized_learner_prequential_code_length_prefixes`（seedは`derive_reinitialized_learner_single_pass_initialization_seed`）で、設定が`reinitialized_and_warm_started_learners`ならwarm startの学習者の途中までの和を`compute_warm_started_learner_prequential_code_length_prefixes`で、この順に1回ずつ計算する。kの候補（最小変化区間以上・n以下の全部の整数）ごとに、各学習者の途中までの和のk番目（先頭k件）を予測的符号長として`select_new_model_learner`へ渡す。
  - nが最小変化区間より短ければ、学習者を計算せず、kの候補は空（今と同じく「現在のモデルのまま」になる）。
- 純粋な計算`select_assignment_by_minimum_description_length`（`methods/fedsda/training_data_assignment/minimum_description_length_assignment_selection.py`）に、必須の引数`change_interval_step_sample_count`（正の整数）を足す。既存の引数`change_interval_grid_sample_count`は、kの最小（最小変化区間）の意味のまま残す（名前は変えない）。kの候補は`range(change_interval_grid_sample_count, n + 1, change_interval_step_sample_count)`とし、入力の符号長の辞書（`new_model_code_lengths_by_change_interval_sample_count`と学習者の名前の辞書）の鍵がその候補とちょうど同じであることを検査する。刻みは呼出し側（`select_alarm_interval_assignment`）が設定から決める: 今の扱いは刻み＝`change_interval_grid_sample_count`（値は基準版15と同じ）、新しい扱いは刻み1。仮説の調べる順（kを短い順、各kで保有モデル→新しいモデル）、厳密に短いときだけ置き換える規則、足す順は変えない。研究ツールの`select_reverse_assignment`（位置の項なし）と同じ規則になる。

### 計算回数（Requirement 5.2）
- 測定用の学習者の計算は、今と同じく警報の処理の中で計算回数の指標（`model_computation_counts`）に入る。新しい扱いでは、学習者ごとに警報1回あたりn回の更新になる。

### 基準版とrunの識別子（Requirement 4.1）
- goldenのJSON（`tests/refactoring/behavior_regression_goldens/v1`〜`v15`）は変えない。設定の項目が増えるので、goldenに保存された設定と比べるときの「後に足された項目」の表（`tests/refactoring/behavior_golden_cases.py`の`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`）へ、版1〜15の全部に、パス`("training_data_assignment_settings", "change_interval_search")`＝`minimum_interval_multiples_with_forward_prequential_code`を足す（`new_model_hypothesis_learners`の扱い——`_NEW_MODEL_HYPOTHESIS_LEARNERS_PATH`——と同じ形）。記録の表には列を足さないので、比べる形から取り除く列はない。
- runの識別子（設定のhash）は、FedSDAの全構成で変わる（項目が増えるため。値は今の扱い）。`tests/refactoring/test_experiment_sweep.py`ほかの、FedSDAの識別子を直書きしたtestを新しい値に合わせる。FedDrift・FedAvgの識別子（`test_feddrift_experiment_sweep.py`、`test_fedavg_experiment_sweep.py`）は変わらないことを確かめる。結果の照合はデータ・間隔・seedで取るので、実行済みの結果との比較には影響しない。

### 研究ツール（Requirement 5.1）
- 研究ツールに、新しい構成を使う変種`every-sample-change-interval-search-configuration`を足す（設定の差し替えをしない）。照合のscript（Git管理外。Implementation Notesに場所を書く）で、手元の小さい条件（MNIST4・SINE2・SEA4・CIRCLE2、間隔50、seed 0、各client 1000標本程度。列Wが最小変化区間より短い警報が1回以上起きる条件を含める）で、研究ツールの変種R1とこの変種の、精度・下り・上り・予測の記録のdigest・最終のパラメータのdigest・計算回数の指標・選択の記録（警報ごとの変化区間の長さ、仮説、行き先、学習者の名前、符号長）が完全に一致する（許容誤差なし）ことを確かめる。
- 研究ツールとの細部の対応（列を新しい順へ並べ替えること、kの候補、seed、2つの学習者を計算する順——初期化し直した学習者→warm startの学習者——、短い列で学習者を使わないこと）は、上の全体runの一致に加えて、製品の部品のtestで直接確かめる（期待値は、研究ツールを呼ばず、今の関数を並べ替えた入力に当てた値や、seedの文字列から直接計算した値で書く）。

### 本規模の実験（Requirement 7）
- 実装・検証・commit・pushの後、uraniumで次のコマンドを流す（`tools/run_server_sweep.sh`は使わず、過去の本規模の実験と同じく`run_experiment_sweep.py`を直接、tmuxで）。実行はユーザーが行い、主担当がコマンドを渡す（内部確認として`--print-plan`で構成を確かめる。提示するコマンドには含めない）。
  `python run_experiment_sweep.py --configurations fedsda-every-sample-change-interval-search --datasets sine2 circle2 sea2 sea4 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study fedsda-every-sample-change-interval-search --label seeds0-9 --purpose "..."`
- 集計は`tools/research_feddrift_delta_comparison.py compare`（既定のFedSDAとの同じseedの対の差、SD、勝ち/負け、通信、MAC）と、選択の記録の集計（警報の回数、選んだkの分布）で行い、IMPROVE-032と実験の台帳に書く。
- 研究ツールの他の変種の意味は変わらない（新しい項目の既定は今の扱い）。

## 旧と違う点
新しい構成を選んだときだけ、警報の後の変化区間の候補と「新しいモデル」の仮説の符号長が変わり、その後の警報・帰属・学習・登録・通信・予測が変わる。既定と既存の構成、FedDrift・FedAvgは変わらない。

## Validation Design
- 純粋な計算: kの候補の刻み（刻み1と刻み5）、入力の辞書の検査、今の扱いの値が変わらないこと。期待値はtestの中で直接書く。
- 途中までの和: 最後の要素が今の関数の値と同じ、各途中までの和が、同じ順序の先頭i+1件を今の関数で測った値と同じ（初期化し直した学習者は同じseedで）、runの乱数と`template_classifier`を変えない、入力の検査。
- seed: 値の例（研究ツールと同じ計算）と検査。
- 接続: 今の扱いで、途中までの和の関数が1回も呼ばれない（呼ばれたら失敗する差し替え）。新しい扱いで、kの候補が1件刻み、列が短いと学習者を使わず「現在のモデルのまま」、選んだkが選択の記録に残る、測定用の学習者のoptimizerの更新回数が学習者ごとにn回。
- 設定の検査、構成の組立て、全構成の組立て、既定の値が変わらないこと。
- 基準版1〜15の照合（値を変えずに）と実旧との照合。
- 研究ツールとの照合（Requirement 5.1）。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜15と、このspecのtest）、fresh smoke。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/training_data_assignment/`: `training_data_assignment_settings.py`、`minimum_description_length_assignment_selection.py`。
- `src/federated_learning_experiments/learning/training/prequential_code_length.py`。
- `src/federated_learning_experiments/runtime/`: `alarm_interval_minimum_description_length_selection.py`、`observed_sample_processing.py`、`fedsda_run_client_settings.py`（組合せの検査）、`fedsda_run_client.py`（受け渡し）、`fedsda_final_configuration_run_settings.py`（既定の明示）。
- `src/federated_learning_experiments/sweep/experiment_configurations.py`（構成）。
- `tests/refactoring/`: 部品と接続のtest、基準版の「後に足された項目」の表、runの識別子のtest、依存境界、`fresh_process_smoke.py`（必要なら）。
- `tools/research_provisional_candidate_expert/run.py`（照合の変種）。
- 文書: `docs/components/model-reuse.md`、`docs/experiments/experiment-sweep.md`（構成の一覧）、`docs/research/improvement-candidates/improve-032-…md`と一覧、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.3 | 設定、受け渡し（今の扱い）、設定の検査と接続のtest |
| 2.1〜2.7 | 途中までの和、seed、受け渡し（新しい扱い）、純粋な計算の刻み、部品と接続のtest |
| 3.1〜3.2 | 構成、構成と既定のtest |
| 4.1〜4.2 | 基準版1〜15の表と照合、呼ばれないことのtest |
| 5.1〜5.2 | 研究ツールの照合の変種と照合のscript、更新回数のtest |
| 6.1 | 文書 |
| 7.1〜7.2 | 本規模の実験のコマンドと集計 |
