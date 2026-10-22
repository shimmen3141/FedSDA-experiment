# 命名と役割

命名の方針（2026-10-22ユーザー指示）: 会話で出ただけの語（「写しの学習者」「2つの学習者」「組み合わせ」など）を使わず、意味の取りやすい語と、研究分野で一般的な語を使う。中心の語は次のとおり。

- warm start（既存のモデルを初期値にして学習を始めること。機械学習で一般的な語）と、reinitialize（初期化し直す）。会話の「写しの学習者」は、warm startの学習者（warm-started learner）と呼ぶ。
- 学習者の選択の符号長（MDLで、有限個の候補に一様な事前分布を置いたときのln K）。
- 再代入の推定（resubstitution estimate）と、k分割交差検証（k-fold cross-validation）、out-of-foldの損失。会話の「交差検証の初期統計」は、初期モデルの損失の統計の推定（initial model loss statistics estimation）と呼ぶ。

## 新しい公開名

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `REINITIALIZED_LEARNER_POLICY_NAME`、`REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME`、`NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES` | `training_data_assignment_settings`の定数 | 設定`new_model_hypothesis_learners`の2つの値（初期化し直した学習者だけ／初期化し直した学習者とwarm startの学習者）と、その組 | 下の学習者の名前は、選択の記録に残す個々の学習者。こちらは設定の値（学習者の組） |
| `validate_new_model_hypothesis_learners` | `training_data_assignment_settings`の関数 | 設定の値が、builtin strの2つの値のどちらかであることを検査する | `validate_alarm_interval_assignment_policy`は選び方の検査 |
| `compute_warm_started_learner_prequential_code_length` | `learning/training/prequential_code_length`の関数 | 現在のモデルの写しを初期化せずに学ばせる学習者（warm startの学習者）の予測的符号長。乱数を使わない | `compute_reinitialized_learner_prequential_code_length`は、概念ごとの部分を初期化し直した学習者。2つは並ぶ関数で、どちらも学習者の名前を持つ |
| `compute_reinitialized_learner_prequential_code_length` | `learning/training/prequential_code_length`の関数 | 現在のモデルの写しの概念ごとの部分を初期化し直した学習者の予測的符号長（基準版10の`compute_new_model_prequential_code_length`を改名。値・処理は同じ。下の改名の表） | `compute_warm_started_learner_prequential_code_length`は初期化しない学習者 |
| `derive_reinitialized_learner_initialization_seed` | `runtime/alarm_interval_minimum_description_length_selection`の関数 | 初期化し直す学習者の初期化のseed（基準版10の`derive_new_model_initialization_seed`を改名。値は同じ） | |
| `REINITIALIZED_LEARNER_NAME`、`WARM_STARTED_LEARNER_NAME`、`NO_NEW_MODEL_LEARNER_NAME` | `minimum_description_length_assignment_selection`の定数 | 選択の記録に残す、「新しいモデル」の仮説で短かった学習者の名前と、該当なし（ほかの仮説） | 上の設定の値とは別の語彙（設定の値`reinitialized_learner`と、学習者の名前`REINITIALIZED_LEARNER_NAME`は、同じ文字列で、同じ学習者を指す） |
| `NewModelLearnerSelection` | 同じmoduleの型（frozen dataclass） | 短かった学習者の名前と、その予測的符号長＋学習者の選択の符号長 | `MinimumDescriptionLengthAssignmentSelection`は仮説の選択の結果 |
| `select_new_model_learner` | 同じmoduleの関数 | 学習者ごとの予測的符号長から、最短の学習者を選び、ln(学習者の数)を足す | `select_assignment_by_minimum_description_length`は仮説を選ぶ |
| `RESUBSTITUTION_ESTIMATION_NAME`、`FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME`、`INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES` | `runtime/initial_model_pretraining`の定数 | 設定`initial_model_loss_statistics_estimation`の2つの値（再代入の推定／5分割交差検証）と、その組 | |
| `INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT` | 同じmoduleの定数 | 交差検証の分割の数（5） | |
| `estimate_cross_validated_loss_statistics` | 同じmoduleの関数 | 事前学習の標本のk分割交差検証で、out-of-foldの有界の損失から、全体と観測クラス別の統計を作る。runの乱数を使わない | `pretrain_initial_model`は初期モデルを学習し、再代入の推定の統計を返す |
| 記録の表の列`alarm_interval_assignment_selection.new_model_learner_name` | runの記録の表 | 選択の記録の、短かった学習者の名前 | |
| 基準版11 | goldenの版 | 2つの新しい扱い（新しい既定）の値 | |

設定の項目`TrainingDataAssignmentSettings.new_model_hypothesis_learners`、`FedsdaRunParticipantSettings.initial_model_loss_statistics_estimation`、型の項目（`PretrainedInitialModel.pretraining_samples`、`MinimumDescriptionLengthAssignmentSelection.new_model_learner_name`、`AlarmIntervalAssignmentSelectionRecord.new_model_learner_name`）、関数の引数は、設計に書く。研究ツールの変種の名前（`default-warm-started-learner-and-cross-validated-initial-statistics`、`product-default`）は、研究ツールの中の名前で、sourceの公開名ではない。

## 既存の公開名の改名（役割は変えない）

| これまで | これから | 理由 |
|---|---|---|
| `compute_new_model_prequential_code_length` | `compute_reinitialized_learner_prequential_code_length` | 学習者が2つになり、仮説（新しいモデル）ではなく、学習者（初期化し直した学習者）の符号長を測る関数になる。warm startの学習者の関数と名前をそろえる（2026-10-22ユーザー指摘） |
| `derive_new_model_initialization_seed` | `derive_reinitialized_learner_initialization_seed` | 同じ理由（初期化し直す学習者のseed） |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `select_alarm_interval_assignment` | 「新しいモデル」の仮説は、初期化し直した学習者だけ | 設定により、2つの学習者の短い方＋ln 2 |
| `select_assignment_by_minimum_description_length`、`MinimumDescriptionLengthAssignmentSelection` | 仮説の選択だけ | 「新しいモデル」を選んだときの学習者の名前も持つ |
| `pretrain_initial_model`、`PretrainedInitialModel` | 初期モデルと再代入の推定の統計 | 事前学習の標本（生成の順）も返す（値と乱数の消費は同じ） |
| `FedsdaRunParticipantFactory.prepare_run` | 初期モデルの統計は再代入の推定 | 設定により、交差検証の統計 |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 10 | 11 |
