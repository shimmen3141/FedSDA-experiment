# Design Document

## Overview
基準版10の既定に、2つの扱いを足して既定にする。研究ツール（`tools/research_provisional_candidate_expert/run.py`）の`_two_new_model_learners_policy`（と`_copied_learner_prequential_code_length`）と`_cross_validated_initial_model_statistics_policy`の計算を、製品の部品として書き直す。

| 設定 | 基準版10 | 新しい既定（基準版11） |
|---|---|---|
| 「新しいモデル」の仮説の学習者（新しい項目`TrainingDataAssignmentSettings.new_model_hypothesis_learners`） | （項目なし。`reinitialized_learner`に当たる） | `reinitialized_and_warm_started_learners` |
| 初期モデルの損失の統計の推定（新しい項目`FedsdaRunParticipantSettings.initial_model_loss_statistics_estimation`） | （項目なし。`resubstitution`に当たる） | `five_fold_cross_validation` |

## Boundary Commitments
### This Spec Owns
2つの設定の項目と検査、warm startの学習者の予測的符号長、学習者の選択（短い方＋ln K）、最小記述長の選択への受け渡しと選択の記録の列、初期モデルの統計の交差検証、参加者の準備（`FedsdaRunParticipantFactory.prepare_run`）での使い分け、最終構成の既定と構成の扱い、旧と比べる設定の入口、基準版のhelper（版1〜10の明示、版11）、既定に依るtest、研究ツールの基点の固定と照合の変種、文書。
### Out of Boundary
検出器、最小記述長の選択の他の仮説と切り替わりの符号長、帰属の保留、候補の学習と採用、登録、クロス評価、サーバのクラスタリングと統合、予測、候補検証、運用の途中の統計、FedDrift・FedAvg、固定旧。
### Allowed Dependencies
新しいmoduleは作らない（既存のmoduleに関数・型・定数を足す）。依存境界の許可へ、新しくimportする名前を登録する。純粋な計算のmoduleは、今と同じくtorchもownerもimportしない。
### Revalidation Triggers
最小記述長の選択を通る全test、参加者の準備を通る全test、既定のbuilderを通るtest、掃引のrunの識別子、基準版のhelper、研究ツールの変種。

## Architecture
### 設定（Requirement 1.1、1.6、2.1、2.3、3.1）
- `TrainingDataAssignmentSettings`に必須の項目`new_model_hypothesis_learners`（値は`reinitialized_learner`と`reinitialized_and_warm_started_learners`）を足す。値の定数（`REINITIALIZED_LEARNER_POLICY_NAME`、`REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME`、組`NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES`）は、同じ設定のmodule（`training_data_assignment_settings.py`。`alarm_interval_assignment_policy`の定数と同じ置き場所）に置く。
- `alarm_interval_assignment_policy`が`detector_change_span_and_mean_loss_increase_tolerance`で、`new_model_hypothesis_learners`が`reinitialized_and_warm_started_learners`の組合せは、clientの設定の検査（`FedsdaRunClientSettings`の、今の`alarm_interval_assignment_policy`と候補検証の組合せの検査と同じ場所）で拒否する（Requirement 1.6）。
- `FedsdaRunParticipantSettings`に必須の項目`initial_model_loss_statistics_estimation`（値は`resubstitution`と`five_fold_cross_validation`）を足す。値の定数（`RESUBSTITUTION_ESTIMATION_NAME`、`FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME`、組`INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES`）と、分割の数の定数`INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT`（＝5）は、`runtime/initial_model_pretraining.py`に置く。`five_fold_cross_validation`で`initial_model_pretraining_settings.pretraining_sample_count`が5より小さければ、参加者の設定の検査で拒否する（Requirement 2.3）。
  - 分割の数を設定の項目にしないのは、評価した値（5）だけを既定にし、使わない数値の設定を増やさないため。値の名前に数を含める。
- 最終構成の既定は、2つとも新しい値。構成`fedsda-with-candidate-validation`は、今の値を明示する（`_replace_candidate_acceptance_and_shared_feature_update`に、2つの項目も一緒に置き換える引数を足す。組合せの検査があるので、片方ずつ置き換えない）。ほかのFedSDAの構成は既定を引き継ぐ。
- 新しい数値の設定は足さない（ln 2は学習者の数から決まる。warm startの学習者は、初期化し直した学習者と同じoptimizerの設定を使う）。

### 学習者の予測的符号長（`learning/training/prequential_code_length.py`、Requirement 1.3）
- 学習者の名前をそろえるため、今の`compute_new_model_prequential_code_length`を`compute_reinitialized_learner_prequential_code_length`へ、seedの`derive_new_model_initialization_seed`（定義は`runtime/alarm_interval_minimum_description_length_selection.py`）を`derive_reinitialized_learner_initialization_seed`へ改名する（値・処理は変えない。基準版10では学習者が1つで、新しいモデルの仮説の符号長と同じだったが、学習者が2つになると、仮説の符号長は`select_new_model_learner`の値になる）。研究ツールの差し替えの対象の名前も合わせる。
- 今の`compute_reinitialized_learner_prequential_code_length`（改名の後）の、標本を順に予測して学ぶ部分を、非公開の関数`_accumulate_prequential_code_length(*, learner, input_features, observed_class_labels, parameter_optimizer_settings)`へ切り出す（処理と浮動小数点の順は今と同じ。今の関数は、初期化し直してからこれを呼ぶ）。
- 新しい関数`compute_warm_started_learner_prequential_code_length(*, template_classifier, input_features, observed_class_labels, parameter_optimizer_settings)`: `template_classifier`の写しを、初期化をせずに`_accumulate_prequential_code_length`へ渡す。乱数を使わない。入力の検査は今の関数と同じ（seedの検査を除く）。研究ツールの`_copied_learner_prequential_code_length`と同じ計算。

### 学習者の選択（純粋な計算。`methods/fedsda/training_data_assignment/minimum_description_length_assignment_selection.py`、Requirement 1.2、1.5）
- 定数: 学習者の名前`REINITIALIZED_LEARNER_NAME`（`"reinitialized_learner"`）、`WARM_STARTED_LEARNER_NAME`（`"warm_started_learner"`）、該当なしを表す`NO_NEW_MODEL_LEARNER_NAME`（`""`）。
- `NewModelLearnerSelection`（frozen dataclass）: `learner_name`（短かった学習者の名前）、`description_length`（その学習者の予測的符号長＋学習者の選択の符号長）。
- `select_new_model_learner(*, prequential_code_lengths_by_learner_name)`: 入力は（学習者の名前、予測的符号長）の組の、空でないtuple（名前は重複しない、符号長は有限）。最小の符号長の学習者（同じ値なら先の学習者）を選び、符号長に`log(学習者の数)`を足す。学習者が1つなら、足す値は0（`log(1)`）で、基準版10の値と同じになる。
  - 研究ツールは`min(fresh, copied) + math.log(2.0)`。同じ値になる（`log(2)`の計算と足す順を揃える）。
- `MinimumDescriptionLengthAssignmentSelection`に項目`new_model_learner_name`を足す（「新しいモデル」を選んだときは`REINITIALIZED_LEARNER_NAME`か`WARM_STARTED_LEARNER_NAME`、ほかの仮説では`NO_NEW_MODEL_LEARNER_NAME`。検査する）。
- `select_assignment_by_minimum_description_length`に引数`new_model_learner_names_by_change_interval_sample_count`（変化区間の長さごとの、短かった学習者の名前。`new_model_code_lengths_by_change_interval_sample_count`と同じ鍵）を足し、選んだ仮説が「新しいモデル」なら、その長さの名前を結果に入れる。選択の規則（符号長の計算、同じ長さの順）は変えない。

### 最小記述長の選択への受け渡し（`runtime/alarm_interval_minimum_description_length_selection.py`、Requirement 1.2、1.4）
- `select_alarm_interval_assignment`に引数`new_model_hypothesis_learners`を足す（`process_observed_sample`が、clientの設定から渡す）。
- 変化区間の長さkごとに、今と同じseedで`compute_reinitialized_learner_prequential_code_length`を計算する。`reinitialized_and_warm_started_learners`なら、続けて`compute_warm_started_learner_prequential_code_length`を計算する（研究ツールと同じ順: 初期化し直した学習者→warm startの学習者）。`select_new_model_learner`で、kの符号長と学習者の名前を決める。
- `reinitialized_learner`なら、warm startの学習者を計算せず、`select_new_model_learner`へ初期化し直した学習者だけを渡す（符号長は基準版10と同じ値）。

### 選択の記録（Requirement 1.5）
- `AlarmIntervalAssignmentSelectionRecord`に列`new_model_learner_name`を足し、記録の表`alarm_interval_assignment_selection.new_model_learner_name`に含める。

### 初期モデルの統計の交差検証（`runtime/initial_model_pretraining.py`、Requirement 2.2、2.4、2.5）
- `PretrainedInitialModel`に項目`pretraining_samples`（事前学習の標本の、生成の順のtuple。学習の並べ替えの前に写す）を足す。FedDrift・FedAvgは読まない。
- 今の`pretrain_initial_model`の、分類器とoptimizerの状態を作り、標本をepochごとに並べ替えて学習する部分を、非公開の関数へ切り出す（乱数の消費の順——分類器の初期化→標本の生成→並べ替え——は今と同じ。標本の生成は切り出さない）。
- 新しい関数`estimate_cross_validated_loss_statistics(*, pretraining_samples, fold_count, initial_model_pretraining_settings, model_architecture_settings, input_feature_count, hidden_layer_widths, class_count, parameter_optimizer_settings)`: Requirement 2.2の手順で、out-of-foldの有界の損失から`ModelAndClassLossStatistics`を作る。組iの分類器は、`isolated_cpu_torch_random_state(random_seed=i)`の中で作って学習し、並べ替えは`random.Random(i)`で行う。研究ツールと同じ計算（研究ツールは、標本の生成を記録して再生し、`pretrain_initial_model`を`fold_count`回呼んでいた。各回の、学習に使った標本での統計の計算は捨てていたので、製品では行わない。乱数を消費しない計算なので、値は変わらない）。
- `FedsdaRunParticipantFactory.prepare_run`: `pretrain_initial_model`の後、`five_fold_cross_validation`なら`estimate_cross_validated_loss_statistics`（`fold_count`は定数5）で統計を作り、clientの組立てとサーバの組立てへ渡す統計を、それに替える。`resubstitution`なら今と同じ。
- FedDrift・FedAvgの参加者の準備は変えない（`PretrainedInitialModel`の項目が増えるだけ）。

### 計算回数（Requirement 5.5）
- 計算回数の計測（`measure_model_computation`）は、全体run（参加者の準備を含む）の間、processの全体でモデルの順伝播とoptimizerの更新を数える（NEW-005で確かめた仕組み）。warm startの学習者と一時的な分類器も、この仕組みで数えられる。testで、それぞれの計算の中だけを計測で囲み、計数が0より大きく、全体runで新しい既定と今の扱いの計数の差がそれを含むことを確かめる。

### 旧と比べる設定の入口（Requirement 4.1）
- `use_legacy_compatible_policies`は、2つの項目を今の扱いにする。実旧との照合に使う設定の組（`test_fedsda_stream_protocol_run.py`の組立て、fresh script）は、今の扱いを明示する。

### 基準版（Requirement 4.2、5）
- 版1〜10: clientの設定の「後に足された項目」の表へ`new_model_hypothesis_learners`＝`reinitialized_learner`を、参加者の設定の「後に足された項目」の表へ`initial_model_loss_statistics_estimation`＝`resubstitution`を、版1〜10とも足す。版1〜10の比べる形から、記録の表の列`alarm_interval_assignment_selection.new_model_learner_name`を取り除く。
- 版11を足す（ケースは版10と同じ21）。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を11へ上げるのは、goldenを置いた後。
- 版10→11の差の説明のtest: 両方の版を実行した結果（`capture_case_once`の版10と版11）を比べる。FedDrift・FedAvgのケースと、候補検証ありの構成のケースは、`run_metrics`・`record_tables`（版11の結果から上の列を取り除いたもの）・`global_parameters`・`coverage`（版11で足す項目を取り除いたもの）が版10と同じ。設定は、別のtestで、新しい既定を引き継ぐFedSDAのケースの2つの項目だけが違うことを確かめる。
- 経路の記録（`coverage`）へ、版11から、選択の記録のうち「新しいモデル」で、warm startの学習者が短かった選択の回数を足し、新しい既定の大きい条件のケースで1回以上を必須の経路にする（Requirement 5.3）。どのケースで起きるかは、実装の前の確認で決める（`fedsda-sine2-adoption-without-validation`と`fedsda-mnist2-adoption-without-validation`から選び、どちらでも起きなければ、ケースの選び直しを決める）。

### 研究ツール（Requirement 6）
- 照合（6.1）: 既定を変える前に、2つの新しい扱いを設定で選ぶだけの変種（`default-warm-started-learner-and-cross-validated-initial-statistics`）を足す。照合のscript（Git管理外、Implementation Notesに場所を書く）で、手元の2500標本の条件（MNIST2・SINE2・CIRCLE2・SEA4、間隔100、seed 0）で、研究ツールの組み合わせの変種とこの変種の、精度・下り・上り・予測の記録のdigest・最終のパラメータのdigestが一致し、計算回数は、学習の計数が一致し、推論の計数の差が、研究ツールだけが行う、各組の一時的な分類器での学習に使った標本の評価（組ごとに学習に使った標本を1件ずつ1回の順伝播。5組で事前学習の標本の数の4倍）の分だけであることを確かめる（製品は、研究ツールが捨てていたこの評価を行わない）。
- 基点の固定（6.2）: 既定を変えた後、研究ツールの全部の変種の設定の組立てで、2つの項目を基準版10の値（`reinitialized_learner`、`resubstitution`）に戻してから、各変種の差し替えを行う（研究ツールの変種は、基準版10の既定を基点として評価してきたため）。製品の既定をそのまま使う変種`product-default`を足す（照合の変種の役割を引き継ぐ）。

## 旧と違う点
既定を使う実行のFedSDAの構成で、初期モデルの損失の統計（検出器の基準）と、警報の後の「新しいモデル」の仮説の符号長が変わる。その後の警報・帰属・学習・登録・通信・予測が変わる。候補検証ありの構成と、FedDrift・FedAvgは変わらない。

## Validation Design
- 純粋な計算: 学習者の選択（短い方、同じ値の順、学習者が1つなら足す値が0、ln 2を足す）、選択の結果の学習者の名前（3種類の仮説で）、検査。期待値はtestの中で直接書く。
- 学習者の計算: warm startの学習者（runの乱数を変えない、現在のモデルを変えない、初期化し直した学習者と同じ手順——初期化だけを除いた同じ値になる場合の照合）、`compute_reinitialized_learner_prequential_code_length`の値が切り出しの前と同じ。
- 交差検証: 組の分け方と足す順、runの乱数を変えない、初期モデルのパラメータとoptimizerの状態を変えない、標本の数が5より少ない設定の拒否、`pretrain_initial_model`の値と乱数の消費が切り出しの前と同じ（今の関数の既存のtestと、実旧との照合）。研究ツールの`_cross_validated_initial_model_statistics_policy`を、testの中で照合の対照に使わない（研究ツールは製品の部品ではない）。研究ツールとの一致は、全体runの照合（6.1）で確かめる。
- 接続: 今の扱いで、warm startの学習者と一時的な分類器が1回も作られないことを、呼ばれたら失敗する差し替えで確かめる。新しい扱いで、選択の記録に学習者の名前が残ること、統計が替わることを確かめる。
- 計算回数（Requirement 5.5）。
- 設定の検査、既定、全構成の組立て。
- 基準版: 版1〜10の照合（値を変えずに）、版11の照合、版10→11の差の説明、必須の経路。
- 研究ツールとの照合と基点の固定（Requirement 6）。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜11と、このspecのtest）、fresh smoke。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/training_data_assignment/`: `training_data_assignment_settings.py`、`minimum_description_length_assignment_selection.py`、`alarm_interval_assignment_selection_record_store.py`。
- `src/federated_learning_experiments/learning/training/prequential_code_length.py`。
- `src/federated_learning_experiments/runtime/`: `alarm_interval_minimum_description_length_selection.py`、`observed_sample_processing.py`、`initial_model_pretraining.py`、`fedsda_run_participant_factory.py`、`fedsda_run_client_settings.py`（組合せの検査）、`fedsda_run_client.py`（受け渡し）、`fedsda_final_configuration_run_settings.py`、`fedsda_run_record_tables.py`（列）、`core/run_provenance.py`。
- `src/federated_learning_experiments/sweep/experiment_configurations.py`（候補検証ありの構成）。
- `tests/refactoring/`: 部品と接続のtest、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v11/`、既定に依るtest、依存境界、`fresh_process_smoke.py`。
- `tools/research_provisional_candidate_expert/run.py`（照合の変種、基点の固定）。
- 文書: `docs/overview/proposed-method.md`、`docs/components/`（再利用と検出器の基準の説明）、`docs/experiments/refactoring-baseline.md`、`docs/experiments/experiment-sweep.md`、研究記録（NEW-006、LEGACY-020、各README）、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.6 | 設定、学習者の予測的符号長、学習者の選択、最小記述長の選択への受け渡し、選択の記録、部品と接続のtest |
| 2.1〜2.5 | 設定、初期モデルの統計の交差検証、参加者の準備、部品と接続のtest |
| 3.1 | 既定、構成の扱い、構成のtest |
| 4.1〜4.2 | 旧と比べる入口、版1〜10の明示と照合 |
| 5.1〜5.5 | 版11、版10→11の差の説明、必須の経路、版1〜10の照合と、作らないことのtest、計算回数のtest |
| 6.1〜6.2 | 研究ツールの照合の変種と照合のscript、基点の固定 |
| 7.1 | 文書 |
