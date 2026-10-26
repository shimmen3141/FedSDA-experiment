# Design Document

## Overview
2つの変更を合わせて基準版13にする。(1) clientの損失の監視（`OverallAndTrueClassLossMonitor`、`methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py`）に、
損失の低下を検出したら基準を下げる扱いを足す。研究ツールの`tools/research_provisional_candidate_expert/loss_monitor_rebaseline.py`（`OverallAndTrueClassLossMonitor`の`__init__`・`reset`・
`observe_loss_after_label_observation`を外から包む）を、製品の部品として監視器の中に書き直す。(2) 最終構成のクロス評価の評価clientの上限を、clientの数にする。

| 設定 | これまで | 新しい既定 |
|---|---|---|
| 監視の基準の扱い（新しい項目`LossChangeDetectionSettings.loss_monitoring_baseline_update_policy`） | （項目なし。`keep_baseline_until_alarm`に当たる） | `lower_baseline_on_loss_decrease` |
| 1つのモデルを評価させるclientの上限（既存`FedsdaRunParticipantSettings.maximum_evaluating_client_count_per_model`） | 3（`_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL`） | 実験のclientの数（`experiment_run_conditions.client_count`） |

## Boundary Commitments
### This Spec Owns
検出の設定の新しい項目と検査（ADWINとの組合せの拒否）、監視器の下がる向きの検出器と基準の引下げ、監視の計算回数への算入、基準を下げた回数の記録とrunの指標、最終構成の既定（2つ）、
候補検証ありの構成とADWINで監視する構成の明示、旧と比べる設定の入口、基準版のhelper（版1〜12の明示、版13、経路の記録の新しい項目）、既定に依るtest、研究ツールの基点の固定と意味の保持、文書。
### Out of Boundary
上がる向きの検出器の判定・閾値・警報の応答、警報の後の判定、帰属・保留、クロス評価の評価と候補の選び方、クラスタリング・統合、予測の混合、通信量の数え方、FedDrift・FedAvg、固定旧と旧golden、
比の閾値の変種、案B、再設定の回数を減らす案。
### Allowed Dependencies
新しいmoduleは無い。依存境界の許可へ、新しくimportする名前を登録する（監視器のmoduleが`bounded_loss_e_sr_detection`の`BoundedLossESRDetector`を既に使う範囲で足りる見込み）。
### Revalidation Triggers
監視器を通る全test、既定のbuilderを通るtest、掃引のrunの識別子、基準版のhelper、研究ツールの既存の変種、研究用の入口。

## Architecture
### 設定（Requirement 1.1、1.6）
- `loss_change_detection_settings.py`に、値の名前と組を置く: `KEEP_BASELINE_UNTIL_ALARM_POLICY_NAME = "keep_baseline_until_alarm"`、
  `LOWER_BASELINE_ON_LOSS_DECREASE_POLICY_NAME = "lower_baseline_on_loss_decrease"`、`LOSS_MONITORING_BASELINE_UPDATE_POLICIES`（今の`SHORT_CHANGE_INTERVAL_ALARM_POLICIES`と同じ形）。
- `LossChangeDetectionSettings`に必須の項目`loss_monitoring_baseline_update_policy`を足し、構築時に検査する（builtin strで2つの値のどちらか。`lower_baseline_on_loss_decrease`は`drift_detector_name="e_sr"`のときだけ。
  今の「`wait_until_change_interval_reaches_minimum`はe-SRのときだけ」と同じ場所・同じ形の検査）。
- 監視器は、設定（`self._loss_change_detection_settings`）からこの値を読む。コンストラクタの引数は変えない。

### 下がる向きの検出器と基準の引下げ（Requirement 1.2〜1.5）
研究ツールの`loss_monitor_rebaseline_policy`（案A）と同じ規則・同じ順序にする（Requirement 5.4の照合で確かめる）。
- 状態: 「損失の低下で基準を下げる扱い」のときだけ、監視器が非公開の下がる向きの検出器（`BoundedLossESRDetector`）を持つ。作り方は、`baseline_loss_mean = 1 − （全体の検出器の基準）`、
  `false_alarm_control_alpha`・`maximum_retained_candidate_count`・`betting_fractions`は全体の検出器と同じ値。全体の検出器の基準は、制限の後の値（`BoundedLossESRDetector`の内部の基準）を使う。
  基準の範囲[0.01, 1 − 10⁻⁶]（監視へ渡る値は全部`select_loss_monitoring_baseline_mean_loss`を通る）から、1 − 基準は[10⁻⁶, 0.99]で、検出器の制限[10⁻⁶, 1 − 10⁻⁶]に掛からない。
- 作成と`reset`: コンストラクタの最後と、`reset`の最後に、下がる向きの検出器を新しい基準で作り直す（「基準を警報まで保つ扱い」では作らず、`None`のまま）。
- 標本ごと（`observe_loss_after_label_observation`、e-SRの経路）: 今の処理で観測（`LossMonitoringObservation`）を作り、`self._last_observation`へ置くところまで今と同じにする。その後、
  1. 下がる向きの検出器に`1 − observed_loss`を観測させる。
  2. 観測の`drift_detected`が真なら、何もしない（警報の応答で`reset`される）。
  3. 偽で、下がる向きの検出器の合図が真で、`current_model_baseline_loss_mean` < 全体の検出器の基準なら、`self.reset(baseline_loss_mean=current_model_baseline_loss_mean)`を呼ぶ
     （全体の検出器・クラス別の検出器・標本位置・`_last_observation`を消し、下がる向きの検出器も作り直す。警報の応答の`reset`と同じ）。基準を下げた回数を1足す。
  4. 偽で、合図が真で、保存平均が基準以上なら、下がる向きの検出器だけを、今の全体の検出器の基準で作り直す。
  5. 返す観測は、手順の前に作った観測（引下げの前の値）。ただし、監視の計算回数（`component_update_count`、`evaluated_candidate_bet_count`）は、下がる向きの検出器の更新1回と、
     その観測の`evaluated_candidate_bet_count`を足した値にする（Requirement 1.4）。引下げが起きた標本かどうか（`baseline_lowered`、真偽）を観測に足す。
- 引下げは警報ではない: 帰属・保留・適応の記録・予測の重みには触れない（監視器の外の処理は変えない）。
- 状態の写し（`get_state_snapshot`）: 下がる向きの検出器の状態を、写しの新しい項目として足す（「基準を警報まで保つ扱い」では`None`）。実行中に写しから復元する経路は無い（研究ツールの実装で確認済み）。
- ADWINの経路: 設定の検査で「損失の低下で基準を下げる扱い」を拒否するので、下がる向きの検出器は作られない。

### 基準を下げた回数の記録（Requirement 1.5、5.3）
- clientは、観測の`baseline_lowered`を数え、runの指標に、全clientの合計を`loss_monitoring_baseline_lowering_count`として足す（置き場所は、今の警報の数などの検出の指標と同じ集計の場所。
  記録の表は足さない）。経路の記録（goldenの`coverage`）に、版13から同じ名前で足す（FedSDAのケースだけ）。

### 評価clientの上限（Requirement 2.1）
- 最終構成の既定（`fedsda_final_configuration_run_settings.py`）の`maximum_evaluating_client_count_per_model`を、`experiment_run_conditions.client_count`にする。定数`_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL`は、
  候補検証ありの構成と版1〜12の明示に使う値（3）として名前を変えて残すか、使う場所に値を書くかは実装で決め、同じ値3の出どころを1つにする。

### 構成（Requirement 3.1）
- 最終構成の既定を`lower_baseline_on_loss_decrease`と上限なしにする。候補検証ありの構成（`sweep/experiment_configurations.py`の`fedsda-with-candidate-validation`）は、
  `keep_baseline_until_alarm`と上限3を明示する（版12の`cross_evaluation_client_selection_policy`の明示の隣）。ADWINで監視する2構成は、`keep_baseline_until_alarm`を明示し（検査で必須）、上限は引き継ぐ。

### 旧と比べる設定の入口（Requirement 4.1）
- testの旧と比べる設定の入口（`use_legacy_compatible_policies`）と、実旧との照合に使う設定の組（`test_fedsda_stream_protocol_run.py`の組立て、fresh script、`test_fedsda_run_client.py`・`test_fedsda_run_server.py`の組立て、
  ほかに`LossChangeDetectionSettings`を組み立てる箇所）は、`keep_baseline_until_alarm`を明示する。上限は、その組が今使っている値を明示する（既定のbuilderから取っていた組は3）。

### 基準版（Requirement 4.2、5.1〜5.3）
- 版1〜12: clientの設定の「後に足された項目」の表へ、`("loss_change_detection_settings", "loss_monitoring_baseline_update_policy")`＝`keep_baseline_until_alarm`を、版1〜12とも足す。
  参加者の設定の上限は、既存の項目で値が変わるので、「版の後に値が変わった項目」の扱い（clientの設定の`_CLIENT_SETTING_VALUES_CHANGED_AFTER_BASELINE_VERSION`と同じ形）を参加者の設定にも用意し、
  版1〜12で上限3を明示する（保存してある設定は上限の項目を持つので、比べる前に除かない）。照合は、既存の版ごとの照合のtestを全部の版で通す。
- 経路の記録に、版13から`loss_monitoring_baseline_lowering_count`を足す（FedSDAのケースだけ）。
- 版13を足す（ケースは版12と同じ22）。必須の経路: 候補検証ありの構成で実行する4ケース（`CANDIDATE_VALIDATION_PATH_CASE_NAMES`）で`loss_monitoring_baseline_lowering_count`が0。
  新しい既定を引き継ぐFedSDAのケースのうち、`fedsda-sine2-adoption-without-validation`と`fedsda-mnist4-warm-started-learner`で1以上（goldenの生成の前に主担当が値を確かめ、0なら
  基準を下げるケースへ選び直し、理由をtasks.mdのImplementation Notesへ書く）。版12までの必須の経路は、版13でも同じに求める。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`を13へ上げるのは、goldenを置いた後。
- 版12→13の差の説明のtest: 版12と版13を実行し（goldenを使わず、その場で）、FedDrift・FedAvgのケースと候補検証ありの構成のケースは、`run_metrics`・`record_tables`・`global_parameters`・
  `coverage`（版13で足した項目を除く）が同じ。新しい既定を引き継ぐFedSDAのケースは、`run_settings`が新しい項目と上限だけ違う。

### 研究ツールとの照合（Requirement 5.4）
- 既定を変える前に（task 1の後）、製品の全体run（基準版12の既定の設定に、`loss_monitoring_baseline_update_policy="lower_baseline_on_loss_decrease"`と上限＝clientの数だけを置き換えた設定。
  一時scriptで`execute_fedsda_experiment_run`と保存を呼ぶ）と、研究用の入口の`product-default-all-evaluating-clients-loss-monitor-decrease-rebaseline`（製品の掃引のraw）を、
  MNIST2と、SINE2・CIRCLE2・SEA4（各1 seed、10 client、1500〜2500標本、間隔50または100。基準の引下げが1回以上、評価clientの候補が3を超える同期を含む条件）で比べる。
- 一致を求める項目: 精度、通信量（`communication_volume`の全項目）、予測の記録の要約、最終のパラメータの要約、`model_computation_counts`・`preparation_model_computation_counts`、
  記録の表の全配列（`numpy.array_equal`。監視の計算回数に当たる列があれば、それを除く）、基準を下げた回数（製品の指標と、研究の診断の`rebaseline_count`）。
- 監視の計算回数: 製品の値から、下がる向きの検出器の分（各標本の更新1回と、その観測の候補×賭け率の数）を引くと、研究の値に一致することを確かめる。
- 結果はtasks.mdのImplementation Notesへ記録する（Git管理外の一時script）。

### 研究ツールの基点（Requirement 6）
- 研究用CLI（`tools/research_provisional_candidate_expert/run.py`の`_execute_job`）: `product-default`系以外の変種を基準版10の既定へ戻す置き換えに、
  `loss_monitoring_baseline_update_policy="keep_baseline_until_alarm"`と上限3を足す。IMPROVE-031の変種（`LOSS_MONITOR_REBASELINE_VARIANTS`）は、変種を当てる前に、
  製品の2つの値を基準版12の値（`keep_baseline_until_alarm`、上限3）に戻す（組み合わせた変種は、その後に上限をclientの数にする。今の扱いと同じ）。
- 研究用の入口（`tools/research_loss_monitor_rebaseline.py`）: 計画を作る間、製品の最終構成のbuilderが作る設定を、IMPROVE-031の変種では基準版12の2つの値に戻す
  （今は上限の定数を差し替えている。builderが定数を使わなくなるので、builderの結果を置き換える形にする）。`product-default`は製品の既定（基準版13）のまま。
- 研究ツールの`product-default-all-evaluating-clients`（上限の撤廃だけ）は、既定を変えた後は製品の既定と同じではない（案Aが既定に入るため）。変種の説明にそう書く。
- 固定の確認（Requirement 6.3）: 既定を替える前後で、研究用CLIの`default`と`product-default-loss-monitor-decrease-rebaseline`（SINE2・seed 0・10 client・500〜600標本・間隔50または100）と、
  研究用の入口の組み合わせた変種（同じ程度の条件）の結果が、実行秒数を除いて同じことを確かめる。

## 旧と違う点
既定を使う実行のFedSDAの構成で、(1) 損失の監視が、今のモデルの損失の水準が固定された基準より下がったと検出したとき、基準を下げる（全体とクラス別の検出器を作り直す）。
その後の概念の変化の検出が早まり、警報・警報の後の判定・帰属・学習が変わる。監視の計算回数が増える。(2) クロス評価で、標本を返せるclientの全部を評価させる（上限のための無作為の選択と乱数の消費が無くなる）。
クロス評価のメッセージと、clientの評価の計算が増える。統合の判定の入力が増え、違う概念の統合が減ることがある。

## Validation Design
testは、重要なもの・後で効くもの・複雑なものに絞る（ユーザー指示）。
- 部品（重要）: 監視器を直接動かし、(i) 下がる向きの検出器の増分が 1 + λ(μ − ℓ)/(1 − μ)（境界の基準0.01と1 − 10⁻⁶を含む）、(ii) 損失が基準より低い列で、有限の標本で1回だけ基準が下がり、返す観測は引下げの前の値、
  (iii) 損失の平均が基準と同じ列では下がらない、(iv) 警報の合図の標本では下がらない、(v) 保存平均が基準以上なら下がる向きの検出器だけ作り直す、(vi) 警報の応答の`reset`で下がる向きの検出器も作り直される、
  (vii) 監視の計算回数への算入、(viii) 「基準を警報まで保つ扱い」で、観測・計算回数・状態の写しが今と同じ。研究ツールの`checks_loss_monitor_rebaseline.py`の部品の検査を参考にしてよい。
- 設定の検査（後で効く）: 不正な値・型の拒否と、ADWINとの組合せの拒否を1つずつ。
- 構成（後で効く）: 既定と全FedSDA構成の2つの値（候補検証ありの構成・ADWINの2構成の明示）。
- 全体run（複雑）: 研究ツールとの照合（一時script）、基準版12→13の差の説明と版13の必須の経路（goldenのtest）。
- 既存の照合: 基準版1〜12、実旧との照合、全pytest、Ruff、Pyright、WSL（goldenの版1〜13と、このspecのtest）。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/loss_change_detection/`: `loss_change_detection_settings.py`、`overall_and_true_class_loss_monitoring.py`。
- `src/federated_learning_experiments/runtime/`: `fedsda_final_configuration_run_settings.py`、基準を下げた回数を集計する場所（clientの処理と、runの指標）。`sweep/experiment_configurations.py`、`core/run_provenance.py`（版の定数）。
- `tests/refactoring/`: 新規`test_fedsda_loss_monitor_rebaseline_and_all_evaluating_clients.py`、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v13/`、
  既存の組立て（`test_fedsda_stream_protocol_run.py`、`test_fedsda_run_client.py`、`test_fedsda_run_server.py`、`fresh_process_smoke.py`ほか`LossChangeDetectionSettings`を組み立てるtest）、依存境界、既定に依るtest。
- `tools/research_provisional_candidate_expert/run.py`、`tools/research_loss_monitor_rebaseline.py`（基点の固定と意味の保持、変種の説明）。研究ツールの専用の検証の期待値（必要なら）。
- 文書: `docs/overview/proposed-method.md`、`docs/experiments/refactoring-baseline.md`、`docs/experiments/experiment-sweep.md`、LEGACY-023・LEGACY-021・実装の不具合の一覧、IMPROVE-031、
  `docs/research/fedsda-v12-v13-comparison-and-candidate-freezing-20261026.md`、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.6 | 新しい項目と検査、下がる向きの検出器と引下げ、計算回数、回数の記録、部品のtest |
| 2.1 | 上限の既定、構成のtest |
| 3.1 | 既定、候補検証ありの構成・ADWINの2構成の明示、構成のtest |
| 4.1〜4.2 | 旧と比べる入口、版1〜12の明示と照合 |
| 5.1〜5.4 | 版13、版12→13の差の説明、引下げの回数の経路、研究ツールとの照合 |
| 6.1〜6.3 | 研究ツールの基点の固定、IMPROVE-031の変種の意味の保持、固定の確認 |
| 7.1 | 文書 |
