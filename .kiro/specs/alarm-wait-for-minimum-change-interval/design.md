# Design Document

## Overview
標本1件の処理（`runtime/observed_sample_processing.py`の`process_observed_sample`）は、検出器の合図（`loss_monitoring_observation.drift_detected`、または、ADWINの補助の判定）があると、警報の処理へ進む。ここへ、「待つ扱いのとき、推定変化区間の件数が最小件数より少なければ、警報としない」という条件を、1か所足す。検出器・警報の処理・完了処理は、変えない。

| 方針の値 | 合図があり、推定変化区間が最小件数より短いとき | 使う所 |
|---|---|---|
| `raise_alarm_immediately` | 警報とする（短い警報として処理し、検出器を再開する。いまの扱い） | 既定。旧との照合、基準版1〜3 |
| `wait_until_change_interval_reaches_minimum` | 警報としない（警報の無い標本として処理する。検出器の証拠は残る） | 比較用（採否は未定）。検出器がe-SRのときだけ |

## Boundary Commitments
### This Spec Owns
検出の設定の1項目、標本1件の処理の、警報を出す条件の1か所、clientから標本1件の処理への値の受渡し、比較用のscriptの第3の指定と集計、待つ扱いのtest。
### Out of Boundary
既定の変更、検出器（`OverallAndTrueClassLossMonitor`）の中身、警報の処理（`handle_alarm_occurrence`以下）、完了処理、保留の扱い（LEGACY-002の方針）、新しい基準版のgolden。
### Allowed Dependencies
既存の部品だけ。`observed_sample_processing.py`は、方針の値の定数を、検出の設定のmoduleから読む（依存境界の登録を足す）。
### Revalidation Triggers
`LossChangeDetectionSettings`を組み立てる全箇所（着手時に`grep`で数える。仕様の時点で、sourceは最終構成の設定の1つ、testは6ファイル前後）と、`process_observed_sample`を呼ぶ全箇所（sourceは`fedsda_run_client.py`の1つ。testは、引数の辞書を作るhelper）。run設定の保存（FedSDAのrunの識別子は、全runで変わる）。

## Architecture
### Existing Architecture Analysis
`process_observed_sample`の流れ（関係する所）: 予測→損失の観測（`loss_change_monitor.observe_loss_after_label_observation`）→候補検証の進行→保留へ標本を足す→`alarm_raised`を決める（検出器の合図、または、ADWINの補助の判定）→警報なら、保留中の学習要求を学習し、警報を記録し（推定変化点は、`警報の位置 − min(保留の件数, 検出器の推定の長さ) + 1`）、`handle_alarm_occurrence`（区間の準備、短い警報かどうかの判定、再利用・候補検証、完了処理＝監視の再開と保留の消費）、保留の標本の切詰め。警報でなければ、容量を超えた最古の保留標本の確定と、学習要求。

短い警報かどうかは、`respond_to_alarm_with_buffered_samples`が、区間の準備の後に、変化区間の件数（`min(保留の件数, 推定の長さ)`）と最小件数を比べて決める。候補検証を保持している間の警報は、区間を分けずに、全部を現行モデルへ取り込む（変化区間の件数を見ない）。

e-SRの検出器は、合図の後も、再開（`reset`）されなければ、証拠（候補の開始点ごとの値）を持ち続け、次の標本でも、値がしきい値以上なら、合図を出す（`bounded_loss_e_sr_detection.py`）。推定の長さは、証拠が最大の候補の開始点から決まる。同じ候補が最大であり続ければ、標本ごとに1ずつ伸び、最大の候補が、より新しい開始点へ替われば、短くなる。値がしきい値を下回れば、合図は出ない（2026-10-13の試行で、再開せずに観測を続けても、例外なく動くことを確かめた）。

ADWINの検出器は、検知したときに、窓の古い側を削る（`adwin_loss_window_detection.py`）。再開せずに待つと、削った後の窓（新しい側の数件）が残り、次の標本で、合図が続くとは限らない。補助の判定（保留の末尾の窓で、現行モデルの損失の悪化を見る）も、窓が25件以上になるまで働かない。待つ扱いをADWINへ使うと、変化を取り逃がす、または、大きく遅れることがありうるので、**この組合せは、設定の検査で拒否する**（要求1.4）。ADWINの構成では、[LEGACY-002の修正のspecの計測](../too-short-alarm-earlier-sample-release/tasks.md)（ADWINの2構成、sine2 seed 0・1、標本2500）で、短い警報が0回だった（検知の後の窓が、推定変化区間になり、最小件数より短くなりにくい）ので、待つ扱いの必要も小さい。

### Components and Interfaces
**設定**（`methods/fedsda/loss_change_detection/loss_change_detection_settings.py`）
- `LossChangeDetectionSettings.short_change_interval_alarm_policy: str`。宣言した値は、fieldのmetadataの`allowed_parameter_values`（定数`SHORT_CHANGE_INTERVAL_ALARM_POLICIES`）。既定値は置かない（組み立てる全箇所が、選ぶ）。値の定数（`RAISE_ALARM_IMMEDIATELY_POLICY_NAME`・`WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME`）は、この設定のmoduleに置く。設定の生成時の検査（`__post_init__`）で、`drift_detector_name`が`adwin`で、方針が「待つ」の組合せを、`RunSettingsValidationError`で拒否する（項目どうしの関係の検査。ほかの設定型の、項目どうしの検査と同じ形）。

**標本1件の処理**（`process_observed_sample`）
- 引数`short_change_interval_alarm_policy: str`を足す（宣言した値でなければ、ほかの入力の検査と同じ段で、どの更新より前に拒否する）。
- `alarm_raised`を決めた直後に、次を足す: 方針が「待つ」で、`alarm_raised`が真で、`min(保留の標本の件数, 推定の長さ) < 最小件数`なら、`alarm_raised`を偽にする。保留の標本の件数は、この標本を足した後の件数（警報の処理が読むのと、同じ保留）。推定の長さは、`loss_monitoring_observation.estimated_change_span_sample_count`（警報の処理が使うのと、同じ値）。警報の処理が変化区間とする件数と、同じ式なので、「待つ扱いで警報になったら、短い警報にはならない」が成り立つ（候補検証を保持していないとき）。
- ADWINの補助の判定（`_current_model_loss_worsened_over_adwin_window`）と、それを呼ぶ条件は、変えない（待つ扱いは、ADWINと組み合わせられないので、補助の判定が働く実行では、新しい条件は働かない）。
- 警報としなかった標本は、既存の「警報なし」の分岐を通る（容量を超えた最古の保留標本の確定、保留の標本の切詰め、学習要求）。検出器は、再開されない（再開は、警報の完了処理だけが行う）。
- 戻り値（`ObservedSampleProcessing`）は、警報なしの標本と同じ形（待ったことを表す項目は、足さない）。待った回数は、保存する記録には残らない（検出器の合図の有無は、記録の表に無い）。testでは、検出器の観測を包んで数える。比較の集計は、待った回数を使わない（短い警報の数・警報の総数・切替までの遅れ・検出の指標で比べる）。

**client**（`runtime/fedsda_run_client.py`）: `process_observed_sample`へ、`run_client_settings.loss_change_detection_settings.short_change_interval_alarm_policy`を渡す。

**最終構成の既定**: `raise_alarm_immediately`（変えない）。派生の構成（ADWINの2構成は、検出の設定を`replace`で替える）も、引き継ぐ。

**比較用のscript**（`run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`）
- 第3の指定`--alarm-wait-levels`（`off`＝いまの扱い、`on`＝待つ扱い。既定は`off`だけ）。`on`の組合せの名前は、これまでの名前の末尾に`__alarm-wait-on`を付ける（`off`は、これまでの名前のまま）。
- このspecで、設定の項目が増えるので、`off`の組合せのrunの識別子も、このspecの前と変わる（挙動は同じ）。このspecの前に実行した`off`の結果（研究室サーバで実行中の、4通りの比較）は、識別子が違っても、集計では、そのまま使える（集計は、組合せのdirectoryの`run_summary.csv`を、dataset・集約間隔・seedで対にする）。待つ扱いを足すときは、`--registration-levels fixed --short-alarm-levels fixed --alarm-wait-levels on`のように、足す組合せだけを指定して実行する（`off`を指定すると、識別子が違うので、済んだrunとして飛ばされずに、もう一度実行される）。
- 集計は、待つ扱いの効果（ほかの2つの扱いを固定して、`off`→`on`）の対の差を足す。短い警報の連鎖の表は、組合せごとに出す（既存）。

## System Flows
待つ扱い、候補検証を保持していないとき: 変化の後、検出器の値がしきい値を超える（推定の長さ3）→警報としない（保留は、そのまま。標本は、容量の30件を超えるまで、どのモデルにも取り込まれない）→次の標本で、推定の長さ4→警報としない→次の標本で、推定の長さ5→警報: 保留を、前区間と、変化区間（5件）に分け、再利用の評価か、候補検証の開始へ進む。検出器は、完了処理で再開される。

待つ間に、検出器の値がしきい値を下回った場合: 合図が出ないので、何も起きない（警報の無い標本が続く）。

## 旧と違う点（待つ扱いを選んだとき。既定では、違いは無い）
1. 短い警報が起きない。短い警報のたびに行っていた、前区間の取込み・評価標本の保存（乱数を使う）・短い警報の記録・検出器の再開が、無くなる。
2. 警報の位置（警報の記録の標本位置）は、推定変化区間が最小件数に達した標本になる（いまの扱いより、最大で、最小件数−1標本、遅い）。再利用・候補検証が始まる位置は、いまの扱いで連鎖が起きていた場合より、早い。
3. 候補検証を保持している間の合図も、推定変化区間が最小件数に達するまで、警報としない（いまの扱いは、すぐに、検証中の警報として、保留の全部を現行モデルへ取り込み、検出器を再開する）。
   警報は、合図があり、推定変化区間の件数が最小件数に達した最初の標本で出る（要求1.7）。証拠が最大の候補の開始点が変わらない間は、推定の長さが1ずつ伸びるので、最初の合図から、最小件数より少ない標本数のうちに出る。最大の候補が、より新しい開始点へ替わると、推定の長さが縮み、それより長くかかることがある。待つ間に、検出器の値がしきい値を下回れば、警報は出ない（一時的な跳ねを、変化として扱わない）。待つ間も、保留の容量（最終構成は30件）を超えた最古の標本は、現行モデルへ確定する（警報の無い標本と同じ）。待つのが長引いても、確定するのは、保留の先頭（推定変化区間より、容量−最小件数ぶん以上、前の標本）である。
4. 検出の指標（警報を検出とみなす指標、変化点の推定の誤差）と、その後の全部（学習標本、モデル数、通信量、精度）が変わる。

## Validation Design
- 設定: 宣言した値の受入れと、それ以外の拒否、保存への出力。
- 標本1件の処理（実体のowner。既存の、標本1件の処理のtestの組立てを使う）:
  - いまの扱い: 既存の、実旧との対照testが、方針を明示して、値を変えずに通る（要求2.3）。
  - 待つ扱い、合図があり、推定変化区間が最小件数より短い標本: 結果と、全ownerの状態が、「同じ入力で、検出器の合図が無かった場合」と同じであること（警報の記録・適応の記録が増えない、検出器が再開されない＝検出器の状態が、観測を1件足しただけの状態、保留・学習要求が、警報なしの標本と同じ）。比べ方: 同じ組立てを2つ作り、片方は、検出器の観測の戻り値の`drift_detected`だけを偽に差し替えて、いまの扱いで処理する（検出器の中の状態は、差し替えない。検出器は、どちらの組でも、同じ損失を観測して、同じ状態——最後の観測の`drift_detected`が真であることを含む——になる）。もう片方は、差し替えずに、待つ扱いで処理する。全ownerの状態（検出器の状態を含む）と、乱数の状態が、一致する。
  - 待つ扱い、推定変化区間が最小件数以上の標本: 結果と、全ownerの状態が、いまの扱いで処理した場合と同じ。
  - 候補検証を保持している間（要求1.5）も、同じ2通りで確かめる。
  - 警報の位置（要求1.7）: (a)待つ扱いで処理した標本の列について、警報の処理が行われるのは、「合図があり、推定変化区間の件数が最小件数以上」の標本の全部で、かつ、それだけであること（標本ごとに、検出器の観測と、保留の件数から、条件を計算して、警報の有無と照らす。全体runの観測でも確かめる）。(b)証拠が最大の候補が変わらない、固定の損失の列（基準より十分に高い損失を続ける）で、最初の合図から警報までの標本数が、最小件数より少なく、その警報が、短い警報でないこと（この列での観測として確かめる。一般の保証ではない）。
  - 設定の検査（要求1.4）: ADWINと、待つ扱いの組合せの拒否。
  - 方針の値の拒否（宣言していない値）が、どの更新より前。
- 全体run（要求4.1・4.2）: `fedsda-sine2-events`の設定（基準版3の扱い）で、方針だけを替えて実行する。警報の応答を観測し（既存の、LEGACY-002のspecのtestと同じ差し込み）、待つ扱いでは、短い警報が0で、警報のたびに（候補検証を保持している間の警報を含めて）、推定変化区間の件数——`min(保留の件数, 推定の長さ)`——が最小件数以上であること、同じ標本の2回目の取込みと、登録されないままの採用が0であること。いまの扱いでは、短い警報が1回以上。待つ扱いの、学習帰属の切替の検出の再現率が、いまの扱い以上で、変化から切替までの遅れの平均が、いまの扱い以下であること（要求4.2。この条件での実測の関係の固定。数値計算の環境で値が違いうるので、大小の関係だけを確かめる。WSLで成り立たなければ、記録して報告する）。
- 旧との照合（要求3）: 実旧と比べるtestの設定の組立て（`LossChangeDetectionSettings`を組み立てる箇所。`valid_run_settings_mapping`ほか）に、いまの扱いを明示する。照合の入口のhelper（`assert_run_client_uses_legacy_compatible_policies`・`use_legacy_compatible_policies`）を、この方針にも広げる。方針は、ownerではなく、clientの設定にあるので、clientへ、読取りのproperty（`FedsdaRunClient.short_change_interval_alarm_policy`。既存の`prediction_combination_strategy`と同じ形）を足して、そこから読む。既定がいまの扱いなので、照合の結果は変わらないが、将来、既定を替えたときに、気づかずに通ることを防ぐ。
- golden（要求3.1）: helperの、「版の後に足された設定の項目」へ、基準版1〜3について、この項目（`run_client_settings`→`loss_change_detection_settings`→`short_change_interval_alarm_policy`、値はいまの扱い）を足す。基準版1〜3のファイルは、変更しない。新しい基準版は、足さない（既定の動作が変わらない）。
- 比較用のscript（要求4.3）: 小さい条件（sine2、client 3、標本1500、seed 2つ）で、`--alarm-wait-levels off on`を、直した2つの扱い（既定）と組み合わせて実行し、集計が、待つ扱いの効果の表を出すこと。
- 手元の比較（要求4.4）: 試行と同じ9条件（sine2 seed 0〜2、circle2・sea4・mnist2のseed 0・1。client 10、標本2500、集約間隔50）を、比較用のscriptで、いまの扱いと待つ扱いで実行する。試行（実行の間だけ差し込んだもの）の「待つ案」の結果と照らし、違う条件があれば、記録する（試行は、推定の長さだけで判定した。このspecは、保留の件数との小さい方で判定するので、保留の件数が推定の長さより少ない場合に、差が出うる）。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/loss_change_detection/loss_change_detection_settings.py`: 項目と、値の定数。
- `src/federated_learning_experiments/runtime/observed_sample_processing.py`: 引数と、警報を出す条件。
- `src/federated_learning_experiments/runtime/fedsda_run_client.py`: 値の受渡し。
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`: 既定（いまの扱い）の明示。
- `run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`: 第3の指定と集計。
- `tests/refactoring/`: 新規`test_alarm_wait_for_minimum_change_interval.py`、`LossChangeDetectionSettings`を組み立てる既存testと`fresh_process_smoke.py`（方針の明示）、`process_observed_sample`の引数を作るhelper、`test_fedsda_run_client.py`（照合の入口のhelper）、`test_run_settings_validation.py`（設定の検査の表）、`behavior_golden_cases.py`（足された項目）、依存境界、掃引のrunの識別子の期待値。
- 文書: IMPROVE-006の記録と一覧（実装と、手元の比較の結果）、`docs/experiments/experiment-sweep.md`（比較用のscriptの指定）、再開案内。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1 | 標本1件の処理の条件、標本1件の処理のtest（合図なしの場合・いまの扱いの場合との一致）、全体runのtest |
| 2 | 設定の項目と2つの値、最終構成の既定、いまの扱いの実旧対照、比較用のscriptの指定 |
| 3 | 設定の組立ての明示、照合の入口のhelper、goldenのhelperの、足された項目 |
| 4 | 全体runのtest（提案構成。ADWINの2構成は、待つ扱いと組み合わせられないので、対象外——既定のまま、変わらない）、比較用のscriptの小さい実行、手元の比較 |
