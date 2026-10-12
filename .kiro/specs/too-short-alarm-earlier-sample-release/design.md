# Design Document

## Overview
保留の設定に、短い警報のあとの保留の扱い（方針）を足す。警報応答の完了処理が、短い警報のとき、方針に応じて、前区間の位置を保留から外す。保留の標本そのもの（`PendingSampleObservationStore`）は、標本1件の処理が、警報の処理の後に、保留の位置に合わせて切り詰める（既存の処理。変えない）。

| 方針の値 | 短い警報のあとの保留 | 使う所 |
|---|---|---|
| `release_earlier_samples_assigned_to_current_model` | 前区間を外し、変化区間を残す | FedSDAの既定 |
| `keep_all_pending_samples` | 全部残す（固定旧と同じ） | 旧との照合、基準版1・2 |

旧と同じ扱いを設定で残す条件（LEGACY-016のspecと同じ）: 分岐が、保留の機能（設定と、保留の位置のowner）と、警報応答の完了処理の1か所に閉じる。既定は直した側で、実験のコマンドの引数には出さない。旧実装を外すときに、値と分岐を一緒に外す（NEW-004の記録へ足す）。

## Boundary Commitments
### This Spec Owns
保留の設定の1項目、保留の位置のownerの「古い側を外す」操作、警報応答の完了処理の短い警報の分岐と記録の検査、最終構成の既定、基準版3のgoldenと、基準版1・2の照合の続け方、影響の計測。
### Out of Boundary
固定旧と旧golden、区間の準備（前区間の評価標本の保存と取込み。`alarm_training_interval_preparation.py`）、検出器の再開の仕方、区間の分け方、容量を超えた標本の確定、FedDrift・FedAvg。
### Allowed Dependencies
既存の部品だけ。sourceの新しいimportは無い（設定の値の定数は、設定のmoduleに置く。保留の位置のownerは、自分の設定から、「短い警報のあとに前区間を外すか」を、真偽のpropertyで返す。完了処理は、そのpropertyを読むだけで、値の定数を読まない）。
### Revalidation Triggers
`TrainingDataAssignmentSettings`を組み立てる全箇所（着手時に`grep`で数える。sourceは最終構成の設定の1つ、testは10ファイル前後）、run設定の保存と検査（`configuration/run_settings_validation.py`が、この設定型を読む）、掃引のrunの識別子（FedSDAの全runで変わる）、完了の記録を読む箇所（`drained_pending_sample_indices`。仕様の時点で、完了処理のmoduleの外に、読む箇所は無い）。

## Architecture
### Existing Architecture Analysis
- 警報のとき、`respond_to_alarm_with_buffered_samples`（`runtime/alarm_buffer_response.py`）が、区間の準備（`prepare_alarm_training_intervals`: 保留を、前区間と変化区間に分け、前区間を、現行モデルの評価標本へ保存し——乱数を使って抜き出す——、現行モデルへ取り込む）を行い、変化区間が最小件数より少なければ、`alarm_change_interval_too_short`を返す。
- `complete_alarm_buffer_response`（`runtime/alarm_response_completion.py`）が、損失の監視を再開し、短い警報でなければ、保留の位置を全部消費する（`drain_pending_sample_indices`）。短い警報では、消費しない。記録（`AlarmResponseCompletion`）は、短い警報で`drained_pending_sample_indices`が空でないと、拒否する。
- 標本1件の処理（`runtime/observed_sample_processing.py`）が、警報の処理の後に、保留の標本を、保留の位置に合わせる（`retain_latest_pending_sample_observations`）。
- 旧`clients/fedsda.py::_resolve_drift`（767〜783行）: `old_data`を評価データへ保存して取り込み、`drift_data`が`MIN_DRIFT_DATA`未満なら、記録して、検出器をresetして戻る（`buffer.clear()`を呼ばない）。
- 短い警報のあと、取込み済みの前区間が保留に残るので、(a)次の警報の区間の準備で、もう一度、前区間として、評価標本へ保存され、取り込まれる、(b)警報のない標本で、容量を超えて確定するときに、もう一度取り込まれる。2026-10-13の計測では、(a)が多い（sine2 seed 0で、(a) 1363件、(b) 308件）。

### Components and Interfaces
**設定**（`methods/fedsda/training_data_assignment/training_data_assignment_settings.py`）
- `TrainingDataAssignmentSettings.pending_sample_retention_after_too_short_change_interval_policy: str`。宣言した値は、fieldのmetadataの`allowed_parameter_values`（定数`PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES`）で、`validate_settings_field_values`が拒否する。既定値は置かない（組み立てる全箇所が、選ぶ）。値の定数は、この設定のmoduleに置く（設定のmoduleは、`core`のほかに依存しない規則）。

**保留の位置のowner**（`PendingTrainingAssignmentBuffer`）
- `releases_earlier_samples_after_too_short_change_interval`: 設定の方針が「前区間を外す」のとき真を返す、読取りのproperty（`capacity_sample_count`と同じく、設定から答える）。
- `release_earliest_pending_sample_indices(*, released_sample_indices)`: 渡された位置が、保留の先頭（最も古い側）の並びと一致することを確かめてから、それだけを外して返す。一致しなければ、何も変えずに拒否する。空のtupleは、何もしない。最後の観測位置は変えない。
  - 既存の`release_sample_indices_exceeding_capacity`（容量を超えた分を外す）、`drain_pending_sample_indices`（全部を外す）は、変えない。

**警報応答の完了処理**（`complete_alarm_buffer_response`。引数は変えない）
- 短い警報のとき、ownerの方針が「前区間を外す」なら、外す位置を、応答の`prepared_alarm_training_intervals.earlier_observations`の位置とする。そうでなければ、空。短い警報でなければ、これまでどおり、保留の位置の全部。
- 順（これまでと同じ）: 検査と記録の組立て（読取りだけ）→損失の監視の再開→保留の位置の消費（短い警報で外すときは`release_earliest_pending_sample_indices`、それ以外は`drain_pending_sample_indices`）。外す位置が、保留の先頭の並びであることは、完了処理が、読取りの段で、明示して確かめる（保留の位置の先頭が、前区間の位置と一致しなければ、監視の再開より前に拒否する。既存の、準備した区間の位置が保留の位置とちょうど一致することの検査に加えて、この分岐の中にも置く）。したがって、完了処理が拒否するときは、監視も保留も変わっていない。`release_earliest_pending_sample_indices`の中の一致の検査は、owner単体の契約で、この経路では拒否しない。
- 記録`AlarmResponseCompletion`の検査: 短い警報のとき、`drained_pending_sample_indices`は、空か、応答の前区間の位置と同じ並び、のどちらか（それ以外は拒否）。短い警報でないときの検査は、これまでどおり。

**変えないもの**
- `AlarmBufferResponse.pending_assignment_buffer_should_be_cleared`（保留を全部消費するか。短い警報では偽）は、名前と意味を変えない。
- 標本1件の処理（`runtime/observed_sample_processing.py`）のsourceは、変えない（警報の処理の後に、保留の標本を、保留の位置に合わせる既存の処理が、前区間の標本を外す）。ただし、この処理が読む保留の中身は変わる（下の「旧と違う点」の4）。候補検証を保持している間の警報（`alarm_during_candidate_validation`）は、全部を現行モデルへ取り込んで、全部を消費する（これまでどおり）。

**最終構成の既定**（`runtime/fedsda_final_configuration_run_settings.py`）: `release_earlier_samples_assigned_to_current_model`。派生の11構成は、引き継ぐ。実験のコマンドの引数には出さない。FedSDAのrunの識別子は、全runで変わる（設定が変わるため）。

## System Flows
短い警報（既定の扱い）: 保留`[e1..ek, c1..cm]`（`m`<最小件数）→準備: `e1..ek`を評価標本へ保存し、現行モデルへ取り込む→完了: 監視を再開、保留の位置から`e1..ek`を外す→標本1件の処理: 保留の標本を`[c1..cm]`に合わせる。以後、`c1..cm`と新しい標本が保留に溜まり、次の警報で、改めて、前区間と変化区間に分けられる（`c1..cm`は、まだ、どのモデルにも取り込まれていない）。

## 旧と違う点（既定の扱いのとき）
1. 短い警報のあと、前区間の標本が、保留から外れる（旧: 残る）。
2. その結果、同じ標本が、次の警報の区間の準備や、容量を超えた確定で、もう一度、評価標本へ保存・取り込まれることが無くなる。学習標本・損失統計（件数・平均）・割当概念計数・評価標本と、評価標本の抜出しが使う乱数の消費が変わり、以後の学習・検出・候補の採否・集約が変わる。
3. 短い警報が1度も起きないrunは、旧と同じ。短い警報が起きても、前区間が空のときは、その警報については、旧と同じ。2つの扱いの状態が初めて分かれるのは、前区間のある短い警報の完了のときで、それより前は、完全に同じ。
4. 保留の標本を読む2か所が、前区間を外した後の保留を読む（要求1.7。処理は変えない）。
   - 警報の推定変化点（`observed_sample_processing.py`: `警報の位置 − min(保留の件数, 検出器の推定の長さ) + 1`）。保留が短くなるので、検出器の推定の長さが保留の件数を超えるときの値（保留の先頭の位置）が、後ろへ動く。取込み済みの標本は、変化区間にできないので、まだ取り込まれていない最古の標本で打ち切るのが、この修正の意図に合う。警報の記録と、変化点の推定の誤差の指標に出る。
   - ADWINで監視する構成（`fedsda-classwise-adwin-detection`・`fedsda-overall-loss-adwin-detection`）の補助の判定（`_current_model_loss_worsened_over_adwin_window`: 保留の件数がADWINの窓の件数以上のとき、保留の末尾の窓で、現行モデルの損失の悪化を見る）。保留が短くなるので、短い警報の直後は、「保留の件数が窓の件数以上」を満たすまでの標本数が変わりうる。窓に入るのは、まだ取り込まれていない標本だけになる。
   どちらも、旧は、取込み済みの前区間を含む保留を読んでいた。値が変わるのは、短い警報の後だけで、変化は受け入れる（取込みを数える別の長さは、持たない）。

旧と同じ扱いでは、違いは無い（実旧との照合testが、これまでどおり確かめる）。

## Validation Design
- 設定: 宣言した値の受入れと、それ以外の拒否、保存への出力。
- owner: 古い側を外す操作（先頭の一部・全部・空、先頭と一致しない並びの拒否と不変、型の拒否、最後の観測位置の不変、外した後の追加・容量の判定）。propertyが設定の値を返すこと。旧と同じ扱いの既存のtest（実旧`_resolve_drift`との対照: 短い警報で保留が全部残る）は、方針を明示して、そのまま通す。
- 完了処理: 短い警報×2つの方針×（前区間あり・なし）で、監視の再開、保留の位置、記録の`drained_pending_sample_indices`。旧と同じ扱いは、既存の実旧対照。既定の扱いは、実旧に正解が無いので、状態を直接確かめ、「保留の位置と記録のほかは、旧と同じ扱いの結果と同じ」（監視・統計・学習標本など）を、同じ入力の2つの実行で比べる。記録の検査（短い警報で、前区間でも空でもない並びを拒否）。更新の前の拒否で、全ownerが不変。
- 標本1件の処理: 既定の扱いで、短い警報の標本を処理した後、保留の標本が変化区間だけになり、続けて、警報のない標本・次の警報を処理しても、同じ標本が2回取り込まれないこと（実体のclientで、短い警報→容量を超えた確定、短い警報→次の警報、の2通り）。短い警報の次の警報で、検出器の推定の長さが保留の件数を超えるとき、警報の記録の推定変化点が、保留の先頭（短い警報の変化区間の先頭）の位置になること（要求1.7。旧と同じ扱いでは、取込み済みの前区間の先頭側の位置になることも、同じ入力で確かめる）。
- ADWINの構成: ADWINで監視する2構成の全体run（短い警報が起きる小さい条件。基準版のADWINのケースは短い警報が起きないので、`fedsda-sine2-events`の設定で、構成だけを替えた条件から探す。起きる条件が見つからなければ、標本1件の処理のtestで、ADWINの補助の判定が、前区間を外した後の保留を読むこと——件数の条件と、窓の中身——を、直接確かめる）で、2回目の取込みが0で、例外なく終わること。
- 観測の方法（新しい記録は足さない）: 「同じ標本の2回目の取込み」は、runの終わりの、clientの学習標本（`ModelTrainingSampleStore`の全モデルの標本）に、同じ標本のobjectが2回以上あるものの、余分の数（全clientの合計）。根拠（コードで確かめた箇所）: 標本のobjectは、観測のたびに1つ作られる（`runtime/fedsda_run_client.py`の、観測標本を学習標本へ直す所）。取込みは、そのobjectを、1つのモデルの学習標本へ足す（`runtime/assigned_training_sample_absorption.py`の`append_model_training_samples`の呼出し）。統合と正式IDの付替えは、標本の列を移すか連結するだけで、objectを複製しない（`learning/training/model_training_sample_store.py`の`reassign_model_training_samples_id`・`remap_model_training_sample_collections`）。候補検証へ渡した変化区間の標本は、採用なら新しいモデルへ足すだけ（`runtime/adopted_candidate_local_adoption.py`）、棄却・再利用なら1つのモデルへ取り込むだけで（`runtime/post_alarm_candidate_validation_resolution.py`）、同じ標本が、正当に2つのモデルへ入る経路は無い。したがって、2回取り込まれた標本は、同じobjectが2つになり、既定の扱いで0を要求できる。2026-10-13に、基準版2の14のFedSDAのケースで数えた: `fedsda-sine2-events`で487（学習標本4897、短い警報18）、`fedsda-sine2-reference-reuse`で467（4900、19）、ほかの12は0（短い警報も0）。FedDrift・FedAvgのrunは、保留の部品を持たないので、数える対象にしない（helperは、FedSDAのrunだけを受け付ける）。testのhelper 1つにまとめ、全体runのtest・基準版のcoverage・影響の計測で、同じものを使う。
- 全体run: `fedsda-sine2-events`の設定（基準版2の扱い——登録は既定——に、この方針だけを替える）で、既定の扱いでは、2回目の取込みが0で、短い警報が1回以上起きること。旧と同じ扱いでは、2回目の取込みが487であること（条件が経路を通ること）。2つの扱いで、指標が違うこと。
- 旧との照合: 実旧と比べるtestの設定の組立て（`tests/refactoring/test_fedsda_run_client.py`の`make_run_client_settings`ほか、`TrainingDataAssignmentSettings`を組み立てるtest）に、旧と同じ扱いを明示する。照合の入口のassert（LEGACY-016のspecで入れた、clientの全状態の照合・全体runの照合・小さい条件の指標の照合の冒頭の確認）を、この方針にも広げる（helperの名前を、2つの扱いを覆う名前へ改める）。旧のgoldenの条件の設定（`make_golden_condition_run_settings`）と、最終構成の既定と旧の設定の全項目の一致のtestは、この項目も、旧と違う値として扱う。順: 先に、明示とassertを入れて、全部のtestが値を変えずに通ることを確かめ、その後で、既定を替える。
- golden（`tests/refactoring/behavior_golden_cases.py`）: 版ごとに、「その版の後に足された設定の項目と、その版の動作に当たる値」を持つ形へ一般化する（版1: 登録の扱いと、この方針。版2: この方針）。設定の組立ては、版の動作に当たる値を入れ、保存してある`run_settings`との比較（読込みの検査と、実行した結果との照合の両方が通る、既存の`comparable_run_settings`）は、値を確かめてから、その項目を除く。この方針は、`run_client_settings`→`training_data_assignment_settings`の中の1項目なので、入れ子の経路で扱う。基準版1・2のファイルは、変更しない。
  - 基準版3: 設定は、最終構成の既定。FedSDAのケースのcoverageへ、「同じ標本の2回目の取込みの数」を足し、0であることを、必須の条件にする（FedDrift・FedAvgのケースのcoverageには、この項目を置かない）。
  - ケース: 基準版2の18ケース（同じ設定）。基準版3の扱いで、必須の経路を通らなくなったケースがあれば、要求3.5のとおり、その経路を通るseedのケースを足す（探し方: `fedsda-sine2-events`の設定で、seed 0〜15を、基準版3の扱いで実行し、必須の経路を全部通る最小のseedを選ぶ。無ければ、範囲を広げ、それでも無ければ、ユーザーへ報告する）。通らなくなったケースの、基準版3での必須の経路は、通る分だけにする（基準版1→2で、`fedsda-sine2-events`に行ったのと同じ）。
  - 版の差の説明: 既存のtest（基準版2と1の差）を、隣り合う版の組（1→2、2→3）へ一般化する。組ごとに、「前の版の実行で、2つの扱いの状態が分かれる事象が起きた数」が0なら、指標・表・パラメータが同じ、を確かめる（一般の主張は、この向きだけ。事象が1回も無ければ、方針を読む唯一の分岐——完了処理の、短い警報で前区間を外す所——が、何も外さないので、全部の状態が同じになる）。事象が1以上のときに値が違うことは、一般には言えない（例: runの最後の標本での短い警報は、その後に読まれない）ので、名前を挙げたケースで、実測の差を確かめる（1→2: `fedsda-sine2-events`。2→3: `fedsda-sine2-events`と`fedsda-sine2-reference-reuse`）。事象は、1→2: 登録されないままの採用（既存。runの終わりの状態から数える）、2→3: 前区間のある短い警報（2回目の取込みが0でも、推定変化点などで差が出うるので、2回目の取込みの数ではなく、状態が分かれる事象そのものを数える）。前区間のある短い警報の数は、基準版のファイルにも記録にも無いので、helperが、版2の設定でケースを実行するときに、警報の処理が呼ぶ完了処理を、その実行の間だけ包んで数える（結果の種別が短い警報で、応答の前区間が1件以上）。包む先は、呼ぶ側のmoduleの名前`federated_learning_experiments.runtime.alarm_occurrence_handling.complete_alarm_buffer_response`（`alarm_occurrence_handling.py`は、この関数を名前でimportしているので、定義した側のmoduleを差し替えても、効かない）。2回目の取込みの数も、同じ実行の終わりの状態から数える。2→3では、`fedsda-sine2-events`と`fedsda-sine2-reference-reuse`で、基準版2の扱いの実行に、前区間のある短い警報と、2回目の取込みが、どちらも1以上あり、基準版2と3の値が違うこと。ケースの数は、定数（版ごとのケース名）から数え、文書の件数に依らない。版2の実行は、基準版2の照合のtestが行う実行と同じなので、ケースごとに1回だけ実行して、両方のtestで使う（既存の、実行の結果のcacheを使う）。
  - 生成: 全taskの後の、独立CLIの実装レビューと、指摘の反映の後に、sourceとtestをcommitしてから、`--baseline-version 3`で、Windowsの基準環境とWSLで生成する（task 5。レビューでsourceを変えた後に生成するため）。現在の版の定数を3にするのは、task 4（基準版3のファイルが無い間、基準版3の、ファイルを要るtestは失敗する。実装レビューへ、その旨を伝える）。
- 影響の計測: このspecのdirectoryへ置く計測script（Git管理。sourceを変えず、設定のこの方針だけを替える。構成は、提案構成の8条件と、ADWINで監視する2構成のsine2 seed 0・1）。結果と、実行したcommitを、LEGACY-002の記録とtasks.mdへ書く。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/training_data_assignment/training_data_assignment_settings.py`: 項目と、値の定数。
- `src/federated_learning_experiments/methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py`: propertyと、古い側を外す操作。
- `src/federated_learning_experiments/runtime/alarm_response_completion.py`: 短い警報の分岐と、記録の検査。
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`: 既定。
- `tests/refactoring/`: `test_pending_training_assignment_buffer.py`、`test_alarm_response_completion.py`、新規`test_too_short_alarm_earlier_sample_release.py`（標本1件の処理と全体run）、`TrainingDataAssignmentSettings`を組み立てる既存test（方針の明示）、`test_fedsda_run_client.py`（照合の入口のhelper）、`test_fedsda_final_configuration_run_settings.py`、掃引の識別子の期待値、`behavior_golden_cases.py`・`test_important_behavior_regression_goldens.py`・`behavior_regression_goldens/v3/`、`fresh_process_smoke.py`、依存境界、設定の検査のtest。
- `.kiro/specs/too-short-alarm-earlier-sample-release/measure_too_short_alarm_policy_effect.py`: 影響の計測。
- 文書: `docs/experiments/refactoring-baseline.md`（基準版3）、`docs/experiments/experiment-sweep.md`、LEGACY-002の記録と一覧、IMPROVE-006の記録（この修正の結果の追記）、NEW-004の記録（外す箇所）、再開案内。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1 | ownerの古い側を外す操作、完了処理の分岐と記録、標本1件の処理のtest、全体runの2回目の取込み0 |
| 2 | 設定の項目と2つの値、最終構成の既定、11構成の既定のtest |
| 3 | 旧との照合の明示と入口のassert、基準版1・2の照合の継続、基準版3と必須の条件、版の差の説明のtest |
| 4 | 8条件×2つの扱いの計測と記録 |
