# Design Document

## Overview
最終構成の既定の2つの値を替え、それに合わせて、派生の構成・基準版・test・文書を整える。2つの扱いの動作は変えない。

| 設定 | これまでの既定 | 新しい既定 |
|---|---|---|
| 登録（`PendingModelRegistrationSettings.pending_model_registration_policy`） | `register_each_pending_model_by_its_own_id` | `register_pending_model_after_waiting_as_current_model` |
| 警報（`LossChangeDetectionSettings.short_change_interval_alarm_policy`） | `raise_alarm_immediately` | `wait_until_change_interval_reaches_minimum` |
| 短い警報のあとの保留 | `release_earlier_samples_assigned_to_current_model` | 変えない |

## Boundary Commitments
### This Spec Owns
`runtime/fedsda_final_configuration_run_settings.py`の2つの既定、`sweep/experiment_configurations.py`のADWINの2構成の検出の設定、基準版のgoldenのhelper（版4の明示、版5、版の差の事象）、既定に依るtest、文書。
### Out of Boundary
2つの扱いの実装、比較script（扱いを明示して組み立てる。説明の文だけ直す）、FedDrift・FedAvg、固定旧。
### Allowed Dependencies
新しいmoduleは無い。最終構成のmoduleのimportは、方針の値の定数が入れ替わる（依存境界の許可の名前を入れ替える）。構成のmoduleは、警報の扱いの定数`RAISE_ALARM_IMMEDIATELY_POLICY_NAME`を新しく読む（許可へ足す）。
### Revalidation Triggers
既定のbuilderを通るtest、掃引のrunの識別子（FedSDAの全runで変わる）、基準版のhelper。

## Architecture
- 最終構成: `build_final_configuration_fedsda_run_settings`が組み立てる`PendingModelRegistrationSettings`と`LossChangeDetectionSettings`の値を替える。設定の検査（`FedsdaRunClientSettings`の、待つ扱いでは保留の容量が最小件数以上、という検査）は、最終構成の値（保留の容量と最小件数）で満たされることを、testで確かめる（構成の派生の全部で、組み立てが例外を出さないこと）。
- ADWINの2構成: `_LOSS_CHANGE_DETECTION_CHANGES_BY_CONFIGURATION_NAME`の替えるfieldへ、`short_change_interval_alarm_policy=raise_alarm_immediately`を足す（ADWINと待つ扱いの組合せは、`LossChangeDetectionSettings.__post_init__`が拒否するので、替えないと、構成の組立てが失敗する）。e-SRの全体の損失の構成（`fedsda-overall-loss-esr-detection`）は、待つ扱いを引き継ぐ。
- 基準版のgolden（`tests/refactoring/behavior_golden_cases.py`）:
  - 「版の後に、既定の値が変わった項目」の表へ、版4の警報の扱い（`raise_alarm_immediately`）を足す。`build_case_settings`は、この表の警報の扱いの経路も扱うように広げる（今は登録の扱いの経路だけ）。版1〜3の警報の扱いは、「版の後に足された項目」の表で、既に`raise_alarm_immediately`を入れている（変えない）。
  - 版5を足す（`SUPPORTED_BASELINE_VERSIONS`、`BASELINE_VERSION`、`CASE_NAMES_BY_BASELINE_VERSION`。ケースは版4と同じ19）。設定は、新しい既定のまま。必須の経路（`assert_required_paths`）は、版5で、版4と同じ分岐を使う（版4で外した経路——`fedsda-sine2-reference-reuse`の別の保有モデルの再利用——は、版5でも外す）。版5の実行で、必須の経路が通らなくなるケースがあれば、版4のときと同じく、同じ設定でseedを替えて、版5の扱いで全部の経路を通るseedを探し、版5だけのケースとして足す（版4のgoldenは変えない）。足すときは、`build_case_settings`のケース名→seedの対応と、`sine2_rich`の組と、`assert_required_paths`の分岐の3か所を広げる。
  - `BASELINE_VERSION`（現在の版）を5へ替えるのは、版5のWindowsとWSLのgoldenを生成して置いた後にする（`GOLDEN_DIRECTORY`と、golden自体の変化を検出するtestが、現在の版のdirectoryを読むため）。生成は`--baseline-version 5`で指定する。
  - 版の差の事象へ、`alarm_wait_situation_count`（警報が上がり、保留の件数と推定変化区間の小さい方が、最小件数より短い回数）を足す。数えるのは、次の版（版5）でそのケースが警報を待つ扱いを使うときだけ（`build_case_settings(ケース, 5)`の警報の扱いが`wait_until_change_interval_reaches_minimum`のとき）。ADWINの2構成は、版4・5とも即時の警報なので、動作は変わらず、事象は定義により0（この場面が実行で起きても数えない。仕様レビュー3回目の指摘）。FedDrift・FedAvgも0。待つ扱いの分岐（`runtime/observed_sample_processing.py`の`process_observed_sample`の中）は、まさにこの条件で警報を抑える（候補検証を保持している間かを問わない）。即時の扱い（版4）では、この条件の警報は、警報の処理（`handle_alarm_occurrence`）へ渡され、その引数（保留の標本の観測、推定変化区間の件数、最小件数）から、同じ式で条件を計算できる。そこで、標本1件の処理のmoduleが名前でimportしている`handle_alarm_occurrence`を、実行の間だけ包んで数える（`try/finally`で戻す。runの参加者の組立ての前に入れる）。応答の種類（`alarm_change_interval_too_short`か、候補検証の保持中の`alarm_during_candidate_validation`か）では数えない（保持中の応答は、短い警報の種類にならないので、数え漏れる。仕様レビュー1回目の指摘）。版4→5の差の説明は、このキーで読む（`BASELINE_VERSION_DIFFERENCES`の`(4, 5)`。事象が1以上のケースを全部、名前で挙げ、名前の無いケースで事象が1以上なら失敗する）。0なら版4と同じ、という根拠: 2つの扱いが分かれるのは、この条件の警報の場面だけで、ほかの警報・検出器の更新・保留の扱いは同じ。版4の扱いでその場面が無ければ、版5の扱いでも、警報を抑える場面が無く、同じ状態の列をたどる。
- 既定に依るtest: 既定を替えた後の全pytestで失敗したものを合わせる。ただし、実旧と比べるtestの期待値は変えない（実旧と比べる設定は、既存の入口——`use_legacy_compatible_policies`・`make_run_client_settings`——で旧と同じ扱いを明示していて、外れると`assert_run_client_uses_legacy_compatible_policies`が失敗する。そこで失敗したら、明示の漏れを直す）（前の2回の既定の変更と同じ。候補: 掃引のrunの識別子の期待値、最終構成の既定の照合、構成の既定のtest、共用のfresh scriptの方針の名前、待つ扱いのspecの「既定は即時」のtest）。

## 旧と違う点
既定を使う実行（掃引——`run_experiment_sweep.py`は`build_final_configuration_fedsda_run_settings`を通る——を含む）の、FedSDAの11構成の値が変わる（ADWINの2構成は、登録の扱いだけが変わる）。比較script（`run_behavior_fix_comparison.py`）は、3つの扱いを明示して組み立てるので、変わらない。基準版4との差は、版4の扱いで短い警報が起きるケースに限る。

## Validation Design
- 構成: FedSDAの11構成の既定の値（登録・警報）を、1つのtestで確かめる（ADWINの2構成は、警報が即時）。全構成の組立てが例外を出さないこと。掃引のコマンドの引数の一覧が変わらないこと（要求1.4）。
- 基準版: 版1〜4の照合、版5の照合、版4→5の差の説明、版5の必須の経路。
- 実旧との照合: 既存のtest（変えない）。
- fresh process: 共用scriptのFedSDAの全体runの流れの、方針の名前のassertを、新しい既定へ合わせる。

## File Structure Plan
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`、`src/federated_learning_experiments/sweep/experiment_configurations.py`、`src/federated_learning_experiments/methods/fedsda/model_registration/pending_model_registration_settings.py`・`methods/fedsda/loss_change_detection/loss_change_detection_settings.py`（既定の注記）。
- `tests/refactoring/`: `behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v5/`、構成の既定のtest、既定に依るtest、`fresh_process_smoke.py`、依存境界。
- 直下: `run_behavior_fix_comparison.py`（説明の文）。
- 文書: `docs/experiments/experiment-sweep.md`、`docs/experiments/refactoring-baseline.md`、`docs/research/improvement-candidates/`（IMPROVE-006の記録と一覧）、LEGACY-016の記録、再開案内。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1 | 最終構成の2つの値、ADWINの2構成の明示、構成の既定のtest |
| 2 | 実旧との照合の既存test、版4の明示と版1〜4の照合 |
| 3 | 版5と、版の差の事象`alarm_wait_situation_count`、版4→5の差の説明のtest |
| 4 | 文書の更新 |
