# Design Document

## Overview
前のspec（`fedsda-reverse-prequential-change-interval-search`）で足した「新しい扱い」を、最終構成の既定にし、基準版16を足す。2つの扱いの計算は変えない。

| 設定 | 基準版15まで（今の扱い） | 基準版16から（新しい扱い） |
|---|---|---|
| `TrainingDataAssignmentSettings.change_interval_search` | `minimum_interval_multiples_with_forward_prequential_code` | `every_sample_with_reverse_prequential_code` |

## Boundary Commitments
### This Spec Owns
最終構成の既定の値、構成の扱い（`fedsda-every-sample-change-interval-search`を既定と同じ構成として残すこと、新しい構成`fedsda-minimum-interval-multiples-change-interval-search`）、最小記述長でない選び方の設定の組での今の扱いの明示、基準版16（版の登録・経路の記録の新しい項目・必須の経路・版15→16の差のtest・golden）、既定に依るtest、研究ツールの基点の固定、文書。
### Out of Boundary
2つの扱いの計算（`select_alarm_interval_assignment`の分岐、予測的符号長の途中までの和、seed、純粋な計算の刻み）、組合せの検査の規則、検出器、帰属・保留、候補・登録・クロス評価・統合・予測、FedDrift・FedAvg、固定旧、版1〜15のgoldenのJSON、本規模の実験。
### Allowed Dependencies
新しいmoduleは作らない。依存境界の許可に新しいimportが要るなら登録する。
### Revalidation Triggers
既定のbuilderと構成の一覧を通る全test、最小記述長の選択を通るtest、掃引のrunの識別子、基準版のhelperと全版のgolden、実旧との照合、研究ツールの変種。

## Architecture
### 既定と構成（Requirement 1）
- `runtime/fedsda_final_configuration_run_settings.py`の既定（今は`change_interval_search=MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME`を明示）を、`EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME`にする。
- `sweep/experiment_configurations.py`:
  - `fedsda-every-sample-change-interval-search`（`EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`）は、既定と同じ設定を返す（分岐の`replace`は、値が既定と同じなので結果は同じ。分岐を残すか、`fedsda-synchronized-local-mixture-prediction`と同じ形にそろえるかは実装で選び、コメントで「基準版16から既定と同じ。これまでのコマンドと結果の記録のために残す」と書く）。
  - 新しい構成`fedsda-minimum-interval-multiples-change-interval-search`（定数`MINIMUM_INTERVAL_MULTIPLES_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`）を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。組立ては、最終構成の既定から`training_data_assignment_settings.change_interval_search`だけを`MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME`に替える。
  - `tests/refactoring/behavior_golden_cases.py`の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ、新しい構成と、それを守る専用のtestのファイル（新しいtestのファイル`tests/refactoring/test_fedsda_every_sample_change_interval_search_default.py`）を足す。
- 構成の一覧を前提にする掃引・CLI・文書のtest（構成の数、名前の一覧）は、失敗したものから合わせる。

### 最小記述長でない選び方の設定の組（Requirement 2）
- 既定を引き継いだうえで`alarm_interval_assignment_policy`を最小記述長でない値（`detector_change_span_and_mean_loss_increase_tolerance`）にする箇所は、組合せの検査で拒否されるので、`change_interval_search`を今の扱いと明示する。対象は、`src/`の旧と比べる設定の入口、実旧との照合の組、testとfresh script（`tests/refactoring/fresh_process_smoke.py`ほか）で、既定に替えた後に失敗するものから見つけて直す（版13のspecで`keep_baseline_until_alarm`を明示した箇所と同じ探し方）。`src/`で明示を足した箇所は、Implementation Notesに一覧で書く。
- 組合せの検査の規則と文言は変えない。

### 基準版16（Requirement 3、5）
- `tests/refactoring/behavior_golden_cases.py`:
  - 版16を登録する（ケースは版15と同じ18ケース。`BASELINE_VERSION`と`BEHAVIOR_BASELINE_VERSION`の16への切替は、goldenの生成・設置と同時に主担当が行う）。
  - 版1〜15の`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`の今の扱いの明示は前のspecのまま（版16には足さない。版16の既定が新しい扱い）。
  - 経路の記録（`_coverage`）に、版16から、FedSDAのケースだけ`non_minimum_interval_multiple_change_interval_selection_count`（記録の表`alarm_interval_assignment_selection.*`で、`selected_hypothesis`が`held_model`か`new_model`で、`change_interval_sample_count`が最小変化区間の倍数でない選択の数。最小変化区間は、そのrunの設定の`minimum_change_interval_sample_count`）を足す。
  - 必須の経路（`assert_required_paths`）: 版16で、`fedsda-sine2-adoption-without-validation`と`fedsda-mnist4-warm-started-learner`の上の回数が1以上（生成の前に主担当が値を確かめ、0なら別の新しい既定を引き継ぐFedSDAのケースへ選び直し、理由をImplementation Notesに書く）。版15までの必須の経路は版16でも求める。
- 版15→16の差のtest（goldenを使わない。版12→13の差のtestと同じ形）: 両方の版の18ケースをその場で実行し、FedDrift・FedAvgのケースは、版16で足す経路の項目を取り除いて、指標・最終のパラメータ・記録の表・経路の記録が同じであること、FedSDAのケースは、設定の差が`change_interval_search`の1か所だけ（版15は今の扱い、版16は新しい扱い）であることを確かめる。
- goldenの生成・設置（`tests/refactoring/behavior_regression_goldens/v16/`）と版の定数の切替は、版16のコードをcommitした後に主担当が行う（Windowsの基準環境とWSL。`docs/experiments/refactoring-baseline.md`の手順）。

### 前のspecの構成との一致（Requirement 4）
- 既定を替える前（このspecの基点のcommit）に、構成`fedsda-every-sample-change-interval-search`で、5条件（MNIST4・SINE2・SEA4・CIRCLE2は間隔50・seed 0・各client 1000標本、SINE2は同じ条件のseed 1）の全体runを実行し、製品の保存関数でrawを保存する（Git管理外。`../../venv/refactoring-tests/every-sample-default-before/`）。
- 既定を替えた後、構成`fedsda`で同じ5条件を実行して保存し（`…-after/`）、要求4.1の(a)〜(c)で比べる一時scriptを使う（Git管理外。場所と結果をImplementation Notesに書く）。

### runの識別子
- FedSDAの既定を引き継ぐ全構成で、設定の値が変わるのでrunの識別子が変わる。識別子を直書きしたtest（`tests/refactoring/test_experiment_sweep.py`ほか）を新しい値に合わせ、FedDrift・FedAvgの識別子が変わらないことを確かめる。
- 構成`fedsda-every-sample-change-interval-search`の識別子は、設定が前のspecと同じなら変わらない。変わる場合は理由（構成の名前の扱いほか）をImplementation Notesに書く。

### 研究ツール（Requirement 6）
- `tools/research_provisional_candidate_expert/run.py`の`_execute_job`:
  - 基準版10を基点にする変種（`_PRODUCT_DEFAULT_BASED_VARIANTS`に無い変種）の基点の固定に、`training_data_assignment_settings.change_interval_search`＝今の扱いを足す。
  - IMPROVE-031の変種（基準版12へ戻す分岐）と、IMPROVE-032の研究用の変種（`REVERSE_PREQUENTIAL_VARIANTS`）は、`change_interval_search`を今の扱いに戻してから変種を当てる（研究用の変種は、選択の関数を差し替え、短い列では製品の選択へ戻すので、戻した値が使われる）。
  - `product-default`と`every-sample-change-interval-search-configuration`は戻さない。ほかの`product-default`系の変種は、版13のspecと同じく製品の既定（基準版16）の上で評価する（意味が変わることをコメントに書く）。
  - 研究用の入口（`tools/research_loss_monitor_rebaseline.py`）の組み合わせた変種も、IMPROVE-031の基準版12への戻しと同じ場所で、今の扱いへ戻す。
- 既定を替える前に、要求6.3の4つの小さい条件（SINE2・seed 0・10 client・600標本・間隔100。研究ツールの`default-reverse-prequential-grid1`は同じ条件で、警報が起きることを確かめる）の結果を保存し、替えた後の結果と比べる。研究ツールの既存の検査（`checks_reverse_prequential.py`ほか）が通ることを確かめる。

### 文書（Requirement 7）
- `docs/overview/proposed-method.md`（既定の表と、基準版16の判断の段落）、`docs/components/model-reuse.md`（最小記述長の節。既定の印を新しい扱いへ）、`docs/experiments/refactoring-baseline.md`（版16の節）、`docs/experiments/experiment-sweep.md`（構成の一覧。`fedsda-every-sample-change-interval-search`は既定と同じ、新しい構成）、IMPROVE-032（状態: 基準版16で既定）と一覧、`.kiro/steering/resume.md`。過去の日付の記録（実験の台帳、IMPROVE-032の結果の節）は書き換えない。

## Validation Design
- 既定と構成のtest: 既定の値が新しい扱い、`fedsda-every-sample-change-interval-search`と`fedsda`の設定が同じ、新しい構成は`change_interval_search`だけが既定と違い、その設定が基準版15の`fedsda`の設定と同じ（版15の設定は、版の設定の組立てのhelperか保存済みの版15のgoldenの設定で取る）、FedSDAの全構成が新しい扱いを引き継ぐ。
- 最小記述長でない選び方の組: 既定を引き継いで最小記述長でない選び方にする組立ての全部が、今の扱いを明示していて例外を出さない。
- 基準版: 版1〜16の照合（WindowsとWSL）、版15→16の差のtest、必須の経路。
- 実旧との照合、`tests/test_regression.py`、`tests/test_proposed_regression.py`。
- 前のspecの構成との一致（要求4）と、研究ツールの前後の一致（要求6.3）。
- 全pytest（Windows、12並列）、Ruff、Pyright（設定の範囲）、fresh smoke。
