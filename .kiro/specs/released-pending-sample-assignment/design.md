# 警報のない標本での帰属確定 — 設計 revision2

## 1. Boundary Commitments

- runtimeの関数`assign_released_pending_samples_to_current_training_model`が、容量を超えた保留標本を、既存の`absorb_assigned_training_samples_into_held_model`で現在の学習帰属のモデルへ確定し、保留位置のownerから解放する。新しいowner・結果recordは作らない（解放した標本のtupleを返す）。
- 既存`PendingTrainingAssignmentBuffer`へ、読取りだけの`get_sample_indices_exceeding_capacity`を足す。

Out of Boundary: 標本の観測と保留への追加、警報の処理、学習要求と共同学習、計算量の診断、保留標本そのものを持つowner、標本1件の処理全体。

## 2. 旧処理との対応（`federated_drift_experiment/clients/fedsda.py`）

| 旧 | 本spec |
| --- | --- |
| `process_one_step`の警報のない分岐（511〜524行）: `while len(self.buffer) > self.fifo_size:`で最古の標本を取り出し、現行モデルで損失を評価、`_update_model_stats`（クラス別を含む）、`train_data_store`へ追加、`_record_model_concept` | 容量を超える位置を読む→最古の側の標本を既存の吸収へ渡す→保留から解放する |
| 同じ分岐の最後の`self.train_step()`（525行） | 範囲外 |
| `_record_model_compute("statistics", ...)` | 範囲外（計算量の診断は未移植） |

旧と順序が違う点: 旧は標本ごとに「損失評価→統計→標本追加→概念計数」を行い、標本を取り出してから評価する。既存の吸収は「全標本の検査と損失評価→標本ごとに標本追加→概念計数→統計」の順で、本specは吸収が済んでから保留を解放する。確定の間にモデルは変わらないので、各標本の損失は同じ値になり、標本・計数・統計・保留の最終状態は旧と一致する（実旧との対照testで示す）。通常は1標本につき解放は高々1件だが、変化区間が不足した警報の後は保留が容量＋1より多く残ることがあり（LEGACY-002）、次の標本で複数件が解放される。その場合も旧と同じく古い順に確定する。

## 3. 契約

`assign_released_pending_samples_to_current_training_model(*, pending_training_assignment_buffer, pending_sample_observations, current_training_model_assignment, held_model_training_state_registry, training_sample_store, model_training_and_assignment_counts_store, loss_statistics_store) -> tuple[IndexedObservedTrainingSample, ...]`

- `pending_sample_observations`: 現在保留されている全標本（位置・標本・診断用概念ID）。保留位置のownerと同じ並びであること。警報のときの既存の応答と同じ形の引数で、保留標本そのものの保持は呼出し側が行う。
- 戻り値: 解放した標本（古い順。渡された値そのもの）。解放がなければ空のtuple。

`PendingTrainingAssignmentBuffer.get_sample_indices_exceeding_capacity() -> tuple[int, ...]`: 容量を超える最古の位置を古い順に返す。状態を変えない。

## 4. 検査と処理順

| 検査（例外） | 位置 | 値の出所 |
| --- | --- | --- |
| 保留位置のownerと現在の学習帰属のownerがexact型（TypeError） | 最初 | 引数。本処理が読むowner |
| 保留標本がexact tuple、各要素がexact `IndexedObservedTrainingSample`、位置がbuiltin int（TypeError） | 更新前 | 引数。frozenな値だが、手で組み立てた入力がありうる |
| 保留標本の位置の並びが保留位置のownerの並びと同じ（ValueError） | 更新前 | 引数とownerの現在の状態 |
| 吸収の検査（吸収先のモデルがあること、ownerの型、解放する標本の形と概念ID、損失評価） | 吸収の中。吸収の更新より前 | 既存の吸収の契約（全標本の検査と損失評価を、状態変更より前に行う） |

処理順: (1)ownerの型、(2)保留標本の型と並び、(3)容量を超える位置を読む（読取りだけ）、(4)解放する標本がなければ空のtupleを返す、(5)最古の側からその件数の標本を吸収へ渡す、(6)保留から解放する、(7)解放した標本を返す。

(5)が拒否した場合、吸収は何も更新しておらず、(6)へ進まないので、どの状態も変わらない（要求2.2）。(6)は(3)と同じ規則で同じ位置を解放し、拒否しない。

解放がないときは吸収を呼ばない（revision2で変更）: 旧は、解放する標本がなければモデル・統計・標本に触れない。既存の吸収は空の標本列でも吸収先のモデルを取得するので、呼ぶと、現在の学習帰属のモデルが保有されていない状態で旧にない例外になる（仕様レビューの指摘）。要求1.2（容量以下なら何も変えず空の結果を返す）のとおり、(1)(2)の検査の後は何も読まずに返す。吸収へ渡すだけのownerの不正は、解放が起こる標本で吸収が拒否する。

確かめないこと: 解放しない保留標本の中身（要求2.3）。保留位置の最終観測位置と標本位置の関係（位置の連続性は保留位置のownerが追加のときに保証する）。

## 5. Allowed Dependencies

`runtime/released_pending_sample_assignment.py`: `ModelAndClassLossStatisticsStore`、`CurrentTrainingModelAssignment`、`HeldModelTrainingStateRegistry`、`IndexedObservedTrainingSample`、`ModelTrainingAndAssignmentCountsStore`、`ModelTrainingSampleStore`、`PendingTrainingAssignmentBuffer`、`absorb_assigned_training_samples_into_held_model`。許可集合を既存と同じ形で登録する（symbolごとの注入契約testは足さない）。保留位置のownerのmoduleの依存は増えない。

## 6. Revalidation Triggers

吸収の引数・検査の位置・更新順、保留位置のownerの容量と解放の規則、現在の学習帰属の読取りが変わったら、本specの対照testを再検証する。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/runtime/released_pending_sample_assignment.py | 新規。関数1つ |
| src/federated_learning_experiments/methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py | 読取りの操作を1つ追加 |
| tests/refactoring/test_released_pending_sample_assignment.py | 新規 |
| tests/refactoring/test_pending_training_assignment_buffer.py | 読取りの操作のtestを追加 |
| tests/refactoring/test_single_run_dependency_boundaries.py | 許可集合の登録 |
| tests/refactoring/fresh_process_smoke.py | 警報の前に、警報のない標本での確定を通す流れを追加 |

## 8. Testing Strategy / Requirements Traceability

oracleは、実旧`FedSDAClient.process_one_step`（警報のない分岐はinlineで、単独のメソッドがない）。吸収のoracle（`build_absorption_oracle`）の実旧clientへ、最後の標本以外を保留済みにしたFIFOを与え、予測・候補検証の観測・検出・学習・計算量の記録を止めて、最後の標本を実旧の標本処理へ渡す。損失評価・統計・標本・概念計数の更新は実物のまま使う。同じ方法は保留位置のownerのtestが実行済み（そちらは統計などもstub）。新しく使う旧メソッドはない。

| 要求 | 証拠 |
| --- | --- |
| 1.1, 3.1 | 2/4 class×（保留が容量未満・容量ちょうど・1件超過・複数件超過）で、標本・割当概念計数・損失統計を既存helperで実旧と照合、保留に残る位置が実旧のFIFOに残る標本と一致、戻り値が渡した最古の標本そのもの。吸収が1回で、その時点で保留が未解放、渡る標本・概念ID・モデルID・ownerが正しいこと |
| 1.2 | 解放がない条件と、解放後の再実行で、全状態が不変。解放がないとき吸収が呼ばれないこと |
| 1.3 | 現在の学習帰属が不変、モデル・optimizer・乱数が不変（既存helper） |
| 1.4 | 保留位置のownerのtest: 読取りが状態を変えず、直後の解放と同じ位置を返す |
| 2.1 | ownerの別の型・派生型、標本の組のlist・tupleの派生型・要素の別の型・派生型・位置のint派生型、並びの不一致（不足・順序・位置の違い）で、吸収が呼ばれず、全状態が不変 |
| 2.2 | 解放対象の最後の標本だけ特徴数が違う・概念IDがintでない条件で、保留を含む全状態が不変。吸収へ渡すだけのownerの型の不正でも同じ |
| 2.3 | 解放しない保留標本の中身が不正でも成功すること |
| 3.2 | 汎用の変異tool（新しい関数と、保留位置のownerの読取り）、依存の許可集合、共用のfresh process script、全pytest、Ruff・Pyright |

未検証として残す: 学習要求と共同学習の接続、標本1件の処理全体、計算量の診断、新全体runのgolden一致。
