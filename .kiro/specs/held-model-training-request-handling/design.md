# 学習要求の記録と保有モデルの共同学習 — 設計 revision5

## 1. Boundary Commitments

- runtimeの新module `held_model_training_request_handling.py` に関数を2つ置く。
  - `record_training_request_and_train_held_models_when_due`: 学習要求を1件記録し、保留が更新間隔に達していれば下の関数へ進む。
  - `train_held_models_for_pending_training_requests`: 保留中の全要求のぶんを共同学習し、共同更新の完了ごとに学習量を計数へ足して、最後に保留を消化する。
- 既存`LocalTrainingRequestSchedule`へ、読取りだけの`has_pending_requests_reaching_update_interval`を足す。
- 新しいowner・結果recordは作らない（共同学習の反復が返す損失を、実行順のtupleで返す）。

Out of Boundary: これらを呼ぶ位置（標本1件の処理、警報の前、ラウンド境界）、共有部を凍結する学習方式、計算量・所要時間・勾配の診断、保有モデルごとに順に学習する旧の基底clientの方式（最終構成で使わない）。

## 2. 旧処理との対応

| 旧（`federated_drift_experiment/clients/`） | 本spec |
| --- | --- |
| `base.py::train_step`: `_pending_updates += 1`、`LOCAL_UPDATE_INTERVAL`以上なら`flush_pending_updates` | 要求の記録→間隔に達したかの読取り→達していれば保留中の要求の学習 |
| `base.py::flush_pending_updates`: 保留が正なら`train_all_held_models(count_multiplier=保留件数)`、その後に保留を0へ | 保留が0件なら何もしない。そうでなければ共同学習と計数→保留の消化 |
| `shared_backbone.py::train_all_held_models`の`joint`の分岐と`_train_heads_together`: `updates_per_sample×count_multiplier`回、毎回batchを抽出して共同更新 | 件数管理が算出した回数だけ、既存`perform_held_model_joint_training_iterations`（実旧との対照は移植済み）を共同更新1回ぶんずつ呼ぶ |
| `_train_heads_together`の中の`model_training_examples[model_id] += len(bx)`、`model_optimizer_steps[model_id] += 1`（共同更新1回ごと、参加モデルごと） | 共同更新が1回完了するたびに、参加モデルごとに`record_completed_model_training`（標本数＝batchの件数、更新回数＝1） |
| `compute_counters`、`_record_model_compute`、`phase_seconds`、`backbone_gradient_diagnostics` | 範囲外（未移植の診断） |
| `train_all_held_models`の`sequential`・`frozen`の分岐 | 範囲外（最終構成は`joint`。新の学習設定は共同学習だけを受け入れる） |

旧と違う点:

- **参加モデルの求め方**: 共同学習の反復は損失しか返さない。反復の間、保有モデルと標本列は変わらないので、参加するモデルは毎回同じで、「保有していて、標本がbatchの件数以上あるモデル（標本列の順）」である。本specはこの条件を、反復へ渡したのと同じsnapshotから求める。反復は、参加モデルがあって共同更新が完了した回だけ損失を1つ返すので、損失が返った回だけ計数する。batchの件数は、損失が返った時点で反復の検査を通った正のintである。条件の重複は[IMPROVE-011](../../../docs/research/improvement-candidates/improve-011-report-participating-models-from-joint-training.md)へ記録する。
- **共同更新1回の途中の失敗**: 旧は、1回の共同更新の中で、モデルごとに個別部のoptimizerを進めた直後にそのモデルの計数を足し、その後に診断（`compute_counters`、`_record_model_compute`）を記録する。本specは、1回の共同更新が完了してから参加モデルの計数を足す。共同更新の合間（抽出、次の回の検査）で失敗した場合は、旧と同じく、完了した回のぶんの更新と計数が残る。1回の共同更新の途中（個別部のoptimizerを順に進めている間と、旧ではその後の診断の記録の間）で失敗した場合だけ、旧はそこまでに進めたモデルの計数が残り、本specではその回の計数は残らない。旧も新も、この例外は標本処理の外まで伝わる。旧は試行単位で例外を捕捉して次の試行へ進む（`trials.py`）が、失敗した試行の結果は返らないので、その後に計数が読まれる経路はない。
- **診断**: 旧が学習の中で進める計算量・所要時間・勾配の診断（`compute_counters`の`optimizer_steps`・`backbone_optimizer_steps`・`head_optimizer_steps`、`_record_model_compute`、`phase_seconds`、`backbone_gradient_diagnostics`）は、新では更新されない（未移植）。
- **乱数**: 旧はmodule全体の`random`、新は借りた`Random`（既存の反復の契約）。

## 3. 契約

```
train_held_models_for_pending_training_requests(*, local_training_request_schedule, held_model_training_state_registry, training_sample_store, model_training_and_assignment_counts_store, batch_sample_count, python_random_generator, local_training_settings, shared_feature_extractor, shared_parameter_optimizer) -> tuple[float, ...]
record_training_request_and_train_held_models_when_due(（同じ引数）) -> tuple[float, ...]
```

- 戻り値: 完了した共同更新の損失（共同学習の反復が返した値を実行順に並べたもの）。学習へ進まなければ空のtuple。
- `batch_sample_count`以降の5つは共同学習の反復へそのまま渡す。共有部の更新は常に行う（`update_shared_features=True`。最終構成の学習方式）。

`LocalTrainingRequestSchedule.has_pending_requests_reaching_update_interval() -> bool`: 保留件数が実行間隔以上か。状態を変えない。既存`record_training_request`の戻り値（実行回数）は、一要求あたりの回数が0のとき、間隔に達していても0になり、達したかどうかを区別できない。旧は回数が0でも間隔に達すれば保留を0へ戻すので（要求1.4）、件数だけで決まる読取りを足す。

## 4. 検査と処理順

| 検査（例外） | 位置 | 値の出所 |
| --- | --- | --- |
| owner 4つがexact型（TypeError。runtimeの既存の関数——警報1回ぶんの処理、警報のない標本での確定——がownerの型の不正に使う例外と同じ。学習の部品の中の検査はValueErrorを使うが、そちらは変えない） | 最初。要求の記録より前 | 引数。本処理が読む・更新するowner |
| 共同学習の反復の検査のうち、保有モデルの対応と抽出の要求（batchの件数、乱数生成器、標本列） | 本処理は反復を共同更新1回ぶんずつ呼ぶので、各回の呼出しの最初に行われる（その回の抽出より前。保有モデルの対応の検査は、反復の中では呼出しごとに1回で、loopの前にある） | 既存の反復と抽出の契約 |
| 共同学習の反復の検査のうち、共同更新の入力（学習の設定、共有部とoptimizer、分類器、抽出したbatch） | 各回の反復の中。その回の抽出の後、更新より前。参加するモデルがある回だけ行われる（参加するモデルがない回は、反復が共同更新を呼ばない）。ここで拒否されると、その回の抽出で乱数は進んでいる（既存の反復の契約どおり。巻き戻さない） | 既存の共同更新の契約 |
| 計数の検査（モデルIDと増分がbuiltin int、増分が非負） | 計数の中。共同更新の後 | 既存の計数の契約。到達しない: モデルIDは標本列のもので、反復の抽出が全標本列のIDを検査済み。増分は反復の検査を通ったbatchの件数と定数1 |
| 保留の消化の検査（現在の全保留件数と等しい正のint） | 件数管理の中。全回の後 | 到達しない: 処理の最初に読んだ保留件数（正であることを確かめた後）をそのまま渡し、その間に件数を変える処理はない |

保留中の要求の学習の処理順: (1)ownerの型、(2)保留件数を読む。0件なら空のtupleを返す、(3)保有モデルの対応と標本列のsnapshotを取る、(4)件数管理が算出した回数だけ、(4a)共同学習の反復を1回ぶん呼び、(4b)損失が返れば、参加モデルごとに計数へ加える、(5)(2)で読んだ件数で保留を消化する、(6)損失を実行順に返す。

要求の記録の処理順: (1)ownerの型、(2)要求を1件記録する、(3)間隔に達していなければ空のtupleを返す、(4)保留中の要求の学習へ進む（ownerの型の検査は重ねて行われる）。

(4a)が失敗した場合、(5)へ進まないので保留は変わらない。それより前に完了した回の更新と計数は残る（要求3.2）。記録済みの要求は残る（旧も`_pending_updates`を足した後で学習が失敗すれば残る）。

算出した回数が0なら(4)は何もしない（学習へ渡すだけの値を読まない）。そのまま(5)で保留を消化する（要求1.4）。

確かめないこと: 学習へ進まないときの、学習へ渡すだけの値（要求3.3。旧も学習に入るまで読まない）。参加するモデルがない回の、共同更新の入力（学習の設定、共有部とoptimizer。反復の契約。旧も参加するモデルがない回は共有部に触れない）。共有部が保有モデルのものであること（反復と共同更新の契約）。

## 5. Allowed Dependencies

`runtime/held_model_training_request_handling.py`: `random.Random`、`torch.optim.Optimizer`、`SharedFeatureExtractor`（以上は型注釈）、`perform_held_model_joint_training_iterations`、`HeldModelTrainingStateRegistry`、`LocalTrainingRequestSchedule`、`LocalTrainingSettings`、`ModelTrainingAndAssignmentCountsStore`、`ModelTrainingSampleStore`。許可集合を既存と同じ形で登録する（symbolごとの注入契約testは足さない）。件数管理のmoduleの依存は増えない。

## 6. Revalidation Triggers

共同学習の反復の引数・参加の条件（保有していて標本がbatchの件数以上）・戻り値（完了した回だけ損失を返す）、件数管理の算出と消化の規則、計数の加算の規則が変わったら、本specの対照testを再検証する。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/runtime/held_model_training_request_handling.py | 新規。公開の関数2つと、ownerの型を確かめる非公開の関数1つ |
| src/federated_learning_experiments/learning/training/local_training_request_schedule.py | 読取りの操作を1つ追加 |
| tests/refactoring/test_held_model_training_request_handling.py | 新規 |
| tests/refactoring/test_local_training_request_scheduling.py | 読取りの操作のtestを追加 |
| tests/refactoring/test_single_run_dependency_boundaries.py | 許可集合の登録 |
| tests/refactoring/fresh_process_smoke.py | 警報の前に学習要求を1件記録し、流れの最後に保留中の要求を学習する |

## 8. Testing Strategy / Requirements Traceability

oracleは、実旧client（`SharedBackboneClassConditionalESRFedSDAClient`）の実物の`train_step`・`flush_pending_updates`・`train_all_held_models`・`_train_heads_together`・`_sample_training_batches`。吸収のoracle（`build_absorption_oracle`）の実旧clientと新ownerへ、学習stepが読む属性（保留件数、一要求あたりの回数、batchの件数、診断と時間の入れ物）と設定（更新間隔、学習方式`joint`、勾配の統合`mean`）を与える。旧のmodule全体の乱数は、新の乱数生成器と同じ状態から始め、実行後の状態を比べる。新しく使う旧メソッドは`train_step`・`flush_pending_updates`・`train_all_held_models`で、前の2つは件数管理のtest、共同学習は反復のtestが実行済み。

| 要求 | 証拠 |
| --- | --- |
| 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 4.1 | 2値・多クラス、更新間隔、一要求あたりの回数（0を含む）、batchの件数（両モデルが参加・1モデルだけ参加・参加なし）を変えた条件で、要求と消化の列（空の消化、端数の消化、連続した要求）の各段の後に、保留件数・乱数の状態・計数（既存helper。項目の順を含む）・全モデルのparameterとgradとoptimizerの状態（既存helper）を実旧と照合。保有していないモデルの標本が学習にも計数にも入らないこと。optimizerの種類を変えた対照 |
| 1.5 | 件数管理のtest: 読取りが状態を変えず、回数が0でも件数だけで決まる |
| 2.1, 2.3 | 共同学習の反復が共同更新1回ぶんずつ呼ばれ、各回の時点で保留が未消化、計数がそれまでに完了した回のぶんだけ増えていること、渡る値が正しいこと。計数への反映のどの時点でも保留が未消化であること。割当概念の計数が不変 |
| 2.2 | 参加するモデルがない条件（どの保有モデルも標本がbatchの件数に満たない）の実旧との対照で、計数が変わらず、計数の項目も増えないこと（上の行の対照に含まれる）。加えて、反復を「損失を返さない」ものへ差し替えた条件で、標本の足りる保有モデルがあっても計数しないこと。後者は実旧をoracleにしていない: 実際の反復では、損失が返らない回は参加するモデルがない回と同じなので、「損失が返った回だけ計数する」という本処理の分岐そのものは、差し替えなければ観測できない |
| 3.1 | 本処理が読むownerそれぞれの別の型・派生型を、2つの関数へ渡して、反復が呼ばれず、保留件数を含む全状態が不変 |
| 3.2, 4.1 | 2回目の共同更新を抽出の後・更新の前に失敗させ（実旧は2回目の抽出の直後に失敗させる）、完了した1回ぶんのparameter・optimizer・計数と、保留件数が実旧と一致すること。batchの件数が不正な条件で、保留件数が実旧と同じ（要求の記録は残る）、計数・モデル・乱数が不変 |
| 1.3, 1.4, 3.3 | 学習へ進まないとき（間隔に達していない、保留が0件、回数が0）、学習へ渡すだけの値が不正でも成功し、保留件数のほかは何も変わらない |
| 4.2 | 汎用の変異tool（新しい関数と、件数管理の読取り）、依存の許可集合、共用のfresh process script、全pytest、Ruff・Pyright |

未検証として残す: これらを呼ぶ位置と標本1件の処理全体、1回の共同更新の途中で失敗した後の計数（旧との差。2節）、計算量・時間・勾配の診断、新全体runのgolden一致。
