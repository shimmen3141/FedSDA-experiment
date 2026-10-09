# 検出力の証拠

Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）、source/test commit `1a03667`。証拠ファイルは元checkoutの`venv/refactoring-tests/held-model-training-request-handling-mutation-evidence/`（Git管理外、このPCだけ）。

## 実装前の失敗（RED）

- 件数管理のtestへ足した読取りの操作のtest（4条件）は、変更前のsourceで4件とも失敗した（操作がない）。既存の70件は成功。
- 依存境界: 新moduleを置いて許可集合を登録する前に、`test_single_run_layers_import_only_allowed_dependencies`が失敗した（新moduleのimportが「この層では許されない依存先」と報告される）。登録後に成功。
- 新しいtest moduleは、sourceがない間は収集で失敗するだけなので記録しない（共通引継ぎ手順）。

## 汎用の変異tool

### 学習要求の処理の関数

```text
python .kiro/settings/scripts/mutation_check.py --source src/federated_learning_experiments/runtime/held_model_training_request_handling.py --functions _validate_training_request_owners,train_held_models_for_pending_training_requests,record_training_request_and_train_held_models_when_due --tests tests/refactoring/test_held_model_training_request_handling.py --evidence <.../handling> --extra ...（9種）
```

結果: 31/35検出（機械的な変異26種と、`--extra`で足した9種）。復元後は101 passed。`report.json`は`handling/`。

未検出の4種は等価と判断した。いずれも、保留中の要求の学習の関数の中で、隣り合う読取りだけの文を入れ替える変異である。

- 「保留が0件なら空のtupleを返す」と、保有モデルの対応のsnapshotの入替え: 保留が0件のときにもsnapshotを取ってから返すことになる。snapshotは保有モデルの一覧を読むだけで、どの状態も変えず、型の検査を通ったownerでは例外を出さない。戻り値は同じ。
- 保有モデルの対応のsnapshotと、標本列のsnapshotの入替え: どちらも読取りだけで、互いに依存しない。
- 標本列のsnapshotと、保有IDの集合を作る文の入替え: 同上（保有IDの集合は、保有モデルの対応のsnapshotだけから作る）。
- 保有IDの集合を作る文と、損失を集める空のlistの代入の入替え: 同上。

手で足した9種（すべて検出）: 計数のownerのexact型検査をisinstanceへ緩める（この検査は複数行にわたるので、toolの機械的な「緩める」変異の対象にならない）、共有部を更新しない、参加の条件のbatchの件数を「より多い」にする、保有していないモデルも計数する、学習標本数を1にする、更新回数を2にする、反復を1度に2回ぶん呼ぶ、保留の消化の件数を1にする、反復の回数を1回にする。

機械的な変異で検出されたものには、ownerの型検査を消す・緩める・次の文（保留件数の読取り、要求の記録）の後へ移す、保留が0件のときの早期returnを消す、反復のloopを消す、損失が返らなかった回を飛ばす文を消す、損失を集める文を消す、計数のloopを消す、保留の消化を消す、要求の記録を消す、間隔に達していないときの早期returnを消す、が含まれる。

仕様化の段階（作業ツリーの複製。規則からの逸脱としてresearch.mdに記録）で未検出だった2種に対して、testを足している: 「計数の反映と保留の消化を入れ替える」（計数への反映の時点で保留が未消化であることの確認）、「損失が返らなかった回を飛ばす文を消す」（反復を差し替えて、損失が返らなかった回を計数しないことの確認。実旧をoracleにしていない。理由は設計8節）。

### 件数管理の読取りの操作

`has_pending_requests_reaching_update_interval`の本体はreturnだけで、toolが機械的に作れる変異がない。`--extra`で3種を足した: 「以上」を「より多い」にする、実行間隔の代わりに一要求あたりの回数と比べる、常に真を返す。

```text
python .kiro/settings/scripts/mutation_check.py --source src/federated_learning_experiments/learning/training/local_training_request_schedule.py --functions has_pending_requests_reaching_update_interval --tests tests/refactoring/test_local_training_request_scheduling.py tests/refactoring/test_held_model_training_request_handling.py --evidence <.../schedule> --extra ...（3種）
```

結果: 3/3検出。復元後は175 passed。`report.json`は`schedule/`。

実行に使ったコマンドの全文（`--extra`を含む）は、各`report.json`の変異の名前と、主担当の一時領域の`draft8/mutate.sh`（Git管理外）にある。変異の後、作業ツリーに差分が残っていないことを`git status --short`で確かめた。

## 依存境界

新moduleの許可集合を登録した（symbolごとの注入契約testは足していない）。許可集合が実際のimportより広くないことは、既存の`test_module_allowed_dependencies_are_all_imported_by_the_module`が確かめる（登録したmoduleの数の下限を60へ上げた）。

## fresh process

共用の`tests/refactoring/fresh_process_smoke.py`へ流れを足した: 警報の前に学習要求を1件記録し（間隔2なので保留されるだけ）、各流れの最後に保留中の要求を学習する（標本を持つ保有モデルだけが3回共同学習され、計数が3ずつ増え、保留が0件になり、もう一度呼ぶと何もしない）。`FRESH PROCESS SMOKE PASSED: 16 flows; legacy and test modules not imported`。pytestからは`test_fresh_process_smoke.py`が別processで実行する。
