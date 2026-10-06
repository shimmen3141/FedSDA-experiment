# 実測: 保有モデルの正式登録確認

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。詳細はdocs/experiments/refactoring-baseline.md。

## Task1

- RED: 未実装moduleによるModuleNotFoundError、1 collection error/1.84秒、exit1。
- GREEN初回73passed/3.27秒、最終対象84passed/2.92秒、exit0。Ruff/format成功、Pyright基準venv明示0 errors/0 warnings。
- 実旧保有分岐32条件（4元一覧順×正式ID0/4/12/大int×補助元あり/なし）、非負現在ID3、正式ID拒否9、各owner拒否28、欠落model1、API順序2、owner派生拒否7、送信ready/empty2。
- モデル参照・元先順序・統計/標本上書き・計数加算・current/pendingと取得済みrecordを対照。parameter/grad不変、呼出順と各入力拒否で全owner snapshot不変を確認。fixtureの旧model/標本は新NN/recordの借用で、実旧confirmのpop/代入/加算を観測する。

## Task2

- AST RED: 注入14 failed/754passed/0.77秒、exit1。generic runtimeは必要なsymbolを拒否し、一部不要stdlibを許可した。新module専用の8symbol guard/resolver/module import拒否へ追加後768passed/0.87秒。
- 12条件実NN接続の初回は12failed/852passed/4.16秒。Task1の旧confirmへ新NNを借用するhelperが、独立実旧NNもobject identityで比較していたため、異なる実装のNNで失敗した。productionの問題ではない。test-only比較モードの名前を追加命名レビューへ戻す。NN数値は既存joint-update oracleで比較する。
- 命名revision2承認後の再実行は、NumPy RNG配列をTensor用再帰helperへ渡したため末尾の配列比較で12failed/852passed/4.56秒。helperと上流testを読み直し、上流と同じ文字列/np.array_equal/末尾tupleの比較へ修正。NN値の不一致ではなくtest側の比較対象の誤りだった。
- 最終GREEN: 対象＋AST864passed/3.97秒、exit0。class2/4×Adam標準/AMSGrad/SGD×共有更新有無の12条件3更新、初回後にnew coordinator/実旧confirm、registryの現在bindingで後続学習。全loss/parameter/grad/共有・個別optimizer/count/各owner ID/順序と3RNG/defaultsが一致。optimizer本体/蓄積state/取得済みrecord/pending snapshot/共有参照も保持。
- fresh `../../venv/refactoring-tests/held_model_registration_confirmation_cpu_smoke.py`成功、class2/4で新学習→7owner確認→現在binding/標本取得→後続学習、計数/optimizer蓄積を確認。旧importなし。上流新CPU smokeを土台にownerへの接続を追加し、独立プロセスで実行。

- 実Luna Task2 APPROVED、独立864passed/fresh新CPU/品質/diff/scan成功、指摘なし。主担当の全対象Ruff成功/127files整形済み、Pyright基準venv明示0 errors/0 warnings、pip/diff成功、固定旧production/golden/回帰test差分空。

Task3とfeature最終GOは未実施。
