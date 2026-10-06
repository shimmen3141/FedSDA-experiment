# 実測: 保有モデルの正式登録確認

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。詳細はdocs/experiments/refactoring-baseline.md。

## Task1

- RED: 未実装moduleによるModuleNotFoundError、1 collection error/1.84秒、exit1。
- GREEN初回73passed/3.27秒、最終対象84passed/2.92秒、exit0。Ruff/format成功、Pyright基準venv明示0 errors/0 warnings。
- 実旧保有分岐32条件（4元一覧順×正式ID0/4/12/大int×補助元あり/なし）、非負現在ID3、正式ID拒否9、各owner拒否28、欠落model1、API順序2、owner派生拒否7、送信ready/empty2。
- モデル参照・元先順序・統計/標本上書き・計数加算・current/pendingと取得済みrecordを対照。parameter/grad不変、呼出順と各入力拒否で全owner snapshot不変を確認。fixtureの旧model/標本は新NN/recordの借用で、実旧confirmのpop/代入/加算を観測する。

Task2/3とfeature最終GOは未実施。
