# 実測: 採用候補の初期ローカル登録

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。詳細はdocs/experiments/refactoring-baseline.md。主担当はClaude Code、独立レビューはcodex exec経由のGPT-6 Luna。

## Task1

- RED: 未実装moduleによるModuleNotFoundError、1 collection error/3.06秒、exit1。
- GREEN初回107passed/1failed/4.52秒。失敗は候補共有部へNaNを注入した拒否testで、test側の不変比較がtorch.equal(NaN,NaN)=Falseになったもの。productionは期待どおり拒否し状態も不変だった。注入値を-infへ変更。
- 最終: 対象108passed/3.96秒、exit0。Ruff check成功/format 129files整形済み、Pyright基準venv明示0 errors/0 warnings。
- 実旧対照48条件（class2/4×保有一覧(4,)/(4,9)/(9,-3,4)×現在ID保有/非保有×保有モデルの共有部が同一/別々×別ID既存保留なし/あり）、統計標本8条件（単一標本/singletonクラス/欠落クラス）、optimizer3種、反映先へ接続済み候補1。
- 拒否: 一時ID9、使用済みID3箇所、待機ラウンド数7、owner4×3型、保有0件1、候補/管理器/特徴/ラベル/非有限parameter15。各拒否で一覧record identity・共有部接続先・全parameter値/grad・全optimizer本体/state・統計・送信保留・現在ID・torch/random/NumPy乱数の不変を確認。呼出順1。
- 保有モデルが別々の共有部を持つ条件で、反映先が現在IDのモデル（非保有時は一覧先頭）の共有部だけになることを実旧と値・参照構造で照合。単一標本では実旧torch.varが自由度0のUserWarningを出すため、当該testだけfilterした（旧はNaN分散を0.1へ置換、新は同値を直接生成）。
- 依存境界の全走査testは新module用exact guard未追加で1件失敗（Task2のAST RED）。
- 実Luna Task1 APPROVED、指摘なし。Lunaのpytest独立実行はsandboxの一時ディレクトリ制約で未実施。
