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

## Task2

- AST RED: 注入契約test追加直後20 failed/788 passed/0.90秒、exit1。generic runtime規則が必要symbolを拒否し、全走査testも新moduleで失敗。新module専用の13symbol guard/束縛symbol解決/module import拒否を追加後808 passed/0.64秒。
- 12条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無）の実NN接続は初回実行でGREEN。保有2モデルの共同更新→独立した共有部を持つ候補だけの学習（実旧ResidualAdapterMLP.updateと新の単一batch共同更新）→旧登録＋待機設定/新登録→3モデルの共同更新2回→現在ID切替え（後続specの責務のためtest-only接続）→実旧confirm/既存新confirm→(4,9,12)での共同更新。各段階でloss、全parameter値/grad、個別・共有optimizer state、共有参照構造、出力、統計、送信保留値、torch/random/NumPy乱数を照合。登録でresetされた候補の個別optimizerが後続学習でstateを蓄積し、正式ID確認後も同じobjectで継続することを確認。
- 最終: 対象＋AST 928 passed/5.11秒、exit0。Ruff check成功/format 129files整形済み、Pyright基準venv明示0 errors/0 warnings。
- fresh `../../venv/refactoring-tests/adopted_candidate_initial_local_registration_cpu_smoke.py`成功、exit0。class2/4で保有モデル学習→候補学習→一時ID登録（共有値反映/出力不変/個別reset/統計/保留）→後続学習→待機満了→正式ID確認→学習。旧importなし。
- 実Luna Task2 APPROVED、独立928 passed/smoke PASS/Ruff成功、指摘なし。
