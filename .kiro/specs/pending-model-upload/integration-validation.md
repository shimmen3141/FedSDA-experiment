# 実測: 新規モデルの送信保留

## Scope
一保留枠/借用snapshot/対応ID/正の待機回数と境界通知/解除。統計取得は上位のtest-only接続。登録確認・ID対応・実送信・新client/全体runは後続。要求/設計/命名rev1が正本。

## Task1
- 先行REDは未存在packageのModuleNotFoundError、1 collection error/2.73秒/exit1。
- GREEN61 passed/1.79秒。9条件（delay1/2/4×ID負/0/正）の実旧具象FedSDAメソッドを用い、空/queue/境界/取得/ready/置換/途中解除/繰返し解除の列を照合。
- frozen field/read-only counter、借用dict/Tensor/順序/旧record保持、17不正条件×空/待機/readyの51拒否と直接record拒否を確認。
- 初回Ruffの未使用loop変数B007を`_`へ修正した。production動作変更なし。
- 最終61 passed/1.92秒、Ruff/format3files、Pyright0 errors/0 warnings。独立Lunaも61passed/品質成功、Task1 APPROVED。
- Lunaの有用な拒否入力前後比較提案を採用。反映後の主担当gateは61 passed/2.50秒、Ruff/format成功。ID/delay参照、key順、Tensor構造/grad/値（NaN含む）不変を追加確認。

## 再実行環境
2026-10-06、共有../../venvの固定Windows CPU基準（docs/experiments/refactoring-baseline.md / environments/golden/windows-cpu）。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
