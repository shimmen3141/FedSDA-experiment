# 実測: 新規モデルの送信保留

## Scope
一保留枠/借用snapshot/対応ID/正の待機回数と境界通知/解除。統計取得は上位のtest-only接続。登録確認・ID対応・実送信・新client/全体runは後続。要求/設計rev1・命名rev2が正本。
全3task承認/check後の別feature統合レビューで実Luna DECISION: GO。11/11要件・設計/ファイル計画・境界/依存・接続/共有状態・全実測にblockerなし。主担当completion gateも本scopeでVERIFIED。

## Task1
- 先行REDは未存在packageのModuleNotFoundError、1 collection error/2.73秒/exit1。
- GREEN61 passed/1.79秒。9条件（delay1/2/4×ID負/0/正）の実旧具象FedSDAメソッドを用い、空/queue/境界/取得/ready/置換/途中解除/繰返し解除の列を照合。
- frozen field/read-only counter、借用dict/Tensor/順序/旧record保持、17不正条件×空/待機/readyの51拒否と直接record拒否を確認。
- 初回Ruffの未使用loop変数B007を`_`へ修正した。production動作変更なし。
- 最終61 passed/1.92秒、Ruff/format3files、Pyright0 errors/0 warnings。独立Lunaも61passed/品質成功、Task1 APPROVED。
- Lunaの有用な拒否入力前後比較提案を採用。反映後の主担当gateは61 passed/2.50秒、Ruff/format成功。ID/delay参照、key順、Tensor構造/grad/値（NaN含む）不変を追加確認。

## 再実行環境
2026-10-06、共有../../venvの固定Windows CPU基準（docs/experiments/refactoring-baseline.md / environments/golden/windows-cpu）。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
## Task2
- test-onlyなのでproduction RED N/A。class2/4×delay1/2/4の6条件で実旧registerと新snapshot/loss/初期統計/store/pendingを接続。
- 登録時snapshot全値/順序/借用参照、保留後のモデル更新非波及、3件の同ID Welford統計更新後の現在全体/クラス別全field、delay countdown/readinessをexact照合した。snapshot固定/統計現在値の区別とTorch/Python/NumPy RNG保持を確認。
- 対象67 passed/3.14秒/exit0、Ruff/format成功。production変更なし。
- 独立Luna67passed/3.16秒、品質/diff成功、Task2/naming2 APPROVED。指摘なし。
- 承認後の主担当gateも67 passed/2.95秒/exit0。

## Task3
- exact依存26注入（21拒否/5許可）の先行RED: 12 failed/14 passed/638 deselected/0.11秒/exit1。
- exact symbol解決とbare import拒否を追加。対象＋AST731 passed/3.59秒/exit0。
- fresh新CPU processでclass2/4のproducer→loss→初期統計/store→pending→delay2→ready→現在統計取得/clearを接続し、旧package非import確認、exit0。
- source214 Python/両goldenのhash: f9858aa95133d1e066a29d0fdf1c7110d897511b8a38d7690b8a145b13467a97。Git追跡パスsort→UTF8相対path+NUL+LF正規化内容+NULで集計。要求/設計rev1・命名rev2の承認hash一致。
- 全Ruff成功、format114 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。固定748c3aaの旧production/両golden/旧回帰test差分は空。
- 全pytest4492 passed/3 skipped/1既存warning/134.37秒/exit0。旧11/最終3goldenも成功、値/許容差変更なし。JUnit pending-upload-full.xmlは4495 tests/0 failures/0 errors/3 skipped。
- skipは既存Windows wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。今回の範囲で新たな旧正常経路の不具合は観測していない。
- 実Luna Task3 APPROVED。承認後の主担当gate: 対象＋AST731 passed/5.59秒/exit0、fresh CPU成功/exit0、diff-check成功。

## 全11要件trace
|要件|証拠|
|---|---|
|1.1|9状態列の初期空/境界no-op|
|1.2|9状態列の正delay queue/再queue/置換、6実NN接続|
|1.3|待機/readyの同一record非消費取得、過去record保持|
|1.4|借用dict/Tensor/値順序、6接続でモデル更新非波及|
|2.1|delay1/2/4の各境界counter/readinessを実旧へ照合|
|2.2|空/readyで追加通知のno-op|
|2.3|途中解除/二回解除と空の有効counter0、次queueで再開|
|2.4|17不正×3状態の51queue拒否、直接record検証、既存/入力保持|
|3.1|9実旧具象FedSDA状態列と6統合、旧無効counterの対応明示|
|3.2|6接続の現在全体/class全統計fieldと登録時snapshotのexact対照|
|3.3|exact AST/fresh非旧import・品質/固定旧差分/全回帰のgate|

```text
python -m pytest tests/refactoring/test_pending_model_upload.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/pending-upload-full-20261006 --junitxml=../../venv/refactoring-tests/pending-upload-full.xml
python ../../venv/refactoring-tests/pending_model_upload_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```
