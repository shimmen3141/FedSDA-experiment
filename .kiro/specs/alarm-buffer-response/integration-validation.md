# 警報時の保留標本への応答 — 統合検証

## 対象と現在の判定

対象source/test commit: `ea61b8a`（本specの実装は`59ded40`、以降は証拠文書）。主担当Codex、2026-10-08、Windows CPUの固定venv。全5taskは独立GPT-6 Luna承認。全回帰は9110 passedで終了し、Task 5と別fresh Lunaのfeature最終判定は**GO、9/9要求を検証、本spec完了**。

実装境界は`runtime/alarm_buffer_response.py`の不変`AlarmBufferResponse`と`respond_to_alarm_with_buffered_samples`。検証中なら保留全件を現行へ吸収し、未検証なら前区間準備→最小件数→公開区間解決を組み立てる。結果5種とFIFO消費判断を返す。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 結果型と初期依存 | 実装前RED、全5有効形/不整合拒否・frozen/kw_only。record＋AST2277 passed、9.76s。exact5、56注入。初回拒否後の修正を別Luna承認 |
| 2 応答組立 | 実装前RED。対象64＋AST2455＝2519 passed、9.65s。exact23、235注入。実NN36正常/8active、2回共同更新、候補開始後の実観測。別Luna承認 |
| 3 検出力 | 実source9変異検出、各回byte復元、復元後64 passed、11.17s。[変異証拠](mutation-evidence.md)。別Luna照合・承認 |
| 4 独立接続 | 対象＋AST2519 passed、20.67s、fresh新CPU12条件。別Lunaが2519 passed、18.20sと新CPU12条件を独立再現・承認。[詳細](cpu-and-dependency-evidence.md) |
| 5 全回帰・品質 | 全pytest 9110 passed / 3 skipped / 2 warnings、416.70s、exit0。Ruff/Pyright/pip checkと同一性検査成功。別LunaがJUnit/hash/空diffを独立照合して承認 |

レビューsession・採否はspec.json/review.md。Task 4の独立再現にcache権限warningと完了後cleanup errorが報告されたため、テスト成功と区別して記録している。

最終担当はreviewed commit `825a206`、全5task・9/9要求・接続/境界・承認hash・全回帰log/JUnitを照合してGO。全suite/品質/fresh CPUの最終担当による再実行は行っていない。Task 4担当の独立対象/CPU再現、Task 5担当の独立JUnit/source hash再現も証拠に含む。未完了taskとblocking指摘はない。最終レビュー証拠は元checkoutのvenv/refactoring-tests/alarm-buffer-response-feature-final-review.md/log。

## 要求対応（9/9）

| 要求 | 実装と意味を持つ検証 |
| --- | --- |
| 1.1 | active分岐で公開吸収だけを使い全件を現行へ適用。空/非空、2/4class、正規/負IDを実旧対照。prepare/resolveは禁止wrapper |
| 1.2 | 公開prepareで評価保存→前区間吸収を済ませてから件数判定。実旧の前区間を含む状態とPython Randomを比較、実呼出し順も記録 |
| 1.3 | 件数直前/ちょうど、全件span/短span、空、容量+1で不足判定。変化区間未処理、旧不足FIFO保持を維持 |
| 1.4 | 準備された変化標本/概念だけを公開resolveへ渡し、吸収後の統計と元metadataを使用。再利用・維持・候補開始のNN結果と後続学習/観測を対照 |
| 2.1 | formal5値、不変kw_only record、実5有効形、型/None/結果とsessionの不整合を拒否 |
| 2.2 | 全経路でFIFOと最終位置は不変、readonly propertyは不足だけFalse。実旧の最終FIFOとの差を境界として照合、消費と判断逆転の変異を検出 |
| 2.3 | 同じactive session、候補/参照値・grad・両optimizer・履歴・pending tuple参照を保持。現行帰属・評価保存状態も実旧一致 |
| 3.1 | 共通構造・位置対応・使う件数の16拒否条件でowner/torch/Python/NumPy/明示Random/FIFO/評価状態不変。activeは未使用設定をobjectへ変更しても成功 |
| 3.2 | 実旧警報5経路、正常torch/Python/NumPy、共同更新と候補観測、9変異、exact依存、新CPU12条件。全9110 passed、旧11/最終3golden、JUnit確認 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r1 | 0d1eadc4547549bb562c723c384f968ddb04a8db43ebb0a6910f7f4160cb65c8 |
| 設計r2 | 6016db437f1113d44c7b4c4b096313a10198dca4decf7900c3708d74a5bec0bc |
| 命名r5 | 1a88576ee07b0e74c8842fc57e40ce5df280332307ae3bc983824a20e9487ffe |
| tasks r1（checkboxを未完了へ戻す） | 964ab0a80f3ad366add18b1e1fd23150989b2fe8bd671e6065ca9c674a2b0903 |
| source全体、ea61b8a、265パス | 49e99d8098e2053386c4e6d592853369569f4c2c62ee1b190a7d761c80cf2790 |

source hashは共通手順のtracked Python＋2golden、パス昇順、UTF8 path＋NUL＋commit内容のLF byte＋NUL。git cat-file --batchで取得し作業ツリーのLF内容一致も確認。全承認hash一致、source/testの175固有識別子に未承認名なし。固定旧`748c3aa`から旧実装/tools/2golden/旧回帰testのdiffは空。

Ruff check成功、format checkは165 files already formatted。Pyright固定venv指定で0 errors/0 warnings/0 informations、pip check成功、git diff --check成功。新TODO/FIXMEや秘密値設定は対象2ファイルに見つからない。

全pytestは**9110 passed / 3 skipped / 2 warnings、416.70s、exit0**。前spec8811＋本対象64＋新注入235＝9110。JUnit9113 testcase、failure/error0、skip3。旧11/最終3goldenは各1pytest entryで全scenarioを検査し、どちらもfailure/error/skipなし（178.754s/50.819s）。既存のPOSIX bash不在の3skipとnested Tensor/TypedStorageの2warningで、追加skip/warningはない。

## コマンド・証拠の場所

OMP/MKL各1thread、TMP/TEMP=元checkoutのvenv/refactoring-tests、MPLCONFIGDIR=venv/matplotlib-cache、FDE_MNIST_DATA_DIR=data/mnist。環境の正本は[基準環境](../../../docs/experiments/refactoring-baseline.md)。

```powershell
../../venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=../../venv/refactoring-tests/alarm-buffer-response-full.xml
../../venv/Scripts/python.exe -m ruff check src tests/refactoring
../../venv/Scripts/python.exe -m ruff format --check src tests/refactoring
../../venv/Scripts/python.exe -m pyright --pythonpath ../../venv/Scripts/python.exe
../../venv/Scripts/python.exe -m pip check
```

外部証拠は元checkoutの`venv/refactoring-tests/`にあるfull.log/full.xml、pyright.log、verification-summary.json、naming-audit.json、各task-review.md（全てalarm-buffer-response接頭辞）。全pytestはユーザー決定に従い主担当の実測とJUnit照合で判定し、独立全再実行を必須としない。

## 境界と制約

- 新client/新全体runのgolden一致は未検証。旧/最終goldenは固定参照実装を守る検査で、本部品だけの検証と区別する。
- 実FIFO消費、event→reset→drainの後始末、切替/再利用計数/通知、session保持/解除、検出episode制御、設定登録、計算量診断は呼出側の後続spec。新state ownerは今回追加しない。
- 位置付きpayloadの意味上の対応は供給側が保証する。sessionとnested recordは借用参照で、深いimmutable化はしない。
- common構造/件数拒否は更新前だが、全設定を全体事前検査しない。prepare後にresolverが拒否すれば完了した前区間更新は残る。実testで固定した保証境界。
- LEGACY-002の不足FIFO保持を維持し、具体的改善は既存の[改善候補](../../../docs/research/improvement-candidates/README.md)と区別する。このspecで研究アルゴリズムの変更は採用していない。
