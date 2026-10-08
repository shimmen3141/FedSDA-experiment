# 統合検証

## 現在の状態
全3taskを独立承認。Task3で対象＋guard2905件/fresh/Ruffを独立再現し、主担当のWindows全回帰とJUnitを照合した。別fresh sessionのfeature最終GOへ進む。

## Task1
対象REDは新module不在でModuleNotFoundError、依存注入REDは18 failed/42 passed。実装と両resolverへのexact guard登録後、復元GREENは2902 passed/7.63s（対象21、依存guard2881）。新規追加は対象21＋guard60で81件。Ruff check/formatは4ファイル成功、正規Pyrightは0 errors/0 warnings。
ID検査をglobal再始動後へ移す実source変異は、bool/None/float/NumPy intの6ケースで状態snapshot assertionが失敗。収集失敗でなく検査順序の不具合を検出し、finallyで原byteへ復元・完全一致確認後にGREENを再測定した。

独立Luna mediumは対象＋guard2902件・Ruff4ファイル・新2module Pyrightを再現してAPPROVED。全pytestの独立再実行なし。

## 環境と成果物
Windows基準 `../../venv/Scripts/python.exe`、OMP/MKL1 thread。TEMP/TMPは既存 `../../venv/refactoring-tests`、MPLCONFIGDIRは `../../venv/matplotlib-cache`。保護設定・venv・goldenを変更しない。
実測logは `../../venv/refactoring-tests/held-adahedge-diagnostic-notification/task1-*.txt`。このPCのGit管理外で、別PCは存在を前提にせず対象test・依存guardを同じ基準環境で追試する。

## 検証の境界
globalと真の概念別oracleの独立保持、およびglobalだけの帰属変更通知が範囲。clientから通知する時機・重複防止、同期再較正、episode、保存診断全体、新全体run、Linuxgoldenは後続。既存golden成功をこれら全体の一致の証拠にしない。

## Task2
実source変異23種類を測定。初回は非等価21のうち19検出、整数派生型受理2変異を見逃したため、既存parameterへ前ID・後ID・概念IDの3条件だけを追加。2変異を再測定して検出し、最終は非等価21/21をassertionで検出した。初回と追加後の計25試行で、検査を更新後へ移動する変異も検出。変更の影響がない他19条件は再実行していない。
等価2種類（exact builtin int検証後の `!=` と `not ==`、dict直接反復とkeys反復）は実測exit0、等価理由を記録し検出率に含めない。各finallyの元byte復元とsource2ファイルの初回hash一致を確認。testだけが3ケース増え、名前と役割の追加なし。names照合は未登録・誤用なし。
復元後は対象24＋guard2881＝2905 passed/11.24s、Ruff check/format・diff check成功。Luna独立再実行も2905 passed/13.04s。

fresh `python -I -S -B` は新srcを明示し、新packageとstdlibだけでglobal/oracle損失更新・pool変更・None/同一通知・異なる変更2回＋同通知再実行・同ID再取得・新ID生成・不正通知3条件/不正概念2条件を接続。global最終は累積{-1:0,2:1}・gap0.5・pool計数1・再始動3、oracleはgap0.3・再始動0、新oracleはgap0.25・再始動0。旧/test/pytest/NumPy/torchとsite-packages importなし。
証拠はtask2-report.md、task2-mutations.json（全23）、task2-mutations-after-coverage.json（欠落2）、各task2-mutant-*.txt、task2-fresh-smoke.py/json、復元GREEN log。いずれも上記Git外directory。別PCで追試する場合は本表の契約を一つずつsourceへ変異させ、対象testのassertion失敗を測り原byteへ復元する。freshは上記操作列を新packageだけで実行し、sys.modulesに旧/test/numericsがないことを確認する。既存脚本がある場合はworktree rootで `python -B <artifact>/task2-mutations.py` と `python -I -S -B <artifact>/task2-fresh-smoke.py`。変異はsourceを書換えるため他の検証と同時実行しない。再実行前に旧証拠を別保存し、JSONを上書きしない。

## Task2レビューの範囲
Luna medium `/root/held_diagnostics_task2_review` が実行したのは対象＋guard2905件、Ruff、diff check。freshと変異は独立実行していない。追加読取りで変異script、初回/追加後JSON、追加後2失敗log、fresh script/JSONを照合し、非等価19＋2＝21/21、等価2、元byte復元hash一致に指摘なし。Task1のsource/test commitは `4ff23f5`、Task2は `24a5953`。

## Task3の主担当測定
対象はcleanなsource/test commit `24a5953`。Windows Python 3.13.15、torch 2.12.1+cpu、NumPy 2.4.6、OMP/MKL各1 thread。TEMP/TMPは `../../venv/refactoring-tests`、MPLCONFIGDIRは `../../venv/matplotlib-cache`、FDE_MNIST_DATA_DIRは `../../data/mnist`。preflightでtorch読込みとTEMPのread/write/delete/cleanup成功を確認。

全pytestは9811 passed/3 skipped/2既存warnings、199.13s、exit0。前spec9727＋今回対象24＋新依存guard60＝9811。JUnit9814 testcase、failure/error0、skipped3。JUnitの `tests.test_regression.test_regression` と `tests.test_proposed_regression.test_proposed_regression` の2 testcaseが成功し、内部で旧11/最終3goldenを照合する（14個のJUnit testcaseではない）。Ruff check成功、format180 files already formatted。正規Pyrightは0 errors/0 warnings/0 informations、pip check成功。

identityはr1の4承認hash一致、固定旧748c3aaからcommit済み/作業ツリーの旧差分空、測定時cleanを確認。tracked Pythonと2goldenのLF source hashは281パス `8f94d784a3a1f0674c6ba76dae1ba114a4c96ca87480945e6041fb94ef99d2e2`。JUnit byte hashは `f4c902f01a23d47a1228cf5f83e4c5ec6deda9425cfa1b27bf627ec778647dc2`。

証拠は同Git外directoryの `full-pytest.txt/xml`、`full-ruff-check.txt`、`full-ruff-format.txt`、`full-pyright.txt`、`pip-check.txt`、`task3-preflight.txt`、`task3-junit-summary.json`、`task3-identity-before.txt`。全pytestは主担当のみ実測（independent_full_suite_rerun=false）。文書担当はログ/JUnit/JSONを読取り照合した。文書編集後の未コミット文書差分と、測定時cleanを区別する。

## 要求8条の証拠対応
対象24件は `tests/refactoring/test_held_adahedge_diagnostic_notification.py`、依存guard2881件（今回追加60）。以下のtest名は対象内。

| 要求 | 観測証拠 |
| --- | --- |
| 1.1 | `test_diagnostic_collection_keeps_distinct_live_owners` で空global・別集合独立・live参照。`global-copy` 変異検出、fresh初期snapshot空。 |
| 1.2 | 同testと `test_held_diagnostic_evidence_matches_real_legacy_notifications` で同ID再取得/別ID・global非共有/損失保持。`eager-concept`/`recreate-same-id`/`share-global-oracle`/`share-oracles` 検出。freshで-7再取得と新4。 |
| 1.3 | owner testで作成順tupleと過去tuple不変。`sorted-ids`/`mutable-ids` 検出。freshは旧tuple(-7,)を保持し(-7,4)を確認。 |
| 2.1 | 実旧通知testで変更ID列・同record再通知のglobal累積/gap消去、再始動+1、pool/oracle/ID順保持。`oracle-target`/`skip-restart`/`double-restart`/`repeat-suppressed` 検出。freshでglobal再始動3/pool1、oracle再始動0。 |
| 2.2 | 実旧通知testとfreshでNone/同IDの全snapshot不変。`equal-restart`/`none-restart` 検出。 |
| 2.3 | `test_diagnostic_notification_rejects_invalid_input_before_restart` と `test_diagnostic_collection_rejects_invalid_concept_id_without_creation` でexact型/IDの先行拒否・状態不変。型/検査順変異を検出。追加後logは通知2 failed/22 passed、概念1 failed/23 passed。fresh不正通知3/概念2を拒否。 |
| 3.1 | `test_diagnostic_notification_preserves_random_states` でPython/NumPy/torch乱数不変。2881 guardとfreshのblocked/nonstdlib imports空で依存境界。予測/学習/session/FIFOの全体進行は未接続。 |
| 3.2 | 実旧clientの取得/update/通知を差し替えず7通知条件×4owner更新・通知後重みを全field比較。JUnit9814件のgolden2 testcase内11+3成功、固定旧/golden差分空。新全体run・保存診断全体の一致は未検証。 |

## Task3の独立照合
Luna medium `/root/held_diagnostics_task3_review` が対象＋guard2905件、fresh stdlib接続、Ruff3ファイルを独立実行してAPPROVED。JUnit/hash/4承認/source281パス/固定旧差分を照合し指摘なし。全pytestとPyright・pip checkは独立再実行していない。全pytestは主担当実測とJUnitに基づく。以後の差分は証拠・進捗文書だけ。
