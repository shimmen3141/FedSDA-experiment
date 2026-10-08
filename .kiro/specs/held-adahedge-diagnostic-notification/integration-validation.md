# 統合検証

## 現在の状態
Task1・Task2を独立承認。Task3の全回帰・feature最終GOへ進む。

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
