# 統合検証

## 現在の状態
Task1を独立承認。Task2の変異・fresh接続とTask3の全回帰・feature最終GOは未実施。

## Task1
対象REDは新module不在でModuleNotFoundError、依存注入REDは18 failed/42 passed。実装と両resolverへのexact guard登録後、復元GREENは2902 passed/7.63s（対象21、依存guard2881）。新規追加は対象21＋guard60で81件。Ruff check/formatは4ファイル成功、正規Pyrightは0 errors/0 warnings。
ID検査をglobal再始動後へ移す実source変異は、bool/None/float/NumPy intの6ケースで状態snapshot assertionが失敗。収集失敗でなく検査順序の不具合を検出し、finallyで原byteへ復元・完全一致確認後にGREENを再測定した。

独立Luna mediumは対象＋guard2902件・Ruff4ファイル・新2module Pyrightを再現してAPPROVED。全pytestの独立再実行なし。

## 環境と成果物
Windows基準 `../../venv/Scripts/python.exe`、OMP/MKL1 thread。TEMP/TMPは既存 `../../venv/refactoring-tests`、MPLCONFIGDIRは `../../venv/matplotlib-cache`。保護設定・venv・goldenを変更しない。
実測logは `../../venv/refactoring-tests/held-adahedge-diagnostic-notification/task1-*.txt`。このPCのGit管理外で、別PCは存在を前提にせず対象test・依存guardを同じ基準環境で追試する。

## 検証の境界
globalと真の概念別oracleの独立保持、およびglobalだけの帰属変更通知が範囲。clientから通知する時機・重複防止、同期再較正、episode、保存診断全体、新全体run、Linuxgoldenは後続。既存golden成功をこれら全体の一致の証拠にしない。
