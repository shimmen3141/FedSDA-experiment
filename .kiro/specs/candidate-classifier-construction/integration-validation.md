# 統合検証

検証日2026-10-07、主担当Codex。検証対象commit **ceb4336**。旧固定基準748c3aa、基準Pythonは元checkoutのvenv/Scripts/python.exe。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1生成/拒否・独立性 | 30passed、Luna独立再現・承認 |
| Task2初期snapshot選択→生成→3batch更新 | 6条件、対象全36passed、Luna独立再現・承認 |
| 対象＋AST | 1074passed、4.30s、exit0。ASTコード部分Luna独立再現・承認 |
| 全pytest | **6412passed / 3skipped / 2warnings、137.46s、exit0** |
| JUnit | 6415testcases、0failures、0errors、3skipped。前回6355＋対象36＋注入21＝6412 |
| Ruff check / format | 全src/tests/refactoring成功、143files formatted |
| Pyright | 基準venv明示の昇格実行で全src0errors/0warnings |
| pip check / diff check | 成功 |
| 固定旧実装・2golden・旧回帰test・tools | 748c3aaからcommit済み/作業中の両方で差分なし |
| 承認文書 | 要求r2・設計r2・命名r4・tasks r2のLF hash一致。tasksのcheckboxだけ進捗更新 |
| Fresh新CPU | 二値/4クラスで候補生成→3回更新、参照不変、旧importなし、exit0。Lunaも独立実行 |

JUnitは`../../venv/refactoring-tests/candidate-classifier-construction-full.xml`。旧回帰・最終回帰の両testcaseにもfailure/error/skipがなく、固定旧11ケース・最終3ケースのgolden照合が成功した。3skippedは既存Windows wrapper。warningsは拒否test準備のnested Tensor prototypeと既存TypedStorage deprecatedで、productionの警告ではない。

環境: Windows CPU、Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1、OMP/MKL各1thread、torch CPU threads1。Ruff0.16.10、Pyright1.1.414。TMP/TEMP/MPLCONFIGDIR/FDE_MNIST_DATA_DIRは共通引継ぎ手順の共有venv/dataを明示。sandboxでのPyright site-packages探索失敗は昇格条件で再検査し、型設定やAnyで回避していない。

## REDと検出力

- Task1: source作成前import失敗1error/exit1。stub/初期値読込み省略/個別parameter逆順はいずれも6failed/exit1、復元hash一致。空共有parameter条件追加はRNG変更で1failed、生成前拒否追加後GREEN。
- Task3: exact注入契約は10failed/11passed。guard追加のみではsymbol resolver登録が不足し8failed、その既存登録一覧へmoduleを追加した後に対象＋AST1074passed。productionの失敗とAST検査器の組立不足を区別する。
- test関数名追加の承認前にTask1test本文を作った手順逸脱はreview.mdへ記録し、正式名承認後に検証した。runtimeの名前・役割は承認済みのまま。

## 同一性と境界

検証commitのtracked Python＋2goldenは243パス、source SHA256:
`6c91132b5d5f6913740fba56f0888a65d9da152a6b8800b42f089844afeb9ea6`。
runtime単体LF SHA256は`50c5e8b311762fa9bfd9e788cfbcde84e622a1474bacdcaff8c2e41e1c6c5dd7`。

生成部品の旧正常値・乱数・新optimizer・参照独立性と既存単回更新への接続を検証した。候補の複数epoch学習/early stopping、参照固定とのsession開始組立、新client/全体runは未完成で後続。旧golden成功は固定旧経路の保全であり、新全体runの一致を主張しない。

空共有部は既知LEGACY-010の制約であり、本生成の事前条件として拒否する。今回、新しい旧実装の不具合は観測していない。LEGACY-014等の確認待ちや既存記録は変更しない。

全pytestの独立再現は共通引継ぎ手順のユーザー決定どおり、主担当の実測とJUnit照合で判定する。Task3と別feature最終GOは独立レビュー結果をreview.md/spec.jsonへ記録してから完了扱いにする。
