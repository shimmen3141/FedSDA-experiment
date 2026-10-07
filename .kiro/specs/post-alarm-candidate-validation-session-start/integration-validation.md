# 統合検証

検証日2026-10-07、主担当Codex。検証対象commit **8cbf2ce**。旧固定基準748c3aa、基準Pythonは元checkoutのvenv/Scripts/python.exe。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1開始runtime | 正常36＋最終6＋履歴8＋任意metadata等4＝54passed、別Luna承認 |
| Task2拒否・不変 | 対象166passed、4変異を検出して元byte復元、別Luna承認 |
| Task3観測・継続更新 | 2/4class×3optimizerの6条件、対象全172passed、別Luna承認 |
| Task4対象＋AST | **1522passed / 1warning、9.91s、exit0**（主担当）。別Luna1522passed/9.18s |
| 全pytest | **7266passed / 3skipped / 2warnings、269.08s、exit0** |
| JUnit | 7269testcases、0failures、0errors、3skipped。前回6868＋対象172＋注入226＝7266 |
| Ruff check / format | 全src/tests/refactoring成功、148files formatted |
| Pyright | 基準venvを明示した昇格実行で全src0errors/0warnings |
| pip check / diff check | 成功 |
| 固定旧実装・2golden・旧回帰test・tools | 748c3aaから差分なし |
| 承認文書 | 要求r2・設計r2・命名r4・tasks r1のLF hash一致。tasksはcheckboxだけ進捗更新 |
| Fresh新CPU | 新public部品だけで2/4classの3epoch学習→4観測、readyは4件目のみ、固定参照全parameter不変、旧importなし。主担当とTask4 Lunaが独立再現 |

環境: Windows CPU、Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1、OMP/MKL各1thread、pytestのtorch CPU threads1。Ruff0.16.10、Pyright1.1.414。TMP/TEMP/MPLCONFIGDIR/FDE_MNIST_DATA_DIRは共通引継ぎ手順の共有venv/dataを明示。

JUnitは`../../venv/refactoring-tests/session-start-full.xml`。旧回帰・最終回帰の両testcaseにfailure/error/skipがなく、旧11ケース・最終3ケースのgolden照合が成功。3skippedは既存Windows wrapper、warningsは拒否test準備のnested Tensor prototypeと既存TypedStorage deprecated。

Fresh smokeの再実行本文と環境は`../../venv/refactoring-tests/session-start-task4-smoke.log`（UTF-16）に保存。旧oracleを使う数値対照と、この旧importなしの起動確認を区別する。

## REDと検出力

- Task1 source追加前: ModuleNotFoundError/collection1error/exit1。
- Task2のstub/順序入替/学習省略/参照live借用の4変異は、それぞれ72/45/30/54failed、全てexit1。各finallyと終端で元byteを復元し、source_bytes_restored=true。script・JSON・logsは`../../venv/refactoring-tests/session_start_mutation_evidence.py`と`session-start-task2-mutation-evidence.json`ほか。
- Task3はtest-only接続検証、production REDはN/A。
- Task4は226注入条件（許可80・拒否146）。最初の180条件はguard追加前92failed/88passed/1124deselected/exit1。Import拒否一覧への登録を一時除去すると46failed/180passed/1124deselected/exit1、finallyでtest元byteへ復元。最終GREEN1522passed。RED/一覧除去/GREEN logsは`../../venv/refactoring-tests/session-start-task4-{red,import-list-red,green}.log`。

## 同一性と境界

runtime単体byte/LF SHA256は`ad4af70509aca22d063a239df6bee70c94c87b8a11a8f739b7fc100efc3b00fc`。Task1 commit cebc254から内容不変。検証commit `8cbf2ce7fc383727989afe643954125d3983a3e4`のtracked Python＋2goldenは248パス、LF総合SHA256は`fab7c4ebcf6d8c037d15afa69e69449f6eb08690e801d5f48be7706376845f4d`。

候補生成→学習→固定参照→空損失収集を実旧の順序/RNGで接続し、借用入力・保有状態・統計・帰属を変更しない。観測と候補の継続更新までtestで接続した。active sessionの設定/交換、検出、区間切出し、初期値選択、採否、登録、通信、新client/全体runは範囲外。旧golden成功を新全体run一致とは扱わない。

今回、新しい旧実装の不具合は観測していない。LEGACY-014等の既存確認待ちは変更しない。動的fixture名の事後承認などの手順逸脱と訂正はreview.mdに記録した。

全pytestの独立再現は共通引継ぎ手順のユーザー決定どおり、主担当の実測とJUnit照合で判定する。Task5は別fresh GPT-6 Luna /root/luna_session_task5 APPROVED。JUnit・承認md/task/runtime hash・固定旧差分を独立確認。全pytest・品質・248path総合hashは独立再実行していない。

別fresh GPT-6 Luna /root/luna_session_feature_final の対象HEAD45adc7eの最終判定はGO。対象＋AST1522passed/1warning/exit0、新CPU2/4class smoke成功を独立再現。JUnit実parse、両goldentest成功、承認/runtime hash、検証commit以後src/tests不変、748c3aa固定旧差分空を独立確認。全pytest・品質・248path総合hashは独立再実行していない。阻害指摘なし、review.md/spec.jsonへ記録しcompletedとする。
