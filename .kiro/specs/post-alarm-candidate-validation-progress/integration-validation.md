# 統合検証

検証日2026-10-07、主担当Codex。検証対象commit **2d513a9c382c5bf55eef81d79bc2c09bec1954d7**。旧固定基準748c3aa。基準Pythonは元checkoutのvenv/Scripts/python.exe。実装・testsをcommitして作業ツリーが空の状態で全回帰を実行した。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1 判定記録 | 2/4/5件×4分岐と不変記録＝13passed、独立Luna承認 |
| Task2 進行 | 対象全52passed。4分岐×2/4class×保留0/3の16条件、非active・未到達・事前拒否・frozen/keyword、独立Luna承認 |
| Task3 実接続 | 対象全61passed。実NN開始→観測→確定→2回共同更新6条件、奇数5件等1条件、確定時ID/可用性2条件、独立Luna承認 |
| Task4 対象＋AST | **1517passed、4.93s、exit0**（対象61＋AST1456）。新規注入106条件、独立Luna承認 |
| 全pytest | **7433passed / 3skipped / 2warnings、142.84s、exit0** |
| JUnit | 7436testcases、0failures、0errors、3skipped。前spec7266＋対象61＋注入106＝7433 |
| Ruff check / format | 全src/tests/refactoring成功、151files formatted |
| Pyright | 基準venvを明示して全src 0errors/0warnings/0informations |
| pip check / diff check | 成功 |
| 固定旧実装・2golden・旧回帰test・tools | 748c3aaからcommit済み・作業ツリーとも差分なし |
| 承認文書 | 要求r2・設計r2・命名r3・tasks r1のLF hash一致。tasksはcheckboxだけ進捗更新 |
| Fresh新CPU | 新public部品だけで2/4classの実学習→非active→未到達→確定→共同更新。固定参照不変、旧/test importなし。主担当とTask4 Lunaが独立再現 |

Windows CPU、Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1、OMP/MKL各1thread。Ruff0.16.10、Pyright1.1.414。TMP/TEMP/MPLCONFIGDIR/FDE_MNIST_DATA_DIRは共通引継ぎ手順の共有venv/dataを明示。全pytestは昇格実行で一時資材と子プロセスのACL制限を回避した。

JUnitはGit管理外`../../venv/refactoring-tests/validation-progress-full.xml`。旧回帰・最終回帰の両testcaseにfailure/error/skipがなく、旧11ケース・最終3ケースのgolden照合が成功。3skippedはPOSIX bashが利用不能なWindows環境による既存test（ablation suiteの一覧1件、server sweep wrapper2件）。warningsは拒否test準備のnested Tensor prototypeと既存TypedStorage deprecated。主担当の全回帰ではpytest cacheproviderを無効化し、新たなcache警告はない。

全回帰ログ、JUnitの独立parse用要約JSON、同一性JSONも同じGit管理外ディレクトリに保存した（validation-progress-full.log / validation-progress-junit-summary.json / validation-progress-identity.json）。fresh smokeはvalidation_progress_cpu_smoke.pyとprogress-task4-smoke.log。別PCではこれらの資材が存在するとは限らず、共通引継ぎ手順の環境・本specのAPI/条件から再実行する。

## REDと検出力

- Task1 source追加前: ModuleNotFoundError、1collectionerror、exit1。共有temp環境指定漏れの訂正はreview.md参照。
- Task2 source追加前: ModuleNotFoundError、1collectionerror、2.87s、pytest exit2。登録・吸収へ影響した制御loss fixtureの初回失敗と修正はreview.md参照。
- Task3はtest-only接続。stub/到達判定反転/開始時ID使用/記録位置誤り/余分なRNG/二重適用の6変異を、47/27/1/25/47/18failedとして検出、各exit1。各finallyと終端で元byte復元、source_bytes_restored=true。script/JSONはvalidation_progress_mutation_evidence.py / validation-progress-mutation-evidence.json。復元後61passed/3.47s/exit0。
- Task4は106注入条件。guard前52failed/54passed/1350deselected/0.22s/exit1、guardと両resolver一覧へ登録後1517passed/exit0。初回patch生成が書込み前のassertで停止した経緯もreview.mdへ記録した。

## 同一性

| 対象 | LF SHA256 |
| --- | --- |
| 要求r2 | 9c2582e9ae9e9e4106182065f0bea60b6e7ede4ae769e2cafc878d70326a84b7 |
| 設計r2 | d14644d30cf2f7d7240ebd1772a13eb8be988553a2edd75589d23793bf3b4b38 |
| 命名r3 | 7367d8eacd4803d7d0bcbfbc5d98a9768c1ac0e4fd057a36a07169a4e153bf42 |
| tasks r1（承認時） | d6c35f73da14d1aef3c96699950b35833f3240a9183b6a3c3a6d22d4d4c43048 |
| 進行runtime（byte/LF同値） | 52fac2754ad51f4a90e4dfbefe8869ccb95c15f229291171a26b24780b3e4610 |
| 不変判定record（byte/LF同値） | a8f1266d7b28e8f091b5956febc1c8262f8a8264875536bf57be892b7332dda4 |
| 検証commitのtracked Python＋2golden、251パス | 24ad958aae77f4656770cf44ef281f2cf2c983b1479a2bc98c3856bb83903d44 |

総合hashは共通引継ぎ手順と同じ、パスを昇順にして各`path UTF8 + NUL + commit内容LF + NUL`を連結したSHA256。今回は指定commitのgit archiveからblob内容を読み、git showを各パスに使う手順と同じ内容を集約した。tasksの現行hashはspec.jsonへ別記し、checkboxを未完了へ正規化したhashが承認値と一致することも確認した。

## 要求と保証範囲

| 要求 | 検証 |
| --- | --- |
| 1.1 | 全未使用引数を不正値にした非active、損失/評価/適用/RNGなし |
| 1.2 | 同session継続、collectionだけ更新、評価/適用/保有一覧取得なし |
| 1.3 | 4分岐16条件、実NN7条件、現在ID/可用性2条件、二重適用等の変異 |
| 2.1 | 実旧判定recordとのmetadata/数値/理由/4導出値対照、不変記録 |
| 2.2 | 実旧適応event/切替位置、適用前ID/metadata/assignment change対照 |
| 2.3 | exact依存境界、active/一覧/通知/episode更新を実装に持ち込まない |
| 3.1 | 事前設定/標本拒否でcollectionと全owner/RNG不変 |
| 3.2 | 実旧loss/全state/optimizer/grad/RNG、奇数/任意metadata/履歴なし/空保留 |
| 3.3 | 全回帰/2golden/品質/固定旧差分/hash/fresh CPUと独立レビュー |

観測→既存評価→判定情報→既存適用を接続した。出力recordを呼出側が保存し、sessionを解除・通知する責任は後続clientへ残す。観測後の評価・適用失敗を巻き戻す保証はない。終端未完了回収、通知owner、episode進行、検出からの開始、新client/新全体runは未実装。旧goldenの成功を新全体runの一致とは扱わない。

新しい旧不具合は実測していない。LEGACY-014（採用時保留標本の概念計数/統計を反映しない）等の確認待ちは維持。全pytestの独立再現は共通引継ぎ手順のユーザー決定に従い、主担当の実測とJUnit照合で判定する。

Task5は独立fresh CLI GPT-6 Luna session `01a11623-a347-79c3-a625-6324d71d8ff2` APPROVED。JUnit・両goldentest・251パス総合hash・承認md/正規化tasks hash・固定旧差分を独立確認。全pytestと品質は主担当実測の確認であり独立再実行ではない。skip理由を具体的に記す非阻害提案を採用した。

別fresh CLI GPT-6 Luna session `01a11627-135f-7220-b22a-3dee16ad8914`の対象HEAD10f6074最終判定はGO（2026-10-08確認）。独立対象＋AST1517passed/5.18s/exit0、新CPU両class成功、9/9要求・全5tasks・設計/境界整合・検証commit以後src/tests不変を確認。全pytest・品質・251パス総合hashはこの最終担当による独立再実行/再計算なし。Task5担当の251パス独立照合と主担当の全回帰証拠を維持する。阻害指摘なし、completedとする。
