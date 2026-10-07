# 統合検証

検証日2026-10-07、主担当Codex。検証対象commit **66ac055**。旧固定基準748c3aa、基準Pythonは元checkoutのvenv/Scripts/python.exe。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1設定 | 26passed、別Luna承認・独立再現 |
| Task2事前検査/固定反復 | 対象124passed、別Luna承認・独立再現 |
| Task3公開方式入口/早期停止 | 対象346passed、別Luna承認・独立再現 |
| Task4生成→学習→継続更新 | 24条件、主担当24passed/346deselected、別Luna追加承認。初回352passedとfresh smokeは独立再現 |
| 対象＋AST | **1494passed / 1warning、10.19s、exit0** |
| 全pytest | **6868passed / 3skipped / 2warnings、144.91s、exit0** |
| JUnit | 6871testcases、0failures、0errors、3skipped。前回6412＋対象370＋注入86＝6868 |
| Ruff check / format | 全src/tests/refactoring成功、146files formatted |
| Pyright | 基準venv明示の昇格実行で全src0errors/0warnings |
| pip check / diff check | 成功 |
| 固定旧実装・2golden・旧回帰test・tools | 748c3aaから差分なし |
| 承認文書 | 要求r2・設計r1・命名r7・tasks r1のLF hash一致。tasksはcheckboxだけ進捗更新 |
| Fresh新CPU | 二値/4classの新候補生成→学習、有限parameter、参照不変、旧importなしで成功。Task4 Luna独立再現 |

JUnitは`../../venv/refactoring-tests/candidate-epoch-training-full.xml`。旧回帰・最終回帰の両testcaseにもfailure/error/skipがなく、固定旧11ケース・最終3ケースのgolden照合が成功した。3skippedは既存Windows wrapper。warningsは拒否test準備のnested Tensor prototypeと既存TypedStorage deprecated。

環境: Windows CPU、Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1、OMP/MKL各1thread、torch CPU threads1。Ruff0.16.10、Pyright1.1.414。TMP/TEMP/MPLCONFIGDIR/FDE_MNIST_DATA_DIRは共通引継ぎ手順の共有venv/dataを明示。

## REDと検出力

- 設定module追加前: ModuleNotFoundError/exit1。
- 学習module追加前: ModuleNotFoundError/exit1。
- 公開入口追加前: ImportError/exit1。
- stub/復元省略/optimizer reset/extra randperm/閾値<=の5変異を全てexit1で検出し、元byteへ復元。Git管理外のscript・JSONは../../venv/refactoring-tests/candidate_epoch_training_red_evidence.pyとcandidate-epoch-training-red-evidence.json。
- Task5の86注入条件: exact guard追加前**47failed/39passed/1038deselected**（candidate-epoch-training-ast-red.txt）、guardとsymbol resolver/broad import拒否への登録後GREEN。REDには許可symbolの解決不足も含まれ、失敗の正負内訳は主張しない。実source2moduleも同じ検査器を通る。

## 同一性と境界

検証commitのtracked Python＋2goldenは246パス、source SHA256:
`34dad8e20b4d3b83e18cc57f0e1c0b0bac86244c7807f79bdf826e4a7524e258`。
学習module単体LF SHA256は`1d67010544b048c123b7151b854974a6fb0ac18c3bfd6c0e2af9b99c73760394`。

生成済み候補の学習を移植し、固定反復/早期停止/省略、0epochの方式差、検証件数丸め、小区間fallback、最良parameterだけの復元、実行量、全parameter/grad/optimizer/RNGを実旧へ照合した。初期値選択→生成→学習→次batch更新の接続までがtest-onlyであり、参照固定とのsession開始組立、新client/全体runは未完成。旧goldenの成功を新全体run一致とは扱わない。

今回、新しい旧実装の不具合は観測していない。最良parameter復元時にoptimizer/gradを戻さない点は保存対象の既存挙動であり、新規不具合とは断定しない。LEGACY-014等の確認待ちと既存記録は変更しない。

全pytestの独立再現は共通引継ぎ手順のユーザー決定どおり、主担当の実測とJUnit照合で判定する。Task5はAPPROVED、別freshセッション `/root/luna_epoch_feature_final`（実GPT-6 Luna）のfeature最終判定はGO。対象HEAD b99a625、対象＋AST1494passed/exit0を独立再現、JUnitと検証commit以後のコード不変を照合。全pytest・品質検査の独立再実行はしていない。指摘なし。review.md/spec.jsonへ記録しcompletedとする。
