# 統合検証

検証日2026-10-08、主担当Codex。検証対象commit **237030b1e52aa7188ffbb989ec3bda64421ffa5e**。旧固定基準748c3aa。実装・testsのcommit後、作業ツリーに変更がない状態で基準Windows CPUの全回帰を実行した。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1 不足判定record | 7passed、独立fresh Luna承認 |
| Task2 終端回収 | 対象全60passed、12正常・現行ID変更2・37拒否・非active・不変返却、独立fresh Luna承認 |
| Task3 実学習接続 | 対象全67passed。実NN2/4class×3optimizerと空保留/metadataなし、独立fresh Luna承認 |
| Task4 対象＋AST | 1649passed/6.30s/exit0（対象67＋AST1582）、新規注入126。独立Lunaも1649passed/13.01s/exit0 |
| 全pytest | **7626passed / 3skipped / 2warnings、166.70s、exit0** |
| JUnit | 7629testcases、0failures、0errors、3skipped。前spec7433＋対象67＋注入126＝7626 |
| Ruff check / format | 全src/tests/refactoring成功、154files already formatted |
| Pyright | 基準venvを明示し全src 0errors/0warnings/0informations |
| pip check / diff check | 成功 |
| 固定旧差分 | 748c3aaから旧実装・tools・2golden・旧回帰testsのcommit済み/作業ツリー差分とも空 |
| 承認文書 | 要求r4・設計r1・命名r2・tasks r1のLF hash一致。tasksはcheckboxを未完了に正規化して承認値と照合 |
| Fresh新CPU | 2/4classの実開始→未到達観測→終端回収→共同更新と非active、固定参照/RNG不変、旧/test importなし。主担当とTask4 Lunaが独立再現 |

環境はWindows CPU、Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1、Ruff0.16.10、Pyright1.1.414。OMP/MKL各1thread、TMP/TEMPは元checkoutのvenv/refactoring-tests、MPLCONFIGDIRはvenv/matplotlib-cache、FDE_MNIST_DATA_DIRはdata/mnistを指定した。基準資料はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu/。

worktreeからのコマンド:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path; $env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --junitxml="$env:TMP/incomplete-finalization-full.xml"
../../venv/Scripts/python.exe -m ruff check src tests/refactoring
../../venv/Scripts/python.exe -m ruff format --check src tests/refactoring
../../venv/Scripts/python.exe -m pyright --pythonpath ../../venv/Scripts/python.exe
../../venv/Scripts/python.exe -m pip check
git diff --check
```

全pytestは一時資材と子プロセスのWindows ACL制限を避けて昇格実行。全pytestの判定は共通引継ぎ手順のユーザー決定に従い主担当実測とJUnit照合を使う。独立した全suite再実行は必須にしない。

Git管理外の証拠はroot venv/refactoring-testsのincomplete-finalization-full.log（PowerShell UTF16）、同-full.xml、同-junit-summary.json、同-identity.json。JUnit内のtest_regression/test_proposed_regression双方にfailure/error/skipなし、旧11・最終3golden成功。3skipはPOSIX bashのないWindowsでの既存条件（ablation一覧1、server sweep wrapper2）。2warningsは既存の拒否test準備によるnested Tensor prototypeとTypedStorage deprecated。新たなskipやgolden更新はない。

fresh smokeは同ディレクトリincomplete_validation_finalization_cpu_smoke.py / incomplete-task4-smoke.log。ログ/scriptはこのPCのみの資材で、別PCでは環境・本specの公開APIと条件に従い再作成する。

## REDと検出力

- Task1 source前: missing module、1collectionerror/3.46s/exit2。GREEN7passed。
- Task2 runtime前: missing runtime、1collectionerror/2.27s/exit2。GREEN60passed。
- Task3 test-only接続。実sourceへの6変異（stub・不足検査無視・開始時ID・位置誤り・余分なRNG・二重吸収）を59/1/4/17/57/14failedで検出、各exit1。finallyと終端で元byteへ復元、runtime SHA256は下記と一致、復元後67passed/4.16s/exit0。script/JSONはincomplete_validation_finalization_mutation_evidence.py / incomplete-finalization-mutation-evidence.json。LunaはJSON/hashを照合し、変異自体は再実行していない。
- Task4初期106注入: guard前42failed/1587passed/6.20s/exit1。Import leaf拒否20条件追加後、126注入とguard/両resolverで1649passed。RED/GREENログはroot venv直下incomplete-task4-ast-{red,green}.log。

## 同一性

| 対象 | LF SHA256 |
| --- | --- |
| 要求r4 | 1a1ff43c1fa09b1b2826df0cd02a5d7c8b9cb1d658277f2d8e68ea91da8a1597 |
| 設計r1 | 95631f6949475c32faa3279a995fc1ea031a1cc732612dea8da8d43111106f08 |
| 命名r2 | 7346a523b478f105c5bb714a53b6b823a70500f0c135aa854598afb072a1fa0d |
| tasks r1承認時 | f0f19b7b30cabe9183030f05309a635a925d55171c9afef616770cc54da0b01d |
| 終端runtime（byte/LF同値） | 86cf36bcb39cc8b7f235d520627fa3f8df0b024adad15142ad6a54a27dce4478 |
| 不足record（byte/LF同値） | 81e133f38cf09f3e06fba0d475c52e7164b9fa9c7b0a2d5a6d004dd636899241 |
| 検証commitのtracked Python＋2golden、254パス | 1e9c00d53198427554df9aa808c2274ba78993ec4f38731b7a40bdfa5a635239 |

総合hashは検証commitのgit archiveの全tracked Python＋2goldenを対象とする。パスを昇順にして各path UTF8 + NUL + blob内容LF + NULを連結したSHA256。共通引継ぎ手順のgit showによる各blob集約と同じ内容。src/testsだけの部分集合ではない。作業ツリーの同254パスが検証commitと同内容であることも照合した。

## 要求と保証範囲

| 要求 | 検証 |
| --- | --- |
| 1.1 | 非activeに不正な未使用引数を与え、他入力に触れずNone/全状態とRNG不変 |
| 1.2 | 保留標本を終端時点の現行IDへ回収、帰属/候補/固定参照/学習状態/collection不変 |
| 1.3 | max(提案位置, 処理件数−1)、処理件数0と提案位置前/後、位置変異検出 |
| 2.1 | 実旧record/不足理由/NaN対応、位置/検出器/学習区間件数/実観測件数、不変型 |
| 2.2 | 実旧eventの同じ現行ID/metadata/位置、解除/通知/episode/帰属変更を持ち込まない |
| 3.1 | 37拒否、後半不正でも全owner/collection/model/optimizer/RNGを変えない |
| 3.2 | 12正常/現行ID変更2/実NN7、実旧損失・全状態・3RNG/optimizer/gradと2回共同更新 |
| 3.3 | exact依存注入、fresh CPU、変異/復元、全回帰/JUnit/旧と最終golden/品質/固定旧差分/hash、独立レビュー |

開始→公開観測→件数不足確定→既存吸収→後続共同学習を同じownerで接続した。返却後のsession解除と一覧記録は呼出側の責任で、同じsessionの再投入による二重吸収を防ぐ責任も後続clientにある。通知/episode操作、検出からsession開始への接続、新client/新全体runは今回の保証外。旧golden成功と、新全体runのgolden一致を混同しない。

新しい旧不具合は実測していない。LEGACY-014等の既存確認待ちは維持する。

Task5は独立fresh CLI GPT-6 Luna session `01a1173e-0556-73a0-81c4-cfb067bf5194` APPROVED。JUnit・両golden・承認hash・254パス総合hash・固定旧差分を独立照合した。Ruff/pip/diffは独立成功、全pytestは主担当実測の確認で独立再実行なし。LunaのPyright実行では依存import解決の問題が出て独立再現できなかったため、これを独立成功とは記録しない。主担当は同じ基準コマンドを再実行し0errors/0warnings/0informations/exit0を確認、incomplete-finalization-quality.log（UTF16）へ保存した。レビュー環境による解決差という原因説明は推測で、原因調査や設定変更は行っていない。別fresh feature最終GOはまだ未完了。
