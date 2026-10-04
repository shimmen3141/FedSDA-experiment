# レビューと採否

## 要求保存前gate
主担当は12条件の数値ID/EARS、構造/出力/空/共有/乱数/grad/異常系/検証と範囲を確認してPASS。
正常数値・時系列は実旧oracleを維持、学習と生成後共有付替えは含めない。以降はユーザー委任のLunaレビューを承認とする。

## 要求Luna指摘の採否
NEEDS_FIXES: 2.2のdtype/deviceと出力shapeが判定できないとの指摘を採用。
CPU float32 stridedと単入力/batchの出力shapeを明記し、旧汎用device/dtype移動の移植とは区別した。境界や他条件への指摘はなかった。修正版を再レビューする。

## 要求修正版Lunaレビュー
PASS: CPU float32 scope・旧.to移植との区別・dtype/device/layout/shapeが定まり、構造・拒否・旧照合・学習除外は検証可能。
主担当はPASSを採用し、修正版requirementsのhashを承認した。

## 設計・命名Lunaレビュー
PASS: 12条件と三つのnn.Moduleの対応、CPU32・共有構造検査・旧初期化順・勾配・標準state_dictは具体的で、学習境界も明確。命名に曖昧さなし。
主担当は指摘なしのPASSを採用し、designとnaming revision1のhashを承認する。

## Taskgraph主担当保存前gate
12条件の割当、1→2→3のDepends、共有source/testによる順次実行、既存venv/設定前提、実測可能な完了条件と独立reviewを確認してPASS。Luna保存前reviewを待つ。

## Taskgraph Luna保存前レビュー
PASS: 三段階の順序と12条件・三module・共有/環境・確率/snapshot接続・golden保持が整合する。
数値IDを個別列挙する指摘はdraftで既に満たしており、そのまま保存する。主担当はPASSを採用してtasksを承認する。独立既存thread再利用。

## 命名revision2 Lunaレビュー
Task2のtest-only5prefixと局所名はPASS。production名・役割不変。
boolのnonzero_expansionをuse_nonzero_expansion_weightsにすると切替意図が明確との任意指摘を採用した。主担当は反映済みrevision2を承認する。

## Task1 初回reviewと修正
実装者READY_FOR_REVIEW、missing-module RED1error/2.05s/exit1→GREEN11 passed/1.62s/exit0。主担当fresh11 passed/1.88s/exit0。
Luna REJECTED: 共有Linearのregistered weightを削除して同形CPU32 Tensorを割り当てると、named_parametersから欠落していても構造検査を通ることを再現した。
主担当は有用な指摘を採用。weight/biasがnamed_parameters(recurse=False)へ同一参照で登録されていることを検査し、malformed拒否とRNG/共有不変の先行RED→GREENを追加する。旧実装の発見ではなく今回の新検査の補完である。

## Task1 再reviewと完了
実修正RED2 failed/11 passed/1.61s/exit1→GREEN13 passed/1.59s/exit0。新名/importは追加していない。
Luna APPROVED、対象13 passed/exit0、placeholder/secret/境界問題なし、未登録weight/bias拒否と共有/CPU RNG不変を確認。指摘なし。
主担当fresh13 passed/1.61s/exit0、source実読/旧基準無差分/diffcheckによりTask1の構造・旧照合をVERIFIED。高度検証とAST/fullは未完了のままTask1を完了する。

## Task2 reviewと完了
実装者READY_FOR_REVIEW、test-only追加40件、53 passed/1.84s/exit0。fake REDは非該当。
zero/nonzero展開の入力/全Parameter.grad旧照合、拒否RNG/input/shared不変、empty batch/reuse/storage、ambient float64/meta/grad/PythonNP保持、旧bounded meanlossとsnapshot選択→標準load接続を確認した。
Luna APPROVED、53 passed/exit0、placeholder/secretなし、test-only境界/勾配/拒否/相互接続問題なし。指摘なし。
主担当fresh53 passed/2.31s/exit0、実diff/source無変更/diffcheckでVERIFIED。AST/fullはTask3のままTask2を完了する。旧新差異・追加の旧不具合は観測していない。

## Task3実測証拠
AST担当READY_FOR_REVIEW（AST部分のみ）。禁止28/許可21先行RED22 failed/297 passed/2.68s/exit1→exactguard後319 passed/2.08s/exit0。主担当fresh319 passed/2.27s/exit0。
主担当fresh CPU smokeはRESIDUAL_ADAPTER_MODEL_ARCHITECTURE_SMOKE_PASS/exit0、旧package importなし。
主担当全testsは3284 passed/3 skipped/1 warning/115.87s/exit0。旧11/最終3goldenを含む。skipは既存Windows非対応wrapper、warningは既存qint8 fixture deepcopyのTypedStorage非推奨。
全suite開始後production/testは変更していない。旧production/golden/旧比較testは748c3aaと差分なし。承認hash/UTF-8/diffcheckも成功。
12条件/実配置/依存と未移植範囲をintegration-validationへ記録した。新たな旧不具合は観測していない。続いてLuna task3review/最終feature gateを行う。

## Task3 reviewと完了
Luna APPROVED、対象/AST319 passed、全3284 passed/3 skipped/1 warning/exit0、placeholder/secretなし、exact依存とfresh smoke/RED証拠を確認。指摘なし。
主担当はPASSを採用し、自ら取得した全回帰/smoke/319件のfresh出力、その後source/test不変更、旧基準差分なし、12条件とUTF-8/diffcheckによりVERIFIED。Task3を完了する。
