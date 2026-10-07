# 統合検証

検証日2026-10-08、主担当Claude Code。検証対象commit **5785533**（`5785533`時点で作業ツリーに変更がない状態で実行）。旧固定基準748c3aa。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1 純粋判定と不変record | 対象73 passed。独立Luna承認 |
| Task2 runtimeの区間評価の組立 | 対象136 passed（実旧対照16＋閾値境界6＋同率6＋拒否35＋Task1の73）。独立Luna承認（1回目CHANGES_REQUESTED→反映後承認） |
| Task3 初期値選択/session開始へのtest-only接続 | 18 passed（2class×3初期化方式×3履歴case）。対象全154 passed。独立Luna承認 |
| Task4 対象＋AST | 1880 passed（対象154＋AST1726、うち新規注入144）。fresh新CPU 2/4class成功 |
| 全pytest | **7924 passed / 3 skipped / 2 warnings、276.07s、exit0** |
| JUnit | 7927 testcases、0 failures、0 errors、3 skipped。前spec7626＋対象154＋注入144＝7924 |
| Ruff check / format | 全src/tests/refactoring成功、157 files already formatted |
| Pyright | 基準venvを明示し全src 0 errors / 0 warnings / 0 informations |
| pip check / git diff --check | 成功 |
| 固定旧差分 | 748c3aaから旧実装・tools・2golden・旧回帰testsの差分は空（commit済み/作業ツリーとも） |
| 承認文書 | 要求r2・設計r2・命名r4・tasks r1のLF hash一致（下表）。tasksはcheckboxを未完了へ戻した内容が承認値と一致 |
| Fresh新CPU | 2/4classで、保有3モデル（負の一時IDを含む）の区間評価→再利用選択、適合なし→評価情報→既存の初期値選択→既存のsession開始→観測1件。評価前後で全parameter/grad/3乱数不変、旧/test importなし |

環境はWindows CPU、OMP/MKL各1thread、TMP/TEMPは元checkoutのvenv/refactoring-tests、MPLCONFIGDIRはvenv/matplotlib-cache、FDE_MNIST_DATA_DIRはdata/mnist。コマンドは[共通引継ぎ手順](../../steering/agent-handoff.md)の「2026-10-07に確立した運用」の検証コマンドと同じ（JUnitは`alarm-interval-reuse-full.xml`）。

全pytestの判定は共通引継ぎ手順のユーザー決定（2026-10-07）に従い、主担当の実測とJUnit照合を使う。レビュー担当による全suiteの再実行は行っていない（各taskレビューでもpytestは未実行と報告されている）。

JUnit内のtest_regression/test_proposed_regressionにfailure/error/skipはなく、旧11・最終3goldenは成功。goldenは更新していない。3 skipはPOSIX bashのないWindowsでの既存条件（ablation一覧1、server sweep wrapper2）。2 warningsは前specと同じ既存のもの（拒否test準備のnested Tensor prototypeとTypedStorage deprecated）。新しいskip・warningはない。

Git管理外の証拠（このPCのみ）: 元checkoutのvenv/refactoring-tests/の`alarm-interval-reuse-full.xml`、`alarm_interval_reuse_cpu_smoke.py`、`alarm_interval_reuse_mutation_evidence.py`。別PCでは本specの公開APIと条件に従い再作成する。

## REDと検出力

- Task1: 実装ファイル作成前に対象testを実行し、collection時のImportError（methodsのmoduleなし）。実装後73 passed。
- Task2: runtime作成前に対象testを実行し、collection時のImportError（runtimeのmoduleなし）。実装後、test側の不具合2件（NaN入力のTensorを値の不変比較へ含めていた。NaNは自身と等しくない）を直して136 passed。productionは無変更。
- 依存境界: 2moduleの注入契約test追加時に49 failed（注入48＋実sourceの許可外依存1）/1677 passed。exact guardと両resolver登録後、依存境界1726 passed。全commitで全suiteをgreenに保つため、tasks.mdではTask4にあるこのRED→guard登録をTask2のcommit（d58c427）へ前倒しした。
- Task3: test-only。追加時点で実装があるため初回からGREEN。検出力は実source変異で確認した。
- 実source変異17種（methods 8: 閾値比較を等値不適合へ、差分の符号、最大選択、同率後着、同率ID最小、零基準の拒否削除、key集合検査の緩和、frozen解除。runtime 9: Pythonの和での平均、基準なしモデルの混入、基準なしモデルでの打切り、保有順の逆、余分なtorch乱数、学習modeの変更、閾値の事前検査削除、保有なし拒否の削除、型検査のisinstance化）を、対象test全体で17/17検出。各変異後にfinallyで元byteへ復元しSHA256一致を確認、復元後154 passed。
- Task3の18条件だけを実行した場合は17種のうち7種（差分の符号、Pythonの和、基準なしモデルの混入/打切り、保有順の逆、余分な乱数、学習mode）を検出。残り10種は「どのモデルも適合しない」接続条件では通らない経路で、Task1/2のtestが検出する。

## 同一性

| 対象 | LF SHA256 |
| --- | --- |
| 要求r2 | f62dc55825c9aa0aaf91ba2584105d555552b30109a1758914b82e668a5d8020 |
| 設計r2 | a906373ac6b8884f00470de4306805831425b9e5454421042b047a61f234e1d2 |
| 命名r4 | 64761b4b82e3e547f3789003aa0cb76541f2c1ad38a4653cf14f5619efd96633 |
| tasks r1承認時 | f6fd70135204fbe4b52215e3c660162979d7aeae5ae1aee77189e469b0549ad1 |
| methodsの純粋判定 | 8ce9b98e6c01ef0dedc2d0ec0f13d39e4311b59494a747bc9084e677074f8017 |
| runtimeの組立 | 6d9b0b0229a475702db3de641bb51901d1d453b38d54a54e02d922a920d5d7fe |
| source全体（commit 5785533、tracked Python＋2golden、257パス） | 4fca6c7b9cc634f1d6dc60952412c87be7ecaaa2e32022a646b0ecb01268f75f |

## 要求との対応

| 要求 | 証拠 |
| --- | --- |
| 1.1 | 実旧_resolve_driftが初期値選択へ渡した評価済み候補列と、新の評価列がID・順序・float値まで一致（2/4class×8履歴case）。履歴の未登録/0件/1件/零平均のモデルも保有順に1回ずつforwardされ、評価列には入らない |
| 1.2 | 純粋判定の閾値直前/等値/直後と履歴より低い場合。実NNの差に対する直前/等値/直後を実旧の適合列と照合 |
| 2.1 | 平均最小・同率先着。同じ値の実モデルで、現行（最後）でもID最小でもなく保有順の先を選ぶことを実旧の選択と照合。負の一時ID、適合なしはNone |
| 2.2 | recordはfrozen/kw_only。成功時に保有順・owner同一性・全parameter/grad・optimizer state・学習mode・履歴統計・入力Tensor・3乱数が不変 |
| 3.1 | 純粋判定58拒否、runtime35拒否（型はTypeError、値はValueError）。状態所有者と閾値の不正は分類器の評価前に拒否。後続モデルの形状拒否は先行2モデルのforward後に起き、全状態と乱数は不変 |
| 3.2 | 上記の実旧対照、3初期化方式×3履歴caseの初期値・session開始・観測の照合、fresh新CPU、全回帰とgolden未更新 |

## 保証範囲と残る制約

- 実旧との対照は、警報処理のうち区間評価・候補列・適合列・選択と、適合なしの場合の初期値選択→session開始まで。評価より後の旧の副作用（吸収・イベント記録・検出器reset・帰属切替）は差し替えており、今回の対照に含めていない。
- 区間の切出し、最小区間件数、前区間の吸収、再利用時の帰属変更と吸収、通知、新client・新全体runへの接続は未実装（後続spec）。新全体runのgolden一致は未検証。
- 実旧の評価済み候補列は、適合がある場合に旧が外へ渡さないため、どのモデルも適合しない閾値（−無限大）で別に実行して観測した。旧は候補を閾値比較より前に追加するので、列は閾値に依らない。
- 手順上の逸脱: Task2のtestで、命名r3に未登録の局所名を使ったままcommitし、命名r4へ事後登録した（独立レビューでMinor指摘、内容は承認）。実装開始前の命名承認という手順は、この局所名については満たしていない。
