# 統合検証

検証日2026-10-08、主担当Claude Code。検証対象commit **b890f0e**（作業ツリーに変更がない状態で実行）。旧固定基準748c3aa。

## 実測

| 検証 | 結果 |
| --- | --- |
| Task1 結果種別と不変record | 実装前RED（ImportError）。独立レビュー承認 |
| Task2 3分岐の組立 | 実装前RED（ImportError）。対象184 passed。独立レビュー承認 |
| Task3 呼出し順・学習継続・実観測（test-only） | 対象241 passed。独立Luna承認 |
| Task4 対象＋AST | 2257 passed（対象241＋AST2016、うち本specの注入290）。fresh新CPU 2/4class×3分岐成功。sourceのimport 25件とguardの一覧が一致、未使用importなし |
| 全pytest | **8455 passed / 3 skipped / 2 warnings、332.48s、exit0** |
| JUnit | 8458 testcases、0 failures、0 errors、3 skipped。前spec7924＋対象241＋注入290＝8455 |
| Ruff check / format | 全src/tests/refactoring成功、159 files already formatted |
| Pyright | 基準venvを明示し全src 0 errors / 0 warnings / 0 informations |
| pip check / git diff --check | 成功 |
| 固定旧差分 | 748c3aaから旧実装・tools・2golden・旧回帰testsの差分は空（commit済み/作業ツリーとも） |
| 承認文書 | 要求r3・設計r3・命名r5・tasks r2のLF hash一致（下表）。tasksはcheckboxを未完了へ戻した内容が承認値と一致 |

環境はWindows CPU、OMP/MKL各1thread、TMP/TEMPは元checkoutのvenv/refactoring-tests、MPLCONFIGDIRはvenv/matplotlib-cache、FDE_MNIST_DATA_DIRはdata/mnist。コマンドは[共通引継ぎ手順](../../steering/agent-handoff.md)の「2026-10-07に確立した運用」の検証コマンドと同じ（JUnitは`alarm-change-interval-resolution-full.xml`）。

全pytestの判定は共通引継ぎ手順のユーザー決定（2026-10-07）に従い、主担当の実測とJUnit照合を使う。レビュー担当による全suiteの再実行は行っていない（各taskレビューでもpytestは未実行と報告されている）。

JUnit内のtest_regression/test_proposed_regressionにfailure/error/skipはなく、旧11・最終3goldenは成功。goldenは更新していない。3 skipはPOSIX bashのないWindowsでの既存条件。2 warningsは前specと同じ既存のもの。新しいskip・warningはない。

Git管理外の証拠（このPCのみ）: 元checkoutのvenv/refactoring-tests/の`alarm-change-interval-resolution-full.xml`、`alarm_change_interval_resolution_cpu_smoke.py`、`alarm_change_interval_resolution_mutation_evidence.py`。

## fresh新CPU

旧実装とtest moduleをimportしないprocessで、2/4class×3分岐（保有3モデル、負の一時IDを含む、現行は最後のモデル）を実行した。

- 再利用: 現行でない負IDのモデルが選ばれ、区間の全標本がそのモデルへ標本順に入り、現行IDが切り替わり、変更記録が返る。概念IDなしの標本は割当概念計数へ入らない。torch乱数は不変。
- 維持: 現行モデルへ吸収し、変更記録なし、現行ID不変。torch乱数は不変。
- 候補検証開始: 標本storeは空のまま、現行ID不変、開始sessionの保留標本は渡した標本列そのもの、区間は標本順の連結、参照は保有順。続けて実観測1件。
- 3分岐とも、保有モデルの全parameterとgrad、Python/NumPy乱数は不変。

## REDと検出力

- Task1: 実装ファイル作成前に対象testを実行し、collection時のImportError。依存境界は注入契約test追加時に21 failed/1793 passed、guard登録後に依存境界＋対象1815 passed。
- Task2: 公開関数の実装前に対象testを実行し、collection時のImportError。依存境界は公開関数ぶんの注入契約へ更新して59 failed/1957 passed、guard更新後に依存境界＋対象2197 passed。全分岐のtestが初回からGREENだったため、下の変異で検出力を確認した。
- Task3: test-only。追加時点で実装があるため初回からGREEN。
- 実source変異25種を対象test全体で25/25検出。各変異後にfinallyで元byteへ復元しSHA256一致を確認、復元後241 passed。内訳: 結果種別の検査削除、frozen解除、選択があっても候補検証を開始、現行を優先して維持（IMPROVE-001相当）、吸収先を現行IDへ、帰属切替えの欠落、吸収の欠落、概念IDを渡さない、標本を逆順に吸収、区間を逆順に連結、結果種別を逆に、初期値選択へ適合列を渡す、初期値を常に現行から、保留標本を空で開始、候補検証開始でも現行へ吸収、余分なtorch乱数、位置情報の渡し違い、区間評価を2回呼ぶ、切替えを吸収より先に行う、保有モデルのsnapshotを逆順に取る、exact Tensor検査のisinstance化、概念ID列の長さ検査削除、帰属が保有外の拒否削除、計数ownerの型検査削除、標本の1行検査削除。
- 初回の変異実行では「標本の1行検査削除」が未検出だった（拒否testが特徴とラベルを同時に2行にしていた）。特徴だけ2行の条件を追加して検出するようにした。
- Task3の3testだけを実行した場合は25種のうち17種を検出。残りは事前検査とrecordの変異で、Task1/2のtestが検出する。
- 「切替えを吸収より先に行う」は最終状態が同じになる変異で、呼出し順のtestだけが検出する。設計は吸収→切替の順を既存の確定処理に合わせて定めており、順序はtestで固定している。

## 同一性

| 対象 | LF SHA256 |
| --- | --- |
| 要求r3 | 715c3d8722797f77caa2ba609289f0631ce91bf9f1a646d8f9b570a94c7f8bf8 |
| 設計r3 | 88244ab34884ec6185af8e82e949c7d49dc60a3dc930a451209e25c4aed799c1 |
| 命名r5 | 1b71690b3cc673a77d6ee0978c86acd2ffd556af400ac82b0c15c7f2bc5f38e1 |
| tasks r2承認時 | e280b69e4764454826e4e7255535774de4171ea7eff14d1b4dcb74518f613f79 |
| runtimeの組立 | d904b786db0d1d60365d707ee62e16d57e2ebd3a2e1d0f31b0f59770ba7da1a7 |
| source全体（commit b890f0e、tracked Python＋2golden、259パス） | d67bebc31e40502a349662204beedfcbc1367e3f717bcb2acc3b689937d4f1ab |

## 要求との対応

| 要求 | 証拠 |
| --- | --- |
| 1.1 | 呼出し順のtest: 区間評価が最初に1回、標本順に連結した特徴・ラベルで呼ばれ、分岐ごとに該当する操作だけが設計の順に1回ずつ呼ばれる |
| 1.2 | 実旧_resolve_driftの再利用分岐と、吸収先・現行ID・変更記録・切替位置を照合（2class/4class×Adam/SGD）。同率で現行が適合していても保有順で先の負IDモデルを選ぶ場合を含む |
| 1.3 | 実旧の維持分岐と照合（現行だけが基準を持つ場合、単一モデルを含む） |
| 1.4 | 実旧の候補検証開始と、候補parameter・optimizer・参照・履歴平均・学習量・保留標本・位置情報を照合（2/4class×3初期化方式×4履歴条件）。標本・計数・統計・現行IDは不変 |
| 2.1 | recordのfrozen/kw_only、正式値以外の拒否（確定処理の値や旧action名を含む）、3結果種別のfieldの組 |
| 2.2 | 標本（同じTensor objectを標本順に末尾へ）、割当概念計数（概念IDなしは加えない）、損失統計を実旧と照合。全parameter/grad、optimizer state、吸収先以外の標本と統計、3乱数が不変 |
| 2.3 | 候補検証開始で保有順・owner同一性、全parameter/grad、optimizer state、学習mode、統計、標本、計数、現行IDが不変。torch乱数は実旧の処理後と一致、Python/NumPy乱数は不変 |
| 3.1 | 共通入力の不正44条件×3分岐条件で例外型が合い、全状態と3乱数が不変。区間の値と許容損失増加量以外は区間評価が0回 |
| 3.2 | 開始だけの入力の不正12条件: 適合なしでは全状態と3乱数を変えずに拒否、再利用・維持では成功し実旧（正常入力）と結果・最終状態が一致 |
| 3.3 | 上記の実旧対照、解決前後を挟んだ共同学習3回の損失・全状態・乱数の一致（2/4class×3optimizer×共有部更新有無×3条件）、開始後の実観測の一致、変異の検出、fresh新CPU、全回帰 |

## 保証範囲と残る制約

- 実旧との対照は、警報処理のうち区間評価の後の分岐の適用まで。旧のイベント記録は記録用に差し替え（actionだけ照合）、検出器resetと推定区間長は差し替えた。区間の切出し、最小区間件数、前区間の処理、候補検証中の警報、FIFOの消費、通知（予測重みの再始動）、active sessionの保持、計算量診断、新client・新全体runへの接続は未実装（後続spec）。新全体runのgolden一致は未検証。
- 吸収と帰属切替えの順は旧（切替→吸収）と逆（吸収→切替）。本specが更新する状態の最終値は同じで、実旧と照合済み。通知の時点だけが変わり、通知は呼出側の責務。
- 未検証: CPU以外のdeviceとnested Tensorの標本の拒否（この基準環境で用意していない。sparse layoutとfloat64/int64は検証済み）。区間が1標本の場合（本specの組立は標本数で分岐しないが、候補の学習を含む上流のsession開始の入力条件に依存する）。差が許容増加量ちょうどの場合は上流の区間評価のtestが実旧と照合済みで、本specでは再検証していない。
- 初期値選択がNoneを返した場合のLookupErrorは到達しない分岐で、testしていない（保有モデルが1件以上あれば既存の初期値選択はNoneを返さない）。
- 開始だけに使う入力は、再利用・維持の場合は検査しない（要求3.2）。不正な設定のまま再利用が成功することがある。
- レビュー担当の代替: 要求・設計・命名・tasks・Task1/2のレビューは、Codexの利用上限中のため、取り決めに従い独立AgentのSonnetが行った。Task3以降はGPT-6 Luna。
- 手順上の逸脱: testの局所名7つを命名表へ登録し忘れ、事後登録した（review.md）。
