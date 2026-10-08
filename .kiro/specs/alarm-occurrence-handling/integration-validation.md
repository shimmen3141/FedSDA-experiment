# 警報1回ぶんの処理の接続 — 統合検証

## 対象と判定

検証対象commit: `e80b368`（本specのsource・testの最終commit。全pytestを実行したcommit）。主担当Claude Code、2026-10-09。実行環境はWindowsの基準環境（固定venv、Python 3.13、torch 2.12.1+cpu、OMP/MKL各1 thread）。判定は末尾の「レビュー」に記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 実装 | 実装前RED（新testは収集失敗、注入契約testはguardなしで58 failed）。GREENは対象40＋依存境界3030＝3070 passed。独立レビューの経緯はreview.md |
| 2 検出力と独立動作 | 実source変異41/41検出（検査を応答の後へ移す変異4種を含む）、復元後40 passed。fresh CPU 10条件成功、警報応答5種類を観測。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | 全pytest 10000 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（11/11）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `handle_alarm_occurrence`。5つの段を記録用wrapperで包み、呼出しが応答→完了→記録→保持→通知の順で各1回であること。段の省略・二重実行・順の入替えの変異を検出 |
| 1.2 | 応答へ渡る進行中のsessionが保持の値（空ならNone）、提案位置が警報位置。候補検証中の条件で、1回目の警報が保持させたsessionを2回目が保持から読む。保持を読まない変異・提案位置を変える変異を検出 |
| 1.3 | 完了と保持へ応答の結果そのもの、記録へ完了の結果そのもの、通知へ区間解決の帰属変更が渡ること（同一object）。他の引数は受け取った値そのもの |
| 1.4 | 戻り値のfieldが完了と記録の結果そのもの。recordはfrozen・kw_only・2 field |
| 1.5 | 依存のexact集合（27 symbol）に候補検証の進行・確定・終端回収とepisodeの制御がない。episode IDは実旧イベントと一致し、応答・完了へ渡さない変異を検出 |
| 2.1 | 5つのowner×（別の型、派生型）の10条件で、どの段も呼ばれず、監視・保留位置・記録・保持・診断が不変 |
| 2.2 | 12条件（警報位置がbool・float・負・最終観測位置の前後、推定変化点が負・bool、episode IDが負・float、検出器名がNone・str派生型・空白）で同じ |
| 2.3 | 引数の集合が「応答の引数−2＋5」で全てkeyword-only・既定値なし。応答が自分で検査する引数は上流の既存testに任せ、重ねて網羅していない |
| 2.4 | 5つの段のそれぞれを失敗させ、後の段が呼ばれないこと、記録の件数が失敗した段の位置に対応すること |
| 3.1 | 2/4class×警報応答5種類: 応答・完了を既存helperで実旧`_resolve_drift`の後の状態（イベント、FIFO、検出器、学習状態）と照合、記録を実旧のイベント・切替位置・再利用件数と照合、保持を実旧のsession属性と照合、診断証拠を実旧の再始動hookの後のAdaHedgeと照合、torch乱数が実旧と一致 |
| 3.2 | 全pytest（旧11・最終3goldenを含む）、固定旧差分が空 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r1 | 39323d2e7092b6c8727979c6528d1b597a8db3215927964b9797a6ec303df68f |
| 設計r2（r1から4節の1行の記述を訂正。処理は同じ） | 6f60eeee5b236e7af9224a52fffc98bac00245359a27155d1af326bd0ac324f6 |
| 命名r1 | 5f9a5d3f11f9bac6f1b75a0e3135a038a1b5005af98f3683b97806e30a40449f |
| tasks r1（checkboxを未完了へ戻した内容） | 315a84bf40233a7e09332b63e987cdb6749e1f2db4727961d3882f7adf78377e |
| source全体、`e80b368`、283パス | bfaa9bf5a5bd7e8bf4e71240f191674eacc986ca05fdb69aaf3efab25a6aee79 |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity --rev e80b368`の計算）。直前の完了spec（`24a5953`、281パス）からの`src`・`tests`の差分は、本specの3ファイル（新source、新test、依存境界test）だけ。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。全pytestの実行時、`src`・`tests`に未コミット差分はない（未コミットだったのは本specの証拠文書だけ）。

全pytestは**10000 passed / 3 skipped / 2 warnings、287.71s、exit 0**。直前の完了spec 9811＋対象40＋注入契約149＝10000。JUnitは10003 testcase、failure 0、error 0、skip 3。`tests.test_regression`と`tests.test_proposed_regression`はどちらも成功。skip 3件（POSIX bashが要る`test_main_ablation_suite`1件と`test_server_sweep_wrapper`2件）と警告2件は以前のspecと同じ既存のもの。

Ruff check成功、format checkは182 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `alarm-occurrence-handling-full.xml`/`.log`、`alarm-occurrence-handling-task1-red-target.log`・`-task1-red-guard.log`・`-task1-green.log`、`alarm-occurrence-handling-mutations.py`と`alarm-occurrence-handling-mutation-evidence/`、`alarm-occurrence-handling-fresh-cpu.py`と`.log`、各レビューの出力。

## 未検証・残る制約

- 候補検証の確定に伴う診断通知、警報のない標本での帰属確定と学習、標本1件の処理全体（応答の前後に標本を観測しない保証、標本位置の連続性を含む）、検出位置の記録、予測側の警報hook、サーバ同期、新全体runのgolden一致は未検証。
- 検出episodeの制御は移植していない（最終構成で無効。主担当の判断、research.md）。episode IDは受け取って記録へ渡すだけで、fresh CPUでは1、実旧対照では上流oracleの値を使っている。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。
- 手順上の事実: 命名の事前登録のためsourceとtestをリポジトリ外で下書きし、承認前に作業ツリーの複製で実行した。worktreeへはtest→RED→注入契約test→RED→guard→srcの順で追加した。変異とfresh CPU（Task 2の内容）は、Task 1の独立レビューより前に実行した（共通引継ぎ手順の「レビューの前に、検査を更新の後へ移す変異を入れる」に従ったため）。

## レビュー

Task 1はClaude Haiku 5.5、設計r2とTask 2・3はGPT-6 Lunaが承認（2026-10-09）。経緯と採否は[review.md](review.md)。
