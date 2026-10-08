# 候補検証の適応記録 — 統合検証

## 対象と判定

検証対象commit: `733994b`（Windows基準環境で全pytestを実行したcommit。本specのsource・testの最終commitは`af3ffa7`）。主担当Claude Code、2026-10-08。判定は末尾の「レビュー」に記録する。

実行環境の経過: 本specの途中から、Windowsの基準環境でtorchの読込みがスマートアプリコントロールにブロックされ、検証をWSL 2 Ubuntu（Python 3.14.4）で進めた。ブロックの解消後、Windowsの基準環境（固定venv、Python 3.13、OMP/MKL各1 thread）で対象test・変異・fresh CPU・全pytestを実行し直した。以下の判定に使う値はWindows基準環境のもの。WSLの値は参考として記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 実装 | 実装前RED、レビュー指摘ごとのRED。Claude Haiku 5.5の独立レビューで承認（経緯はreview.md） |
| 2 検出力と独立動作 | Windows基準: 変異51/51検出、fresh CPU 10条件成功。WSLでも同じ結果。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | Windows基準: 全pytest 9663 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（10/10）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `record_completed_candidate_validation`。2/4class×旧4確定条件×保留0/3件の16条件で、記録の全fieldを実旧`_finalize_forward_validation`のイベントと照合 |
| 1.2 | `record_incomplete_candidate_validation_finalization`。2/4class×観測0/3件×処理済み件数2種の8条件で実旧`finalize_incomplete_forward_validation`のイベントと照合。結果値は到達時の棄却と別 |
| 1.3 | storeの切替位置と2種類の件数を全10結果で確認。実旧`local_switch_positions`の増分と照合。候補検証の結果で件数が増えないこと |
| 1.4 | 既存`test_alarm_adaptation_recording.py`の40件（field名だけ置換）が成功。適応結果の先頭5値が警報応答の5結果 |
| 2.1 | 到達時25条件・終端11条件・storeの型で、正常な記録を先に持つstoreのsnapshotが不変（更新前の拒否）。検査を更新の後へ移す変異を検出 |
| 2.2 | 記録の前後で3種の乱数の状態全体が不変。記録の後に完了情報と上流の全状態を実旧と再照合。依存のexact 9 symbolにsession・通知・学習帰属ownerがない |
| 2.3 | 検査する範囲（写す値、写す値を決める値、位置と提案位置）と、検査しない2点（推定変化点の順序、採否評価と確定結果の対応）を実装がそのとおりに扱うことを、独立レビューが実コードで確認 |
| 3.1 | 上の実旧対照（24条件）と、切替位置・件数の照合 |
| 3.2 | 全pytest（旧11・最終3goldenを含む）、固定旧差分が空 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r2 | 94b8ba5f568f598f3d39ae07197448fd501b577bd859b8d3002aec47ecd2fe80 |
| 設計r4 | eb3a09ecb9f5a7166d9d7a7e4a1cf7aa0855e2166b58e72666401e1589862e67 |
| 命名r2 | d71cf7abe49f062b1b22688d7498bf5ae5843c208f1039324ce04c1b46c3e09b |
| tasks r2（checkboxを未完了へ戻した内容） | 2276e5a386b4acbad06eacd3007e4279f5084de79953fa5824940aa08953ac11 |
| source全体、`733994b`、276パス | f2348d1bede38ac4ebd4f1aa59468250821f35f7740a9831cf7926f21e73e69e |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity`と同じ計算）。直前の完了spec（`9b72182`、271パス）からの`src`・`tests`の差分は、候補検証の適応記録と候補検証sessionの保持と進行の2 specの9ファイルだけ。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。検証時の`git status --short`は空。

全pytestは**9663 passed / 3 skipped / 2 warnings、327.73s、exit 0**（Windows基準環境、commit `733994b`）。直前の完了spec 9363＋候補検証の適応記録（新test 71＋注入契約70）＋候補検証sessionの保持と進行（対象41＋注入契約118）＝9663。JUnitは9666 testcase、failure 0、error 0、skip 3。`tests.test_regression`（160.8s）と`tests.test_proposed_regression`（41.3s）はどちらも成功。skip 3件（POSIX bashが要る`test_main_ablation_suite`1件と`test_server_sweep_wrapper`2件）と警告2件（nested tensor、TypedStorage）は以前のspecと同じ既存のもの。

Ruff check成功、format checkは175 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

参考（WSL、同じcommit）: 9663 passed / 3 failed。失敗は、Python 3.14の構文解析の違いによる既存test 1件と、golden回帰2件。golden回帰の不一致はOS・Pythonの違いによるもので（旧実装は固定旧から無変更。Windows基準では成功）、goldenの更新・許容誤差の変更・skipは行っていない。WSLでは、Windowsでskipになる3件が実行されて成功する。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `held-candidate-validation-progress-full.xml`/`.log`（Windows基準の全pytest。2 specで共用）、`held-candidate-validation-progress-full-wsl.xml`/`.log`、`candidate-validation-adaptation-recording-mutations.py`、`candidate-validation-adaptation-recording-mutation-evidence-windows/`と`-wsl/`、`candidate-validation-adaptation-recording-fresh-cpu.py`と`.log`・`-wsl.log`、各レビューの出力。

## 未検証・残る制約

- session保持は次のspec（held-candidate-validation-progress）で扱った。通知と保存診断、検出episode、標本ごとのclient進行、新全体runのgolden一致は未検証。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。Task 1のレビュー担当（Haiku、読取り専用）はtestを実行していない。
- 手順上の事実: 命名の事前登録のためsourceとtestをリポジトリ外で下書きし、承認前に作業ツリーの複製で実行した。worktreeへはtest→RED→srcの順で追加した。Task 1の実装と改訂後のREDはWSLで確認し、Windows基準ではREDを取り直していない（最終のsourceでの成功と変異の検出だけをWindowsで確認した）。

## レビュー

