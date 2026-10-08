# 候補検証sessionの保持と進行 — 統合検証

## 対象と判定

検証対象commit: `733994b`（Windows基準環境で全pytestを実行したcommit。本specのsource・testの最終commitは`733994b`）。主担当Claude Code、2026-10-08。判定は末尾の「レビュー」に記録する。

実行環境の経過: 本specの途中から、Windowsの基準環境でtorchの読込みがスマートアプリコントロールにブロックされ、検証をWSL 2 Ubuntu（Python 3.14.4）で進めた。ブロックの解消後、Windowsの基準環境（固定venv、Python 3.13、OMP/MKL各1 thread）で対象test・変異・fresh CPU・全pytestを実行し直した。以下の判定に使う値はWindows基準環境のもの。WSLの値は参考として記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 実装 | 実装前RED、レビュー指摘ごとのRED。Claude Haiku 5.5の独立レビューで承認（経緯はreview.md） |
| 2 検出力と独立動作 | Windows基準: 変異30/30検出（29種は1回目で全件検出、独立レビューの指摘で終端回収の「検査を上流の後へ移す」1種を追加して検出）、fresh CPU 10条件成功。WSLでも先の29種とfresh CPUは同じ結果。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | Windows基準: 全pytest 9663 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（12/12）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `CandidateValidationSessionHolder`。保持・解除・拒否（空で解除、保持中に同じ/別のsession、sessionでない値）と拒否後の状態不変 |
| 1.2 | `apply_alarm_response_to_validation_session_holder`。2/4class×警報応答5種類で、反映後の保持の有無が実旧`_forward_validation`の有無と一致。保持されるのは応答のsessionそのもの |
| 1.3 | 候補検証中の応答で保持が空・別のsession、開始・維持・不足の応答でsessionを保持中、の各条件で保持が不変 |
| 2.1 | `advance_held_candidate_validation`。2/4class×旧4確定条件で、記録が実旧イベントと一致、保持が空、実旧のsessionもNone、解除の時点で記録が追加済み。未到達3標本で保持と実旧のsessionが維持され記録が増えない |
| 2.2 | `finalize_held_incomplete_candidate_validation`。2/4class×観測0/3件で記録が実旧イベントと一致、保持が空。解除後の再回収は何も変えない |
| 2.3 | 保持が空のとき、owner以外の引数を読まずに何も更新しない（記録・乱数が不変） |
| 2.4 | 依存のexact集合（1 symbolと20 symbol）に通知・episode・警報応答の実行と記録がない |
| 3.1 | 2関数×2ownerで、型の拒否が上流の呼出しより前（上流を呼んだら失敗するmock）。検査を上流の呼出しの後へ移す変異を検出 |
| 3.2 | 反映の拒否12条件（別の型、正式な5値でない結果種別、応答だけ・応答と区間解決の両方を差し替えた結果種別とsessionの不対応、holderの状態と合わない応答）で保持が不変 |
| 3.3 | 上流へ渡すだけの引数の検査は上流の既存testに任せ、重ねて網羅していない。既知の限界（標本位置と提案位置の順序は上流が検査しない）を設計4節へ記録 |
| 4.1 | 上の実旧対照（警報応答10条件、到達時8条件、未到達、終端4条件）。記録→解除の順を呼出し記録で確認 |
| 4.2 | 全pytest（旧11・最終3goldenを含む）、固定旧差分が空 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r1 | da50bb750a31a9cc685672cd249a04ff1d58454bace385e7f85c3056b91234d9 |
| 設計r3 | 7364bc940cbb98b9df0b0578ec4e894846800dee4831d73b81b53f6e83963faf |
| 命名r2 | c2937a681bae408c1eaccca87cdad76518e8ffeb20cdf1ec1d586e92fe52b684 |
| tasks r2（checkboxを未完了へ戻した内容） | 26ebd8f3b4bb511d9aa32cfa33de32d9f33fcd442141196c39805bbd287b6c54 |
| source全体、`733994b`、276パス | f2348d1bede38ac4ebd4f1aa59468250821f35f7740a9831cf7926f21e73e69e |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity`と同じ計算）。直前の完了spec（`9b72182`、271パス）からの`src`・`tests`の差分は、候補検証の適応記録と候補検証sessionの保持と進行の2 specの9ファイルだけ。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。検証時の`git status --short`は空。

全pytestは**9663 passed / 3 skipped / 2 warnings、327.73s、exit 0**（Windows基準環境、commit `733994b`）。直前の完了spec 9363＋候補検証の適応記録（新test 71＋注入契約70）＋候補検証sessionの保持と進行（対象41＋注入契約118）＝9663。JUnitは9666 testcase、failure 0、error 0、skip 3。`tests.test_regression`（160.8s）と`tests.test_proposed_regression`（41.3s）はどちらも成功。skip 3件（POSIX bashが要る`test_main_ablation_suite`1件と`test_server_sweep_wrapper`2件）と警告2件（nested tensor、TypedStorage）は以前のspecと同じ既存のもの。

Ruff check成功、format checkは175 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

参考（WSL、同じcommit）: 9663 passed / 3 failed。失敗は、Python 3.14の構文解析の違いによる既存test 1件と、golden回帰2件。golden回帰の不一致はOS・Pythonの違いによるもので（旧実装は固定旧から無変更。Windows基準では成功）、goldenの更新・許容誤差の変更・skipは行っていない。WSLでは、Windowsでskipになる3件が実行されて成功する。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `held-candidate-validation-progress-full.xml`/`.log`（Windows基準の全pytest。2 specで共用）、`held-candidate-validation-progress-full-wsl.xml`/`.log`、`held-candidate-validation-progress-mutations.py`、`held-candidate-validation-progress-mutation-evidence-windows/`と`-wsl/`、`held-candidate-validation-progress-fresh-cpu.py`と`.log`・`-wsl.log`、各レビューの出力。

## 未検証・残る制約

- 通知と保存診断、検出episode、標本ごとのclient進行（応答→完了処理→記録→保持の反映の並べ方、応答と完了の間に標本を観測しない保証、標本位置の連続性）、新全体runのgolden一致は未検証。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。Task 1のレビュー担当（Haiku、読取り専用）はtestを実行していない。
- 手順上の事実: 命名の事前登録のためsourceとtestをリポジトリ外で下書きし、承認前に作業ツリーの複製で実行した。worktreeへはtest→RED→srcの順で追加した。Task 1の実装と改訂後のREDはWSLで確認し、Windows基準ではREDを取り直していない（最終のsourceでの成功と変異の検出だけをWindowsで確認した）。

## レビュー

Task 1〜3の独立レビュー承認と、別sessionのfeature最終GO（2026-10-08）。経緯と採否は[review.md](review.md)。
