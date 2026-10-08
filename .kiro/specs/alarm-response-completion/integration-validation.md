# 警報応答の完了処理 — 統合検証

## 対象と判定

検証対象のsource/test commit: `def37e4`（以降のcommitは証拠・進捗文書だけ）。主担当Claude Code、2026-10-08、Windows CPUの固定venv。判定は末尾の「レビュー」に記録する。

実装は`runtime/alarm_response_completion.py`の不変record `AlarmResponseCompletion`と`complete_alarm_buffer_response`。完了した警報応答を受け、検査→現行モデルの統計から監視の基準平均を選択→recordの組立→損失監視のreset→保留位置のdrain（変化区間の不足ではdrainしない）を行い、呼出側の記録・通知に必要な情報を返す。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 完了recordと完了処理 | 実装前RED（moduleなしの収集失敗、guardなしの注入契約27 failed）。設計の改訂ごとに拒否条件のtestを先に追加してRED（5 failed、1 failed）を記録。対象57＋依存境界2542＝2599 passed。Haiku 5.5の独立レビュー5回目で承認（経緯はreview.md） |
| 2 検出力と独立動作 | 実source変異33種を33/33検出、各回byte復元、復元後57 passed。fresh新CPU 2/4class×5経路の10条件成功。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | （Task 2・3のレビュー1回目の指摘でtestのNumPy乱数の比較を状態全体へ改めたので、その commit で取り直した値）全pytest 9254 passed / 3 skipped / 2 warnings、155.63s、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（10/10）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | 応答後の現行モデルの全体統計から既存の基準選択で基準平均を選び、監視をresetする。実旧`_reset_drift_detectors`の後の全体・class別検出器の全状態と基準を24条件＋2回目の警報4条件で照合。統計なし・0件・下限未満・中間・上限の5条件。再利用では切替先モデルの吸収後の統計が渡ることを呼出し記録で確認 |
| 1.2 | 消費指示のある応答で保留位置を全件消費し、最終観測位置を維持する。実旧FIFOのclearと照合（再利用・維持・候補検証開始・候補検証中）。2回目の警報の「保留0件」は、実際の流れ（警報標本を先に保留する）では起こらない境界条件として確認している |
| 1.3 | 不足の応答で保留位置を保持し、監視だけをresetする。実旧（LEGACY-002の残留を含む）と照合 |
| 1.4 | reset→drainの呼出し順を記録で確認。成功時と拒否時に、保有モデルの値とgrad・optimizer・統計・学習標本・計数・帰属・3乱数が不変。完了処理の後の標本・統計・帰属・保有モデルを既存の照合helperで実旧と比較。候補検証sessionは参照一致と、sessionのAPIをimportしないこと（exact依存）で確認 |
| 2.1 | recordの全fieldを実旧`AdaptationEvent`（position・old/new model・estimated_change_point・episode_id）、FIFO、reset後の基準と照合。frozen/kw_onlyと受理集合の拒否17条件。recordは更新の前に組み立て、整合性の検査を更新前に終える |
| 2.2 | 切替位置と、検出episodeへの操作記録の要否を、実旧`local_switch_positions`と戻り値（1のときだけ操作あり）と照合 |
| 2.3 | 一覧・計数・episodeを更新せず、通知もsession保持もしない。依存のexact 8 symbolに該当ownerがない。再利用計数は応答結果からの対応表で実旧と照合 |
| 3.1 | 応答・4 ownerの型、警報位置・推定変化点・episode IDの型と値の13条件で、更新前に拒否し全状態が不変 |
| 3.2 | 警報位置と最終観測位置の不一致（未観測のFIFOを含む）、準備済み区間と保留位置の不一致（応答後の追加、消費後の再適用）、応答の変更後IDと現在の帰属の不一致、結果種別と変更記録の不対応2方向、exact型でない変更記録、boolのID 2種を、更新前に拒否 |
| 3.3 | 実旧`_resolve_drift`（イベント記録・検出器reset・FIFO clearを差し替えない）との照合28条件、完了後の同じ損失列に対する監視のe値・警報・推定区間長・全状態の照合、変異33種、exact依存、新CPU 10条件、全9254 passed、旧11・最終3golden |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r3 | d68a4564a4d1889fffbcfd1b8b0c01feeb301cb0757f93daa417704bd29dc4b0 |
| 設計r5 | f9877d26b6d1c6a71a71f3b41ea53bac1e9ff18cc1334bc752ad389a62e77c99 |
| 命名r5 | 3c71de90f706a46bcf801ec7ff99cbeb9c49e315d4b35199d092e488356861db |
| tasks r3（checkboxを未完了へ戻した内容） | a3242e8cb825072ae5c3d0fe4759216c339dbbab979aa7f74f73228ffb82f190 |
| source全体、`def37e4`、267パス | b11a144ab4312c63f4dfd84492c5dc81c9ab11f7264caca9de5e32453fd9db2b |

source hashは共通引継ぎ手順の`source_sha256`（tracked Python＋2golden）。同じ手順で前specの`ea61b8a`は265パス・`49e99d80…`となり、前specの記録値を再現した。`ea61b8a`から`def37e4`までのsrc/testsの差分は、新src 1・新test 1・依存境界testの3ファイルだけ（`9335d5a`から`def37e4`はtest 1ファイルのNumPy乱数の比較だけ）（既存srcの変更なし）。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。検証時の`git status --short`は空。

全pytestは**9254 passed / 3 skipped / 2 warnings、155.63s、exit 0**。前spec 9110＋対象57＋注入契約87＝9254。JUnitは9257 testcase、failure 0、error 0、skip 3。`tests.test_regression`（78.4s）と`tests.test_proposed_regression`（20.4s）はどちらも成功。skip 3件（POSIX bashがない環境の`test_main_ablation_suite`1件と`test_server_sweep_wrapper`2件）と警告2件（nested tensor、TypedStorage）は前specと同じ既存のもの。

Ruff check成功、format checkは167 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

## コマンドと証拠の場所

環境: OMP/MKL各1 thread、TMP/TEMP=元checkoutの`venv/refactoring-tests`、MPLCONFIGDIR=`venv/matplotlib-cache`、FDE_MNIST_DATA_DIR=`data/mnist`。手順は共通引継ぎ手順の「検証コマンド」。

```bash
$PY -m pytest tests -q -p no:cacheprovider --junitxml="$TMP/alarm-response-completion-full.xml"
$PY -m ruff check src tests/refactoring; $PY -m ruff format --check src tests/refactoring
$PY -m pyright --pythonpath ../../venv/Scripts/python.exe; $PY -m pip check
$PY ../../venv/refactoring-tests/alarm-response-completion-fresh-cpu.py
```

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `alarm-response-completion-full.xml`/`.log`、`-task1-red.log`/`-red-r4.log`/`-red-r5.log`、`-fresh-cpu.py`/`.log`、`-mutations.py`、`-mutation-evidence-r6/`、各レビューの出力（`-spec-review-r1`〜`r5`、`-naming-r2-review`、`-task1-review-r1`〜`r4`など）、承認済みの命名r2・r4の全文、下書き。

## 未検証・残る制約

- 適応イベント一覧・切替位置一覧・再利用計数・検出episodeのowner、学習帰属変更の通知、候補検証sessionの保持と解除は未実装（後続spec）。新client・新全体runは未接続で、新全体runのgolden一致は未検証。
- 応答と完了処理の対応のうち、構造から検出できない誤用は呼出側の保証（research.md）: 候補検証中・不足・保留が空だった応答への再適用、候補検証中の応答の後に標本を追加して警報位置を進めた呼出し。
- 手で組み立てた変更記録の`True == 1`で比較を通過する入力そのもの（現行IDが1の場合）はtestしていない。型検査の削除は別の条件で検出される。
- メモリ不足などの実行環境の失敗時の部分更新（reset済み・未消費）は保証外。CPU以外のdeviceは対象外（完了処理はTensorを扱わない）。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。Task 1のレビュー担当（Haiku、読取り専用）はtestを実行していない。
- 手順上の事実: 命名の事前登録のためsourceとtestをリポジトリ外で下書きし、承認前にリポジトリ外で実行した。worktreeへはtest→RED→srcの順で追加した。Task 1は独立レビューで4回差し戻され、要求r3・設計r5・命名r5まで改訂した。

## レビュー

