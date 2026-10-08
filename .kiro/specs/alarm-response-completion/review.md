# 独立レビューと検証記録

主担当はClaude Code。レビュー担当の選択は共通引継ぎ手順（2026-10-08）に従う。

## 要求r2・設計r2・命名r1・tasks r2 — APPROVED

- 選択: 要求・設計・命名・tasksは文書と命名の通常のレビューなのでGPT-6 Lunaを選んだ。代替は行っていない。`codex exec -m gpt-6-luna -c 'model_reasoning_effort="high"' --sandbox read-only --ephemeral`で起動し、ログのmodel行は`gpt-6-luna`、`reasoning effort: high`を確認した。モデルの同定は起動指定とログの行に基づく。
- 1回目（session `01a119e0-fb80-7633-878b-4eaee76ce359`、4文書のr1を対象）: 要求APPROVED、設計REJECTED、命名APPROVED、tasks APPROVED。旧`_resolve_drift`との対応、基準選択、reset/drainの実コード、依存7 symbolの方向、外部下書きのAST束縛名と命名表の照合に問題なしと報告。指摘は次の3件と付記1件で、すべて採用した。
  1. 要求Minor: 候補検証中の応答も準備済み区間を持たないので、再適用を3.2では検出できない。→ 要求の末尾の注記へ、候補検証中の応答と保留が空だった応答の再適用は検出できないことを追記し、3.2の括弧書きを「準備済み区間を持ち、保留が1件以上あったもの」とした（要求r2）。
  2. 設計Major: recordの消費位置が負の値も受理できる。→ 受理集合へ「要素は0以上、負はValueError」を追加し、下書きの検査とrecord検査testへ反映した（設計r2）。
  3. 設計Minor: drainのtuple複製が資源不足で失敗すれば部分更新が起こりうるので、無条件の主張は導けない。→「公開APIの契約上の拒否による部分更新はない。実行環境の失敗は保証外」へ限定した（設計r2）。
  4. tasksへの付記: Task 1のrecord受理集合testへ非負位置の拒否を明示する。→ Task 1へ明記した（tasks r2）。
- 2回目（別session `01a119ec-e854-7841-a707-d27f2bdd98cf`、要求r2・設計r2・tasks r2を対象）: 3段階ともAPPROVED、指摘なし。命名r1は内容を変更しておらず、設計変更による再確認の必要はないと報告された。
- 命名r1は1回目で承認され、以後byteを変更していない（LF hashはspec.json）。
- 手順上の事実: 命名を事前登録するため、runtimeとtestをリポジトリ外で下書きし、リポジトリ外で実行して実旧oracleの実行可能性を確かめた（research.md）。下書きは元checkoutの`venv/refactoring-tests/alarm-response-completion-draft/`にある。承認の時点でworktreeに新src・新testはない。仕様化の段階でworktreeのtestは実行していない。
- 外部証拠: 元checkoutの`venv/refactoring-tests/alarm-response-completion-spec-review-r1.md`/`.log`、`alarm-response-completion-spec-review-r2.md`/`.log`。

## Task 1 — 実施記録とレビューの経緯

選択: 可変状態の更新順序と複数部品の接続を含む実装のレビューなので、Claude Haiku 5.5を選んだ。`claude -p <依頼文> --model claude-haiku-5-5 --effort high --output-format json --no-session-persistence --tools Read,Glob,Grep --permission-mode plan`で起動し、JSON応答の`is_error=false`と`modelUsage`の実モデル`claude-haiku-5-5`を確認した。effortのhighは指定値で、JSONは実効値を出力しない。読取り専用のため、レビュー担当はtest・Ruff・gitを実行していない。文書と命名の改訂はGPT-6 Luna（effort high、ログで確認）へ依頼した。代替は行っていない。

### 最初の実装（commit `a70674b`、設計r2）

- 実装前RED: 対象testはmoduleなしで収集失敗（ModuleNotFoundError）、注入契約testはguardなしで27 failed/53 passed。GREEN: 対象51＋依存境界2535＝2586 passed、Ruff・Pyright成功。実source変異25種を25/25検出。
- Haiku 1回目（session `3c10b991-7ecf-4449-a954-aa23c28d0832`）: CHANGES_REQUESTED。Blocker・Majorなし、Minor 6件。旧5経路との対応、前後IDの導出、検査が更新より前にあること、依存のexact一致、oracleが実旧メソッドを使うことは問題なしと報告。採否:
  1. test内の局所名`drain_pending_sample_indices`が命名表にない → 採用。既存メソッド名の同義再利用として命名r2へ明記（主担当のAST照合は、既存srcに同じ語があると新規名として検出しない。局所束縛の役割までは照合できていなかった）。
  2. 拒否testの不変確認に保有モデル・学習標本・計数がない → 採用。既存helper（`snapshot_alarm_change_interval_resolution_state`・`assert_alarm_change_interval_resolution_state_unchanged`）で、拒否時と成功時の両方を確かめる。候補検証sessionの中身は、完了処理がsessionのAPIをimportしないこと（exact依存）と参照一致で確かめる。
  3. 順序testの期待値が本番関数と同じ → 採用（後述の2回目の指摘で、期待値を応答が記録した切替先モデルの統計から作る形へ改めた）。
  4. 推定区間長の差し替えが完了後の比較まで残る → 採用。実旧警報処理のfinallyで外す。
  5. 候補検証中の応答で応答後に標本を追加した場合は検出できない → 採用。research.mdへ呼出側の保証として記録。
  6. spec.jsonの進捗が古い → 採用。
- 命名r2: Luna（session `01a119fd-2862-74d0-b14f-ca0508f0623b`）がAPPROVED、指摘なし。承認されたr2の全文は元checkoutの`venv/refactoring-tests/alarm-response-completion-naming-r2.md`。

### 設計の改訂（r3→r4）

- Haiku 2回目（session `f5fdc97b-880f-46df-8060-eed6577984a2`、上の反映後）: CHANGES_REQUESTED。Major 1件、Minor 4件。
  - Major: 手で組み立てた応答（結果種別が再利用なのに帰属変更の記録がない、またはその逆）は既存の応答とその区間解決のconstructorを通過し、完了処理はreset・drainの後でrecordの検査により例外になる。設計の「検査後に片方だけ更新されない」、要求3.2と一致しない。→ 事実と確認して採用。recordを更新の前に組み立てる設計へ改めた（設計r3、要求r3）。既存の応答関数が返す応答では起こらない経路だが、設計の主張と実装を一致させる。
  - Minor: testの照合（`assert_response_completion_matches_legacy`）が最終観測位置のNoneを許す → 採用（testの照合を警報位置との厳密な一致へ改めた。sourceの検査は設計どおりで変更していない。下の3回目の指摘1を参照）。順序testの期待値が本体と同じ経路 → 採用（応答の`assigned_model_id`の統計から作る）。位置引数の拒否はkw_onlyの確認であることのコメント → 採用。完了後のsession参照の確認 → 採用。
- Luna 3回目（session `01a11a02-9756-7f70-8fff-4d35949d976f`、要求r3・設計r3・命名r3）: 要求r3 APPROVED、設計r3 REJECTED、命名r3 REJECTED。
  - 設計P1: 手で組み立てた変更記録の不正なID（`True == 1`で比較を通過）を更新前に拒否できる保証がない。変更記録のexact型と両IDのbuiltin int検査を定めること。→ 採用。設計r4で、比較より前に変更記録のexact型と両IDの型を検査する。新moduleの依存は8 symbol（`TrainingModelAssignmentChange`を追加）になった。
  - 命名P2: r2承認の証拠がローカル正本にない。→ 採用。spec.jsonへr2の承認を記録し、r2の全文を保存した（r3からr3の補足節を除いて復元し、LF hashが承認値と一致することを確認）。
- Luna 4回目（session `01a11a07-e367-7632-a95e-c24e29648754`、設計r4・命名r4・tasks r3）: 3段階ともAPPROVED、指摘なし。下書きsourceで検査の順序、保存済みr2との比較でr2の名前と役割が変わっていないことを確認したと報告。
- 現在の承認: 要求r3、設計r4、命名r4、tasks r3（hashとsessionはspec.json。過去の承認revisionは`previous_approved_revisions`）。

### 改訂後の実装（設計r4）

- 改訂分のRED（srcは`a70674b`のまま）: 新しい拒否条件5件が失敗（5 failed/51 passed）。8番目のsymbolの注入条件は4 failed/83 passed。記録は元checkoutの`venv/refactoring-tests/alarm-response-completion-task1-red-r4.log`。
- GREEN: 対象56＋依存境界2542＝2598 passed。Ruff check/format成功（167 files）、Pyright 0 errors。fresh新CPU 2/4class×5経路の10条件成功。
- 検出力: 実source変異28種（recordの組立を更新の後へ戻す、変更記録の型検査2種の削除を追加）。1回目の実行で「応答と現在の帰属の対応検査の削除」が未検出だった。原因はtestが応答後の帰属を変更前のモデルへ戻しており、recordの検査（再利用なのにIDが同じ）でも同じ例外になるためで、どちらでもないIDへ変えるようtestを直した。修正後は28/28検出、各回元byteへ復元、復元後56 passed。証拠は`venv/refactoring-tests/alarm-response-completion-mutation-evidence-r4/report.json`。
- worktreeのsrcとtestは、Luna 4回目が読んだ下書きから、上のtestの修正（未検出変異への対応）だけが変わっている。新しい名前は追加していない（AST照合で新規名0件）。
- 全pytestはこのtaskでは実行していない（Task 3で実行する）。

### Haiku 3回目（session `2f214285-3203-475d-b1c3-fff2c5f88147`、HEAD `95bb98e`）

CHANGES_REQUESTED。Blocker・Majorなし。すべての検査がreset・drainより前にあること、旧5経路との対応、基準平均の規則、依存8 symbolの一致、oracleが実旧を実行すること、各拒否条件が意図した検査で拒否されること、過去のMajorとMinorの解消を確認したと報告された。指摘はMinor 3件と既知の境界1件。

1. 「sourceの最終観測位置の検査がNoneを許すのに、review.mdは採用（警報位置との一致へ）と記録している」→ 記録の書き方が不正確だったので訂正した（提案B）。2回目のMinorはtestの照合についての指摘で、反映したのもtestだけである。sourceがNoneを受理するのは承認済みの設計r4（4節手順2「最終観測位置がNoneでなければ」）どおりで、変更しない。理由: Noneは保留FIFOが一度も標本を観測していない状態で、比較する位置がない。旧は警報標本をFIFOへ追加してから警報処理へ入るので、接続後の流れではこの状態で完了処理は呼ばれない。この状態で準備済み区間を持つ応答が渡されれば、区間と保留位置の一致検査が働く。Noneの拒否へ厳格化する案（提案A）は、設計・要求の改訂と新しいtest入力が必要になる一方、部分更新や誤った記録を防ぐ効果がないため採らない。未観測のFIFOに対するtestはない（未検証として記録する）。
2. 「現行IDが1のときの`True == 1`の経路を試していない」→ 不採用（記録のみ）。上流oracleの保有モデルIDに1がなく、1にするには上流oracleの変更が要る。型検査の削除は、現在のtest（boolの変更後ID）がTypeErrorを期待して検出する（変異28/28）。`True == 1`で比較を通過する入力そのものは未検証。
3. 「boolの変更前IDは、完了処理の型検査が消えてもrecordのconstructorが同じTypeErrorを出すので区別できない」→ 不採用（記録のみ）。どちらも更新前の拒否で、挙動は同じ。完了処理側の型検査の削除は変更後IDの条件で検出される。
4. 既知の境界（候補検証中の応答の後に標本を追加して位置を進めた場合）→ research.mdの記録を維持する。

### Haiku 4回目（session `3fe66984-98df-49e7-a7b0-1163deb5eb7d`）と設計r5

CHANGES_REQUESTED、Major 1件。上の3回目の指摘1で主担当が選んだ「Noneの受理を維持する」は、要求3.2の文言（警報位置が最終観測位置と一致しないとき拒否）と両立しないと指摘された。記録と実装・設計・testの一致、指摘2・3の不採用理由は妥当と報告された。

→ 採用し、3回目の指摘1の判断を改めた。一度も標本を観測していない保留FIFO（最終観測位置がNone）も拒否する。設計r5（4節手順2）、命名r5（testで既存3 symbolをimportする補足。新しい名前なし）。Luna（session `01a11a24-022b-7261-8d69-37c92e1dc580`、effort high）が設計r5・命名r5をAPPROVED、指摘なし。既存の応答関数が返す5種の応答で、正常な呼出しが拒否されないことを実コードで確認したと報告された。承認されたr4の命名全文は元checkoutの`venv/refactoring-tests/alarm-response-completion-naming-r4.md`。

実施: 拒否条件`alarm_without_any_observed_sample`（未観測のFIFOと、区間が空の不足の応答を手で組み立てた入力）をtestへ先に追加し、srcが設計r4のままで1 failed/56 passedのREDを確認してからsrcを改めた。GREEN: 対象57＋依存境界2542＝2599 passed、Ruff・Pyright成功、fresh新CPU 10条件成功。実source変異29種（未観測FIFOの受理へ戻す変異を追加）を29/29検出、各回元byteへ復元、復元後57 passed。証拠は`venv/refactoring-tests/alarm-response-completion-mutation-evidence-r5/report.json`、REDの記録は`alarm-response-completion-task1-red-r5.log`。

### Haiku 5回目（session `21354972-36d9-4b5b-a539-950fcf1c1b0a`、HEAD `9335d5a`）— Task 1 APPROVED

Blocker・Majorなし。4回目のMajorの解消、すべての検査がreset・drainより前にあること、追加した拒否条件が最終観測位置の検査で拒否されること、依存8 symbolの一致を確認したと報告された。読取り専用のためtest・Ruff・Pyrightは実行していない（主担当の実測で判定）。任意のMinor 3件の採否:

1. 拒否testが例外の種類だけを見る（`match=`で拒否理由を固定できる）→ 不採用。各拒否条件が意図した検査で拒否されることは、検査を1つずつ削除する変異（29種）がすべて検出されることで確かめている。メッセージ文字列を契約にしない。
2. 2回目の警報の「保留0件」は現実の流れではなく境界条件 → 採用。integration-validation.mdでは境界条件として記載する。
3. kw_onlyの確認で`asdict`による複製を避ける → 不採用（動作は同じ。承認済みのtestを変更しない）。

Task 1を完了とした。

## Task 2・3 — レビューの経緯

選択: 機械的な証拠の照合と再現が中心なのでGPT-6 Luna（effort high、ログで確認）。対象testとsmokeを実行させるため`--sandbox workspace-write`で起動した。

- 1回目（session `01a11a32-89ea-74f1-be7c-90cf5cab6916`、HEAD `8c936d4`、検証対象`9335d5a`）: Task 2・3ともCHANGES_REQUESTED、Major 1件。「対象testとfresh CPU scriptのNumPy乱数の比較が状態の配列だけで、位置とGaussian cacheを比べていない。『3種の乱数状態が不変』の主張と、乱数消費の変異を省いた理由が裏付けられない」。→ 事実と確認して採用。testとscriptを状態全体の比較へ改め、乱数消費の変異4種を追加した（33/33検出）。testを変更したので、commit `def37e4`で全pytest・品質検査・source hashを取り直した（9254 passed、件数は同じ）。
- 同じ回の独立実行（レビュー担当）: 対象＋依存境界2599 passed、fresh CPU 10条件PASS、Ruff check/format成功、JUnit 9257 testcase・failure 0・error 0・skip 3と2つのgolden回帰testcaseの成功、固定旧差分が空、承認hash 4件とsource hash（267パス）の一致を再現・照合した。全pytest・Pyright・pip checkは再実行していない。これらは`9335d5a`に対する結果で、`def37e4`については2回目で確認する。
- 2回目（別session `01a11a40-c221-7511-a079-c422a3b38e67`、HEAD `bb9bc16`、検証対象`def37e4`）: Task 2・3ともAPPROVED、指摘なし。NumPy乱数の比較が状態全体になったこと、追加した乱数消費の変異4種（Gaussian cacheだけの変異を含む）が妥当で検出されていること、統合証拠の件数（実旧照合28条件、拒否条件12＋9、record拒否17条件）と未検証事項の記載を確認したと報告された。独立実行: 対象＋依存境界2599 passed、fresh CPU 10条件PASS、Ruff check/format成功。照合: JUnit 9257 testcase・failure 0・error 0・skip 3と2つのgolden回帰testcaseの成功、固定旧差分が空、`ea61b8a..def37e4`のsrc/tests差分が3ファイル、承認hash 4件の一致。全pytestの再実行・変異scriptの実行・Pyright・pip checkは行っていない（全pytestは2026-10-07のユーザー決定の基準で判定）。レビュー担当のpytest終了時に一時ディレクトリ削除のPermissionErrorが出たが、exitは0で、`git status --short`は空だった。
- 2回目の依頼の前に、統合証拠の3.1の拒否条件数の誤記（13→12）を主担当が見つけて訂正した。

Task 2・3を完了とした。外部証拠: 元checkoutの`venv/refactoring-tests/alarm-response-completion-task23-review-r1.md`/`.log`、`-r2.md`/`.log`。

## feature最終レビュー — GO

選択: 文書・証拠・実装の全体照合なのでGPT-6 Luna（effort high、ログで確認）。これまでのどのレビューとも別のsession `01a11a44-7ac9-76c2-8c96-bfb757d54ed6`、read-only、対象HEAD `9802719`。判定はGO、指摘なし。要求10項目すべて適合、設計r5の境界と依存8 symbolの一致、旧`_resolve_drift`との対応、承認の流れと記録、進捗の整合、手順上の逸脱と未検証事項の記載、固定旧差分が空であること、IMPROVE-006が移植へ混ざっていないことを確認したと報告された。

最終レビュー担当が実行したもの: Ruff check/format（成功）。実行できなかったもの: 対象test＋依存境界testとfresh CPU smoke（read-only環境で一時ディレクトリを作れず失敗。Task 2・3のレビュー担当が同じ内容を独立に再現済み）。実行していないもの: 全pytest、Pyright、pip check。JUnit・全pytestのlog・変異report・smokeのscriptとlogは読んで記載値と照合した。全pytestは基準（主担当実測＋JUnit照合）で判定した。主担当は承認hash・source hash・固定旧差分を照合してcompletedへ更新した。
