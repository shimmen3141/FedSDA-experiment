# 独立レビューと検証記録

主担当はClaude Code。レビュー担当の選択は共通引継ぎ手順に従う（2026-10-09のユーザー指示により、Haikuのeffortの既定は`medium`）。実行環境はWindowsの基準環境（Python 3.13、torch 2.12.1+cpu）。

## 要求r1・設計r1・命名r1・tasks r1 — APPROVED

- 選択: 文書と命名の通常のレビューなのでGPT-6 Luna（`model_reasoning_effort="medium"`、read-only、ephemeral。ログのmodel行と`reasoning effort: medium`を確認）。代替は行っていない。依頼文には、4文書のLF hash、設計レビューの確認項目（検査がすべて上流の進行より前にあるか、通知の中の検査が通常の入力で拒否になる具体例を作れないか、既存の関数へ足す判断・通知を解除の後に置く判断・保持がないときも診断のownerを検査する判断、要求との1文ずつの照合）を入れた。
- 結果（session `01a11ca2-c9b8-7003-89df-5744bc3266fe`）: 4段階ともAPPROVED。Minor以上の指摘なし（各段階の「任意」は確認内容の説明で、提案は「不要」）。候補検証の確定で再始動するのは採用と他モデルの再利用だけであること（最終構成の前提と、shadowの分岐を移植しない境界のもとで）、通知の中の型検査が通常の入力で拒否になる具体例は確認できないこと、既存の関数へ足す判断・通知の位置・保持がないときの検査の拡張が要求と一貫すること、ownerの型の拒否のtestの改め方と、文言だけの修正に専用のtestを足さない判断が許容できることを確認したと報告された。レビュー担当はtestを実行していない。下書きの実行結果の独立再現も行っていない。
- 手順上の事実: 命名の事前登録のため、sourceとtestの変更をリポジトリ外のpatchとして下書きし、`git archive HEAD`で作った作業ツリーの複製へ適用して、Windowsの基準環境のPythonで実行した（worktreeは変更していない。対象47件が成功、Ruff・Pyright成功）。`spec_checks.py names`は、複製で報告なし。承認の時点でworktreeのsource・testは未変更。変異scriptも、レビューの完了を待つ間に同じ複製で試行した（40/40検出。worktreeでの実行はTask 2で記録する）。
- 外部証拠: 元checkoutの`venv/refactoring-tests/held-candidate-validation-diagnostic-notification-spec-review.md`/`.log`。

## Task 1 — 実施記録

- 実装前RED: 変更後のtestは変更前のsourceで12 failed / 35 passed。注入契約test（13条件）はguardなしで4 failed（`venv/refactoring-tests/held-candidate-validation-diagnostic-notification-task1-red-target.log`・`-red-guard.log`）。
- GREEN（commit `42fc7b8`）: 対象47＋依存境界3043＋alarm-occurrence-handlingのtest 40＝3130 passed。Ruff check/format・Pyright・`spec_checks.py names`が成功。sourceとtestは、仕様の承認時にLunaが読んだ下書きとbyteが同じ。
- レビューの前に、実source変異40種（本specの10種＋既存30種の再実行）を40/40検出し、fresh新CPU 8条件が成功した。
- commit `42fc7b8`の全pytestは10019 passed / 3 skipped（下の反映より前のcommitの値。判定には使わない）。

### Haiku 1回目（session `f5e0b67f-f6bb-45ef-96a3-c397dabcfffa`、source/test commit `42fc7b8`）— TASK 1: CHANGES_REQUESTED

選択: 既存の接続の更新順序と検査の位置を変える実装のレビューなのでClaude Haiku 5.5（`--effort medium`指定。2026-10-09のユーザー指示による既定。JSONの`is_error=false`と`modelUsage`の`claude-haiku-5-5`で確認。実効effortは出力されないので指定値）。代替は行っていない。読取り専用で、testとgitを実行していない（変更前との差分は取れていないと報告された）。検査の位置、再始動の回数（採用・再利用で1回、他は0回）、完了情報の読取りで例外になる経路がないこと、通常の入力で通知の検査が拒否にならないこと（記録の処理が同じ型の検査を先に行う）、oracle、依存22 symbolは問題なしと報告された。指摘と採否:

1. Major: 要求2.1の「保持がないときも同じ」を確かめるtestがない。ownerの型の拒否のtestは常にsessionを保持した状態で、保持なしのtestは有効な診断のownerを渡している。「診断の型検査を、保持があるときだけ行う」変異はどの変異でも検出されない（保持なしでは上流が呼ばれないので、上流のmockでも検出できない、と指摘された）。→ 事実と確認して採用。ownerの型の拒否のtestへ「保持あり・保持なし」の条件を足し（10条件→20条件。parametrize引数`session_is_held`）、変異`n_diagnostic_type_check_only_with_held_session`を足して検出を確かめた（41/41）。sourceは変更していない。補足: 既存の関数は保持がなくても上流をNoneで呼ぶので、保持なしの条件でも「上流を呼んだら失敗するmock」は働く（追加した変異は、そのmockが呼ばれて検出される）。設計8節・命名（`session_is_held`）・tasksの条件数をrevision2へ改め、再レビューへ出す。
2. Minor（記録）: 再開案内の「次に直すこと」がNEW-001を未修正とする記述のままである。→ 採用。Task 3で、NEW-001の記録と再開案内を修正済みへ改める（tasks.mdのTask 3に記載済みの作業）。
