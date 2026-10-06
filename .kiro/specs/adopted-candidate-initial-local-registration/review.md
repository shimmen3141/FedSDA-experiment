# 独立レビューと採否

主担当はClaude Code（claude-opus-5-5）。独立レビューは`codex exec -m gpt-6-luna --sandbox read-only`で別プロセスのGPT-6 Lunaを起動した（実行ログのmodel行はgpt-6-luna）。Sonnet代替は使用していない。session IDはspec.jsonへ記録。

要求revision1: 実Luna CHANGES_REQUESTED。(1)一時IDの重複確認が学習状態一覧だけで、統計store・送信保留の同ID上書きを防げない→採用、2.2で3箇所を事前確認。(2)損失評価失敗時に共有値・接続・optimizer resetが残る→採用。(3)登録途中の失敗時の部分状態が未定義→採用。(2)(3)は全検証と数値生成を状態変更より前に終える契約（2.6/2.7）へ変更し、1.2/1.3を接続後の値との一致という観測可能な契約にした。保有0件LookupError・保有済みID拒否・後続へ残す境界は妥当との確認。

要求revision2: 実Luna APPROVED、hash一致を確認、指摘なし。mode/乱数依存層がなく、反映前評価への並べ替えが観測可能な損失・統計・snapshotを変えないことを確認。

設計revision1・命名revision1: 実Luna 設計APPROVED/命名APPROVED、指摘なし。手順4〜6を反映前に置く値の一致、snapshotのkey順/独立性、同一共有部の自己複写、8〜10が検証済み値だけを受けることを既存APIと照合。対照には旧SharedBackboneClassConditionalESRFedSDAClient / SharedBackboneRestartingSoftRoutingFedSDAClient（共有表現mixinの反映＋BaseClient._register_trained_new_model）が適切との助言→Task1の対照clientに採用。

task graph revision1: 実Luna CHANGES_REQUESTED。Task2の「12NN条件」が不明確→採用、組合せを明記。

task graph revision2: 実Luna APPROVED、指摘なし。主担当は4承認のhashを再計算して照合し実装を開始。

設計revision2・命名revision2・tasks revision3: 実Luna APPROVED、指摘なし。型注釈用の既存2型（HeldModelTrainingState/SharedFeatureExtractor）の許可追加、Task1のtest用追加名、tasksの「13symbol」修正を確認。productionの名前追加なし。

Task1: 実Luna APPROVED、指摘なし。実装が設計手順1〜10と検証順に一致し、testが実旧SharedBackboneClassConditionalESRFedSDAClientの登録＋待機設定を実行して対照していること、拒否時不変の観測範囲、要求2.1〜2.6の入力区分を確認。Lunaはread-only sandboxで一時ディレクトリを作成できず、pytestの独立実行は未実施（No usable temporary directory found）。主担当の実測は対象108 passed、Ruff/format成功、Pyright 0/0。依存境界の全走査testは新module用guard未追加で1件失敗しており、Task2のAST REDとして扱う。

命名revision3: 実Luna APPROVED、指摘なし。Task2のtest用追加名（保有モデル状態の対照helper、両実装の共同更新closure等）を確認。productionの名前追加なし。

Task2: 実Luna APPROVED、指摘なし。13symbol＋__future__.annotationsだけを許可するexact guardと注入契約、12条件testが実旧の_train_heads_together/ResidualAdapterMLP.update/_register_trained_new_model/BaseClient.confirm_model_registrationを実行して全段階を対照していること、fresh CPU smokeの旧importなしを確認。Lunaはworkspace-write sandboxでPowerShellから独立実行し、対象＋AST 928 passed、smoke PASS、Ruff check/format成功（指定したbash起動はGit BashのWin32 error 5で不可）。主担当はレビュー前後のgit status一致（Lunaによるファイル変更なし）を確認。

Task3（1回目）: 実Luna CHANGES_REQUESTED。Lunaの独立全pytestがsandboxのPermissionErrorで1 failed/5649 passed/3 skipped/28 errorsとなり記録と一致しない→採用。主担当実測とLuna再実行の区別、未確認である旨をintegration-validationへ追記。対象＋AST928/smoke/Ruff/JUnit集計/旧差分空/承認hashはLunaが独立に一致確認。

Task3（2回目）: 実Luna CHANGES_REQUESTED。worktree内の一時ディレクトリでもsandbox制限で同じ結果となり、独立再現は未達。失敗・errorは全て書込み拒否の環境起因で、コード起因の失敗は確認されないとのLuna判断→再実行結果を記録へ追記（採用）。sandboxを外した実行はユーザー承認がないため行わない。

Task3（3回目）: 実Luna APPROVED、指摘なし。前specと同じ基準（主担当実測＋JUnit照合、対象testの独立実行、旧差分空）で承認。JUnit内のgolden回帰2 testcaseに失敗/error/skipがないことをLunaが照合。残る制約: codex sandboxの書込み拒否により、独立レビュー側では全pytestの成功を再現できておらず、主担当の実測とJUnitを根拠にしている。
