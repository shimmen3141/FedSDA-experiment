# 独立レビューと採否

主担当はClaude Code（claude-opus-5-5）。独立レビューは`codex exec -m gpt-6-luna --sandbox read-only`で別プロセスのGPT-6 Lunaを起動した（実行ログのmodel行はgpt-6-luna）。Sonnet代替は使用していない。session IDはspec.jsonへ記録。

要求revision1: 実Luna CHANGES_REQUESTED。(1)一時IDの重複確認が学習状態一覧だけで、統計store・送信保留の同ID上書きを防げない→採用、2.2で3箇所を事前確認。(2)損失評価失敗時に共有値・接続・optimizer resetが残る→採用。(3)登録途中の失敗時の部分状態が未定義→採用。(2)(3)は全検証と数値生成を状態変更より前に終える契約（2.6/2.7）へ変更し、1.2/1.3を接続後の値との一致という観測可能な契約にした。保有0件LookupError・保有済みID拒否・後続へ残す境界は妥当との確認。

要求revision2: 実Luna APPROVED、hash一致を確認、指摘なし。mode/乱数依存層がなく、反映前評価への並べ替えが観測可能な損失・統計・snapshotを変えないことを確認。

設計revision1・命名revision1: 実Luna 設計APPROVED/命名APPROVED、指摘なし。手順4〜6を反映前に置く値の一致、snapshotのkey順/独立性、同一共有部の自己複写、8〜10が検証済み値だけを受けることを既存APIと照合。対照には旧SharedBackboneClassConditionalESRFedSDAClient / SharedBackboneRestartingSoftRoutingFedSDAClient（共有表現mixinの反映＋BaseClient._register_trained_new_model）が適切との助言→Task1の対照clientに採用。

task graph revision1: 実Luna CHANGES_REQUESTED。Task2の「12NN条件」が不明確→採用、組合せを明記。

task graph revision2: 実Luna APPROVED、指摘なし。主担当は4承認のhashを再計算して照合し実装を開始。

設計revision2・命名revision2・tasks revision3: 実Luna APPROVED、指摘なし。型注釈用の既存2型（HeldModelTrainingState/SharedFeatureExtractor）の許可追加、Task1のtest用追加名、tasksの「13symbol」修正を確認。productionの名前追加なし。

Task1: 実Luna APPROVED、指摘なし。実装が設計手順1〜10と検証順に一致し、testが実旧SharedBackboneClassConditionalESRFedSDAClientの登録＋待機設定を実行して対照していること、拒否時不変の観測範囲、要求2.1〜2.6の入力区分を確認。Lunaはread-only sandboxで一時ディレクトリを作成できず、pytestの独立実行は未実施（No usable temporary directory found）。主担当の実測は対象108 passed、Ruff/format成功、Pyright 0/0。依存境界の全走査testは新module用guard未追加で1件失敗しており、Task2のAST REDとして扱う。
