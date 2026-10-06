# 独立レビューと採否

主担当はClaude Code（claude-opus-5-5）。独立レビューは`codex exec -m gpt-6-luna`で別プロセスのGPT-6 Lunaを起動した（実行ログのmodel行はgpt-6-luna）。session IDはspec.jsonへ記録。

初回（要求r1・設計r1・命名r1・tasks r1を段階別に一括依頼）: 実Luna 要求APPROVED、設計/命名/tasks CHANGES_REQUESTED。
- (1)research.mdの「サーバ側の同値一時ID衝突は未確認」→採用。Lunaが旧サーバの回収/確認経路を確認し、clientごとに処理され衝突しないとの指摘。調査範囲を明記して修正（要求本文は変更なし）。
- (2)実旧clientの生成が採番単体のoracleには重い。実__init__/_alloc_temp_idを呼ぶ最小fixtureを先に確認し、式をtestへ複製しない→採用。`BaseClient(client_id, {0: object()}, verbose=False)`で実行できることを確認し設計・taskへ明記。
- (3)AST guardの許可ノード・相対/別名/他future importの扱いが不明確→採用。許可はImportFromの`__future__.annotations`だけと明記し、禁止例を列挙。stdlib単独起動は補助検証と位置付け。
- (4)spec.jsonがない→採用。承認状態とhashを記録して作成。
- (5)temporary_model_id/registered_global_model_id/current_training_model_idとの違いが未記載→採用。比較表を追加。
- (6)Task2の全回帰は採番部品の完了条件として過大、統合ゲートへ分ける提案→部分採用。worktree規約と全前specの運用（各featureの最終taskで固定環境の全回帰を記録）を維持し、含める理由をresearch・tasksへ明記した。taskを分離する変更は行わない。登録接続testの必要性と依存方向の説明は採用。

再レビュー（設計r2・命名r2・tasks r2）: 実Luna 3段階ともAPPROVED、指摘なし。4ファイルのhash一致、指摘6の部分採用（全回帰taskをfeatureに残す）は規約と前例に沿い妥当との確認。主担当はhashを再計算して照合し実装を開始。
