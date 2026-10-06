# Claude Codeからのリファクタリング作業

共通規約・現在地・引継ぎ手順を以下から読み込む。spec・承認・進捗をClaude専用に複製しない。

@AGENTS.md
@.kiro/steering/resume.md
@.kiro/steering/agent-handoff.md

## Codexとのツール差分

- Claude版cc-sddは`.claude/skills/`、Codex版は`.agents/skills/`。両方とも3.1.0。要求・設計・命名・task・レビュー記録と設定は共通の`.kiro/`を使用する。
- Claudeでは`/kiro-spec-init`、`/kiro-spec-requirements`、`/kiro-spec-design`、`/kiro-spec-tasks`、`/kiro-impl`等を利用する。Codex側の起動方法・ツール名をClaudeへそのまま適用しない。
- 2026-10-07のユーザー指示により、Claude担当時の承認レビューにはSonnetを使用する。独立した`Agent`へSonnetモデルを指定し、実際のモデル・対象revision/hash・結果・指摘の採否を共通specへ記録する。Codex担当時はGPT-6 Lunaを使用する。
- cc-sddのレビュー委任先もSonnetを指定する。主担当の自己レビュー、同梱の推奨モデル・自動承認・inline/offで承認を代替しない。実装担当と独立したレビュー担当を使用する。
- Codex用の`collaboration`、thread ID、`.codex/agents/`設定はClaudeに引き継げる実行手段と仮定しない。Sonnetを実際に呼べない場合はレビュー待ちを共通specへ記録し、実行可能なセッションへ引き継ぐ。
- Claude側の権限・sandbox・認証は独立している。Codex側で動いたGitや検証コマンドの権限がClaudeにもあると仮定しない。秘密値を確認・出力しない。
- 自動memoryや会話履歴は補助情報。正本と矛盾したらGitの実状態と対象specを確認し、個人memoryだけで承認状態を変更しない。

生成元の汎用CLAUDE.mdは採用せず、この入口を使用する。導入方法と再生成時の注意は共通引継ぎ手順を参照する。
