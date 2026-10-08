# Claude Codeからのリファクタリング作業

共通規約・現在地・引継ぎ手順を以下から読み込む。spec・承認・進捗をClaude専用に複製しない。

@AGENTS.md
@.kiro/steering/resume.md
@.kiro/steering/agent-handoff.md

## Codexとのツール差分
- effortはHaikuを`high`、Lunaを`medium`に明示指定する。代替で使うときも各モデルのこの指定とし、具体的な起動方法は共通引継ぎ手順を参照する。

- Claude版cc-sddは`.claude/skills/`、Codex版は`.agents/skills/`。両方とも3.1.0。要求・設計・命名・task・レビュー記録と設定は共通の`.kiro/`を使用する。
- Claudeでは`/kiro-spec-init`、`/kiro-spec-requirements`、`/kiro-spec-design`、`/kiro-spec-tasks`、`/kiro-impl`等を利用する。Codex側の起動方法・ツール名をClaudeへそのまま適用しない。
- 2026-10-08のユーザー指示により、日常的・分量の多いレビューはGPT-6 Luna、やや複雑・難易度の高い実装のレビューはClaude Haiku 5.5を優先し、利用不能時は他方へ代替する。具体的な選択・起動・記録は共通引継ぎ手順を正本とする。Claudeの`Agent`を使う場合も実モデルがHaiku 5.5であることを確認し、別世代に解決されるaliasは承認担当として扱わない。
- cc-sddのレビュー委任先もこの優先順に従う。主担当の自己レビュー、同梱の推奨モデル・自動承認・inline/offで承認を代替しない。実装担当と独立したレビュー担当を使用する。
- Codex用の`collaboration`、thread ID、`.codex/agents/`設定はClaudeに引き継げる実行手段と仮定しない。LunaとHaiku 5.5の両方を実際に呼べない場合はレビュー待ちを共通specへ記録し、実行可能なセッションへ引き継ぐ。Sonnetへの自動代替は行わない。
- Claude側の権限・sandbox・認証は独立している。Codex側で動いたGitや検証コマンドの権限がClaudeにもあると仮定しない。秘密値を確認・出力しない。
- 自動memoryや会話履歴は補助情報。正本と矛盾したらGitの実状態と対象specを確認し、個人memoryだけで承認状態を変更しない。

生成元の汎用CLAUDE.mdは採用せず、この入口を使用する。導入方法と再生成時の注意は共通引継ぎ手順を参照する。
