# Claude・Codex共通の引継ぎ手順

## 作業場所と正本

両方のエージェントで、このリファクタリングworktreeを作業フォルダとして開く。
元checkoutのmainとは区別する。Claude用に別のspecや進捗台帳を作らない。

| 情報 | 正本・入口 |
| --- | --- |
| 共通規約・方針 | [AGENTS.md](../../AGENTS.md)、[refactoring-policy.md](../../docs/research/refactoring-policy.md) |
| 現在地・次の候補 | [resume.md](resume.md)、[roadmap.md](roadmap.md) |
| 採用構成・依存方向・配置 | [product.md](product.md)、[tech.md](tech.md)、[structure.md](structure.md) |
| 個別仕様・承認・task | 対象specのREADMEにある正本一覧、spec.json、tasks.md、requirements.md、design.md、naming.md |
| レビューと検証証拠 | 対象specのreview.md、integration-validation.md等、Gitの対象commitと実diff |
| 旧実装の問題 | [implementation-findings](../../docs/research/implementation-findings/README.md) |
| 開発手順の問題 | worktreeのdevelopment-findings/ |

resumeは要約であり、承認状態を単独で決める根拠にしない。Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。
CodexはAGENTS.mdからroadmap・resumeへ進む。共通方針を入口ファイルへ複製しない。

## 開始時

1. 作業ディレクトリ、`git branch --show-current`、`git rev-parse HEAD`、`git status --short`を確認する。
   期待するブランチは`refactor/architecture`。元checkoutや別ブランチなら実装前に作業場所を確認する。
2. resumeとroadmapを読み、対象specのREADME→spec.json/tasks.md→要求/設計/命名→レビュー/検証を読む。
   未完了task・レビュー待ちがあれば、新しいspecより先にその状態を復元する。
3. 既存の未コミット差分の作成者・対象を確認する。別作業の差分を破棄・上書き・一括stageしない。
4. 対象revision・内容hashと承認記録が合うか確認し、差分と検証対象commitを照合する。
   過去の別revisionの承認や、別commitでのテスト成功を流用しない。
5. 必要なcc-sdd skillを読み、共通規約に従って次の一単位を進める。

同じworktreeを編集する主担当は一度に一人とする。交代前に旧セッションの編集・子エージェントの処理が終了していることを確認する。
セッション固有のthread IDは証拠の来歴であり、次のツールで再利用できる接続先ではない。

## 承認と独立レビュー

要求→設計・命名→tasks→実装task→統合検証・別feature最終GOの流れを維持する。
[命名レビュー規約](../settings/rules/naming-review.md)と各specの承認契約に従う。
2026-10-07のユーザー訂正指示により、担当ツールを問わずGPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューと、
主担当による有用な指摘の反映を承認として扱う。要求・設計・命名・tasks・実装task・別feature最終GOの各ゲートへ適用する。
利用可能な接続・モデルを確認し、モデルを明示してレビュー担当を起動する。
実際の担当モデル・代替時のLuna利用不能理由・対象revision/hash・結果・指摘の採否を記録する。
Claude担当であることだけをSonnet使用の理由にせず、Lunaへの接続がない、呼出上限、実行エラー等の事実を記録する。
自己レビューやcc-sddの自動承認・inline/offで置き換えない。`kiro-spec-quick --auto`や`-y`も承認を代替しない。
旧Luna承認記録は変更しない。

指定された独立レビュー担当を利用できないセッションでは、対象specの既存review.mdへ次を記録する。

- レビュー待ちの段階・task番号、対象ファイル、revision・内容hash、対象commit/未コミット差分。
- 確認してほしい契約・命名・判断、実施した検証と未実施項目。
- 前提の承認状態、未解決の指摘、レビュー後に進める具体的な次作業。

未承認段階のapprovedをtrueにせず、未レビューの実装taskを完了扱いしない。
独立した承認済み作業は進められるが、未承認の名前・仕様を先取りして実装しない。
独立レビューを実行可能なセッションへ記録を引き継ぎ、実レビューの結果と指摘の採否・理由を保存する。
レビュー後に対象が変わった場合はrevision/hashを更新し、必要な承認を取り直す。

### レビュー依頼の例

```text
このworktreeのresume.mdと対象specのREADME・spec.json・review.mdを読み、
review.mdに記録したレビュー待ち対象をGPT-6 Lunaに独立レビューさせてください。
Lunaを利用できない場合はSonnetで代替し、利用不能理由と実際のモデルを記録してください。
対象revision/hashと実差分を照合し、有用な指摘を反映・再確認してください。
承認状態と次の作業を正本へ記録してください。
```

## 検証・成果物・終了時

- 検証手順は[code-quality.md](../../docs/research/code-quality.md)、
  [refactoring-baseline.md](../../docs/experiments/refactoring-baseline.md)、対象specを参照する。
  このPCの共有Pythonはworktreeから`../../venv/Scripts/python.exe`。別PCでは基準環境を構築して選択する。
- Pyrightには必要なら選択したPythonを`--pythonpath`で指定する。
  Windows sandboxでルート探索にアクセス拒否がある場合、Ruffの現行全対象は`src tests/refactoring`。
  日本語ファイルを読むPowerShellでは`Get-Content -Encoding UTF8`を使用する。
- 固定旧実装とgoldenを環境差だけで更新しない。旧golden成功・部品の新旧一致・新全体runの一致を区別して記録する。
- results、保留中の3資料、venv、ローカルのログ/JUnitはGitで共有されるとは限らない。
  検証記録へ結果・環境・対象commitを残し、別PCで必要な証拠がなければ再実行または証拠の所在を確認する。
- 中断前に対象specへ「完了済み/進行中/レビュー待ち/検証待ち」と実差分・次の一手を記録し、resumeを更新する。
  未検証の成功や未レビューの承認を記録しない。spec.jsonの既存構造を守り、別台帳を増やさない。
- 許可済みのcommit/pushは対象ファイルを明示し、日本語メッセージ・AI coauthorなしで行う。
  commit後にGit状態とpush結果を確認する。保留資料を自動でstageしない。

### 次の主担当への開始依頼

```text
このリファクタリングworktreeのAGENTS.md、.kiro/steering/resume.md、
.kiro/steering/agent-handoff.mdを読み、Gitと対象specから現在地を確認してください。
未完了・レビュー待ちを優先し、共有規約とLuna優先・利用不能時Sonnet代替の独立レビュー条件を維持して続けてください。
```

## cc-sdd導入の来歴と再生成

2026-10-07、既存Codex Skills版と同じcc-sdd 3.1.0のClaude Skills版を導入した。
両方17 skills。ツール固有のskillと補助テンプレートは各ツールのディレクトリに置き、
プロジェクトの仕様・承認・方針は共通の`.kiro/`とdocsで管理する。

別の作業用ディレクトリへ`npx cc-sdd@3.1.0 --claude-skills --lang ja --overwrite force`で生成し、
生成された`.claude/skills/`だけを本worktreeへ取り込んだ。
生成元のCLAUDE.mdと`.kiro/settings/`は取り込まず、既存spec・承認・カスタム規約を保持した。
更新時も隔離した場所で生成してdiffを確認する。本worktreeへforce上書きで再導入しない。

Claudeのimport構文は[公式memory文書](https://code.claude.com/docs/en/memory#import-additional-files)、
cc-sddの各エージェント対応は[公式リポジトリ](https://github.com/gotalab/cc-sdd)を参照する。
Claude実行時に`/context`で入口とimportの読み込みを確認する。ファイル導入だけで実行・読み込み確認済みとは扱わない。

### 導入確認記録（2026-10-07）

- 17 skills・33ファイルを生成元とバイト単位で照合し、新しい相対リンク・CLAUDE.mdのimport先を確認した。
- 既存spec・旧新コード・テスト・Codex skillsは変更なし。コード変更がないため数値回帰は再実行していない。
- GPT-6 Luna独立レビューで、共通規約の旧一律Luna指定を担当ツール別へ統一する指摘を採用し、再レビューPASS。
  自動承認オプションを代替にしない旨も明記した。Claudeの起動・Sonnet委任の実行確認は未実施。
  その後のユーザー訂正により、現行方針は上記のLuna優先・利用不能時Sonnet代替へ変更した。この導入時レビューは来歴として保持する。
