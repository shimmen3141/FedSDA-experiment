# cc-sddとリファクタリングの入口

## 作業場所

- ブランチ: `refactor/architecture`
- 基準commit: `748c3aa`
- worktree: 元のcheckout直下の`.worktrees/refactoring/`
- 現在: 要求・命名の人間レビュー待ち。新src実装は未作成。

## 読む順序

1. [リファクタリング方針](../docs/research/refactoring-policy.md)
2. [全体の進行](steering/roadmap.md)
3. [最初の要求仕様](specs/configuration-foundation/requirements.md)
4. [関数・変数等の命名と役割](specs/configuration-foundation/naming.md)

## 運用

cc-sdd 3.1.0をCodex Skills・日本語で導入した。版・npm integrity・導入コマンドは
`installation.json`。導入済みskillは`.agents/skills/kiro-*/SKILL.md`にある。
このworktreeをIDE/Codexのプロジェクトルートとして開いて使う。
元のcheckoutを開いたセッションでは、新worktreeのskillが自動検出されるとは扱わない。

現在は`kiro-steering`、`kiro-spec-init`、`kiro-spec-requirements`の手順で初期文書を作成した。
`spec.json`の要求・命名・設計・task承認は全てfalse、実装準備完了もfalseである。

人間が要求と命名を確認した後に、設計とtaskを作成してレビューする。
設計で追加した関数・変数も命名表へ追記し、そのrevisionを承認してから実装する。
実装時は担当taskを指定して一単位ずつ進める。
自動承認フラグで命名レビューを省略しない。

## 基準環境・成果物

worktree作成ではvenv・results・MNIST cache・保留資料は複製されない。
現時点の調査用Pythonは元checkoutの`venv/Scripts/python.exe`を使っている。
新実装の検証環境は保存済み環境定義から用意し、テスト開始時に明示する。
未配置のデータ・結果を黙って新規生成せず、元資料の保存先を確認する。
保留3資料のコミット判断は元checkoutの`AGENTS.md`に従う。

今回の初期化で確認するのは、導入ファイル・JSON・参照リンク・未承認gateの整合性である。
新実装の回帰テストや自律実装を実行した、という意味ではない。
