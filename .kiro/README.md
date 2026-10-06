# cc-sddとリファクタリングの入口

## 作業場所

- ブランチ: `refactor/architecture`
- 基準commit: `748c3aa`
- worktree: 元のcheckout直下の`.worktrees/refactoring/`
- 現在の承認・進捗・次のtask: 対象specの`spec.json`を参照する。入口文書へ進捗を重複記載しない。
- 正本と再開手順: [対象specの入口](specs/configuration-foundation/README.md)。候補・履歴を現在の実装契約と混同しない。

現在地と次の作業は[再開案内](steering/resume.md)、担当交代の手順と検証コマンドは[共通引継ぎ手順](steering/agent-handoff.md)が入口。以下は導入初期（設定基盤・単一run基盤）の案内で、来歴として残す。

## 読む順序

1. [リファクタリング方針](../docs/research/refactoring-policy.md)
2. [全体の進行](steering/roadmap.md)
3. [対象specの正本一覧と再開手順](specs/configuration-foundation/README.md)

specでは承認・進捗、要件、設計、命名、taskを順に確認する。
レビューと候補表は根拠・履歴として必要な箇所だけ参照する。

次段階の仕様は[単一runの実行順序・SINEデータ供給](specs/single-run-execution/README.md)。
設定基盤の初回完了と、次specの要件承認・実装完了を区別する。

## 運用

cc-sdd 3.1.0をCodex Skills・日本語で導入した。版・npm integrity・導入コマンドは
`installation.json`。導入済みskillは`.agents/skills/kiro-*/SKILL.md`にある。
このworktreeをIDE/Codexのプロジェクトルートとして開いて使う。
元のcheckoutを開いたセッションでは、新worktreeのskillが自動検出されるとは扱わない。

要求・設計・taskの人間承認を確認してから実装する。
命名の正本は対象specの`naming.md`、承認revisionと履歴は`spec.json`。
命名の採否の理由は`luna-naming-review.md`に記録する。
追加する関数・変数も命名表へ追記し、そのrevisionをgpt-6-lunaレビューと主担当の有用指摘の反映で承認してから実装する。
実装時は担当taskを指定して一単位ずつ進める。
自動承認フラグで命名レビューを省略しない。

## 基準環境・成果物

worktree作成ではvenv・results・MNIST cache・保留資料は複製されない。
検証用Pythonは元checkoutの`venv/Scripts/python.exe`を共有し、保存済みのPython・数値ライブラリのビルド・依存全体・golden hashと一致することを照合した。
worktreeからは`../../venv/Scripts/python.exe`、MNISTは`FDE_MNIST_DATA_DIR`で元checkoutの`data/mnist/`を明示参照する。
未配置のデータ・結果を黙って新規生成せず、元資料の保存先を確認する。
保留3資料のコミット判断は元checkoutの`AGENTS.md`に従う。

検証コマンドと観測結果は対象specの`review.md`へ記録する。
旧経路のgolden検証は参照実装の保全を確認するもので、新しいアルゴリズムの同値性を意味しない。
