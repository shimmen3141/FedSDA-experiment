# cc-sddとリファクタリングの入口

## 作業場所

- ブランチ: `refactor/architecture`
- 基準commit: `748c3aa`
- worktree: 元のcheckout直下の`.worktrees/refactoring/`
- 現在: ユーザーの再開指示により、承認済み命名revision 11へ既存設定を移行するtask 9.1を先行し、その後task 2.6〜3.3を進める。
- 正本と再開手順: [対象specの入口](specs/configuration-foundation/README.md)。候補・履歴を現在の実装契約と混同しない。

## 読む順序

1. [リファクタリング方針](../docs/research/refactoring-policy.md)
2. [全体の進行](steering/roadmap.md)
3. [最初の要求仕様](specs/configuration-foundation/requirements.md)
4. [関数・変数等の命名と役割](specs/configuration-foundation/naming.md)
5. [設定基盤の設計](specs/configuration-foundation/design.md)
6. [gpt-6-lunaのレビューと対応](specs/configuration-foundation/luna-review.md)
7. [初回13タスクと後続計画](specs/configuration-foundation/tasks.md)

## 運用

cc-sdd 3.1.0をCodex Skills・日本語で導入した。版・npm integrity・導入コマンドは
`installation.json`。導入済みskillは`.agents/skills/kiro-*/SKILL.md`にある。
このworktreeをIDE/Codexのプロジェクトルートとして開いて使う。
元のcheckoutを開いたセッションでは、新worktreeのskillが自動検出されるとは扱わない。

現在は`kiro-steering`、`kiro-spec-init`、`kiro-spec-requirements`の手順で初期文書を作成した。
要求と命名revision 2は人間が承認済み。命名承認は`spec.json`の`approvals.naming.approval_history`へ記録した。
設計・独立レビューで追加した名前はrevision 4として人間が承認済み。
初回taskの実装開始と、既存dataset名を維持したrevision 5も人間が承認済み。
具体化した内部関数・テスト名はrevision 6の第9節へ分離し、gpt-6-lunaレビューと主担当の反映を完了した。
revision 7〜9の配置・テスト名も同じ手順で承認済み。採否の理由は`specs/configuration-foundation/luna-naming-review.md`に記録した。

人間が設計と追加命名を確認したため、初回taskを作成して独立確認を完了した。
承認済みの名前でパッケージ境界・検証環境を作成し、task 1.1から実装に着手した。
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
