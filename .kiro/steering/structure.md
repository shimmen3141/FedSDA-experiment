# 構造と命名

## 現状と目標

worktreeには参照元と同じ既存パッケージがcheckoutされている。
新パッケージは命名レビュー後にsrc配置で作成する案。既存名の維持は要件にしない。
旧コードは段階的に移植し、完成した新ブランチの実行経路から除く。

## 設計原則

機能別配置と依存方向を組み合わせる。runtimeが具体方式を組み立てる。
判断部はCLI・保存・描画へ依存しない。評価は記録を読み、保存は結果を受け取る。
各機能の状態所有者を一つ決め、クライアント全体の可変状態を共有しない。

## 命名

ファイル・関数・変数はsnake_case、型はPascalCase。
model、expert、assignment、leader、candidateの違いを説明する。
`sample_index`、`round_index`、`*_sample_count`、`*_byte_count`で単位を明示する。
名前だけで意味が分かるかを、実装前に人間が判断する。

新しい設定・選択肢の正式名は一つとし、旧名aliasを設けない。
詳細は`docs/research/refactoring-policy.md`と各specの`naming.md`を参照する。
