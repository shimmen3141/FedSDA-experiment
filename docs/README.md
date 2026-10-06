# 文書一覧

文書を用途ごとに分類する。最終提案はSwitching SoftRouting構成である。
各ディレクトリには代替方式・旧構成の説明も含むため、配置だけで最終提案の採用要素とは判断しない。

## 読む順序と正本

リファクタリングの再開は[resume](../.kiro/steering/resume.md)、
Claude・Codexの交代は[共通引継ぎ手順](../.kiro/steering/agent-handoff.md)から確認する。

1. [最終提案構成](overview/proposed-method.md): 採用要素、固定設定、処理順、主張の範囲。
2. [主要ablation](experiments/ablation-plan.md): 比較目的と既存成果の対応。
3. components: 調べたい個別機能の実装と代替方式。
4. reference: 設定の意味・コード初期値・依存関係。
5. research: 過去の検討、非採用案、今後の候補。

最終構成の定義はoverview、実験の条件・結果の来歴はexperiments、選択肢の実装範囲はcomponentsとreferenceで管理する。
各文書の冒頭で対象構成と採用状況を示す。`reference/options.md`は自動生成のため、冒頭への手作業の追記も行わない。

## 全体説明・処理の概観 — overview

- [FedSDAとFedDriftの構成比較](overview/fedsda-feddrift-comparison.md): 最終提案構成と本リポジトリの比較実装の違い。
- [最終提案構成](overview/proposed-method.md): 論文で扱うSwitching構成の正本。
- [1サンプルのコマ送り図](overview/fedsda-processing-flow.html): 作成中のフロー図。コミット保留のローカル資料。
- [実装用仕様書](overview/fedsda-algorithm.md): ADWIN中心の初期構成の全体説明。最終Switching構成とは区別する。
- [シーケンス図](overview/sequence-diagrams.md): 複数方式の通信処理。削除済み方式も設計比較用に含む。

## 個別機能 — components

- [ドリフト検出](components/drift-detection.md)
- [共有バックボーン・概念別モデル](components/shared-backbone.md)
- [モデルルーティング](components/model-routing.md)
- [SoftRoutingの予測レイヤー](components/soft-routing.md)
- [既存モデル再利用](components/model-reuse.md)
- [新規モデル作成](components/new-model-creation.md)
- [モデル統合](components/model-consolidation.md)
- [Shadow tournament](components/shadow-tournament.md): 実験的な新規モデル作成方針。最終提案の採用方針ではない。

## 実験条件・比較・成果物管理 — experiments

- [主要ablationの計画と実行状況](experiments/ablation-plan.md)
- [既存成果の監査と論文用索引](experiments/experiment-results-audit.md): 日付付き確認記録。コミット保留のローカル資料。
- [固定baselineの構成](experiments/baselines.md)
- [リファクタリング前の回帰基準](experiments/refactoring-baseline.md): 旧11ケースと最終構成3ケース、基準環境、検証手順。
- [実験設定・掃引計画・manifest](experiments/experiment-configuration.md)
- [データ特性](experiments/dataset-characteristics.md)
- [評価指標](experiments/metrics.md)
- [FedDrift元論文との相違点](experiments/differences-from-feddrift.md)

## 設定の参照資料 — reference

- [オプション依存構造](reference/options.md): 自動生成。`python -m tools.generate_option_docs`で更新する。
- [ハイパーパラメータ・変数](reference/hyperparameters.md)

## 研究上の検討資料 — research

- [リファクタリングの再開案内](../.kiro/steering/resume.md): 現在地、次の候補、読む文書。
- [新実装の品質検査](research/code-quality.md): Ruff・Pyright・commit hook・CIの設定と実行手順。
- [リファクタリング方針案](research/refactoring-policy.md): 責務・レイヤー・命名、新APIへの移行、選択肢の追加・削除とcc-sddの評価。
- [研究バックログ](research/research-backlog.md): 設計候補・非採用案・今後の課題。
- [実装の不具合・改善事項](research/implementation-findings/README.md): 再現条件・影響・移植時の扱い・将来修正を追跡する入口。
- [Meta-switchingの先行研究・差分・新規性に関する調査報告](<research/FedSDA Meta-switchingの先行研究・差分・新規性に関する調査報告.pdf>): Meta-switchingを対象とした検討資料。コミット保留のローカル資料。

コミット保留の3資料と`results/`はgit cloneだけでは取得できない。コミット対象は後続のユーザー指示で決める。
リファクタリングworktreeにはこれらを複製していない。該当リンクの資料は元checkoutで参照する。
