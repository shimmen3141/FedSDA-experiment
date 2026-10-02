# 文書一覧

文書を用途ごとに分類する。最終提案はSwitching SoftRouting構成である。
各ディレクトリには代替方式・旧構成の説明も含むため、配置だけで最終提案の採用要素とは判断しない。

## 全体説明・処理の概観 — overview

- [1サンプルのコマ送り図](overview/fedsda-processing-flow.html): 作成中のフロー図。
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
- [既存成果の監査と論文用索引](experiments/experiment-results-audit.md)
- [固定baselineの構成](experiments/baselines.md)
- [実験設定・掃引計画・manifest](experiments/experiment-configuration.md)
- [データ特性](experiments/dataset-characteristics.md)
- [評価指標](experiments/metrics.md)
- [FedDrift元論文との相違点](experiments/differences-from-feddrift.md)

## 設定の参照資料 — reference

- [オプション依存構造](reference/options.md): 自動生成。`python -m tools.generate_option_docs`で更新する。
- [ハイパーパラメータ・変数](reference/hyperparameters.md)

## 研究上の検討資料 — research

- [研究バックログ](research/research-backlog.md): 設計候補・非採用案・今後の課題。
- [Meta-switchingの先行研究・差分・新規性に関する調査報告](<research/FedSDA Meta-switchingの先行研究・差分・新規性に関する調査報告.pdf>): Meta-switchingを対象とした検討資料。

成果のまとめは実験索引・監査から、機能の詳細確認はcomponentsから読む。
最終Switching構成の全体説明を整備する際は、overviewに正本文書を置く。
