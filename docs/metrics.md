# 実験指標の構成

実験指標の正規ID、分類、適用範囲、CSV列順は
[`experiment_spec/metrics.py`](../federated_drift_experiment/experiment_spec/metrics.py)を正本とする。
指標を追加するときは、実験スクリプトへ列名を直接追加せず、このスキーマへ登録する。

## 1. 三つの利用階層

| 階層 | 用途 | 例 |
|---|---|---|
| `primary` | 手法の主要な比較・結論 | accuracy、stable accuracy、precision、recall、F1、モデル通信量、最終モデル数 |
| `secondary` | コスト、規模、実行環境を含む補助評価 | 軽量メッセージ、計算量、実行時間、平均・最大モデル数 |
| `diagnostic` | 特定機能の原因分析 | 変化点誤差、FP種別、適応操作、仮モデルの採否理由 |

主要な結論は`primary`を中心に述べ、`secondary`でコストとの交換関係を確認する。
`diagnostic`は該当機能を使用した実験だけで解釈し、全手法の優劣を直接表す値としては扱わない。

## 2. 適用範囲

| スキーマ値 | 意味 |
|---|---|
| `all_methods` | Obliviousを含む全手法で定義できる |
| `adaptive_methods` | ドリフトに応じてモデルを切り替える手法で意味を持つ |
| `change_point_estimators` | 変化開始点を明示的に推定する検出器で意味を持つ |
| `fedsda_methods` | FedSDA固有の警報・適応・サーバ割当で意味を持つ |
| `forward_creation_policies` | forward検証を使う仮モデル作成方式でのみ意味を持つ |

現行CSVは既存結果との互換性のため、適用外の列も保持し、多くの場合`0`を格納する。
したがって、`0`を「測定済みで事象がなかった」と解釈する前に、スキーマの`applicability`を確認する。
将来この表現を変更する場合は、CSVのschema versionを上げて欠損値と実測ゼロを区別する。

## 3. 指標群

| group | 内容 | 主な指標 |
|---|---|---|
| `predictive_performance` | 予測性能 | `accuracy`, `stable_accuracy` |
| `detection` | 真のドリフトとの照合 | `precision`, `recall`, `f1`, `avg_delay`, `total_detect` |
| `communication` | モデル転送と軽量メッセージ | `comm_models_*`, `comm_messages_*` |
| `model_population` | モデル数の終値・時系列要約 | `final_model_count`, `mean_model_count`, `max_model_count`, `model_count_auc` |
| `runtime` / `compute` | 時間とモデル計算回数 | `runtime_seconds`, `client_compute_seconds_*`, `compute_*` |
| `change_point` | 推定変化点の誤差 | `change_point_mae`, `change_point_bias`, `change_point_estimate_count` |
| `alarm_episode` / `false_positive` | 警報エピソードと誤検出の内訳 | `alarm_*`, `switch_fp_*` |
| `adaptation_action` | 再利用・新規作成・維持の判断 | `adaptation_*`, `model_reuse_*` |
| `provisional_model` | 仮モデル方式の採否・標本数・棄却理由 | `provisional_*` |
| `server_mapping` | サーバ割当変更 | `server_mapping_change_count` |

コードから用途別に選ぶ場合は`metrics_in_profile("core")`、`"detection"`、
`"adaptation"`、`"resource"`、`"all"`を使用できる。profileはCSV列を削るものではなく、
表や分析で表示する指標集合を揃えるためのものである。

## 4. CSVとNPZの役割

- CSVには、seed間で平均・標準偏差を計算するスカラー要約を保存する。
- NPZには、回復曲線、クライアント別・ラウンド別計算量、モデル数時系列、SoftRoutingの重みなど、
  後から再集計するための系列を保存する。
- 実行時間は環境依存なので、アルゴリズム比較では`compute_model_examples_total`や
  `compute_optimizer_steps_total`も併記する。
- `comm_models_total`はモデル転送回数であり、モデル構造が同じ場合の比較を前提とする。
  共有バックボーン + 概念別ヘッドを導入する場合は、転送パラメータ数またはバイト数を別途追加する。

## 5. 次に追加する診断指標

次の二群は有用だが、現時点では未実装であり、既存のCSV指標には含めない。

1. **モデル別学習量**: モデルごとの学習サンプル数、保存データ数、optimizer step数。
2. **モデル対予測相補性**: 一致率と、`iのみ正解`、`jのみ正解`、`両方正解`、`両方不正解`のクロス表。

いずれも複数モデルを持つ方式に依存する診断値である。詳細系列はNPZへ保存し、CSVにはモデル間の
中央値・最大値など、研究上の問いが固まった後に必要最小限の要約だけを追加する。これにより、
手法固有の詳細値が主要比較表を過度に肥大化させることを防ぐ。
