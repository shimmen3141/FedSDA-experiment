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
| `model_learning` | モデル別の割当データ・学習サンプル・optimizer stepの分布 | `model_assigned_samples_*`, `model_training_examples_*`, `model_optimizer_steps_*` |
| `model_complementarity` | 同一標本上のモデル対正誤表とoracle選択余地 | `model_pair_*` |
| `dominance_pruning` | クロス評価で支配されたモデルの除去回数 | `dominated_model_prune_count` |
| `soft_routing` | SoftRoutingが全モデル中の正解可能性を回収できた割合 | `routing_*` |

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

## 5. モデル多様性と学習断片化の診断

モデル別学習量は、最終時点のモデルIDごとにクライアント間で集約する。CSVには総量・平均・最小値・
変動係数（CV）だけを保存し、クライアント×モデルの詳細はNPZの`model_diagnostic_*`へ保存する。
CVが大きい、または最小値が小さい場合は、一部モデルへ十分な学習量が届いていない可能性がある。
`model_training_examples_total`は最終的に保持されたモデルへ帰属できる学習量であり、棄却された仮モデルや
一時的shadowの学習は含めない。それらを含む全計算量は`compute_training_examples_total`で確認する。

モデル対予測相補性はFedSDAの既存クロス評価と同じ標本で計測する。追加のモデル通信は行わないが、
対象モデル自身の予測を得るforward計算は追加される。NPZの`model_pair_*`には、候補のみ正解、対象のみ
正解、両方正解、両方不正解の件数を保存する。CSVの主な要約は次のとおり。

- `model_pair_correctness_disagreement_rate`: 片方だけが正解した割合。
- `model_pair_oracle_gain_rate`: 良い方の単一モデルに対し、標本ごとにoracle選択した場合の改善上限。
- `model_pair_both_correct_rate`: 両モデルが正解した割合。

支配モデル除去を有効にした場合、`dominated_model_prune_count`は、通常の
距離クラスタリングとは別に優勢モデルへ再割当されたモデル数を数える。
判定には直近のクロス評価だけを使い、過去ラウンドの古い証拠は混ぜない。

SoftRoutingでは、予測時にすでに計算している全保持モデルの出力から次を記録する。

- `routing_oracle_accuracy`: 実混合または少なくとも一つの保持モデルが正解した割合。
- `routing_mixture_accuracy`: 実際の重み付き混合が正解した割合。
- `routing_leader_accuracy`: 最大重みモデル単体が正解した割合。
- `routing_oracle_gain_rate`: oracle accuracyと実混合accuracyの差。
- `routing_oracle_recovery_rate`: oracleが正解可能だった標本のうち実混合も正解した割合。
- `routing_missed_oracle_count`: 正解モデルが存在したのに実混合が誤答した件数。

NPZには`history_routing_oracle_correct`と`history_routing_leader_correct`も保存する。
これらの指標は追加のモデルforwardや通信を発生させない。

これらは診断専用であり、現時点ではクラスタリング判定を変更しない。精度改善を主張する指標ではなく、
モデルを残す利益と学習量断片化の原因を切り分けるために使用する。
