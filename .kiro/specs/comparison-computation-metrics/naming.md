# 命名表: comparison-computation-metrics

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## module・class

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `held_model_count_record_store` / `HeldModelCountRecordStore` | `evaluation/` | clientの、標本ごとの保有モデル数の列を持つ |
| `computation_cost_summary` | `evaluation/` | 計算量のまとめの型と、計算 |
| `ServerComputationCounts` | 同上 | サーバのパラメータの積和演算の数（集約、統合、診断のパラメータ距離） |
| `ComputationCostSummary` | 同上 | 比較に使う計算量（合計、内訳、標本あたり・保有モデル×標本あたり） |
| `RoundModelComputationCounts` | `runtime/fedsda_measured_run_execution.py` | ラウンド1つの、ローカルの処理と同期の、モデルの計算 |

## 関数・メソッド

| 名前 | 役割 |
|---|---|
| `HeldModelCountRecordStore.append_held_model_count` | 標本1件ぶんの保有モデル数を足す |
| `HeldModelCountRecordStore.snapshot_held_model_counts` | 標本の順の列を返す |
| `summarize_computation_cost` | 計数から、計算量のまとめを作る |

## 判断が必要な点

- 「held model sample」は、「保有モデル×標本」の延べ数（各clientが、各標本の時点で保有していたモデル数の合計）。
- 「local processing」は、ラウンドの中の、標本の処理と、区間末の学習。「synchronization」は、サーバの同期の間（clientが行う評価と再較正を含む）。実行の枠の、既存の段の名前（`sample_processing`、`server_synchronization`）と対応する。
- サーバの計数の「multiply-accumulate」は、パラメータの値1つの「重みを掛けて足す」「2乗して足す」。clientの全結合層の積和演算と、同じ単位として足せる。
- `weighted_parameter_multiply_accumulate_count`（集約）と`consolidation_parameter_multiply_accumulate_count`（統合）: どちらも加重和だが、処理が違うので、名前で分ける。
