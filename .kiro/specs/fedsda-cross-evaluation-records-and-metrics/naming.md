# 命名表: fedsda-cross-evaluation-records-and-metrics

sourceの公開する名前（module、class、公開の関数）と、保存する表の名前を載せる。fieldと列の名前は、design.md。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `cross_evaluation_metric_calculations` | `evaluation/` | クロス評価の記録からの、指標の計算 |
| `ModelPairPredictionComplementarityMetrics` / `summarize_model_pair_prediction_complementarity` | 同上 | モデル対の予測の相補性（正誤を比べた評価の、食い違い・上限の改善・両方正解の割合） |
| `FedsdaRunMetrics.model_pair_prediction_complementarity_metrics` | `runtime/fedsda_run_metric_derivation.py` | 上のまとまり |

## 保存する表の名前

追加: `client_cross_evaluation`、`client_cross_evaluation_class`。

## 判断が必要な点

- 「complementarity（相補性）」は、2つのモデルの正誤が食い違い、片方だけが正解する標本があること（旧の文書の語「予測相補性」）。
- 「per sample selection gain」は、旧の「oracle gain」。標本ごとに、正しいほうのモデルを選べたら、という仮定の上限なので、何を選ぶかが分かる名前にした。
- 「client cross evaluation」は、既存の`ClientCrossEvaluationRecord`の語。
