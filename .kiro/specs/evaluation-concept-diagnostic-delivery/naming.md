# 命名表: evaluation-concept-diagnostic-delivery

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。新しい公開の名前はない（既存の操作と関数へ、引数を足すだけ）。確認は、全taskの後の独立レビューで受ける。

## 関数・メソッド（引数を足すもの）

| 名前 | 足す引数 | 役割 |
|---|---|---|
| `RunClientOperations.process_observed_sample` | `evaluation_concept_id` | その標本の真の概念ID。診断専用 |
| `run_stream_protocol_intervals` | `evaluation_concept_traces` | clientごとの概念列。標本位置の値を、clientへ渡す |

## 判断が必要な点

- 引数名は、`FedsdaRunClient.process_observed_sample`の既存の任意の引数`evaluation_concept_id`、全体runの結果の`evaluation_concept_traces`と、同じ語にする。「evaluation」は、評価用の真値（手法が使えない情報）を表す、実行の枠の既存の語。
