# 命名表: evaluation-concept-diagnostic-delivery

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。新しい公開の名前は、検査の関数1つ。ほかは、既存の操作と関数へ、引数を足すだけ。確認は、全taskの後の独立レビューで受ける。

## 関数（新規）

| 名前 | module | 役割 |
|---|---|---|
| `validate_evaluation_concept_traces_match_observed_streams` | `execution/stream_protocol_execution_loop.py` | 概念列が、観測列と、同じclientの順・同じ標本の数で対応することを確かめる（区間の進行が、最初に呼ぶ）。参加者の検査`validate_prepared_run_participants`とは、対象が違う |

## 関数・メソッド（引数を足すもの）

| 名前 | 足す引数 | 役割 |
|---|---|---|
| `RunClientOperations.process_observed_sample` | `evaluation_concept_id` | その標本の真の概念ID。診断専用 |
| `run_stream_protocol_intervals` | `evaluation_concept_traces` | clientごとの概念列。標本位置の値を、clientへ渡す |

## 判断が必要な点

- 引数名は、`FedsdaRunClient.process_observed_sample`の既存の任意の引数`evaluation_concept_id`、全体runの結果の`evaluation_concept_traces`と、同じ語にする。「evaluation」は、評価用の真値（手法が使えない情報）を表す、実行の枠の既存の語。
