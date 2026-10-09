# 命名表: model-cross-evaluation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `client_model_cross_evaluation` | module（runtime） | client 1つが、渡されたモデルを、手元の標本で評価する処理 | `model_cross_evaluation`は、サーバが、モデルの全部の組をclientへ評価させて集計する処理 |
| `ModelPairCorrectnessCounts` | frozen dataclass | 同じ標本での、渡されたモデル（candidate）と対象のモデル（target）の正誤の4つの数 | `ModelPairUniqueCorrectnessCounts`は、IDの向きにそろえて、片方だけが正解した数を足し合わせたもの |
| `ClientModelCrossEvaluation` | frozen dataclass | clientの評価1回の結果（件数、損失の和、2乗和、正誤の数、クラス別） | `ModelCrossEvaluation`は、サーバのクロス評価1回の結果 |
| `model_cross_evaluation` | module（runtime） | サーバのクロス評価 | 上 |
| `CrossEvaluationLossSums` | frozen dataclass | （評価する側、対象）の組の、件数・損失の和・2乗和 | `BoundedLossMoments`は、平均と偏差平方和で持つ損失統計 |
| `ModelPairUniqueCorrectnessCounts` | frozen dataclass | モデルの対（小さいID、大きいID）の、評価した標本数と、それぞれの側だけが正解した数（全体とクラス別） | 上 |
| `ModelCrossEvaluation` | frozen dataclass | クロス評価1回の結果（モデルIDの列、損失の統計の表、対ごとの正誤の集計） | 上 |
| `cross_evaluation_record_store` | module（evaluation） | クロス評価の診断の記録 | `communication_volume_record_store`は通信量 |
| `ClientCrossEvaluationRecord` | frozen dataclass | clientの評価1回の診断の記録（ラウンド、client、評価する側、対象、結果の数値） | `ClientModelCrossEvaluation`は、clientが返す結果（ラウンドとclientを持たない） |
| `CrossEvaluationRecordStore` | class | 診断の記録を、評価の順に持つ | — |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `evaluate_candidate_model_on_target_model_samples` | 渡されたパラメータ、対象のモデルID、正誤の比較の有無、clientのowner、標本の上限、乱数生成器 | `ClientModelCrossEvaluation` | 対象のモデルの標本を選び、渡されたモデルの損失の統計と、保有する対象のモデルとの正誤の比較を返す | 借りた乱数、torchの乱数 |
| `cross_evaluate_global_models` | clientの列、グローバルモデルのowner、通信量のowner、診断の記録のowner、モデルIDの列、ラウンド、clientの上限、乱数生成器 | `ModelCrossEvaluation` | モデルの全部の組を、保有するclientへ評価させ、通信量と記録を足し、表と対の集計を返す | 通信量、診断の記録、乱数 |
| `FedsdaRunClient.get_cross_evaluation_held_model_ids` | なし | `frozenset[int]` | クロス評価で保有とみなすモデルID（評価標本・学習データ・保有モデル・現在の学習帰属のIDの和集合） | なし |
| `FedsdaRunClient.evaluate_candidate_model_on_target_model_samples` | 渡されたパラメータ、対象のモデルID、正誤の比較の有無 | `ClientModelCrossEvaluation` | 自分のownerと設定で、評価を行う | 同上 |
| `CrossEvaluationRecordStore.append_client_cross_evaluation_record`・`snapshot_client_cross_evaluation_records` | 記録／なし | なし／記録のtuple | 記録を足す／評価の順の写しを返す | 記録／なし |

## 判断が必要な点

- 「candidate」は、評価のためにサーバから渡されたモデル（旧の`candidate_model_id`）。候補検証の「候補（candidate）」と同じ語だが、ここでは「対象（target）のモデルの標本で評価される側」の意味で、旧の診断の列名と合わせる。型名・関数名では、`candidate_model`と`target_model`を必ず対で使う。
- 「cross evaluation」は、モデルを、別のモデルの標本で評価すること（旧の`_cross_evaluate`）。
- 束の新しいfield `maximum_cross_evaluation_sample_count`は、clientが1回の評価に使う標本の上限（旧`EVAL_MAX_SAMPLES`）。評価標本の保存の上限`maximum_stored_evaluation_sample_count_per_model`と区別する。
