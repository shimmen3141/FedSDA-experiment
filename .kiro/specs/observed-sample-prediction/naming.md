# 命名表: observed-sample-prediction

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `sample_prediction_record_store` | module（evaluation） | 標本ごとの予測の記録と、そのowner | `adaptation_record_store`は適応（警報への応答と候補検証の確定）の記録、`loss_change_alarm_record_store`は監視の値と警報の位置の記録 |
| `SamplePredictionRecord` | frozen dataclass | 標本1件の予測の結果と診断の項目（位置、概念ID、観測クラス、正否、最大重みのモデル、実効モデル数） | `ObservedSamplePrediction`は予測の関数の戻り値で、この記録に加えて確率と重みを持つ |
| `SamplePredictionRecordStore` | class | `SamplePredictionRecord`を観測順に持つ。予測や判断をしない | `PendingSampleObservationStore`は保留中の標本そのものを持つ |
| `observed_sample_prediction` | module（runtime） | 観測した標本1件の予測・記録・更新をつなぐ関数と、その結果 | `observed_sample_processing`は標本1件の処理の全体（本moduleの関数を最初に呼ぶ） |
| `ObservedSamplePrediction` | frozen dataclass | 予測の関数の結果: 記録、予測クラス、結合した確率、モデル別の確率、正規化後のFixed-Shareの重みとglobalの診断重み、モデル別の損失 | `ObservedSampleProcessing`は標本1件の処理の結果（これをfieldに持つ） |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `predict_observed_sample_and_update_prediction_weights` | 標本、Fixed-Shareの重みのowner、診断証拠の保持集合、記録のowner、保有モデルのregistry、現在の学習帰属 | `ObservedSamplePrediction` | 保有する全モデルの確率をFixed-Shareの重みで結合して予測し、記録を足し、ラベル観測後に重みを更新する。診断証拠（globalと真の概念別）も同じ損失で更新する | 3つのownerと記録のowner。モデルと乱数は変えない |
| `SamplePredictionRecordStore.append_sample_prediction_record` | 記録 | なし | 記録を検査して末尾へ足す | 記録のlist |
| `SamplePredictionRecordStore.snapshot_sample_prediction_records` | なし | 記録のtuple | 保持している記録の写しを返す | なし |
| `SamplePredictionRecordStore.last_recorded_sample_index` | なし（property） | intまたはNone | 最後に足した記録の位置。予測の関数が、更新の前に位置の連続を確かめるために読む | なし |

## 判断が必要な点

- 関数名は「予測」と「予測重みの更新」だけを言い、診断証拠の更新と記録の追加は名前に含めていない。全部を並べると名前が長くなりすぎるので、主な役割（予測と、次の予測を決める重みの更新）を名前にし、残りはdocstringと本表に書いた。
- `any_model_or_combined_prediction_is_correct`などの記録のfieldは、旧の`oracle`・`leader`という語（何のoracleか、何のleaderかが名前から分からない）を使わず、何が正しいかを書いた。
