# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `fedsda-local-parameter-prediction`（`LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME`） | 構成の名前 | 既定から、予測に使うパラメータ（`prediction_model_parameter_source`）だけを、局所更新した現在の値（`current_local_parameters`）にした構成（混ぜる予測の寄与を比べる用）。全モデルの結合予測は既定と同じ。`fedsda-single-model-prediction`（結合をせず、現在の学習帰属のモデルだけで予測する）とは別。名前は、替える設定の値に合わせた |
| 基準版6 | goldenの版 | 新しい既定（混ぜる予測・T120）の値 |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `fedsda`（既定） | 局所の予測・T30 | 混ぜる予測・T120 |
| `fedsda-synchronized-local-mixture-prediction` | 既定から、予測だけを混ぜる値にした構成 | 既定と同じ設定（名前だけが違う。既存の結果との連続のために残す） |
| `fedsda-single-model-prediction` | 既定の予測に使うパラメータを引き継ぐ | 局所を明示する（混ぜる値と組み合わせられないため） |
| `BASELINE_VERSION` | 5 | 6 |
