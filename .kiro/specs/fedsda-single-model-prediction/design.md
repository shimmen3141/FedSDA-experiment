# Design Document: fedsda-single-model-prediction

## 変更

- `PredictionCombinationSettings.prediction_combination_strategy`へ、値`current_training_model_prediction`を足す。
- `runtime/observed_sample_prediction.py`: この方式のとき、`_predict_with_current_training_model_only`へ分ける。現在のモデルだけを評価し（共有部1回、概念固有部1回）、記録を足す。重みの保持と証拠には、触れない。
- `runtime/fedsda_run_client_settings.py`: この方式は、再較正の方針が`keep_weights_without_recalibration`のときだけ受け付ける（作り直す重みがない。再較正をすると、保留中の標本を、全保有モデルで評価する計算が、余分に入る）。
- `AdaHedgeWeightEvidenceCollection`へ、`evidence_is_updated_by_prediction`（既定は真）を足す。clientは、この方式のとき、偽で作る。学習帰属の変更での始め直し（`restart_adahedge_weight_evidence_after_training_assignment_change`）は、偽の集まりには、何もしない。
- `FedsdaRunClient.prediction_combination_strategy`（読み取り）を足し、指標の導出が、結合して予測した標本の数を、この方式のclientでは0にする。
- 掃引: 構成`fedsda-single-model-prediction`を足す。

## 記録の、重みに関する項目

この方式では、重みが無い。記録の項目を「無い」（欠損）にすると、項目の型と、表の列の型（真偽→欠損つきの整数）が、どの構成でも変わる。そこで、「現在のモデルに重み1」の値を入れる。旧も、重みで結合するclientが、結合を止めている標本で、同じ入れ方をする（`_record_hard_prediction`: 正否の列は現在のモデルの正否、最大重みのモデルは現在のモデル、実効モデル数と最大の重みは1）。

この方式では、「どれかのモデルか実際の予測が正解」「確信度が最大のモデルが正解」も、現在のモデルの正否になる（ほかのモデルを評価しないため）。予測の内訳の指標を、提案構成と比べるときは、この点に注意が要る（文書`docs/experiments/experiment-sweep.md`に書く）。

## 保有モデル×標本の延べ数

指標`held_model_sample_count`は、標本ごとの保有モデル数の合計で、予測の方式に依らない。重みで結合する構成では、予測で評価した標本の延べ数と同じ。この方式では、予測で評価するのは、標本ごとに1モデルなので、同じにならない（実旧の、予測の評価の数は、処理した標本数）。

## 検証

- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を足す（旧の側は、構成名を替え、再較正を「しない」）。小さい条件の7つで、実旧の全体runと照合する。固有の経路のtestと、設定の組の拒否のtestを足す。
- 照合の土台（`tests/refactoring/test_fedsda_run_metric_derivation.py`の`compare_small_run_with_real_legacy`）: 実旧が、重みと予測の内訳の記録を持たない構成（`legacy_records_prediction_weight_diagnostics`が偽）では、実旧に無い3つの離散列（重みでの結合の正否・最大重みのモデル・結合の有効）と、予測の内訳の指標を、比べない。代わりに、実旧にそれらが無いこと（保存された列の集合、内訳の標本数0、重みの保持が無いこと）を確かめる。予測の正否と、予測したモデルは、離散列（`history_accuracy`、`history_model_id`）で照合する。
- `tests/refactoring/test_training_assignment_adahedge_weight_evidence_restart.py`: 予測の処理が更新しない集まりは、始め直さないこと。
