# Design Document

## Overview
予測の設定の値に`mixture_of_synchronized_and_local_parameters`を足す。clientは、2つの候補の重みのowner（`SynchronizedAndLocalPredictionMixture`）を、写しのownerと同じく属性として持つ。この値のとき、clientは、配布の後に写しを取り、標本の処理へ、写しのownerと混ぜる重みのownerの両方を渡す。予測の関数は、混ぜる重みのownerを受け取ったとき、同期の結合予測（今の`parameters_at_last_synchronization`の経路）に加えて、局所の結合予測を作り、混ぜた確率で予測と記録の`combined_prediction_is_correct`を決め、記録の後に混ぜる重みを更新する。

## Boundary Commitments
### This Spec Owns
- `methods/fedsda/prediction_combination/prediction_combination_settings.py`: 値。
- 新しいmodule `methods/fedsda/prediction_combination/synchronized_and_local_prediction_mixture.py`: 混ぜる重みのowner。
- `runtime/observed_sample_prediction.py`・`runtime/observed_sample_processing.py`: 省略可能な引数`synchronized_and_local_prediction_mixture`（Noneか、exactな型。ほかの型の検査と同じく、状態を変える前に確かめる）。混ぜる重みのownerを受け取るときは、写しのownerも必要（無ければ`ValueError`）。
- `runtime/fedsda_run_client.py`: 構成子の引数と属性、配布の後の写しの条件（同期の時点か、混ぜる値）、標本の処理への受渡し。
- `runtime/fedsda_run_client_settings.py`: 混ぜる値と`current_training_model_prediction`の組合せの拒否。
- `sweep/experiment_configurations.py`: 構成名と組み立て。
- `tests/refactoring/behavior_golden_cases.py`: `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`への登録。構成の一覧を使うtest（`test_fedsda_configuration_variants.py`）、設定の検査のtestの許す値、依存境界の許可、コマンドの引数の一覧のtest（構成名は引数ではないので変わらない）。
- 新しいtest `tests/refactoring/test_fedsda_synchronized_local_mixture_prediction.py`。
- 文書。
### Out of Boundary
学習・検出・候補の検証・統計・アップロード・再較正、既定。
### Allowed Dependencies
新しいmoduleは、`math.exp`と`torch.Tensor`（型）だけを使う。依存境界の許可へ、新しいmoduleと、それを読むmodule（予測・標本の処理・client）の新しい依存を足す。
### Revalidation Triggers
予測の関数と標本の処理の引数、clientの構成子。

## Architecture
### 混ぜる重みのowner（`SynchronizedAndLocalPredictionMixture`）
- 構成子: `share_rate: float`（`1/T`。0より大きく1未満の有限のfloat）。重みは`(0.5, 0.5)`（同期、局所）から始める。学習率は定数2.0（`LEARNING_RATE`）。
- `mix_prediction_probabilities(*, synchronized_probabilities, local_probabilities) -> Tensor`: `synchronized_probabilities * w_sync + local_probabilities * w_local`（この順の演算。診断と同じ値にするため）。
- `update_weights_after_loss_observation(*, synchronized_loss: float, local_loss: float) -> None`（既存の`FixedShareWeightController.update_weights_after_loss_observation`と語を揃える）: `(w_s·exp(−2·l_s), w_l·exp(−2·l_l))`を和で割り、`((1−a)·u_s + a·u_l, (1−a)·u_l + a·u_s)`（aは`share_rate`）。演算はPythonのfloat（倍精度）。
- 損失の式（診断と同じ値にするため固定する）: 結合予測の確率のtensor（float32）から、真のクラスの確率をPythonのfloatへ直し（2値で真のクラスが1なら`float(v[0])`、0なら`1.0 - float(v[0])`。多クラスなら`float(v[k])`）、損失は`1.0 - p`（倍精度）。今の損失の関数（`compute_model_mean_bounded_losses_after_label_observation`。float32の演算）は使わない。式は、真のクラスが0の2値でも`1.0 - (1.0 - float(v[0]))`のように、確率をいったん作ってから1から引く形のまま書き、`float(v[0])`へ簡略化しない（最後の桁が変わるため）。この式は、ownerのmodule関数`true_class_loss_of_prediction_probabilities(*, prediction_probabilities, observed_class_id, class_count) -> float`にし、予測の関数が使う。testの中の診断の差し替えは、この関数を使わず、診断の式を書き写して使う（関数の誤りも検出できるように）。関数そのものは、手で計算した値（2値の両方のクラス、多クラス）との単体testで確かめる。
- 寿命: 重みは、clientを組み立てるときに`(0.5, 0.5)`で作り、以後、clientの寿命の間、作り直さない（警報・配布・集約・再較正のどれでも変えない。診断と同じ）。
- `snapshot_weights() -> tuple[float, float]`: 読取り。

### 予測
- `predict_observed_sample_and_update_prediction_weights(..., synchronized_and_local_prediction_mixture=None)`: 結合して予測する2つの設定の経路で、今の同期の結合予測（`combined_prediction_probabilities`、予測に使う重み`prediction_weights_by_model_id`）を作った後、混ぜる重みのownerがあれば:
  1. 局所の確率: 保有モデルの現在の分類器で、ID昇順で最初のモデルの共有特徴を計算し、各モデルの確率へ変換する（`convert_model_outputs_to_prediction_probabilities`）。この標本の学習は予測の後なので、現在の分類器は、診断が元の予測の前に計算した値と同じ。
  2. 局所の結合予測: `combine_model_prediction_probabilities`（同じ予測に使う重み）。
  3. 混ぜた確率で、予測クラスと正否（`predict_class_labels_from_prediction_scores`。混ぜた確率は、この関数の検査を省ける条件——このmoduleで検査した確率か、`combine`の結果——に当たらないので、`prediction_scores_are_validated=False`で検査する）を決め、記録の`combined_prediction_is_correct`と`any_model_or_combined_prediction_is_correct`、戻り値の`predicted_class_labels`・`combined_prediction_probabilities`を、混ぜた確率のものにする。
  4. 記録を足した後、モデルの重みの更新（今のまま。同期の側の損失）と、混ぜる重みの更新（同期・局所の結合予測の、上の損失の式）を行う。
- 記録の項目の意味: 混ぜた確率の側になるのは、`combined_prediction_is_correct`（混ぜた確率の予測の正否）と`any_model_or_combined_prediction_is_correct`（混ぜた確率の正否と、同期の側のモデルごとの正否の和）の2項目と、それらから作る指標（予測の精度、定常精度、期間ごとの精度の診断ほか）。それ以外の項目（`fixed_share_weighted_prediction_is_correct`・`adahedge_weighted_prediction_is_correct`・最大重みのモデルと、その正否ほか）は、同期の結合予測の側で、`parameters_at_last_synchronization`と同じ値。そのため、Fixed-Shareの重みで予測する構成でも、`combined_prediction_is_correct`と`fixed_share_weighted_prediction_is_correct`は食い違いうる。
- 現在のモデルだけで予測する経路では、混ぜる重みのownerを受け取らない（設定の検査が組合せを拒否する。関数は、その経路で受け取ったら`ValueError`）。
- 混ぜる重みのownerは、写しのownerと一緒にだけ渡す（無ければ`ValueError`）。

### client
- 構成子へkeyword引数`synchronized_and_local_prediction_mixture`、属性。`assemble_fedsda_run_client`で、`share_rate=1/T`（Fixed-Shareの時間尺度）で作る。
- 写しを取る条件と、標本の処理へ写しのownerを渡す条件を、「`parameters_at_last_synchronization`か、混ぜる値」にする。混ぜる重みのownerは、混ぜる値のときだけ渡す。

### 構成
`SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME = "fedsda-synchronized-local-mixture-prediction"`。一覧の末尾。組み立ては、`_replace_prediction_combination_settings(prediction_model_parameter_source="mixture_of_synchronized_and_local_parameters")`。

## Validation Design
`tests/refactoring/test_fedsda_synchronized_local_mixture_prediction.py`:
1. owner: 最初は半々。更新の式（手計算の値と一致）、分け合いで片方が0にならない、`share_rate`の範囲外の拒否。
2. 設定: 値の受理、`current_training_model_prediction`との組合せの拒否。構成のrun設定が、提案構成と予測に使うパラメータの項目だけ違う。帰属の保留の容量の軸（例: 60）と組み合わせた計画で、混ぜる値と容量60・時間尺度60のrun設定ができる。
3. 診断との一致（要求1.5）: 診断の差し替えは、testの中で書き直す。局所の確率は、元の予測の関数（写しのstate_dictを読み込む前）より前に、現在の分類器で計算する（診断と同じ順）。混ぜる重みの値も、標本ごとに実装のownerの値と一致することを確かめる。小さい条件（同期の時点の予測の構成のtestと同じ条件と、帰属の保留の容量60の条件）で、新しい構成のrunの各標本の`combined_prediction_is_correct`と、同期の時点の予測の構成に診断の差し替え（同じ混ぜ方）を当てたrunの、混ぜた予測の正否が一致する。混ぜた確率の側の2項目（`combined_prediction_is_correct`・`any_model_or_combined_prediction_is_correct`）は、オラクルの混ぜた確率の予測の正否（と、同期の側のモデルごとの正否の和）と比べ、それ以外の記録の項目は、同期の時点の予測の構成のrunと一致する。AdaHedgeの重みで予測する構成にも、同じ照合を1つ置く。
4. 検出力: 同期の時点の予測の構成とは、`combined_prediction_is_correct`が少なくとも1標本で違う（違いが出る小さい条件を、実装の前に下調べで選ぶ）。
5. 型と組合せ: 混ぜる重みのownerだけを渡す、ほかの型を渡す、現在のモデルだけの経路で渡すと、状態を変える前に拒否する。

既存のtest: 同期の時点の予測の構成のtest、goldenの版1〜5、構成の一覧を使うtest、依存境界、全pytest。WSLで、新しいtestとgoldenの版1〜5。

## 旧と違う点
既存の構成の動作・値・識別子は変わらない（設定の項目は増えず、値が増えるだけ）。
