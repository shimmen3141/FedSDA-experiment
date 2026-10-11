# Design Document: prediction-probability-validation-reuse

## Overview

**Purpose**: 標本1件の予測の中で、同じ確率のtensorを、何度も検査し直すのをやめて、実行を速くする。

**Impact**: `learning/prediction/class_probability_calculations.py`の3つの関数へ、keywordの引数を足す。`runtime/observed_sample_prediction.py`が、それを使う。

## 変更

| 関数 | 足す引数（既定は偽） | 真のときに省くもの | 真でも行うもの |
|---|---|---|---|
| `combine_model_prediction_probabilities` | `prediction_probabilities_are_validated` | モデル別の確率のtensorの中身の検査 | クラス数、モデルIDの型、全モデルの標本数の一致、重み、確率と重みのIDの一致 |
| `compute_model_mean_bounded_losses_after_label_observation` | `prediction_probabilities_are_validated` | 同上 | クラス数、モデルIDの型、全モデルの標本数の一致、観測ラベル |
| `predict_class_labels_from_prediction_scores` | `prediction_scores_are_validated` | 予測のtensorの中身の検査 | クラス数 |

- 引数は、builtin boolだけを受け入れる（それ以外は`TypeError`）。
- 真にしてよいのは、呼出し側が、同じtensorを、このmoduleの関数で検査した場合（`convert_model_outputs_to_prediction_probabilities`の戻り値）と、検査済みの確率と、検査済みの重み（`normalize_model_prediction_weights`の戻り値）から、`combine_model_prediction_probabilities`で作った結合の結果を、予測クラスの決定へ渡す場合。

### 標本1件の予測での使い方

1. 保有モデルの出力を、`convert_model_outputs_to_prediction_probabilities`で、モデル別の確率にする（出力と、変換後の確率の検査は、ここだけ。不正なら、状態を変える前に拒否する。これまでと同じ）。
2. その確率を、損失の計算、モデルごとの予測クラス、3通りの結合へ、「検査済み」として渡す。
3. 結合の結果（検査済みの確率の、検査済みの重みでの、凸結合）を、予測クラスの決定へ、「検査済み」として渡す。

変換の結果が、検査を通る値であること: 2値は、検査した確率の写し。多クラスは、変換の中で、変換後の確率（softmaxの結果）も、確率の条件（有限、0〜1、行の和）で検査してから返す（独立レビューの指摘: 変更の前は、変換は、logitだけを検査し、変換後の確率は、後段の関数が検査していた。後段の検査を省くので、変換の中へ移した）。

## 検証の方針

- 関数: 引数が真のとき、結果が、偽のときと同じ（2値・多クラス）。真のとき、tensorの検査の関数が呼ばれない。重み・ラベル・クラス数の不正は、真でも拒否する。boolでない値を拒否する。
- 標本1件の予測: tensorの検査の回数が、標本1件につき、保有モデルの数と同じ。実旧の予測との照合と、不正な出力の拒否のtestが、そのまま成功する。
- 全体runの照合（goldenの3ケース、小さい条件）と、全pytest。速度の実測。
